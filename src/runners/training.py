import argparse
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import inspect
import json
import logging
from pathlib import Path
import random
import secrets
import time
from typing import Optional, Tuple, List, Dict, Type

import deepxde as dde
import numpy as np
import torch
import yaml
from deepxde.nn.pytorch import DeepONetCartesianProd

from src.model_definition.classical_orthogonal_deeponet import OrthoONetCartesianProd
from src.model_definition.classical_res_ortho_deeponet import ResONetCartesianProd, ResOrthoONetCartesianProd
from src.utils.common import apply_overrides
from src.utils.data_handling import DataHandler
from src.utils.training import create_decay_and_hold_scheduler, LRLogger, plot_training_inputs, plot_model_outputs


# --- Configuration ---

class ModelType(Enum):
    """Enum for supported model architectures to prevent typos."""
    ORTHO_ONET = "OrthoONet"
    RES_ORTHO_ONET = "ResOrthoONet"
    DEEP_ONET = "DeepONet"
    RES_ONET = "ResONet"


class LossFunc(Enum):
    """Enum for loss functions to prevent typos"""
    MSE = "MSE"
    L2_rel = "mean l2 relative error"


@dataclass
class Config:
    """Configuration schema for the training script."""
    # Data options
    data_dir: str = "antiderivative"
    fourier_features: bool = False
    online: bool = False
    bootstrap: bool = False

    # Model/ensemble options
    model_name: Optional[str] = None
    ensemble_size: int = 0
    ensemble_name: Optional[str] = None
    ensemble_member: Optional[int] = None
    model_type: ModelType = ModelType.ORTHO_ONET
    orthogonal_implementation: str = "optimized"

    # Training hyperparameters
    iterations: int = 30_000
    lr: float = 0.001
    batch_size: int = 512
    loss: LossFunc = LossFunc.MSE

    # Layer sizes
    layers: List[int] = field(default_factory=lambda: [10, 10])

    # Scheduler options
    decay_gamma: float = 0.996
    minimum_lr: float = 5e-5

    # Reproducibility and Debugging
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    verbose: bool = False
    display_every: int = 1_000
    allow_tf32: bool = True

    def __post_init__(self):
        """Ensure string values from YAML/overrides are converted to Enum."""
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)

        if isinstance(self.loss, str):
            self.loss = LossFunc(self.loss)

        if self.ensemble_member is not None:
            if self.ensemble_size <= 0:
                raise ValueError("ensemble_member requires ensemble_size > 0")
            if not 0 <= self.ensemble_member < self.ensemble_size:
                raise ValueError("ensemble_member must be in [0, ensemble_size)")
        if self.orthogonal_implementation not in {"optimized", "legacy"}:
            raise ValueError("orthogonal_implementation must be 'optimized' or 'legacy'")


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Utility functions ---

def set_seeds(seed: int):
    """Set all relevant seeds for reproducibility."""
    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --- Training Workflow Class ---

def _get_network_map() -> dict[ModelType, Type[OrthoONetCartesianProd | ResOrthoONetCartesianProd |
                                               DeepONetCartesianProd | ResONetCartesianProd]]:
    """Returns a mapping from ModelType Enum to network classes."""
    return {
        ModelType.ORTHO_ONET: OrthoONetCartesianProd,
        ModelType.RES_ORTHO_ONET: ResOrthoONetCartesianProd,
        ModelType.DEEP_ONET: dde.nn.pytorch.DeepONetCartesianProd,
        ModelType.RES_ONET: ResONetCartesianProd,
    }


class TrainingRunner:
    """Encapsulates the entire model training workflow."""

    def __init__(self, config: Config):
        """
        Initializes the TrainingRunner.

        Args:
            config (Config): The main configuration object.
        """
        self.config = config
        if config.orthogonal_implementation == "legacy":
            from src.legacy.classical_orthogonal_layer import LegacyOrthoLayer
            import src.model_definition.classical_orthogonal_NN as ortho_module
            import src.model_definition.classical_res_ortho_deeponet as residual_module

            ortho_module.OrthoLayer = LegacyOrthoLayer
            residual_module.OrthoLayer = LegacyOrthoLayer
        torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.allow_tf32
        self.data_handler = self._setup_data_handler()
        self.data_manifest = self._build_data_manifest()
        self.base_output_dir = self._setup_output_directory()
        self.network_map = _get_network_map()

    def run(self):
        """Main execution method to start the training process."""
        if self.config.ensemble_size > 0:
            self._run_ensemble()
        else:
            self._run_single_model()

    def _setup_data_handler(self) -> DataHandler:
        """Initializes the DataHandler and loads data."""
        handler = DataHandler(
            data_dir=self.config.data_dir,
            fourier_features=self.config.fourier_features,
            online=self.config.online
        )
        handler.load_and_process_data()
        return handler

    def _setup_output_directory(self) -> Path:
        """Determines and creates the base output directory for the run."""
        if self.config.ensemble_size > 0:
            name = self.config.ensemble_name or f"ensemble_seed{self.config.seed}"
            output_dir = Path("models", "ensembles", name)
        else:
            name = self.config.model_name or f"model_seed{self.config.seed}"
            output_dir = Path("models", name)

        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Base output directory set to: {output_dir}")
        return output_dir

    def _build_data_manifest(self) -> Dict:
        """Fingerprint the exact processed splits used by this training run."""
        manifest = {"data_dir": self.config.data_dir, "splits": {}}
        for split, dataset in self.data_handler.datasets.items():
            source = self.data_handler.data_path / f"{split}.npz"
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest["splits"][split] = {
                "source": source.as_posix(),
                "sha256": digest.hexdigest(),
                "branch_shape": list(dataset["X"][0].shape),
                "trunk_shape": list(dataset["X"][1].shape),
                "target_shape": list(dataset["y"].shape),
            }
        return manifest

    def _run_single_model(self):
        """Trains a single model instance."""
        logging.info(f"--- Training single model (seed={self.config.seed}) ---")
        self._train_one_instance(self.base_output_dir, self.config.seed)

    def _run_ensemble(self):
        """Train all members serially, or one member for fleet parallelism."""
        base_seed = self.config.seed
        indices = (
            [self.config.ensemble_member]
            if self.config.ensemble_member is not None
            else range(self.config.ensemble_size)
        )
        for i in indices:
            seed_i = base_seed if i == 0 else int(
                np.random.SeedSequence([base_seed, i]).generate_state(1, dtype=np.uint32)[0]
            )
            model_dir = self.base_output_dir / f"model_{i}"
            logging.info(f"--- Training ensemble model {i + 1}/{self.config.ensemble_size} (seed={seed_i}) ---")
            self._train_one_instance(model_dir, seed_i)

    def _train_one_instance(self, model_dir: Path, seed: int):
        """The core logic to train and save a single model instance."""
        set_seeds(seed)
        model_dir.mkdir(parents=True, exist_ok=True)

        x_train, y_train = self._bootstrap_data()
        dde_data, plot_data = self._prepare_dde_data(x_train, y_train)

        if self.config.verbose:
            plot_training_inputs(
                x_train=plot_data['x_train'], y_train=plot_data['y_train'],
                model_dir=model_dir, x_train_plot=plot_data['x_train_plot']
            )

        model = self._create_and_compile_model(dde_data)

        callbacks = [LRLogger(display_every=self.config.display_every)] if self.config.decay_gamma else []

        training_start = time.perf_counter()
        losshistory, _ = model.train(

            iterations=self.config.iterations,
            display_every=self.config.display_every,
            callbacks=callbacks,
            batch_size=self.config.batch_size
        )
        training_seconds = time.perf_counter() - training_start

        if self.config.verbose:
            plot_model_outputs(
                model=model, x_test=plot_data['x_test'], y_test=plot_data['y_test'],
                model_dir=model_dir, x_test_plot=plot_data['x_test_plot']
            )

        self._save_artifacts(model, losshistory, model_dir, seed)
        timing = {
            "training_seconds": training_seconds,
            "iterations": self.config.iterations,
            "iterations_per_second": self.config.iterations / training_seconds,
            "orthogonal_implementation": self.config.orthogonal_implementation,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        with (model_dir / "timing.json").open("w") as handle:
            json.dump(timing, handle, indent=2)

    def _bootstrap_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Applies bootstrapping to the training data if configured."""
        x_train_full, y_train_full = self.data_handler.get_split('train')
        if self.config.bootstrap:
            logging.info("Bootstrapping dataset...")
            n_train = y_train_full.shape[0]
            indices = np.random.choice(n_train, n_train, replace=True)
            # Bootstrap branch and output, but keep trunk fixed
            x_train = (x_train_full[0][indices], x_train_full[1])
            y_train = y_train_full[indices]
            return x_train, y_train
        return x_train_full, y_train_full

    def _prepare_dde_data(self, x_train, y_train) -> Tuple[dde.data.Data, Dict]:
        """
        Prepares data for DeepXDE, handling reshaping for online/offline cases.
        """
        x_test, y_test = self.data_handler.get_split('test')

        x_train_plot = self.data_handler.datasets['train']['X0_plot']
        x_test_plot = self.data_handler.datasets['test']['X0_plot']

        if self.config.online:
            # For online/autoregressive data, each window is a sample.
            # Reshape from (signals, windows, features) to (total_windows, features).
            x_train = (x_train[0].reshape(-1, x_train[0].shape[-1]), x_train[1].reshape(-1, x_train[1].shape[-1]))
            y_train = y_train.reshape(-1, y_train.shape[-1])

            x_test = (x_test[0].reshape(-1, x_test[0].shape[-1]), x_test[1].reshape(-1, x_test[1].shape[-1]))
            y_test = y_test.reshape(-1, y_test.shape[-1])

            dde_data = dde.data.Triple(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
            plot_data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
                         'x_train_plot': x_train_plot.reshape(-1, x_train_plot.shape[-1]),
                         'x_test_plot': x_test_plot.reshape(-1, x_test_plot.shape[-1])}
        else:
            dde_data = dde.data.TripleCartesianProd(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
            plot_data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
                         'x_train_plot': x_train_plot, 'x_test_plot': x_test_plot}

        return dde_data, plot_data

    def _create_network(self, m: int, dim_x: int) -> torch.nn.Module:
        """
        Creates a neural network instance based on the config.
        """
        layer_sizes_branch = [m] + [int(layer) for layer in self.config.layers]
        layer_sizes_trunk = [dim_x] + [int(layer) for layer in self.config.layers]
        logging.info(f"Branch layers: {layer_sizes_branch}, Trunk layers: {layer_sizes_trunk}")

        network_class = self.network_map.get(self.config.model_type)
        if not network_class:
            raise ValueError(f"Invalid model type: {self.config.model_type}")

        # Prepare arguments for the network class
        # Use inspect to pass only the arguments the specific class constructor accepts
        sig = inspect.signature(network_class.__init__)
        params = {}
        if 'layer_sizes_branch' in sig.parameters: params['layer_sizes_branch'] = layer_sizes_branch
        if 'layer_sizes_trunk' in sig.parameters: params['layer_sizes_trunk'] = layer_sizes_trunk
        if 'activation' in sig.parameters: params['activation'] = 'silu'
        if 'online' in sig.parameters: params['online'] = self.config.online
        if 'kernel_initializer' in sig.parameters: params['kernel_initializer'] = "He uniform"

        return network_class(**params)

    def _create_and_compile_model(self, data: dde.data) -> dde.Model:
        """Initializes the network, creates the dde.Model, and compiles it."""
        m = data.train_x[0].shape[1]
        dim_x = data.train_x[1].shape[1]

        net = self._create_network(m, dim_x)
        model = dde.Model(data, net)

        # Compile with or without LR scheduler
        if self.config.decay_gamma and self.config.minimum_lr:
            scheduler_fn = create_decay_and_hold_scheduler(
                initial_lr=self.config.lr, gamma=self.config.decay_gamma, min_lr=self.config.minimum_lr
            )
            model.compile("adam", lr=self.config.lr, metrics=["mean l2 relative error"],
                          decay=("lambda", scheduler_fn), loss=self.config.loss.value)
        else:
            model.compile("adam", lr=self.config.lr, metrics=["mean l2 relative error"],
                          loss=self.config.loss.value)

        return model

    def _save_artifacts(self, model: dde.Model, losshistory, model_dir: Path, seed: int):
        """Saves all model and config files for a training run."""
        model.save(str(model_dir / "model_checkpoint"))
        dde.utils.external.save_loss_history(losshistory, str(model_dir / "loss_history.txt"))

        if self.config.model_type in [ModelType.ORTHO_ONET, ModelType.RES_ORTHO_ONET]:
            for name, param in model.net.named_parameters():
                np.savetxt(model_dir / f"{name}.txt", param.cpu().detach().numpy())

        (model_dir / "seed.txt").write_text(str(seed))
        with (model_dir / "config_used.json").open("w") as f:
            # Use dataclasses.asdict for clean serialization
            json.dump(asdict(self.config), f, indent=4, default=str)
        with (model_dir / "data_manifest.json").open("w") as handle:
            json.dump(self.data_manifest, handle, indent=2)
        logging.info(f"Model and artifacts saved to {model_dir}")


# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="DeepONet Training Script")
    parser.add_argument("--config", type=str, default="default_antiderivative",
                        help="Config file name in configs/training")
    parser.add_argument("--override", nargs='*', help="Overrides in key=value format (e.g., lr=0.005 iterations=50000)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config_path = Path("configs/training") / f"{args.config}.yaml"
    if not config_path.exists():
        logging.error(f"Configuration file not found at {config_path}")
        return

    config = load_config(str(config_path))

    if args.override:
        apply_overrides(config, args.override)

    runner = TrainingRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
