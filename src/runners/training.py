import argparse
import json
import logging
import os
import inspect
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import deepxde as dde
import numpy as np
import torch
import yaml

from src.model_definition.classical_orthogonal_deeponet import OrthoONetCartesianProd
from src.model_definition.classical_res_ortho_deeponet import ResONetCartesianProd, ResOrthoONetCartesianProd
from src.utils.common import apply_overrides
from src.utils.data_handling import DataHandler
from src.utils.training import create_decay_and_hold_scheduler, LRLogger, model_input_plotting, model_output_plotting


# --- Config and logging ---

@dataclass
class Config:
    """Configuration schema for training."""

    # Data options
    bootstrap: bool = False
    data_dir: str = "antiderivative"
    fourier_features: bool = False

    # Model/ensemble options
    model_name: str = None
    ensemble: int = 0
    ensemble_name: str = None

    # Training hyperparamters
    iterations: int = 30_000
    lr: float = 0.001
    display_every: int = 1_000
    model_type: str = "OrthoONet"
    batch_size: int = 512
    online: bool = False

    # Layer sizes
    layers: List[int] = field(default_factory=lambda: [10, 10])

    # Scheduler
    decay_gamma: float = 0.996
    minimum_lr: float = 5e-5

    seed: int = field(default_factory=lambda: secrets.randbits(32))
    verbose: bool = False


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# --- Utility functions ---

def set_seeds(seed: int):
    """Set all relevant seeds for reproducibility."""
    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --- Training Workflow Class ---

class TrainingRunner:
    """Encapsulates the entire model training workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.data_handler = DataHandler(config.data_dir, fourier_features=self.config.fourier_features)

        if self.config.ensemble > 0:
            name = self.config.ensemble_name or f"ensemble_seed{self.config.seed}"
            self.base_output_dir = Path("models", "ensembles", name)
        else:
            name = self.config.model_name or f"model_seed{self.config.seed}"
            self.base_output_dir = Path("models", name)

        logging.info(f"Base output directory set to: {self.base_output_dir}")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Main execution method to start the training process."""
        if self.config.ensemble > 0:
            self._run_ensemble()
        else:
            self._run_single_model()
        logging.info("Training complete.")

    def _run_single_model(self):
        """Trains a single model instance."""
        logging.info(f"Training single model (seed={self.config.seed})...")
        self._train_one_instance(self.base_output_dir, self.config.seed)

    def _run_ensemble(self):
        """Trains an ensemble of models, each with a different seed."""
        base_seed = self.config.seed
        for i in range(self.config.ensemble):
            # Use the base seed for the first model, then random seeds for the rest
            seed_i = base_seed if i == 0 else secrets.randbits(32)
            model_dir = self.base_output_dir / f"model{i}"

            logging.info(f"Training ensemble model {i + 1}/{self.config.ensemble} (seed={seed_i})...")
            self._train_one_instance(model_dir, seed_i)

    def _train_one_instance(self, model_dir: Path, seed: int):
        """
        The core logic to train and save a single model instance.
        """
        set_seeds(seed)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Use pre-loaded and pre-normalized data from the handler
        x_train_full, y_train_full = self.data_handler.x_train, self.data_handler.y_train

        # Optional bootstrapping
        if self.config.bootstrap:
            logging.info("Bootstrapping dataset")
            n_train = y_train_full.shape[0]
            indices = np.random.choice(n_train, n_train, replace=True)
            x_train = (x_train_full[0][indices], x_train_full[1])  # Fixed trunk bootstrapping
            y_train = y_train_full[indices]
        else:
            x_train, y_train = x_train_full, y_train_full

        # DataHandler object transformers data, so no transformation is required
        x_test, y_test = self.data_handler.x_test, self.data_handler.y_test

        model = self._setup_dde_model(x_train, y_train, x_test, y_test, model_dir)

        # Create an instance of the logger if scheduler exists
        lr_logger_callback = None
        if self.config.decay_gamma and self.config.minimum_lr:
            # Set display every in case user forgets
            if self.config.display_every is None:
                self.config.display_every = 1_000
            lr_logger_callback = LRLogger(display_every=self.config.display_every)

        # Train
        losshistory, _ = model.train(
            iterations=self.config.iterations,
            disregard_previous_best=True,
            display_every=self.config.display_every,
            callbacks=[lr_logger_callback] if lr_logger_callback is not None else None,
            batch_size=self.config.batch_size
        )

        if self.config.verbose:
            model_output_plotting(model, x_test, y_test, model_dir, self.data_handler.x_test_plot)

        # Save all results
        self._save_artifacts(model, losshistory, model_dir, seed)

    def _setup_dde_model(self, x_train: Tuple, y_train: np.ndarray, x_test: Tuple, y_test: np.ndarray,
                         model_dir: Path) -> dde.Model:
        """Boilerplate for setting up the DeepXDE model."""

        # Identify if dataset is for autoregression
        data = None
        if x_train[0].shape[0] == x_train[1].shape[0]:
            data = dde.data.Triple(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
        else:
            data = dde.data.TripleCartesianProd(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)

        if self.config.verbose:
            # Make directory to store plots
            os.makedirs(model_dir / "training_plots", exist_ok=True)

            # x_train_plot is used for the x ticks
            model_input_plotting(x_train, y_train, model_dir, self.data_handler.x_train_plot)

        m = x_train[0].shape[1]
        dim_x = x_train[1].shape[1]

        # Ensure layer size isn't a float
        self.config.layers = [int(layer) for layer in self.config.layers]

        layer_sizes_branch = [m] + self.config.layers
        layer_sizes_trunk = [dim_x] + self.config.layers

        logging.info(f"Layers for branch: {layer_sizes_branch}")
        logging.info(f"Layers for trunk: {layer_sizes_trunk}")

        net = None
        if self.config.model_type == "OrthoONet":
            net = OrthoONetCartesianProd(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation='silu',
                online=self.config.online
            )
        elif self.config.model_type == "ResOrthoONet":
            net = ResOrthoONetCartesianProd(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation="silu",
                online=self.config.online
            )
        elif self.config.model_type == "DeepONet":
            net = dde.nn.pytorch.DeepONetCartesianProd(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation="silu",
                kernel_initializer="He uniform"
            )
        elif self.config.model_type == "ResONet":
            net = ResONetCartesianProd(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation="silu",
                online=self.config.online
            )
        else:
            logging.error("Invalid model type provided. Please specify from 'OrthONet', 'ResOrthoONet', 'DeepONet' or "
                          "'ResONet'")
            exit(1)

        model = dde.Model(data, net)

        # Scheduler
        has_scheduler = False
        if self.config.decay_gamma and self.config.minimum_lr:
            has_scheduler = True
        elif self.config.decay_gamma or self.config.minimum_lr:
            logging.error("If creating scheduler, please specify both decay gamma and minimum lr")
            exit(1)

        if has_scheduler:
            custom_scheduler_fn = create_decay_and_hold_scheduler(
                initial_lr=self.config.lr,
                gamma=self.config.decay_gamma,
                min_lr=self.config.minimum_lr
            )
            decay_schedule = ("lambda", custom_scheduler_fn)
            model.compile("adam", lr=self.config.lr, metrics=["mean l2 relative error"], decay=decay_schedule)
        else:
            model.compile("adam", lr=self.config.lr, metrics=["mean l2 relative error"])

        return model

    def _save_artifacts(self, model: dde.Model, losshistory, model_dir: Path, seed: int):
        """File-saving operations."""
        # Save DDE-specific files
        model.save(str(model_dir / "model_checkpoint"))
        dde.utils.external.save_loss_history(losshistory, str(model_dir / "loss_history.txt"))

        # Save weights as plain text for easy loading in simulation
        for name, param in model.net.named_parameters():
            np.savetxt(model_dir / f"{name}.txt", param.cpu().detach().numpy())

        # Save the seed and config for reproducibility
        (model_dir / "seed.txt").write_text(str(seed))
        with open(model_dir / "config_used.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=4)
        logging.info(f"Model and artifacts saved to {model_dir}")


# --- Main Entry Point ---

def main():
    """Parses arguments and starts the training runner."""
    parser = argparse.ArgumentParser(description="DeepONet Training Script")
    parser.add_argument("--config", type=str, default="default_antiderivative", help="Config file name in configs/training")
    parser.add_argument("--override", nargs='*', help="Overrides in key=value format (e.g., lr=0.005 iterations=50000)")
    args = parser.parse_args()

    config_path = Path("configs/training") / f"{args.config}.yaml"
    if not config_path.exists():
        logging.error(f"Configuration file not found at {config_path}")
        return

    config = load_config(str(config_path))
    apply_overrides(config, args.override)

    runner = TrainingRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
