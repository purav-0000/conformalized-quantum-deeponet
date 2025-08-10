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
import deepxde.nn.pytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.classical_orthogonal_deeponet import OrthoONetCartesianProd
from src.classical_res_ortho_deeponet import ResONetCartesianProd, ResOrthoONetCartesianProd
from src.utils.common import apply_overrides
from src.utils.data_handling import DataHandler


# --- Config and logging ---

@dataclass
class Config:
    """Configuration schema for training."""
    bootstrap: bool = False
    data_dir: str = "data_ode_simple"
    ensemble: int = 0
    ensemble_name: Optional[str] = None
    iterations: int = 30000
    lr: float = 0.001
    model_name: Optional[str] = None
    seed: Optional[int] = field(default_factory=lambda: secrets.randbits(32))

    # Custom layer sizes
    branch_hidden: int = 10
    trunk_hidden: int = 10
    shared_output: int = 10  # Output layer size must be shared
    layers: List[int] = field(default_factory=lambda: [10, 10])



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


def create_decay_and_hold_scheduler(initial_lr, gamma, min_lr):
    """
    Scheduler that returns a multiplicative factor
    """
    # Calculate the minimum multiplicative factor that corresponds to the minimum LR
    min_factor = min_lr / initial_lr

    def scheduler(step):
        # Calculate the decay factor for the current step
        decay_factor = gamma ** step

        # Return the larger of the two factors: the decayed factor or the minimum factor
        return max(decay_factor, min_factor)

    return scheduler


class LRLogger(dde.callbacks.Callback):
    """A callback to print the learning rate."""

    def __init__(self, reps=50):
        super().__init__()
        # Print epoch every 'reps'
        self.reps = reps

    def on_epoch_end(self):
        current_epoch = self.model.train_state.epoch

        if current_epoch % self.reps == 0:
            current_lr = self.model.opt.param_groups[0]['lr']
            print(f"└─> [LR at Epoch {current_epoch}]: {current_lr:.6f}")


# --- Training Workflow Class ---

class TrainingRunner:
    """Encapsulates the entire model training workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.data_handler = DataHandler(config.data_dir)

        # NEW: Centralized path management
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
            n_train = y_train_full.shape[0]
            indices = np.random.choice(n_train, n_train, replace=True)
            x_train = (x_train_full[0][indices], x_train_full[1])  # Fixed trunk bootstrapping
            y_train = y_train_full[indices]
        else:
            x_train, y_train = x_train_full, y_train_full

        # DataHandler object transformers data, so no transformation is required
        x_test, y_test = self.data_handler.x_test, self.data_handler.y_test

        model = self._setup_dde_model(x_train, y_train, x_test, y_test)

        # Create an instance of the logger for epoch 50
        reps = 50
        lr_logger_callback = LRLogger(reps=reps)
        # Train
        losshistory, _ = model.train(iterations=self.config.iterations, disregard_previous_best=True, display_every=reps,
                                     callbacks=[lr_logger_callback])

        # Debugging and error analysis
        model.net.eval()

        logging.info("Starting error analysis...")

        # Plotting distribution of errors + worst and best predictions
        with torch.no_grad():
            # 1. Get predictions for the entire test set
            y_pred = model.predict(x_test)

            # 2. Calculate L2 relative error for each sample
            errors = (np.linalg.norm(y_pred - y_test, axis=1) / np.linalg.norm(y_test, axis=1))

            # 3. Plot the histogram of errors
            plt.figure(figsize=(10, 6))
            plt.hist(errors, bins=50, alpha=0.75)
            plt.title("Distribution of L2 Relative Errors on Test Set")
            plt.xlabel("L2 Relative Error")
            plt.ylabel("Number of Samples")
            plt.grid(True)
            plt.show()

            # 4. Find and plot the worst and best predictions
            sorted_indices = np.argsort(errors)
            worst_indices = sorted_indices[-3:]  # Top 3 worst
            best_indices = sorted_indices[:3]  # Top 3 best

            logging.info(f"Worst L2 errors: {errors[worst_indices]}")
            logging.info(f"Best L2 errors: {errors[best_indices]}")

            # Plot worst predictions
            for i, index in enumerate(worst_indices):
                plt.figure(figsize=(10, 6))
                plt.plot(x_test[1][:, 0], y_test[index], 'r-', label=f'Ground Truth (Error: {errors[index]:.2%})')
                plt.plot(x_test[1][:, 0], y_pred[index], 'b--', label='Prediction')
                plt.title(f"Worst Prediction #{i + 1} (Sample Index: {index})")
                plt.legend()
                plt.grid(True)
                plt.show()

            # Plot best predictions
            for i, index in enumerate(best_indices):
                plt.figure(figsize=(10, 6))
                plt.plot(x_test[1][:, 0], y_test[index], 'g-', label=f'Ground Truth (Error: {errors[index]:.2%})')
                plt.plot(x_test[1][:, 0], y_pred[index], 'b--', label='Prediction')
                plt.title(f"Best Prediction #{i + 1} (Sample Index: {index})")
                plt.legend()
                plt.grid(True)
                plt.show()

        # Plot 5 samples with prediction
        with torch.no_grad():
            for i in range(5):
                index = np.random.randint(len(x_test[0]))
                plt.figure(figsize=(10, 6))
                plt.plot(x_test[1][:, 0], y_test[index], c='purple', label=f'Ground Truth (Error: {errors[index]:.2%})')
                plt.plot(x_test[1][:, 0], y_pred[index], 'b--', label='Prediction')
                plt.title(f""
                          f"(Sample Index: {index})")
                plt.legend()
                plt.grid(True)
                plt.show()

        # Plot outputs of branch to check if similar
        for i in range(20):
            index = np.random.randint(len(x_train[0]))
            plt.plot(
                model.net.branch(torch.tensor(x_train[0][index:index + 1], dtype=torch.float32)).detach().cpu().squeeze().numpy(),
                label=f'func {i}')
        plt.title("Branch outputs per function")
        plt.legend()
        plt.savefig("Different_outputs.png")
        plt.show()

        # Save all results
        self._save_artifacts(model, losshistory, model_dir, seed)


    def _setup_dde_model(self, x_train: Tuple, y_train: np.ndarray, x_test: Tuple, y_test: np.ndarray) -> dde.Model:
        """Boilerplate for setting up the DeepXDE model."""
        data = dde.data.TripleCartesianProd(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)

        # Plotting 5 random data samples to ensure correct data is being fed to the model
        for i in range(5):
            index = np.random.randint(len(x_test[0]))
            plt.figure(figsize=(10, 6))
            plt.plot(np.linspace(0, 2.0, len(x_train[0][index])), x_train[0][index], 'r-', label=f'Branch input')
            plt.plot(x_test[1][:, 0], y_train[index], 'b--', label='Ground truth')
            plt.title(f"(Sample Index: {index})")
            plt.legend()
            plt.grid(True)
            plt.show()

        m = x_train[0].shape[1]
        dim_x = x_train[1].shape[1]

        layer_sizes_branch, layer_sizes_trunk = None, None
        if self.config.layers is None:
            layer_sizes_branch = [m, self.config.branch_hidden, self.config.shared_output]
            layer_sizes_trunk = [dim_x, self.config.trunk_hidden, self.config.shared_output]
        else:
            self.config.layers = [int(layer) for layer in self.config.layers]
            layer_sizes_branch = [m] + self.config.layers
            layer_sizes_trunk = [dim_x] + self.config.layers

        # Error checking
        print("Layers for branch:", layer_sizes_branch)
        print("Layers for trunk:", layer_sizes_trunk)


        net = ResOrthoONetCartesianProd(
            layer_sizes_branch=layer_sizes_branch,
            layer_sizes_trunk=layer_sizes_trunk,
            activation="silu",
        )

        """
        net = deepxde.nn.pytorch.DeepONetCartesianProd(
            layer_sizes_branch=layer_sizes_branch,
            layer_sizes_trunk=layer_sizes_trunk,
            activation="silu",
            kernel_initializer="Glorot uniform"
        )

        
        net = ResONetCartesianProd(
            layer_sizes_branch=layer_sizes_branch,
            layer_sizes_trunk=layer_sizes_trunk,
            activation="silu",
        )
        """
        model = dde.Model(data, net)

        # Scheduler parameters
        # Defined for the problem, not added to config yet
        DECAY_GAMMA = 0.998
        MINIMUM_LR = 3e-3

        custom_scheduler_fn = create_decay_and_hold_scheduler(
            initial_lr=self.config.lr,
            gamma=DECAY_GAMMA,
            min_lr=MINIMUM_LR
        )
        decay_schedule = ("lambda", custom_scheduler_fn)

        model.compile("adam", lr=self.config.lr, metrics=["mean l2 relative error"], decay=decay_schedule)
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
    parser.add_argument("--config", type=str, default="default", help="Config file name in configs/training")
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