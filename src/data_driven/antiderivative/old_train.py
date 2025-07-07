import argparse
import json
import logging
import os
import random
import secrets
from dataclasses import dataclass, field
from typing import Optional

import deepxde as dde
import numpy as np
import torch
import yaml

from src.classical_orthogonal_deeponet import OrthoONetCartesianProd
from src.utils.common import apply_overrides


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


# --- Training function ---
def train_model(
    config: Config,
    data_dir: str,
    model_dir: str,
    seed: int,
    lr: float,
    iterations: int,
    bootstrap: bool = False
):
    """Train a single DeepONet model."""
    set_seeds(seed)

    # Load dataset
    x_train_full, y_train_full, x_val, y_val, x_test, y_test, _ = load_dataset(data_dir)
    x_cal, y_cal = load_calibration_dataset(data_dir)

    # Normalize input bounds for both branch and trunk inputs
    bounds = normalize_bounds(x_train_full, x_test, x_val, x_cal)

    # Optional bootstrapping of training samples
    if bootstrap:
        n_train = y_train_full.shape[0]
        indices = np.random.choice(n_train, n_train, replace=True)
        x_train = (x_train_full[0][indices], x_train_full[1])
        y_train = y_train_full[indices]
    else:
        x_train = x_train_full
        y_train = y_train_full


    # Normalize data to [-1, 1]
    x_train = (
        transform_input(x_train[0], bounds["branch_min"], bounds["branch_max"]),
        transform_input(x_train[1], bounds["trunk_min"], bounds["trunk_max"])
    )
    x_test = (
        transform_input(x_test[0], bounds["branch_min"], bounds["branch_max"]),
        transform_input(x_test[1], bounds["trunk_min"], bounds["trunk_max"])
    )

    # Prepare data loader
    data = dde.data.TripleCartesianProd(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)

    # Model definition
    m = x_train[0].shape[1]
    dim_x = x_train[1].shape[1]
    layer_sizes_branch = [m, config.branch_hidden, config.shared_output]
    layer_sizes_trunk = [dim_x, config.trunk_hidden, config.shared_output]

    net = OrthoONetCartesianProd(
        layer_sizes_branch=layer_sizes_branch,
        layer_sizes_trunk=layer_sizes_trunk,
        activation="silu"
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=["mean l2 relative error"])

    # Train
    losshistory, train_state = model.train(iterations=iterations, disregard_previous_best=True)

    # Save outputs
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model_checkpoint"))
    dde.utils.external.save_loss_history(losshistory, os.path.join(model_dir, "loss_history.txt"))

    # Save weights
    for name, param in model.net.named_parameters():
        np.savetxt(os.path.join(model_dir, f"{name}.txt"), param.cpu().detach().numpy())

    # Save training config and seed
    with open(os.path.join(model_dir, "seed.txt"), "w") as f:
        f.write(str(seed))
    with open(os.path.join(model_dir, "config_used.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)


# --- Main entry point ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default", help="Config file name")
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")
    args = parser.parse_args()
    args.config = os.path.join("configs/training", args.config + ".yaml")
    config = load_config(args.config)
    apply_overrides(config, args.override)

    data_path = os.path.join("data", config.data_dir)

    if config.ensemble > 0:
        base_seed = config.seed
        ensemble_name = config.ensemble_name or f"ensemble_seed{base_seed}"
        ensemble_dir = os.path.join("models", "ensembles", ensemble_name)

        for i in range(config.ensemble):
            seed_i = base_seed if i == 0 else secrets.randbits(32)
            model_name = f"model{i}"
            model_dir = os.path.join(ensemble_dir, model_name)
            logging.info(f"Training ensemble model {i+1}/{config.ensemble} (seed={seed_i})...")
            train_model(config, data_path, model_dir, seed_i, config.lr, config.iterations, config.bootstrap)
    else:
        model_name = config.model_name or f"seed{config.seed}"
        model_dir = os.path.join("models", model_name)
        logging.info(f"Training single model (seed={config.seed})...")
        train_model(config, data_path, model_dir, config.seed, config.lr, config.iterations, config.bootstrap)


def transform_input(x, min_val, max_val):
    d = x.shape[1]
    x = 2 * (x - min_val) / (max_val - min_val) - 1
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x ** 2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)


def normalize_bounds(x_train, x_test, x_val, x_cal):
    def get_min_max(idx):
        arrays = [x_train[idx], x_test[idx], x_val[idx], x_cal[idx]]
        concatenated = np.concatenate(arrays, axis=0)
        return np.min(concatenated, axis=0), np.max(concatenated, axis=0)

    branch_min, branch_max = get_min_max(0)
    trunk_min, trunk_max = get_min_max(1)

    return {
        "branch_min": branch_min,
        "branch_max": branch_max,
        "trunk_min": trunk_min,
        "trunk_max": trunk_max,
    }


def load_dataset(self):
    train = np.load((self + '/picked_aligned_train.npz'), allow_pickle=True)
    val = np.load((self + '/picked_aligned_val.npz'), allow_pickle=True)
    test = np.load((self + '/picked_aligned_test.npz'), allow_pickle=True)

    return (train['X0'].astype(np.float32), train['X1'].astype(np.float32)), train['y'].astype(np.float32), \
        (val['X0'].astype(np.float32), val['X1'].astype(np.float32)), val['y'].astype(np.float32), \
        (test['X0'].astype(np.float32), test['X1'].astype(np.float32)), test['y'].astype(np.float32), \
        test['X0_plot'].astype(np.float32)

def load_calibration_dataset(self):
    cal = np.load((self + '/picked_aligned_calibration.npz'), allow_pickle=True)

    return (cal['X0'].astype(np.float32), cal['X1'].astype(np.float32)), cal['y'].astype(np.float32)

if __name__ == "__main__":
    main()


