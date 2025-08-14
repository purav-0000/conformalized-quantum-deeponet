import argparse
import logging
import os
import secrets
from dataclasses import dataclass, field

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils.common import apply_overrides

# --- Config Dataclass ---

@dataclass
class Config:

    # DeepONet parameters
    n_sensors_branch: int = 20
    n_sensors_trunk: int = 50

    # Splits
    train: int = 1000
    test: int = 200
    calibration: int = 200

    # Geenration parameters
    nx: int = 201
    nt: int = 20001
    xmax: float = 1.0
    tmax: float = 1.0
    grf_periodicity: float = 1.0
    length_scale: float = 1.5
    interp: str = "cubic"

    # Misc
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    n_jobs: int = 4
    output_dir: str = "data/processed_data/advection"

# --- Numerical Solver ---

def numerical_solver(u0, nx, nt, xmax=1.0, tmax=1.0):
    """
    Solves the 1D advection equation u_t + u_x = 0 using an upwind scheme
    with periodic boundary conditions.
    """
    dt = tmax / (nt - 1)
    dx = xmax / (nx - 1)

    # Initialize solution array
    u = np.zeros((nx - 1, nt))
    u[:, 0] = u0[:-1]  # Set initial condition, excluding the last point due to periodicity

    # Create the differentiation matrix for the upwind scheme (v*u_x)
    I = np.eye(nx - 1)
    I1 = np.roll(I, 1, axis=0)
    A = (I - I1) / dx

    # Time-stepping loop
    for n in range(nt - 1):
        u[:, n + 1] = u[:, n] - dt * np.dot(A, u[:, n])

    # Re-apply periodic boundary by concatenating the first row to the end
    u = np.concatenate([u, u[0:1, :]], axis=0)
    return u


# --- Core Data Generation Functions ---

def generate_sample(config: Config):
    """Generates a single sample of initial conditions and its corresponding solution."""
    space = dde.data.GRF(
        T=config.grf_periodicity,
        kernel="ExpSineSquared",
        length_scale=config.length_scale,
        N=config.nx,
        interp=config.interp,
    )
    # Generate one random function for the initial condition u0(x)
    u0 = space.random(1)[0]
    # Solve the advection equation for this initial condition
    s = numerical_solver(u0, config.nx, config.nt, config.xmax, config.tmax)
    return u0, s


def generate_and_save_split(split_name: str, config: Config):
    """Generates and saves a full data split (train, test, or calibration)."""
    num_samples = getattr(config, split_name)

    # Generate all samples in parallel
    results = Parallel(n_jobs=config.n_jobs)(
        delayed(generate_sample)(config)
        for _ in tqdm(range(num_samples), desc=f"Generating '{split_name}' data")
    )

    u0_all, s_all = zip(*results)
    u0_all = np.array(u0_all, dtype=np.float32)  # Shape: (num_samples, nx)
    s_all = np.array(s_all, dtype=np.float32)  # Shape: (num_samples, nx, nt)

    # Define spatial and temporal grids for the full solution
    x_full = np.linspace(0, config.xmax, config.nx)
    t_full = np.linspace(0, config.tmax, config.nt)

    # --- Create datasets for DeepONet ---
    # The branch net input `X0` is the initial condition u0(x) at sensor locations.
    # The trunk net input `X1` is the (x, t) coordinate pairs.
    # The output `y` is the solution u(x, t) at those (x, t) locations.

    # 1. Define sensor locations for branch and trunk networks
    idx_branch = [round(config.nx / config.n_sensors_branch) * i for i in range(config.n_sensors_branch - 1)] + [
        config.nx - 1]
    idx_trunk_x = [round(config.nx / config.n_sensors_trunk) * i for i in range(config.n_sensors_trunk - 1)] + [
        config.nx - 1]
    idx_trunk_t = [round(config.nt / config.n_sensors_trunk) * i for i in range(config.n_sensors_trunk - 1)] + [
        config.nt - 1]

    # 2. Create branch input (subsampled initial conditions)
    X0 = u0_all[:, idx_branch]

    # 3. Create trunk input (grid of (x, t) sensor locations)
    x_trunk = x_full[idx_trunk_x]
    t_trunk = t_full[idx_trunk_t]
    xx, tt = np.meshgrid(x_trunk, t_trunk)
    X1 = np.vstack((np.ravel(tt), np.ravel(xx))).T

    # 4. Create output y (solution sampled at trunk locations)
    s_sampled = s_all[:, idx_trunk_x][:, :, idx_trunk_t]
    y = s_sampled.reshape(num_samples, -1)

    # --- Save Data ---
    os.makedirs(config.output_dir, exist_ok=True)
    save_path = os.path.join(config.output_dir, f"{split_name}.npz")
    np.savez_compressed(save_path, X0=X0, X1=X1, y=y, X0_plot=idx_branch)
    logging.info(f"Successfully saved '{split_name}' data to {save_path}")


# --- Helper Functions and Entry Point ---

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    secrets.randbits(128)  # Consume some randomness from secrets
    logging.info(f"Using random seed: {seed}")


def load_config(config_path: str) -> Config:
    """Loads a YAML configuration file into a Config object."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate datasets for the 1D Advection equation.")
    parser.add_argument("--config", type=str, default="default_advection",
                        help="Name of the config file in configs/data_generation")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run a small test and plot results without saving .npz files")
    parser.add_argument("--override", nargs='*', help="Override config values, e.g., train=2000 n_jobs=16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config_path = os.path.join("configs", "data_generation", args.config + ".yaml")
    config = load_config(config_path)

    # Apply overrides from command line
    apply_overrides(config, args.override)
    if args.seed is not None:
        config.seed = args.seed
    if args.dry_run:
        config.dry_run = True
        config.n_jobs = 1  # Force single-threaded execution for dry run

    set_seed(config.seed)

    for split in ["train", "test", "calibration"]:
        generate_and_save_split(split, config)


if __name__ == "__main__":
    main()