import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
import secrets
from typing import List, Tuple

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import yaml

from src.utils.common import apply_overrides

# --- Constants ---

SPLITS: List[str] = ["train", "test", "calibration"]
OUTPUT_DIR = Path("data/processed_data/advection")


# --- Configuration ---

@dataclass
class Config:

    # DeepONet sensor parameters
    n_sensors_branch: int = 20
    n_sensors_trunk: int = 50

    # Data split sizes
    train: int = 1000
    test: int = 200
    calibration: int = 200

    # Numerical solver parameters
    nx: int = 201
    nt: int = 20001
    xmax: float = 1.0
    tmax: float = 1.0

    # GRF parameters
    grf_periodicity: float = 1.0
    length_scale: float = 1.5
    interp: str = "cubic"

    # Miscellaneous
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    n_jobs: int = 4


# --- Utilities ---

def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Numerical Solver ---

def numerical_solver(u0: np.ndarray, nx: int, nt: int, xmax: float, tmax: float) -> np.ndarray:
    """
    Solves the 1D advection equation u_t + u_x = 0 using an upwind scheme.

    This solver assumes periodic boundary conditions.

    Args:
        u0 (np.ndarray): The initial condition u(x, 0) of shape (nx,).
        nx (int): Number of spatial points.
        nt (int): Number of time steps.
        xmax (float): The spatial domain limit [0, xmax].
        tmax (float): The temporal domain limit [0, tmax].

    Returns:
        np.ndarray: The solution u(x, t) with shape (nx, nt).
    """
    dt = tmax / (nt - 1)
    dx = xmax / (nx - 1)
    u = np.zeros((nx - 1, nt))
    u[:, 0] = u0[:-1]  # Use all but the last point due to periodicity

    # Differentiation matrix for upwind scheme (for positive velocity c=1)
    I = np.eye(nx - 1)
    I_shifted = np.roll(I, 1, axis=0)
    A = (I - I_shifted) / dx

    # Time-stepping using Forward Euler
    for n in range(nt - 1):
        u[:, n + 1] = u[:, n] - dt * np.dot(A, u[:, n])

    # Re-enforce periodicity for the final output
    return np.concatenate([u, u[0:1, :]], axis=0)


# --- Core logic ---

def generate_sample(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single sample pair (initial condition, full solution).

    Args:
        config (Config): The configuration object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the initial condition
        `u0` and the corresponding solution `s`.
    """
    space = dde.data.GRF(
        T=config.grf_periodicity,
        kernel="ExpSineSquared",
        length_scale=config.length_scale,
        N=config.nx,
        interp=config.interp,
    )
    u0 = space.random(1)[0]
    s = numerical_solver(u0, config.nx, config.nt, config.xmax, config.tmax)
    return u0, s


def downsample_indices(total_points: int, n_sensors: int) -> np.ndarray:
    """
    Generates uniformly spaced indices for downsampling.

    Args:
        total_points (int): The total number of points in the original grid.
        n_sensors (int): The number of sensor locations to select.

    Returns:
        np.ndarray: A sorted array of indices.
    """
    return np.linspace(0, total_points - 1, n_sensors, dtype=int)


def generate_and_save_split(split_name: str, config: Config):
    """
    Generates, processes, and saves a full data split.

    Args:
        split_name (str): The name of the split (e.g., 'train').
        config (Config): The configuration object.
    """
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

    # Prepare dataset for DeepONet
    # Define sensor locations for branch and trunk networks
    idx_branch = downsample_indices(config.nx, config.n_sensors_branch)
    idx_trunk_x = downsample_indices(config.nx, config.n_sensors_trunk)
    idx_trunk_t = downsample_indices(config.nt, config.n_sensors_trunk)

    # Create branch input (subsampled initial conditions)
    X0 = u0_all[:, idx_branch]

    # Create trunk input (grid of (x, t) sensor locations)
    x_trunk = x_full[idx_trunk_x]
    t_trunk = t_full[idx_trunk_t]
    xx, tt = np.meshgrid(x_trunk, t_trunk)
    X1 = np.vstack((np.ravel(tt), np.ravel(xx))).T

    # Create output y (solution sampled at trunk locations)
    s_sampled = s_all[:, idx_trunk_x][:, :, idx_trunk_t]
    y = s_sampled.reshape(num_samples, -1)

    # Save data
    save_path = OUTPUT_DIR / f"{split_name}.npz"
    # X0_plot is unnecessary for advection, but required for DataHandler class
    np.savez_compressed(save_path, X0=X0, X1=X1, y=y, X0_plot=x_full[idx_branch])
    logging.info(f"Successfully saved '{split_name}' data to {save_path}")


# --- Entry point ---

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(
        description="Generate datasets of the 1D Advection equation for DeepONet training.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="default_advection",
                        help="Name of the config file in 'configs/data_generation'")
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Configuration Loading and Override
    config_path = Path("configs/data_generation") / f"{args.config}.yaml"
    config = load_config(str(config_path)) if args.config else Config()

    if args.override:
        apply_overrides(config, args.override)

    # Set random seed
    np.random.seed(config.seed)
    logging.info(f"Using random seed: {config.seed}")

    for split in SPLITS:
        generate_and_save_split(split, config)



if __name__ == "__main__":
    main()