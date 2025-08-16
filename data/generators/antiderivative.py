import argparse
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import secrets
from typing import List, Union, Callable, Tuple, Dict

import numpy as np
from deepxde.data.function_spaces import GRF
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm
import yaml

from src.utils.common import apply_overrides

# --- Constants ---

SPLITS: List[str] = ["train", "test", "calibration"]
OUTPUT_DIR = Path("data/processed_data/antiderivative")


# --- Config dataclass ---

@dataclass
class Config:

    # DeepONet sensor parameters
    Nu: int = 30
    Nv: int = 10

    # Numerical solver parameters
    M: int = 1000
    downsample: str = "random"
    integrator: str = "custom"

    # GRF parameters
    amplitude: Union[List[float], float] = field(default_factory=lambda: [0.5, 1.5])
    length_scale: Union[List[float], float] = field(default_factory=lambda: [0.5, 1.3])
    interp: str = "cubic"

    # Splits
    train: int = 1500
    test: int = 500
    calibration: int = 500

    # Miscellaneous
    n_jobs: int = 8
    noise: float = 1e-4
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    verbose: bool = False


# --- Utilities ---

def _to_range(value):
    """Convert scalar or list to a range (2-element list)."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value
    return [value, value]


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Integrators ---

def compute_numerical_solution(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Solves u' = v with u(0)=0 using backward differences.

    This method constructs a lower-triangular system and solves it.

    Args:
        x (np.ndarray): The 1D spatial grid.
        v (np.ndarray): The function to be integrated (source term).

    Returns:
        np.ndarray: The integrated function u.
    """
    h = x[1] - x[0]
    N = len(x)
    # This creates a lower bidiagonal matrix for the backward difference operator.
    K = np.eye(N - 1) - np.eye(N - 1, k=-1)
    b = h * v[1:]
    u_solution = np.linalg.solve(K, b)
    return np.concatenate(([0], u_solution))


def scipy_integrator(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Computes the trapezoidal integral of v over x with u(x[0])=0.

    Args:
        x (np.ndarray): The 1D spatial grid.
        v (np.ndarray): The function to be integrated.

    Returns:
        np.ndarray: The integrated function u.
    """
    return np.concatenate(([0], cumulative_trapezoid(v, x)))


INTEGRATORS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "custom": compute_numerical_solution,
    "scipy": scipy_integrator,
}


# --- Core logic ---

def generate_sample(M: int, interp: str, length_range: list, amp_range: list, integrator: Callable)\
                    -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single (v, u) sample pair from a Gaussian Random Field.

    Args:
        M (int): Number of points in the high-fidelity grid.
        interp (str): Interpolation method for the GRF.
        length_range (list): Range [min, max] for the GRF length scale.
        amp_range (list): Range [min, max] for the GRF amplitude.
        integrator (Callable): The integration function to compute u from v.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The generated v and u functions.
    """
    length = np.random.uniform(*length_range)
    amplitude = np.random.uniform(*amp_range)
    # The GRF space defines the function's properties.
    space = GRF(1, kernel='RBF', length_scale=length, N=M, interp=interp)
    x = np.ravel(space.x)
    v = np.ravel(amplitude * space.random(1))
    u = integrator(x, v)
    return v, u


def downsample_indices(M: int, N: int, mode: str = "random") -> np.ndarray:
    """
    Generates N indices for downsampling from a grid of size M.

    Args:
        M (int): The original number of points.
        N (int): The target number of points.
        mode (str): 'random' or 'uniform'. Defaults to "random".

    Returns:
        np.ndarray: A sorted array of N indices.
    """
    if mode == "uniform":
        return np.linspace(0, M - 1, N, dtype=int)
    elif mode == "random":
        return np.sort(np.random.choice(M, N, replace=False))
    else:
        raise ValueError(f"Unknown downsampling mode: '{mode}'")


def save_split_data(filename: Path, v: np.ndarray, u: np.ndarray, x: np.ndarray, Nv: int, Nu: int,
                    downsample_mode: str):
    """
    Downsamples and saves a data split to a compressed .npz file.

    Args:
        filename (Path): Name of saved file
        v (np.ndarray): Input function.
        u (np.ndarray): Output function.
        x (np.ndarray): The 1D spatial grid.
        Nv (int): Number of branch coordinates.
        Nu (int): Number of trunk coordinates.
        downsample_mode (str): 'random' or 'uniform'.
    """

    idx_v = downsample_indices(len(x), Nv, downsample_mode)
    idx_u = downsample_indices(len(x), Nu, downsample_mode)

    xv = x[idx_v]
    xu = x[idx_u]
    v_downsampled = v[:, idx_v]
    u_downsampled = u[:, idx_u]

    # X0 = branch input, X1 = trunk input, y = target, X0_plot = for high resolution plotting of the branch
    np.savez_compressed(filename, X0=v_downsampled, X1=xu.reshape(-1, 1), y=u_downsampled, X0_plot=xv)
    logging.info(f"Saved {v.shape[0]} samples to {filename}")


def generate_and_save_split(split_name: str, config: Config, integrator: Callable):
    """
     Generates and saves a full data split.

     Args:
         split_name (str): Refer to SPLITS list defined as a global variable.
         config (Config): Configuration object.
         integrator (Callable): The integration function to compute u from v.
     """
    num_samples = getattr(config, split_name)

    # Generate samples in parallel for efficiency
    results = Parallel(n_jobs=config.n_jobs)(
        delayed(generate_sample)(config.M, config.interp, _to_range(config.length_scale),
                                 _to_range(config.amplitude), integrator)
        for _ in tqdm(range(num_samples), desc=f"Generating {split_name}")
    )

    v_all, u_all = zip(*results)
    v_all = np.array(v_all, dtype=np.float32)
    u_all = np.array(u_all, dtype=np.float32)
    x = np.ravel(GRF(1, 'RBF', length_scale=1.0, N=config.M, interp=config.interp).x)

    # Add a small amount of noise to simulate real-world sensor data
    if config.noise > 0:
        v_all += np.random.normal(0, config.noise, size=v_all.shape)
        u_all += np.random.normal(0, config.noise, size=u_all.shape)

    if config.verbose:

        # Plot 3 samples
        for i in range(3):
            plt.figure(figsize=(12, 6))
            plt.plot(x, v_all[i], label='Underlying v (input function)', color="blue")
            plt.plot(x, u_all[i], label='Underlying u (output function)', color="red")

            # Plot what the model sees
            idx_v = downsample_indices(len(x), config.Nv, config.downsample)
            idx_u = downsample_indices(len(x), config.Nu, config.downsample)

            xv = x[idx_v]
            xu = x[idx_u]

            v_idx = v_all[:, idx_v]
            u_idx = u_all[:, idx_u]

            plt.plot(xv, v_idx[i], label='Downsampled v (input function)', color="blue", alpha=0.9, linestyle="dashed")
            plt.plot(xu, u_idx[i], label='Downsampled u (output function)', color="red", alpha=0.9, linestyle="dashed")

            plt.title(f"{split_name} sample {i}")
            plt.xlabel("x")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.close()

    save_split_data(
        OUTPUT_DIR / f"{split_name}.npz",
        v_all, u_all, x, config.Nv, config.Nu, config.downsample
    )


# --- Entry point ---

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(
        description="Generate GRF-based data for DeepONet training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="default_antiderivative",
                        help="Name of the config file in 'configs/data_generation'."
    )
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Configuration loading and overrides
    config_path = Path("configs/data_generation") / f"{args.config}.yaml"
    config = load_config(str(config_path)) if args.config else Config()

    if args.override:
        apply_overrides(config, args.override)

    # Set random seed
    np.random.seed(config.seed)
    logging.info(f"Using random seed: {config.seed}")

    integrator = INTEGRATORS[config.integrator]

    for split in ["train", "test", "calibration"]:
        generate_and_save_split(split, config, integrator)


if __name__ == "__main__":
    main()
