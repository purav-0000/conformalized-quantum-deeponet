import argparse
import os
import logging
import secrets
from dataclasses import dataclass, field
from typing import List, Union, Optional

import numpy as np
import yaml
from deepxde.data.function_spaces import GRF
from joblib import Parallel, delayed
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

from src.utils.common import apply_overrides

# Set DeepXDE backend
os.environ['DDE_BACKEND'] = 'pytorch'


# --- Integrators ---

def compute_numerical_solution(x, v, N):
    """Solve u' = v with u(0)=0 using backward differences (lower-triangular system)."""
    h = x[1] - x[0]
    K = np.eye(N - 1) - np.eye(N - 1, k=-1)
    b = h * v[1:]
    u = np.linalg.solve(K, b)
    return np.concatenate(([0], u))


def scipy_integrator(x, v):
    """Compute trapezoidal integral of v over x with u(x[0])=0."""
    return np.concatenate(([0], cumulative_trapezoid(v, x)))


INTEGRATORS = {
    "custom": compute_numerical_solution,
    "scipy": scipy_integrator,
}


# --- Config dataclass ---

@dataclass
class Config:
    M: int = 1000
    Nu: int = 30
    Nv: int = 10
    amplitude: Union[List[float], float] = field(default_factory=lambda: [0.5, 1.5])
    length_scale: Union[List[float], float] = field(default_factory=lambda: [0.5, 1.3])
    interp: str = "cubic"
    noise: float = 1e-4
    n_jobs: int = 8
    test: int = 500
    train: int = 1500
    calibration: int = 500
    downsample: str = "random"
    integrator: str = "custom"
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    dry_run: bool = False


# --- Core functions ---

def generate_sample(M, interp, length_range, amp_range, integrator):
    length = np.random.uniform(*length_range)
    amplitude = np.random.uniform(*amp_range)
    space = GRF(1, kernel='RBF', length_scale=length, N=M, interp=interp)
    x = np.ravel(space.x)
    v = np.ravel(amplitude * space.random(1))
    u = integrator(x, v)
    return v, u


def downsample_indices(M, N, mode="random"):
    if mode == "uniform":
        return [round(M / N) * i for i in range(N - 1)] + [M - 1]
    return sorted(np.random.choice(M, N, replace=False))


def save_split(filename, v, u, x, Nv, Nu, downsample_mode):
    idx_v = downsample_indices(len(x), Nv, downsample_mode)
    idx_u = downsample_indices(len(x), Nu, downsample_mode)
    xv = x[idx_v]
    xu = x[idx_u]

    v_idx = v[:, idx_v]
    u_idx = u[:, idx_u]

    np.savez_compressed(filename, X0=v_idx, X1=xu.reshape(-1, 1), y=u_idx, X0_plot=xv)


def generate_and_save_split(split_name: str, config: Config, integrator):
    num_samples = getattr(config, split_name)

    if config.dry_run:
        logging.info(f"[DRY RUN] Generating 3 samples for {split_name}")
        num_samples = 3

    results = Parallel(n_jobs=config.n_jobs)(
        delayed(generate_sample)(config.M, config.interp, _to_range(config.length_scale),
                                 _to_range(config.amplitude), integrator)
        for _ in tqdm(range(num_samples), desc=f"Generating {split_name}")
    )

    v_all, u_all = zip(*results)
    v_all = np.array(v_all, dtype=np.float32)
    u_all = np.array(u_all, dtype=np.float32)
    x = np.ravel(GRF(1, 'RBF', length_scale=1.0, N=config.M, interp=config.interp).x)

    v_all = v_all[:, :] + np.random.normal(0, config.noise, size=(v_all.shape[0], v_all.shape[1]))
    u_all = u_all[:, :] + np.random.normal(0, config.noise, size=(u_all.shape[0], u_all.shape[1]))

    if config.dry_run:
        import matplotlib.pyplot as plt
        os.makedirs("data/processed_data/antiderivative", exist_ok=True)

        for i in range(min(3, len(v_all))):
            plt.figure(figsize=(8, 4))
            plt.plot(x, v_all[i], label='Underlying v (source term)', color="blue")
            plt.plot(x, u_all[i], label='Underlying u (solution)', color="red")

            # Plot what the model sees
            idx_v = downsample_indices(len(x), config.Nv, config.downsample)
            idx_u = downsample_indices(len(x), config.Nu, config.downsample)

            xv = x[idx_v]
            xu = x[idx_u]

            v_idx = v_all[:, idx_v]
            u_idx = u_all[:, idx_u]

            plt.plot(xv, v_idx[i], label='Indexed v (source term)', color="blue", alpha=0.9, linestyle="dashed")
            plt.plot(xu, u_idx[i], label='Indexed u (solution)', color="red", alpha=0.9, linestyle="dashed")

            plt.title(f"{split_name} sample {i}")
            plt.xlabel("x")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"data/processed_data/antiderivative/{split_name}_sample_{i}.png")
            plt.close()

        logging.info(f"Saved sample plots for dry run in data/processsed_data/antiderivative/")
        return

    os.makedirs("data/processed_data/antiderivative", exist_ok=True)

    # No need for this right now
    """
    np.savez_compressed(
        f'data/processed_data/antiderivative/full_aligned_{split_name}.npz',
        X0=v_all, X1=x.reshape(-1, 1), y=u_all
    )
    """

    save_split(
        f'data/processed_data/antiderivative/{split_name}.npz',
        v_all, u_all, x, config.Nv, config.Nu, config.downsample
    )


def _to_range(value):
    """Convert scalar or list to a range (2-element list)."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value
    return [value, value]


# --- Seed and YAML loading ---

def set_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logging.info(f"Using random seed: {seed}")


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(description="Generate GRF-based data for DeepONet training.")
    parser.add_argument("--config", type=str, default="default_antiderivative", help="Config file name"
    )
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--dry_run", action="store_true", help="Run a small test without saving files")
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.config = os.path.join("configs/data_generation", args.config + ".yaml")
    config = load_config(args.config) if args.config else Config()
    apply_overrides(config, args.override)

    if config.dry_run or args.dry_run:
        config.n_jobs = 1

    if args.seed is not None:
        config.seed = args.seed
    if args.dry_run:
        config.dry_run = True

    set_seed(config.seed)
    integrator = INTEGRATORS[config.integrator]

    for split in ["train", "test", "calibration"]:
        generate_and_save_split(split, config, integrator)


if __name__ == "__main__":
    main()
