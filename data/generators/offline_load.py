# scripts/preprocess_vqp_data.py

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import secrets

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

from src.utils.common import apply_overrides

# --- Configuration Dataclass ---

@dataclass
class Config:
    """Configuration for VQP data preprocessing."""
    # Paths
    source_data_dir: str = "data/raw_data/load"
    output_data_dir: str = "data/processed_data/offline_load"

    # Data Source
    input_variable_key: str = 'V'
    output_variable_key: str = 'P'

    # Preprocessing Hyperparameters
    # None sets to full domain
    time_domain_limits: list[float] = field(default_factory=lambda: [0.0, 10.0])
    filter_strength_divisor: float = 20.0
    input_resolution: int = 100
    output_resolution: int = 30

    # Splitting Ratios
    train_split: float = 0.8
    cal_split: float = 0.1
    test_split: float = 0.1

    # Reproducibility & Debugging
    random_seed: int = field(default_factory=lambda: secrets.randbits(32))
    verbose: bool = False


# --- Utility Functions ---

def load_config(yaml_path: str) -> Config:
    """Loads configuration from a YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found at: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


###########################################################################################
# NOTE!!!     HERE THE U VARIABLE CORRESPONDS TO THE INPUT
# NOTE!!!     AND P CORRESPONDS TO THE OUTPUT.
# NOTE!!!     P IS NOT NECESSARILY ACTIVE POWER BUT CAN ALSO BE REACTIVE POWER (Q)
###########################################################################################

# --- Core Logic Functions ---

def load_and_slice_data(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads raw data, selects the correct source file, and slices the time domain."""
    source_dir = Path(config.source_data_dir)
    source_file_4001 = source_dir / "data_intersection_res_4001.npz"
    source_file_2001 = source_dir / "data_intersection_res_2001.npz"

    # Load both files
    data_4001 = np.load(source_file_4001)
    data_2001 = np.load(source_file_2001)

    # Both files exist: Downsample the 4001 data to the 2001 time grid
    logging.info(f"Loading and combining data from {source_file_4001} and {source_file_2001}.")

    # Use the low-resolution time grid as the target
    t_target = data_2001['t']
    U_low_res = data_2001[config.input_variable_key]
    P_low_res = data_2001[config.output_variable_key]

    # Get the high-resolution source data that needs downsampling
    t_source = data_4001['t']
    U_high_res = data_4001[config.input_variable_key]
    P_high_res = data_4001[config.output_variable_key]

    logging.info("Downsampling high-res data to the low-res time grid via interpolation...")

    # Create interpolation functions for the high-res U and P
    interp_U = interp1d(t_source, U_high_res, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    interp_P = interp1d(t_source, P_high_res, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate", assume_sorted=True)

    # Perform the interpolation to get the downsampled data
    U_high_res_resampled = interp_U(t_target)
    P_high_res_resampled = interp_P(t_target)

    # Concatenate along the trajectory axis (axis=0).
    # We place the (now downsampled) high-res data first for consistency.
    U_raw = np.concatenate([U_high_res_resampled, U_low_res], axis=0)
    P_raw = np.concatenate([P_high_res_resampled, P_low_res], axis=0)
    t_raw = t_target  # The unified, low-resolution time vector

    logging.info(
        f"Final combined shapes: {config.input_variable_key}={U_raw.shape}, {config.output_variable_key}={P_raw.shape}, t={t_raw.shape}"
    )

    # Slice time domain
    logging.info(f"Slicing raw data to time domain: {config.time_domain_limits}s...")
    limits = config.time_domain_limits
    assert limits[0] < limits[1], "time_domain_limits must be [min, max]."

    time_mask = (t_raw >= limits[0]) & (t_raw <= limits[1])
    if not np.any(time_mask):
        raise ValueError(f"The specified time domain {limits} is outside the data's range.")

    U_raw = U_raw[:, time_mask]
    P_raw = P_raw[:, time_mask]
    t_raw = t_raw[time_mask]
    logging.info(
        f"New shapes after slicing: {config.input_variable_key}={U_raw.shape}, {config.output_variable_key}={P_raw.shape}, t={t_raw.shape}")

    return U_raw, P_raw, t_raw


def filter_and_downsample(U_raw: np.ndarray, P_raw: np.ndarray, t_raw: np.ndarray, config: Config):
    """Applies a low-pass filter and downsamples the data."""
    logging.info(f"Applying low-pass filter with strength divisor: {config.filter_strength_divisor}...")
    sampling_interval = t_raw[1] - t_raw[0]
    sampling_freq = 1 / sampling_interval
    cutoff_freq = sampling_freq / config.filter_strength_divisor
    b, a = butter(4, cutoff_freq, btype='low', fs=sampling_freq)

    # No filtering for input 'V'
    # U_filtered = filtfilt(b, a, U_raw, axis=1)
    U_filtered = U_raw

    P_filtered = filtfilt(b, a, P_raw, axis=1)

    logging.info(
        f"Downsampling {config.input_variable_key} to {config.input_resolution} and {config.output_variable_key} to {config.output_resolution} points..."
    )

    # Downsample input signal (for branch net)
    u_indices = np.linspace(0, U_raw.shape[1] - 1, config.input_resolution, dtype=int)
    U_downsampled = U_filtered[:, u_indices]
    t_downsampled_for_u = t_raw[u_indices]

    # Downsample output signal (for trunk net)
    # AGAIN, EVERYTHING NAMED FOR P BUT CAN ALSO BE Q
    p_indices = np.linspace(0, P_raw.shape[1] - 1, config.output_resolution, dtype=int)
    P_downsampled = P_filtered[:, p_indices]
    t_downsampled_for_p = t_raw[p_indices]

    return U_filtered, P_filtered, U_downsampled, P_downsampled, t_downsampled_for_u, t_downsampled_for_p


def visualize_sample(U_raw, U_filtered, U_downsampled, t_raw, t_downsampled_for_u,
                     P_raw, P_filtered, P_downsampled, t_downsampled_for_p,
                     config: Config, sample_idx=0):
    """Plots and saves the effect of preprocessing on a single sample trajectory."""
    plt.figure(figsize=(14, 6))

    # Plot Input Variable (U)
    plt.subplot(1, 2, 1)
    plt.plot(t_raw, U_raw[sample_idx], color='cyan', label=f'Original {config.input_variable_key}')
    plt.plot(t_raw, U_filtered[sample_idx], color='blue', label=f'Filtered {config.input_variable_key}')
    plt.plot(t_downsampled_for_u, U_downsampled[sample_idx], 'bo', label=f'Downsampled {config.input_variable_key}')
    plt.title(f'{config.input_variable_key} Trajectory #{sample_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel(config.input_variable_key)
    plt.legend()
    plt.grid(True)

    # Plot Output Variable
    plt.subplot(1, 2, 2)
    plt.plot(t_raw, P_raw[sample_idx], color='orange', alpha=0.7, label=f'Original {config.output_variable_key}')
    plt.plot(t_raw, P_filtered[sample_idx], color='red', label=f'Filtered {config.output_variable_key}')
    plt.plot(t_downsampled_for_p, P_downsampled[sample_idx], 'ro', label=f'Downsampled {config.output_variable_key}')
    plt.title(f'{config.output_variable_key} Trajectory #{sample_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel(config.output_variable_key)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Main Orchestration ---
def run_preprocessing(config: Config):
    """Executes the full data preprocessing workflow."""
    logging.info("Starting data preprocessing...")
    np.random.seed(config.random_seed)

    # --- 1. Load and Slice ---
    U_raw, P_raw, t_raw = load_and_slice_data(config)

    # --- 2. Filter and Downsample ---
    U_filtered, P_filtered, U_downsampled, P_downsampled, t_u, t_p = filter_and_downsample(
        U_raw, P_raw, t_raw, config
    )

    # --- 3. Structure for DeepONet ---
    logging.info("Structuring data for vectorized DeepONet...")
    X0_branch = U_downsampled.astype(np.float32)
    X1_trunk = t_p.reshape(-1, 1).astype(np.float32)
    Y_target = P_downsampled.astype(np.float32)

    logging.info(f"Final shapes: X0={X0_branch.shape}, X1={X1_trunk.shape}, Y={Y_target.shape}")
    assert X0_branch.shape[0] == Y_target.shape[0], "Batch sizes of branch and target must match."
    assert X1_trunk.shape[0] == Y_target.shape[1], "Trunk coordinates must match number of output points."

    # --- 4. Split Data ---
    logging.info("Splitting data into train, validation, calibration, and test sets...")
    num_trajectories = X0_branch.shape[0]
    rng = np.random.default_rng(config.random_seed)
    shuffled_indices = rng.permutation(num_trajectories)

    # Calculate the end-points for train and calibration sets
    train_end = int(num_trajectories * config.train_split)
    cal_end = train_end + int(num_trajectories * config.cal_split)

    splits = {
        "train": shuffled_indices[:train_end],
        "calibration": shuffled_indices[train_end:cal_end],
        "test": shuffled_indices[cal_end:],
    }

    # --- 5. Save Processed Data ---

    # Suffix P or Q for output directory
    output_dir = Path(config.output_data_dir + "_" + config.output_variable_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving processed files to {output_dir}...")

    for split_name, split_indices in splits.items():
        logging.info(f"Processing {split_name} split with {len(split_indices)} trajectories.")
        filename = output_dir / f'{split_name}.npz'

        save_dict = {
            'X0': X0_branch[split_indices],
            'X1': X1_trunk,  # Trunk is shared across all samples
            'y': Y_target[split_indices],
            'X0_plot': t_u
        }

        np.savez(filename, **save_dict)

    logging.info("All data splits saved successfully.")

    # --- 6. Visualize a Sample ---
    if config.verbose:
        # Choose a random index from the test set to visualize
        random_test_trajectory_idx = np.random.choice(splits['test'])
        visualize_sample(
            U_raw, U_filtered, U_downsampled, t_raw, t_u,
            P_raw, P_filtered, P_downsampled, t_p,
            config, sample_idx=random_test_trajectory_idx
        )

    logging.info("\nPreprocessing complete!")


# --- Entry Point ---
def main():
    """Parses arguments, loads config, and starts the preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess VQP data for DeepONet.")
    parser.add_argument(
        "--config",
        type=str,
        default="default_offline_load",
        help="Name of the config file (without .yaml) in the 'configs' directory."
    )
    parser.add_argument(
        "--override",
        nargs='*',
        help="Optional config overrides in key=value format (e.g., input_resolution=150)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    config_path = os.path.join("configs/data_generation", args.config + ".yaml")
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logging.error(e)
        return

    if args.override:
        apply_overrides(config, args.override)

    run_preprocessing(config)


if __name__ == '__main__':
    main()
