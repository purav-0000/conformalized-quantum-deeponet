import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import secrets
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from data.generators.offline_voltage import _perform_fourier_analysis
from src.utils.common import apply_overrides


# --- Constants ---

INPUT_DIR = Path("data/raw_data/load")
OUTPUT_DIR = Path("data/processed_data/offline_load")


# --- Configuration ---

@dataclass
class Config:

    # Data Source Keys
    input_variable_key: str = 'V'
    output_variable_key: str = 'P'

    # Preprocessing Hyperparameters
    time_domain_limits: List[float] = field(default_factory=lambda: [0.0, 10.0])
    filter_strength_divisor: float = 20.0
    filter_input_signal: bool = False  # Allows enabling/disabling filtering on the input
    input_resolution: int = 30
    output_resolution: int = 100

    # Splitting Ratios
    train_split: float = 0.8
    cal_split: float = 0.1
    # Test split is implicitly calculated as 1.0 - train_split - cal_split

    # Reproducibility & Debugging
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    verbose: bool = False


# --- Utility Functions ---

def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Core Logic Functions ---

def _load_and_slice_data(config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads raw data files, downsamples high-res data, and combines them.

    Args:
        config (Config): The configuration object.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the combined
        input data, output data, and the unified time grid.
    """
    source_file_4001 = INPUT_DIR / "data_intersection_res_4001.npz"
    source_file_2001 = INPUT_DIR / "data_intersection_res_2001.npz"

    if not source_file_2001.exists() or not source_file_4001.exists():
        raise FileNotFoundError(f"Required data files not found in {INPUT_DIR}")

    data_4001 = np.load(source_file_4001)
    data_2001 = np.load(source_file_2001)
    logging.info("Combining data from high-resolution and low-resolution sources.")

    t_target = data_2001['t']
    input_low_res = data_2001[config.input_variable_key]
    output_low_res = data_2001[config.output_variable_key]

    t_source = data_4001['t']
    input_high_res = data_4001[config.input_variable_key]
    output_high_res = data_4001[config.output_variable_key]

    logging.info("Downsampling high-res data to the low-res time grid via interpolation...")
    interp_input = interp1d(t_source, input_high_res, axis=1, kind='linear', assume_sorted=True)
    interp_output = interp1d(t_source, output_high_res, axis=1, kind='linear', assume_sorted=True)

    input_high_res_resampled = interp_input(t_target)
    output_high_res_resampled = interp_output(t_target)

    input_combined = np.concatenate([input_high_res_resampled, input_low_res], axis=0)
    output_combined = np.concatenate([output_high_res_resampled, output_low_res], axis=0)

    return input_combined, output_combined, t_target


# --- Plotting and debugging ---

def _filter_and_downsample(
    input_raw: np.ndarray, output_raw: np.ndarray, t_raw: np.ndarray, config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a low-pass filter and downsamples the data to target resolutions.

    Args:
        input_raw (np.ndarray): The raw input signal data.
        output_raw (np.ndarray): The raw output signal data.
        t_raw (np.ndarray): The time grid for the raw signals.
        config (Config): The configuration object.

    Returns:
        Tuple containing filtered and downsampled signals and their time grids.
    """
    logging.info("Applying low-pass filter and downsampling...")
    sampling_freq = 1 / (t_raw[1] - t_raw[0])
    cutoff_freq = sampling_freq / config.filter_strength_divisor
    b, a = butter(4, cutoff_freq, btype='low', fs=sampling_freq)

    input_filtered = filtfilt(b, a, input_raw, axis=1) if config.filter_input_signal else input_raw
    output_filtered = filtfilt(b, a, output_raw, axis=1)

    input_indices = np.linspace(0, input_raw.shape[1] - 1, config.input_resolution, dtype=int)
    input_downsampled = input_filtered[:, input_indices]
    t_downsampled_input = t_raw[input_indices]

    output_indices = np.linspace(0, output_raw.shape[1] - 1, config.output_resolution, dtype=int)
    output_downsampled = output_filtered[:, output_indices]
    t_downsampled_output = t_raw[output_indices]

    return (input_filtered, output_filtered, input_downsampled, output_downsampled,
            t_downsampled_input, t_downsampled_output)


def _plot_sample(
        raw_data: dict, filtered_data: dict, downsampled_data: dict, config: Config, sample_idx: int
):
    """
    Plots and saves the effect of preprocessing on a single sample trajectory.

    Args:
        raw_data (dict): Dictionary of raw signals and time grids.
        filtered_data (dict): Dictionary of filtered signals.
        downsampled_data (dict): Dictionary of downsampled signals and time grids.
        config (Config): The configuration object.
        sample_idx (int): The index of the sample to visualize.
    """

    plt.figure(figsize=(16, 6))
    key_in, key_out = config.input_variable_key, config.output_variable_key

    # Plot Input Variable
    plt.subplot(1, 2, 1)
    plt.plot(raw_data['t'], raw_data['input'][sample_idx], color='cyan', label=f'Original {key_in}')
    plt.plot(raw_data['t'], filtered_data['input'][sample_idx], color='blue', label=f'Filtered {key_in}')
    plt.plot(downsampled_data['t_input'], downsampled_data['input'][sample_idx], 'bo', label=f'Downsampled {key_in}')
    plt.title(f'Input Trajectory #{sample_idx} ({key_in})')
    plt.xlabel('Time (s)')
    plt.ylabel(key_in)
    plt.legend()
    plt.grid(True)

    # Plot Output Variable
    plt.subplot(1, 2, 2)
    plt.plot(raw_data['t'], raw_data['output'][sample_idx], color='orange', alpha=0.7, label=f'Original {key_out}')
    plt.plot(raw_data['t'], filtered_data['output'][sample_idx], color='red', label=f'Filtered {key_out}')
    plt.plot(downsampled_data['t_output'], downsampled_data['output'][sample_idx], 'ro', label=f'Downsampled {key_out}')
    plt.title(f'Output Trajectory #{sample_idx} ({key_out})')
    plt.xlabel('Time (s)')
    plt.ylabel(key_out)
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.close()


def run_workflow(config: Config):
    """
    Executes the full data preprocessing workflow.

    Args:
        config (Config): The configuration object containing all parameters.
    """
    logging.info("Starting data preprocessing...")

    # Load and combine data from source files
    input_raw, output_raw, t_raw = _load_and_slice_data(config)

    # Slice to the specified time domain
    limits = config.time_domain_limits
    time_mask = (t_raw >= limits[0]) & (t_raw <= limits[1])
    if not np.any(time_mask):
        raise ValueError(f"The specified time domain {limits} is outside the data's range.")

    input_sliced = input_raw[:, time_mask]
    output_sliced = output_raw[:, time_mask]
    t_sliced = t_raw[time_mask]

    # Filter and downsample the sliced data
    (input_filtered, output_filtered, input_downsampled, output_downsampled,
     t_input_final, t_output_final) = _filter_and_downsample(
        input_sliced, output_sliced, t_sliced, config
    )

    # Structure data for DeepONet
    logging.info("Structuring data for DeepONet...")
    X0_branch = input_downsampled.astype(np.float32)
    X1_trunk = t_output_final.reshape(-1, 1).astype(np.float32)
    Y_target = output_downsampled.astype(np.float32)
    logging.info(f"Final shapes: X0={X0_branch.shape}, X1={X1_trunk.shape}, Y={Y_target.shape}")

    # For cross-checking fourier features in model
    # _perform_fourier_analysis(Y_target, X1_trunk)

    # Split data into train, calibration, and test sets
    num_trajectories = X0_branch.shape[0]
    shuffled_indices = np.random.permutation(num_trajectories)

    # Calculate the end-points for train and calibration sets
    train_end = int(num_trajectories * config.train_split)
    cal_end = train_end + int(num_trajectories * config.cal_split)

    splits: Dict[str, np.ndarray] = {
        "train": shuffled_indices[:train_end],
        "calibration": shuffled_indices[train_end:cal_end],
        "test": shuffled_indices[cal_end:],
    }

    # Save the final datasets
    for name, idx in splits.items():
        logging.info(f"Processing '{name}' split with {len(idx)} signals.")
        save_path = Path(str(OUTPUT_DIR) + "_" + config.output_variable_key) / f"{name}.npz"
        np.savez_compressed(
            save_path,
            X0=X0_branch[idx],
            X1=X1_trunk,
            y=Y_target[idx],
            X0_plot=t_input_final
        )
    logging.info("All files saved successfully!")

    # Visualize a sample if in verbose mode
    if config.verbose:
        sample_idx = np.random.choice(splits['test'])
        _plot_sample(
            raw_data={'input': input_sliced, 'output': output_sliced, 't': t_sliced},
            filtered_data={'input': input_filtered, 'output': output_filtered},
            downsampled_data={'input': input_downsampled, 'output': output_downsampled,
                              't_input': t_input_final, 't_output': t_output_final},
            config=config,
            sample_idx=sample_idx
        )


# --- Entry Point ---

def main():

    parser = argparse.ArgumentParser(
        description="Generate load prediction data for DeepONet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="default_offline_load",
                        help="Name of the config file in 'configs/data_generation'")
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Config and overrides
    config_path = Path("configs/data_generation") / f"{args.config}.yaml"
    config = load_config(str(config_path)) if args.config else Config()

    if args.override:
        apply_overrides(config, args.override)

    # Dir creation
    os.makedirs(Path(str(OUTPUT_DIR) + "_"  + config.output_variable_key), exist_ok=True)

    # Set the seed for reproducibility
    np.random.seed(config.seed)
    run_workflow(config)


if __name__ == '__main__':
    main()
