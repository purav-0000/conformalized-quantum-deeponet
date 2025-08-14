# Standard imports
import argparse
import logging
import os
from dataclasses import dataclass, field
import secrets
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils.common import apply_overrides

# --- Configuration ---

@dataclass
class Config:
    """Configuration for sliding window dataset generation."""
    # Data Paths
    input_data_path: str = "data/raw_data/voltage_dataset.npz"
    output_data_dir: str = "data/processed_data/offline_voltage_sliding_window"

    # Raw Data Parameters
    num_nodes: int = 7
    num_data_per_client: int = 407

    # Data Parameters
    memory_window_size: int = 100  # Number of past time steps to use as input (tau)
    prediction_horizon: int = 1  # Number of future time steps to predict (fixed at 1 for t+1)
    time_domain_limits: list[float] = field(default_factory=lambda: [4.0, 6.0])

    # Splitting Ratios
    max_samples: int = 10000
    train_split: float = 0.8
    cal_split: float = 0.1
    test_split: float = 0.1

    # Reproducibility and Debugging
    seed: Optional[int] = field(default_factory=lambda: secrets.randbits(32))
    verbose: bool = True

    # Filters dataset based on the variance of the *entire* signal before windowing
    variance_filter_percentile: float = 1.0


def load_config(yaml_path: str) -> Config:
    """Loads configuration from a YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Core Logic Functions ---

def raw_data_processor(voltage_data, config: Config):
    """
    Loads raw client data and concatenates it into a single centralized dataset.
    This function is simplified as `clear_time` is not used in the sliding window model.
    """
    client_voltage_data = {}

    # Split by client
    for client in range(config.num_nodes):
        start_idx = client * config.num_data_per_client
        end_idx = (client + 1) * config.num_data_per_client - 1
        client_voltage_data[client] = voltage_data[:, start_idx:end_idx]
        logging.info(f"Size of raw data for client {client} is {client_voltage_data[client].shape}")

    # Recenterallize the dataset
    central_voltage = np.concatenate([client_voltage_data[c] for c in range(config.num_nodes)], axis=0)

    logging.info(f"Size of centralized voltage dataset is {central_voltage.shape}")
    return central_voltage


def creating_DeepONet_dataset_cartesian(signals_database, time_mask, config: Config):
    """
    Creates a dataset based on a sliding window approach.

    For each signal, it generates pairs of (input_window, target_value).
    - input_window: A sequence of `memory_window_size` data points.
    - target_value: The single data point immediately following the window.
    """
    num_signals = signals_database.shape[0]

    # Assuming that raw_data_processor shaped according to (clients?, config.num_data_per_client - 1)
    original_time = np.linspace(-0.1, 12.0, num=config.num_data_per_client - 1)
    original_time = original_time[time_mask]

    # All memory windows
    U_data = []
    # Time of each point in the memory window (for plotting)
    U_time = []

    # All coordinates to be evaluated
    Y_data = []
    # Ground truth
    G_data = []

    num_signals = signals_database.shape[0]

    # The total length needed for one sample is window_size + horizon
    # The selection of signal 0 here is arbitrary
    max_start_index = len(signals_database[0]) - config.memory_window_size - config.prediction_horizon + 1
    for i in range(num_signals):
        signal = signals_database[i]

        U_data_window = []
        U_time_window = []
        Y_data_window = []
        G_data_window = []

        for start_idx in range(max_start_index):
            end_idx = start_idx + config.memory_window_size

            # The input is the memory window
            window = signal[start_idx:end_idx]

            # The target is the single point after the window
            target = signal[end_idx]

            U_data_window.append(window)
            U_time_window.append(original_time[start_idx:end_idx])
            Y_data_window.append(original_time[end_idx])
            G_data_window.append(target)
        U_data.append(U_data_window)
        U_time.append(U_time_window)
        Y_data.append(Y_data_window)
        G_data.append(G_data_window)

    # Convert lists to numpy arrays
    U_data = np.array(U_data, dtype=np.float32)
    U_time = np.array(U_time, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32).reshape(num_signals, -1, 1)  # Reshape for model
    G_data = np.array(G_data, dtype=np.float32).reshape(num_signals, -1, 1)

    logging.info('Generated data shapes for Sliding Window Model:')
    logging.info(f'U_data (Input Windows) shape: {U_data.shape}')
    logging.info(f'Y_data (Target Values) shape:  {Y_data.shape}')
    logging.info(f'G_data (Target Values) shape:  {G_data.shape}')

    if config.verbose:
        for i in range(3):
            # Plot one example to verify the logic
            plt.figure(figsize=(12, 6))
            sample_signal_to_plot = np.random.randint(0, num_signals)
            sample_window_to_plot = np.random.randint(0, max_start_index)

            # Recreate the time axis for plotting
            window_time = U_time[sample_signal_to_plot, sample_window_to_plot]
            target_time = Y_data[sample_signal_to_plot, sample_window_to_plot]

            plt.plot(window_time, U_data[sample_signal_to_plot, sample_window_to_plot], 'bo-',
                     label=f'Input Window (t - {config.memory_window_size} to t-1)')
            plt.plot(target_time, G_data[sample_signal_to_plot, sample_window_to_plot], 'r*', markersize=12,
                     label='Target Value (at t)')

            plt.title(f"Data Generation Check (Sliding Window) - Sample Signal: {sample_signal_to_plot}, "
                      f"Window: {sample_window_to_plot}")
            plt.xlabel("Time Steps within Window")
            plt.ylabel("Voltage")
            plt.legend()
            plt.grid(True)
            plt.show()

    return U_data, Y_data, G_data, U_time


# --- Main Orchestration ---
def run_generation(config: Config):
    """Executes the full data generation workflow."""
    logging.info("Starting sliding window data generation process...")

    # Load Raw Data
    logging.info(f"Loading raw data from {config.input_data_path}")
    try:
        raw_data = np.load(config.input_data_path)
    except FileNotFoundError:
        logging.error(f"Input data file not found at {config.input_data_path}. Exiting.")
        return

    voltage_data = raw_data['voltage_data']

    # Shuffle the signals before processing
    np.random.shuffle(voltage_data)

    # --- Step 2: Process and Centralize Data ---
    centralized_voltage = raw_data_processor(voltage_data, config)
    logging.info(f"Centralized voltage data shape: {centralized_voltage.shape}")

    # Constrain to time limits
    time_mask = None
    if config.time_domain_limits:
        original_time = np.linspace(-0.1, 12.0, num=centralized_voltage.shape[1])
        logging.info(f"Slicing raw data to time domain: {config.time_domain_limits}s...")
        limits = config.time_domain_limits
        assert limits[0] < limits[1], "time_domain_limits must be [min, max]."

        time_mask = (original_time >= limits[0]) & (original_time <= limits[1])
        if not np.any(time_mask):
            raise ValueError(f"The specified time domain {limits} is outside the data's range.")

        centralized_voltage = centralized_voltage[:, time_mask]
        logging.info(
            f"New shape after slicing: {centralized_voltage.shape}")

    # Filter Signals by Variance
    if 0.0 < config.variance_filter_percentile < 1.0:
        logging.info(
            f"Filtering signals to keep those with the lowest {config.variance_filter_percentile * 100:.0f}% variance.")

        # Calculate variance across the time axis for each signal
        signal_variances = np.var(centralized_voltage, axis=1)
        variance_threshold = np.quantile(signal_variances, config.variance_filter_percentile)
        keep_mask = signal_variances <= variance_threshold

        original_count = centralized_voltage.shape[0]
        centralized_voltage = centralized_voltage[keep_mask]
        logging.info(f"Filtering complete. Kept {centralized_voltage.shape[0]} out of {original_count} signals.")
    else:
        logging.info("No variance filtering applied.")

    # Create Sliding Window Dataset
    u_data, y_data, g_data, u_time = creating_DeepONet_dataset_cartesian(
        signals_database=centralized_voltage,
        time_mask=time_mask,
        config=config
    )

    # Split and Save Data
    num_samples = u_data.shape[0]

    logging.info(f"Splitting and saving the data to {config.output_data_dir}")
    os.makedirs(config.output_data_dir, exist_ok=True)

    shuffled_indices = np.random.permutation(num_samples)

    train_end = int(num_samples * config.train_split)
    cal_end = train_end + int(num_samples * config.cal_split)

    splits = {
        "train": shuffled_indices[:train_end],
        "calibration": shuffled_indices[train_end:cal_end],
        "test": shuffled_indices[cal_end:],
    }

    for split_name, indices in splits.items():
        logging.info(f"Processing {split_name} split with {len(indices)} samples.")
        filename = os.path.join(config.output_data_dir, f'{split_name}.npz')

        save_dict = {
            'X0': u_data[indices].astype(np.float32),
            'X1': y_data[indices].astype(np.float32),
            'y': g_data[indices].astype(np.float32),
            'X0_plot': u_time[indices].astype(np.float32)
        }

        np.savez(filename, **save_dict)

    logging.info("All files have been saved successfully!")


# --- Entry Point ---

def main():
    """Parses arguments and starts the data generation."""
    parser = argparse.ArgumentParser(description="Generate voltage prediction data for DeepONet.")
    parser.add_argument("--config", type=str, default="default_online_voltage",
                        help="Config file name in configs/data_generation")
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S')

    config_path = os.path.join("configs/data_generation", args.config + ".yaml")
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at {config_path}")
        return

    config = load_config(config_path)
    if args.override:
        apply_overrides(config, args.override)

    # Set seeds for reproducibility
    np.random.seed(config.seed)

    run_generation(config)


if __name__ == "__main__":
    main()
