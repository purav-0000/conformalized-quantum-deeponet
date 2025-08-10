# Standard imports
import argparse
import logging
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import interpolate

from src.utils.common import apply_overrides


# Assuming this utility exists based on your example
# from src.utils.common import apply_overrides


# --- Configuration ---

@dataclass
class Config:
    # Data Paths
    input_data_path: str = "data/data_voltage/dataset.npz"
    output_data_dir: str = "data/voltage_prediction"

    # Raw Data Parameters
    num_nodes: int = 7
    num_data_per_client: int = 407

    # DeepONet Data Generation Parameters
    n_sensors: int = 100
    nLocs: int = 30
    memory_ranging: float = 0.4
    max_time: float = 8.0
    max_clear: float = 0.9

    # Splitting Ratios
    train_split: float = 0.7
    val_split: float = 0.1
    test_split: float = 0.1

    # Reproducibility and Debugging
    rng1: int = 12345
    rng2: int = 54321
    plot_checking_verbose: bool = False

    # Filters dataset, keeping only lowest variance parcentile
    variance_filter_percentile: float = 1.0


def load_config(yaml_path: str) -> Config:
    """Loads configuration from a YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Core Logic Functions ---

def raw_data_processor(voltage_data, clear_time, config: Config):
    """
    Loads raw client data and concatenates it into a single centralized dataset.
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

    # Tile clear time to match central_voltage
    central_clear_time = np.tile(clear_time, config.num_nodes)

    logging.info(f"Size of centralized voltage dataset is {central_voltage.shape}")
    logging.info(f"Size of centralized clear time dataset is {central_clear_time.shape}")
    return central_voltage, central_clear_time


def creating_DeepONet_dataset_cartesian(train_database, clearing_time, config: Config):
    """
    Creates a dataset suitable for dde.data.TripleCartesianProd.
    """
    np.random.seed(config.rng1)
    num_signals = train_database.shape[0]
    original_time = np.linspace(-0.1, 12.0, num=train_database.shape[1])

    U_train = np.zeros((num_signals, config.n_sensors))
    Y_train = np.zeros((config.nLocs, 1))
    G_train = np.zeros((num_signals, config.nLocs))

    trunk_time_start = config.max_clear + config.memory_ranging
    trunk_time_end = config.max_time
    fixed_trunk_locations = np.linspace(trunk_time_start, trunk_time_end, config.nLocs)
    Y_train[:, 0] = fixed_trunk_locations

    time_u_full = np.linspace(-0.1, config.max_clear + config.memory_ranging + (0.1 * config.memory_ranging), num=10000)
    sensors_locs = np.linspace(time_u_full[0], time_u_full[-1], config.n_sensors)

    # Plot these indices if verbose is true
    plot_indices_for_verbose = np.random.randint(num_signals, size=5)
    for i in range(num_signals):
        interpolated_signal = interpolate.interp1d(original_time, train_database[i], copy=False, assume_sorted=True)
        cut_section = clearing_time[i] + config.memory_ranging
        u_full_values = interpolated_signal(time_u_full)
        # u_full_values[time_u_full > cut_section] = 0

        f_u = interpolate.interp1d(time_u_full, u_full_values, copy=False, assume_sorted=True)
        U_train[i, :] = f_u(sensors_locs)
        G_train[i, :] = interpolated_signal(fixed_trunk_locations)

        if config.plot_checking_verbose and i in plot_indices_for_verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(original_time, train_database[i], 'm', label="Original Voltage Signal")
            plt.plot(time_u_full, u_full_values, 'b', label="Masked Input `u`")
            plt.plot(sensors_locs, U_train[i, :], '*k', ms=8, label=f"{config.n_sensors} Branch Sensor Values")
            plt.plot(fixed_trunk_locations, G_train[i, :], '.g', ms=10, label=f"{config.nLocs} Output G_values")
            plt.axvline(x=cut_section, color='r', ls='--', label=f"Cut Section for signal {i}")
            plt.title(f"Data Generation Check (Cartesian Product) - Signal {i}")
            plt.xlabel("Time")
            plt.ylabel("Voltage")
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

    logging.info('Generated data shapes for Cartesian Product Model:')
    logging.info(f'U_train (Branch) shape: {U_train.shape}')
    logging.info(f'Y_train (Trunk) shape:  {Y_train.shape}')
    logging.info(f'G_train (Output) shape: {G_train.shape}')

    logging.info("Analyzing frequencies in the G_train output signals...")
    num_signals, n_locs = G_train.shape

    if n_locs > 1:
        # Determine the sampling interval from the trunk locations
        sampling_interval = fixed_trunk_locations[1] - fixed_trunk_locations[0]

        # Calculate the frequencies corresponding to the FFT output
        # We only need the positive frequencies for the one-sided power spectrum
        frequencies = np.fft.fftfreq(n_locs, d=sampling_interval)[:n_locs // 2]

        # Accumulate power spectra from all signals
        total_power_spectrum = np.zeros(n_locs // 2)
        for i in range(num_signals):
            signal = G_train[i, :]
            fft_values = np.fft.fft(signal)
            # Compute power (squared magnitude) for positive frequencies
            power = np.abs(fft_values[0:n_locs // 2]) ** 2
            total_power_spectrum += power

        # Average the power spectrum across all signals
        average_power_spectrum = total_power_spectrum / num_signals

        # Plot the averaged power spectrum for visual inspection
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies[1:], average_power_spectrum[1:])  # Ignore DC component for better scaling
        plt.title('Averaged Power Spectrum of Output Signals (G_train)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)
        plt.yscale('log')  # Use a log scale to see smaller peaks
        plt.show()

        # Identify and log the top 5 dominant frequencies (excluding the DC component)
        top_indices = np.argsort(average_power_spectrum[1:])[-15:][::-1] + 1
        dominant_frequencies = frequencies[top_indices]
        dominant_powers = average_power_spectrum[top_indices]

        logging.info("Top 5 Dominant Frequencies in G_train:")
        for freq, power in zip(dominant_frequencies, dominant_powers):
            logging.info(f"  - Frequency: {freq:.2f} Hz, Power: {power:.2e}")
    else:
        logging.warning("Not enough data points in G_train to perform frequency analysis.")

    return U_train, Y_train, G_train


# --- Main Orchestration ---
def run_generation(config: Config):
    """Executes the full data generation workflow."""
    logging.info("Starting data generation process...")

    # --- Step 1: Load Raw Data ---
    logging.info(f"Loading raw data from {config.input_data_path}")
    raw_data = np.load(config.input_data_path)
    voltage_data = raw_data['voltage_data']
    clear_time = raw_data['clear_time']
    # Shuffle
    indices = np.random.permutation(voltage_data.shape[0])
    voltage_data = voltage_data[indices]
    clear_time = clear_time[indices]

    # --- Step 2: Process and Centralize Data ---
    clients_data_voltage, clients_data_clear_time = raw_data_processor(voltage_data, clear_time, config)
    logging.info(f"Sliced data by clients and recentralized. Voltage shape: {clients_data_voltage.shape}")

    # --- Step 3: Create DeepONet Cartesian Dataset ---
    u_data, y_data, g_data = creating_DeepONet_dataset_cartesian(
        train_database=clients_data_voltage,
        clearing_time=clients_data_clear_time,
        config=config
    )

    # Remove hard samples
    if 0.0 < config.variance_filter_percentile < 1.0:
        logging.info(
            f"Filtering dataset to keep samples with the lowest {config.variance_filter_percentile * 100:.0f}% variance.")

        output_variances = np.var(g_data, axis=1)
        variance_threshold = np.quantile(output_variances, config.variance_filter_percentile)
        keep_mask = output_variances <= variance_threshold

        # Store original data and indices for plotting before filtering
        if config.plot_checking_verbose:
            g_data_unfiltered = g_data.copy()  # Make a copy before it's modified
            kept_indices = np.where(keep_mask)[0]
            discarded_indices = np.where(~keep_mask)[0]

        # Apply the filter
        original_count = u_data.shape[0]
        u_data = u_data[keep_mask]
        g_data = g_data[keep_mask]
        logging.info(f"Filtering complete. Kept {u_data.shape[0]} out of {original_count} samples.")

        # Visualization of the filtering effect
        if config.plot_checking_verbose:
            logging.info("Displaying diagnostic plots for variance filtering...")

            # 1. Plot the histogram of all variances and the threshold
            plt.figure(figsize=(10, 6))
            plt.hist(output_variances, bins=50, alpha=0.75, label="Variance Distribution")
            plt.axvline(x=variance_threshold, color='r', linestyle='--', linewidth=2,
                        label=f"Threshold at {variance_threshold:.2e} ({config.variance_filter_percentile * 100:.0f}th percentile)")
            plt.title("Distribution of Output Variances for Filtering")
            plt.xlabel("Variance of g_data")
            plt.ylabel("Number of Samples")
            plt.legend()
            plt.grid(True)
            plt.show()

            # 2. Plot a few examples of signals that were KEPT (low variance)
            logging.info(f"Plotting up to 3 examples of KEPT low-variance samples...")
            for i in range(min(3, len(kept_indices))):
                idx = kept_indices[i]
                plt.figure(figsize=(10, 6))
                plt.plot(y_data, g_data_unfiltered[idx, :], '.-g')
                plt.title(f"KEPT Sample (Index: {idx}) | Variance: {output_variances[idx]:.2e} (LOW)")
                plt.xlabel("Time")
                plt.ylabel("Voltage")
                plt.grid(True)
                plt.ylim(0, 1.2)
                plt.show()

            # 3. Plot a few examples of signals that were DISCARDED (high variance)
            logging.info(f"Plotting up to 3 examples of DISCARDED high-variance samples...")
            for i in range(min(3, len(discarded_indices))):
                idx = discarded_indices[i]
                plt.figure(figsize=(10, 6))
                plt.plot(y_data, g_data_unfiltered[idx, :], '.-r')
                plt.title(f"DISCARDED Sample (Index: {idx}) | Variance: {output_variances[idx]:.2e} (HIGH)")
                plt.xlabel("Time")
                plt.ylabel("Voltage")
                plt.grid(True)
                plt.ylim(0, 1.2)
                plt.show()

    else:
        logging.info("No variance filtering applied (percentile is 1.0 or invalid).")

    # --- Step 4: Split and Save Data ---
    logging.info(f"Splitting and saving the data to {config.output_data_dir}")
    os.makedirs(config.output_data_dir, exist_ok=True)

    num_signals = u_data.shape[0]
    shuffled_indices = np.random.permutation(num_signals)

    train_end = int(num_signals * config.train_split)
    val_end = int(num_signals * (config.train_split + config.val_split))
    test_end = int(num_signals * (config.train_split + config.val_split + config.test_split))

    splits = {
        "train": shuffled_indices[:train_end],
        "val": shuffled_indices[train_end:val_end],
        "test": shuffled_indices[val_end:test_end],
        "calibration": shuffled_indices[test_end:],
    }

    X1_shared_trunk = y_data.astype(np.float32)

    for split_name, indices in splits.items():
        logging.info(f"Processing {split_name} split with {len(indices)} signals.")
        filename = os.path.join(config.output_data_dir, f'picked_aligned_{split_name}.npz')

        save_dict = {
            'X0': u_data[indices].astype(np.float32),
            'X1': X1_shared_trunk,
            'y': g_data[indices].astype(np.float32)
        }
        if split_name == "test":
            save_dict['X0_plot'] = u_data[indices].astype(np.float32)

        np.savez(filename, **save_dict)

    logging.info("All files have been saved successfully!")


# --- Entry Point ---

def main():
    """Parses arguments and starts the data generation."""
    parser = argparse.ArgumentParser(description="Generate voltage prediction data for DeepONet.")
    parser.add_argument("--config", type=str, default="default_voltage_prediction",
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
    np.random.seed(config.rng2)

    run_generation(config)


if __name__ == "__main__":
    main()