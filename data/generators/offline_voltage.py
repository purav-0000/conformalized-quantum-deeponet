import argparse
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import secrets
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.fft import dct
import yaml

from tqdm import tqdm

from src.utils.common import apply_overrides


# --- Constants ---

INPUT_FILE = Path("data/raw_data/voltage_dataset.npz")
OUTPUT_DIR = Path("data/processed_data/offline_voltage")

# --- Configuration ---

class FilterMode(Enum):
    """Defines how the variance filter should operate."""
    KEEP_LOWEST = "keep_lowest"
    KEEP_HIGHEST = "keep_highest"


@dataclass
class Config:

    # Raw Data Parameters
    num_nodes: int = 7
    num_data_per_client: int = 407

    # DeepONet Data Generation Parameters
    n_sensors: int = 100       # Number of sensors for the branch network
    n_locs: int = 30           # Number of locations for the trunk network
    memory_ranging: float = 0.4
    max_time: float = 8.0
    max_clear: float = 0.9

    # Splitting Ratios
    train: float = 0.8
    calibration: float = 0.1
    # Test split is implicitly (1 - train_split - cal_split)

    # Variance Filtering
    frequency_filter_percentile: float = 1.0     # 1.0 means no filtering
    filter_mode: FilterMode = FilterMode.KEEP_HIGHEST

    # Reproducibility and Debugging
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    verbose: bool = False

    def __post_init__(self):
        """Ensure string values from YAML/overrides are converted to Enum."""
        if isinstance(self.filter_mode, str):
            self.filter_mode = FilterMode(self.filter_mode)


# --- Utilities ---

def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Core Logic Functions ---


def _load_and_process_raw_data(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads raw data, shuffles it, and centralizes the client data.

    Args:
        config (Config): The configuration object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the centralized voltage and
        clearing time datasets.
    """
    logging.info(f"Loading raw data from {INPUT_FILE}")
    raw_data = np.load(INPUT_FILE)
    voltage_data = raw_data['voltage_data']
    clear_time = raw_data['clear_time']

    # Shuffle the entire dataset before processing
    indices = np.random.permutation(voltage_data.shape[0])
    voltage_data = voltage_data[indices]
    clear_time = clear_time[indices]

    # Storing client-by-client into one centralized array.
    client_voltages = []
    for client in range(config.num_nodes):
        start_idx = client * config.num_data_per_client
        # The original script had -1 for some reason added to end_idx
        end_idx = (client + 1) * config.num_data_per_client
        client_voltages.append(voltage_data[:, start_idx:end_idx])

    central_voltage = np.concatenate(client_voltages, axis=0)
    central_clear_time = np.tile(clear_time, config.num_nodes)

    logging.info(f"Centralized voltage dataset shape: {central_voltage.shape}")
    return central_voltage, central_clear_time


def _prepare_deeponet_data(
        signals_db: np.ndarray, clearing_time: np.ndarray, config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a dataset suitable for a Cartesian Product DeepONet.

    This function interpolates signals and samples them at specified branch
    and trunk locations.

    Args:
        signals_db (np.ndarray): The database of input signals (e.g., voltage).
        clearing_time (np.ndarray): The clearing time for each signal.
        config (Config): The configuration object.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - U_data: Branch network inputs.
        - Y_data: Trunk network inputs (shared locations).
        - G_data: Network outputs.
        - sensor_locs: The locations of the branch sensors.
    """

    num_signals = signals_db.shape[0]
    original_time = np.linspace(-0.1, 12.0, num=signals_db.shape[1])

    # Define trunk locations (Y_data)
    trunk_time_start = config.max_clear + config.memory_ranging
    trunk_time_end = config.max_time
    fixed_trunk_locations = np.linspace(trunk_time_start, trunk_time_end, config.n_locs)
    Y_data = fixed_trunk_locations.reshape(-1, 1)

    # Define branch sensor locations
    time_u_full = np.linspace(-0.1, config.max_clear + config.memory_ranging + (0.1 * config.memory_ranging), num=10000)
    sensor_locs = np.linspace(time_u_full[0], time_u_full[-1], config.n_sensors)

    U_data = np.zeros((num_signals, config.n_sensors))
    G_data = np.zeros((num_signals, config.n_locs))

    plot_indices_for_verbose = np.random.randint(0, num_signals, size=3)
    for i in tqdm(range(num_signals), desc="Preparing DeepONet data"):
        # Create an interpolator for the full, original signal
        interpolated_signal = interpolate.interp1d(original_time, signals_db[i], copy=False, assume_sorted=True)

        # Create the branch input signal `u` by masking the original signal
        cut_section = clearing_time[i] + config.memory_ranging
        u_full_values = interpolated_signal(time_u_full)

        # Sample the branch and trunk data
        f_u = interpolate.interp1d(time_u_full, u_full_values, copy=False, assume_sorted=True)
        U_data[i, :] = f_u(sensor_locs)
        G_data[i, :] = interpolated_signal(fixed_trunk_locations)

        # Verbose plotting
        if config.verbose and i in plot_indices_for_verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(original_time, signals_db[i], 'm', label="Original Voltage Signal")
            plt.plot(time_u_full, u_full_values, 'b', label="Masked Input `u`")
            plt.plot(sensor_locs, U_data[i, :], '*k', ms=8, label=f"{config.n_sensors} Branch Sensor Values")
            plt.plot(fixed_trunk_locations, G_data[i, :], '.g', ms=10, label=f"{config.n_locs} Output G_values")
            plt.axvline(x=cut_section, color='r', ls='--', label=f"Cut Section for signal {i}")
            plt.title(f"Data Generation Check (Cartesian Product) - Signal {i}")
            plt.xlabel("Time")
            plt.ylabel("Voltage")
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

    logging.info(
        f"U_data (Branch) shape: {U_data.shape}, Y_data (Trunk) shape: {Y_data.shape}, G_data (Output) shape: {G_data.shape}")
    return U_data, Y_data, G_data, sensor_locs
"""

def _prepare_deeponet_data(
        signals_db: np.ndarray, clearing_time: np.ndarray, config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    Creates a dataset for DeepONet to predict post-fault dynamics
    from pre-fault dynamics.

    num_signals = signals_db.shape[0]
    original_time = np.linspace(-0.1, 12.0, num=signals_db.shape[1])

    # --- THIS IS THE KEY CHANGE (Part 1) ---
    # Define trunk locations (Y_data) as RELATIVE time since fault.
    # e.g., [0.0, 0.03, 0.06, ..., 3.0]
    # We'll assume a new config value: config.post_fault_duration
    # For this example, let's set it to 3.0
    post_fault_duration = 3.0
    Y_data = np.linspace(0.0, post_fault_duration, config.n_locs).reshape(-1, 1)

    # We also define the sensor locations for the input window.
    # Let's assume input window is from t=-0.1 up to the fault.
    # We'll sample at config.n_sensors points.
    # Note: sensor_locs_relative is not strictly needed, but good for plotting.
    sensor_locs_relative = np.linspace(-0.1, 0.0, config.n_sensors)  # Sample from t=-0.1 to t=0 (relative to fault)
    # Let's adjust this to be more like your code: window from -0.1 to fault
    pre_fault_start_time = -0.1
    # We will pass the *relative* sensor locations to the branch net
    # to be consistent. But wait, no, the branch net takes a *function*.
    # So we just sample the function. Let's simplify.

    U_data = np.zeros((num_signals, config.n_sensors))
    G_data = np.zeros((num_signals, config.n_locs))
    # This will store the *actual* sensor locations for each sample, for plotting
    sensor_locs_actual = np.zeros((num_signals, config.n_sensors))

    plot_indices_for_verbose = np.random.randint(0, num_signals, size=3)

    for i in tqdm(range(num_signals), desc="Preparing DeepONet data"):
        # Create an interpolator for the full, original signal
        interpolated_signal = interpolate.interp1d(original_time, signals_db[i], copy=False, assume_sorted=True,
                                                   bounds_error=False,
                                                   fill_value=(signals_db[i][0], signals_db[i][-1])
                                                   )

        fault_time = clearing_time[i]

        # --- THIS IS THE KEY CHANGE (Part 2) ---

        # 1. Branch Input (U_data) - PRE-FAULT
        # Sample from pre_fault_start_time (e.g., -0.1) up to the fault_time.
        input_sensor_locs = np.linspace(pre_fault_start_time, fault_time, config.n_sensors)

        # 1. Get the original data
        original_signal = interpolated_signal(input_sensor_locs)

        # 2. Apply the Discrete Cosine Transform (Type II is standard)
        # This transforms 30 signal points into 30 frequency coefficients
        dct_coeffs = dct(original_signal, type=2, norm='ortho')

        # 3. Use this as your new branch input
        U_data[i, :] = dct_coeffs.astype(np.float32)
        sensor_locs_actual[i, :] = input_sensor_locs  # Save for plotting

        # 2. Trunk Output (G_data) - POST-FAULT
        # Sample from fault_time up to (fault_time + post_fault_duration).
        # This corresponds to the *relative* Y_data we defined earlier.
        output_actual_locs = np.linspace(fault_time, fault_time + post_fault_duration, config.n_locs)
        G_data[i, :] = interpolated_signal(output_actual_locs)

        # Verbose plotting
        if config.verbose and i in plot_indices_for_verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(original_time, signals_db[i], 'm', ls='--', label="Original Voltage Signal")

            # Plot the input data
            plt.plot(input_sensor_locs, U_data[i, :], 'b.-', label=f"{config.n_sensors} Branch Input Values (U_data)")

            # Plot the output data
            plt.plot(output_actual_locs, G_data[i, :], 'g.-', ms=10, label=f"{config.n_locs} Output Values (G_data)")

            # Plot the fault line
            plt.axvline(x=fault_time, color='r', ls='--', label=f"Fault Time (t={fault_time:.2f})")

            plt.title(f"Data Generation (Corrected) - Signal {i}")
            plt.xlabel("Absolute Time")
            plt.ylabel("Voltage")
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

    logging.info(
        f"U_data (Branch) shape: {U_data.shape}, Y_data (Trunk) shape: {Y_data.shape}, G_data (Output) shape: {G_data.shape}")

    # Note: returning sensor_locs_actual (which is different for each sample) is tricky.
    # For a general plot, maybe just return the locations for the first sample.
    return U_data, Y_data, G_data, sensor_locs_actual[0, :]
"""


def _split_and_save_data(
    u_data: np.ndarray, y_data: np.ndarray, g_data: np.ndarray, sensor_locs: np.ndarray, config: Config
):
    """
    Splits the dataset into train, calibration, and test sets and saves them.

    Args:
        u_data (np.ndarray): Branch data.
        y_data (np.ndarray): Trunk data.
        g_data (np.ndarray): Output data.
        sensor_locs (np.ndarray): Branch sensor locations for plotting.
        config (Config): The configuration object.
    """
    logging.info(f"Splitting and saving the data to {OUTPUT_DIR}")

    num_signals = u_data.shape[0]
    indices = np.random.permutation(num_signals)

    train_end = int(num_signals * config.train)
    cal_end = train_end + int(num_signals * config.calibration)

    splits: Dict[str, np.ndarray] = {
        "train": indices[:train_end],
        "calibration": indices[train_end:cal_end],
        "test": indices[cal_end:],
    }

    for name, idx in splits.items():
        logging.info(f"Processing '{name}' split with {len(idx)} signals.")
        save_path = OUTPUT_DIR / f'{name}.npz'
        np.savez_compressed(
            save_path,
            X0=u_data[idx].astype(np.float32),
            X1=y_data.astype(np.float32),
            y=g_data[idx].astype(np.float32),
            X0_plot=sensor_locs.astype(np.float32)
        )
    logging.info("All files saved successfully!")


# --- Plotting and Debugging ---

def _filter_by_frequency(
        u_data: np.ndarray, g_data: np.ndarray, config: Config, y_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters the dataset based on the spectral centroid of the output signals.

    The spectral centroid represents the "center of mass" of the signal's
    power spectrum. Signals with lower centroids are dominated by low
    frequencies, while signals with higher centroids have more high-frequency
    content. This provides a continuous metric for effective filtering.

    Args:
        u_data (np.ndarray): Branch data.
        g_data (np.ndarray): Output data.
        config (Config): The configuration object.
        y_data (np.ndarray): Trunk data (for plotting and sampling interval).

    Returns:
        The filtered u_data and g_data.
    """
    if not (0.0 < config.frequency_filter_percentile < 1.0):
        logging.info("No frequency filtering applied (percentile is not between 0.0 and 1.0).")
        return u_data, g_data

    # Calculate the dominant frequency for each signal
    num_signals, n_locs = g_data.shape

    # Calculate sampling interval from the trunk coordinates
    unique_coords = np.unique(y_data, axis=0)
    sampling_interval = unique_coords[1, 0] - unique_coords[0, 0]

    # Pre-calculate frequency bins and Hann window
    frequencies = np.fft.fftfreq(n_locs, d=sampling_interval)[:n_locs // 2]
    hann_window = np.hanning(n_locs)
    spectral_centroids = np.zeros(num_signals)

    logging.info("Calculating dominant frequency for each signal...")
    for i in range(num_signals):
        signal = g_data[i, :]
        windowed_signal = signal * hann_window
        fft_values = np.fft.fft(windowed_signal)
        power_spectrum = np.abs(fft_values[:n_locs // 2]) ** 2

        # Calculate spectral centroid: sum(freq * power) / sum(power)
        # Add a small epsilon to avoid division by zero for silent signals
        spectral_centroids[i] = np.sum(frequencies * power_spectrum) / (np.sum(power_spectrum) + 1e-9)

    # Apply the percentile filter
    if config.filter_mode == FilterMode.KEEP_LOWEST:
        threshold = np.quantile(spectral_centroids, config.frequency_filter_percentile)
        keep_mask = spectral_centroids <= threshold
        percentage = config.frequency_filter_percentile * 100
        desc = f"KEEPING samples with the LOWEST {percentage:.0f}% dominant frequencies"
    else:  # KEEP_HIGHEST
        threshold = np.quantile(spectral_centroids, 1 - config.frequency_filter_percentile)
        keep_mask = spectral_centroids >= threshold
        percentage = config.frequency_filter_percentile * 100
        desc = f"KEEPING samples with the HIGHEST {percentage:.0f}% dominant frequencies"

    logging.info(f"Filtering dataset to {desc}.")

    if config.verbose:
        _plot_frequency_filter_diagnostics(
            spectral_centroids, threshold, desc, g_data, keep_mask, y_data
        )

    original_count = u_data.shape[0]
    filtered_u = u_data[keep_mask]
    filtered_g = g_data[keep_mask]
    logging.info(f"Filtering complete. Kept {filtered_u.shape[0]} of {original_count} samples.")
    return filtered_u, filtered_g


def _plot_frequency_filter_diagnostics(
        metric_values: np.ndarray, threshold: float, desc: str, g_unfiltered: np.ndarray,
        mask: np.ndarray, y_data: np.ndarray
):
    """
    Saves diagnostic plots for the frequency-based filtering step.

    Args:
        metric_values (np.ndarray): Array of the metric (e.g., spectral centroid) for each signal.
        threshold (float): The metric value used as the filter cutoff.
        desc (str): A description of the filtering action for plot titles.
        g_unfiltered (np.ndarray): The complete, unfiltered output data array.
        mask (np.ndarray): The boolean mask indicating which samples were kept.
        y_data (np.ndarray): The trunk coordinates for the x-axis of the plots.
    """

    # Distribution of dominant frequencies
    plt.figure(figsize=(10, 6))
    plt.hist(metric_values, bins=100, alpha=0.75, label="Spectral Centroid Distribution")
    plt.axvline(x=threshold, color='r', ls='--', lw=2, label=f"Threshold at {threshold:.2f} Hz")
    plt.title(f"Distribution of Spectral Centroids ({desc})")
    plt.xlabel("Spectral Centroid metric")
    plt.ylabel("Number of Samples")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

    # Examples of kept and discarded signals
    kept_indices = np.where(mask)[0]
    discarded_indices = np.where(~mask)[0]

    for i in range(min(3, len(kept_indices))):
        idx = kept_indices[i]
        plt.figure(figsize=(10, 6))
        plt.plot(y_data, g_unfiltered[idx, :], '.-g')
        plt.title(f"KEPT Sample (Index: {idx}) | Spectral Centroid metric: {metric_values[idx]:.2f}")
        plt.xlabel("Time")
        plt.ylabel("Voltage")
        plt.grid(True)
        plt.show()
        plt.close()

    for i in range(min(3, len(discarded_indices))):
        idx = discarded_indices[i]
        plt.figure(figsize=(10, 6))
        plt.plot(y_data, g_unfiltered[idx, :], '.-r')
        plt.title(f"DISCARDED Sample (Index: {idx}) | Spectral Centroid metric: {metric_values[idx]:.2f}")
        plt.xlabel("Time")
        plt.ylabel("Voltage")
        plt.grid(True)
        plt.show()
        plt.close()


def _perform_fourier_analysis(
    y_train: np.ndarray, y_data: np.ndarray
):
    """
    Performs a Fourier analysis on the training set targets.

    Args:
        y_train (np.ndarray): The training target signals.
        y_data (np.ndarray): The trunk coordinates (used to find sampling interval).
    """
    logging.info("Verbose mode: Performing Fourier analysis on training set targets...")

    # Calculate Sampling Interval from Trunk Coordinates
    unique_coords = np.unique(y_data, axis=0)
    if len(unique_coords) < 2:
        logging.warning("Not enough unique trunk coordinates to determine sampling interval.")
        return
    sampling_interval = unique_coords[1, 0] - unique_coords[0, 0]

    # Compute Average Power Spectrum with Windowing
    num_signals, n_locs = y_train.shape
    frequencies = np.fft.fftfreq(n_locs, d=sampling_interval)[:n_locs // 2]
    total_power_spectrum = np.zeros(n_locs // 2)

    power_spectra_all_signals = np.zeros((num_signals, n_locs // 2))

    hann_window = np.hanning(n_locs)

    for i in range(num_signals):
        signal = y_train[i, :]

        windowed_signal = signal * hann_window
        fft_values = np.fft.fft(windowed_signal)
        power = np.abs(fft_values[:n_locs // 2]) ** 2
        power_spectra_all_signals[i] = power
        total_power_spectrum += power

    avg_power_spectrum = total_power_spectrum / num_signals

    # Find and Log Dominant Frequencies
    # Exclude the DC component (index 0) for finding dominant peaks
    top_indices = np.argsort(avg_power_spectrum[1:])[-5:][::-1] + 1
    dominant_freqs = frequencies[top_indices]
    logging.info(f"Top 5 dominant frequencies in training set: {np.round(dominant_freqs, 2)} Hz")

    # Plot the results for verification
    with plt.style.context('seaborn-v0_8-whitegrid'):
        # Plot the power spectrum
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies[1:], avg_power_spectrum[1:])
        plt.title('Averaged Power Spectrum (with Hann Window)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)
        plt.yscale('log')
        plt.show()
        plt.close()

        # Plot an example of a windowed signal
        example_idx = np.random.randint(0, num_signals)
        plt.figure(figsize=(10, 6))
        plt.plot(y_data, y_train[example_idx, :], '.-', label='Original Signal')
        plt.plot(y_data, y_train[example_idx, :] * hann_window, 'r-', label='Signal after Hann Window')
        plt.title(f"Example of Hann Window Application (Signal #{example_idx})")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

    # Might be of use in the future
    # Investigate the Highest Frequency Contributors
    """
    # Find the index of the highest frequency value within the dominant_freqs array
    idx_of_highest_freq = np.argmax(dominant_freqs)

    # Use that index to get the actual highest frequency value
    highest_freq = dominant_freqs[idx_of_highest_freq]

    # Use that same index to get the corresponding bin index from the original power spectrum
    highest_freq_bin_index = top_indices[idx_of_highest_freq]

    # Find which signals have the most power in that specific frequency bin
    power_at_highest_freq = power_spectra_all_signals[:, highest_freq_bin_index]
    top_contributor_indices = np.argsort(power_at_highest_freq)[-3:][::-1]

    for i, signal_idx in enumerate(top_contributor_indices):
        plt.figure(figsize=(10, 6))
        plt.plot(y_data, y_train[signal_idx, :], '.-')
        plt.title(f"Signal #{signal_idx}, a Top Contributor to {highest_freq:.2f} Hz Component")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()
        plt.close()
    """


# --- Main Workflow ---

def run_workflow(config: Config):
    """
    Executes the full data generation and processing pipeline.

    Args:
        config (Config): The configuration object.
    """
    logging.info("Starting data generation workflow...")

    # Load and process raw data
    voltage_db, clear_time_db = _load_and_process_raw_data(config)

    # Prepare data for DeepONet
    u_data, y_data, g_data, sensor_locs = _prepare_deeponet_data(voltage_db, clear_time_db, config)

    # Filter dataset by variance
    u_data, g_data = _filter_by_frequency(u_data, g_data, config, y_data)

    if config.verbose:
        # Perform Fourier analysis on the training portion of the data
        num_train_samples = int(g_data.shape[0] * config.train)

        # This is performed only on the training set
        # Frequency filtering is performed on the entire dataset
        _perform_fourier_analysis(g_data[:num_train_samples], y_data)

    # Split and save the final datasets
    _split_and_save_data(u_data, y_data, g_data, sensor_locs, config)


# --- Entry Point ---

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Generate voltage prediction data for DeepONet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="default_offline_voltage",
                        help="Name of the config file in 'configs/data_generation'")
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config_path = Path("configs/data_generation") / f"{args.config}.yaml"
    config = load_config(str(config_path)) if args.config else Config()

    if args.override:
        apply_overrides(config, args.override)

    np.random.seed(config.seed)
    run_workflow(config)


if __name__ == "__main__":
    main()