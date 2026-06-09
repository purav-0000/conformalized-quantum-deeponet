import argparse
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import secrets
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml

from src.utils.common import apply_overrides


# --- Constants ---

INPUT_FILE = Path("data/raw_data/voltage_dataset.npz")
OUTPUT_DIR = Path("data/processed_data/online_voltage")


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

    # Sliding Window Parameters
    memory_window_size: int = 100
    prediction_horizon: int = 1  # Fixed at 1 for predicting the next time step (t+1)
    time_domain_limits: Optional[List[float]] = field(default_factory=lambda: [4.0, 6.0])

    # Splitting Ratios
    train: float = 0.8
    calibration: float = 0.1
    # Test split is implicitly calculated as 1.0 - train_split - cal_split

    # Reproducibility and Debugging
    seed: Optional[int] = field(default_factory=lambda: secrets.randbits(32))
    verbose: bool = True

    # Filters dataset based on the variance of the *entire* signal before windowing
    frequency_filter_percentile: float = 1.0
    filter_mode: FilterMode = FilterMode.KEEP_LOWEST

    def __post_init__(self):
        """Ensure string values from YAML/overrides are converted to Enum."""
        if isinstance(self.filter_mode, str):
            self.filter_mode = FilterMode(self.filter_mode)


# --- Utility Functions ---

def load_config(yaml_path: str) -> Config:
    """Loads configuration from a YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Core Logic Functions ---

def _load_and_centralize_raw_data(config: Config) -> np.ndarray:
    """
    Loads raw client data and concatenates it into a single centralized dataset.

    Args:
        config (Config): The configuration object.

    Returns:
        np.ndarray: The centralized voltage dataset.
    """
    logging.info(f"Loading raw data from {INPUT_FILE}")
    raw_data = np.load(INPUT_FILE)
    voltage_data = raw_data['voltage_data']

    # Shuffle the entire dataset before processing
    indices = np.random.permutation(voltage_data.shape[0])
    voltage_data = voltage_data[indices]

    # Storing client-by-client into one centralized array.
    client_voltages = []
    for client in range(config.num_nodes):
        start_idx = client * config.num_data_per_client
        # The original script had -1 for some reason added to end_idx
        end_idx = (client + 1) * config.num_data_per_client
        client_voltages.append(voltage_data[:, start_idx:end_idx])

    central_voltage = np.concatenate(client_voltages, axis=0)

    logging.info(f"Centralized voltage dataset shape: {central_voltage.shape}")
    return central_voltage


def _create_sliding_window_dataset(
        signals: np.ndarray, sliced_time_grid: np.ndarray, config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a dataset from signals using a sliding window approach.

    Args:
        signals (np.ndarray): A 2D array of time-series data (num_signals, num_timesteps).
        sliced_time_grid (np.ndarray): Time grid constrained by user specified time limits
        config (Config): The configuration object.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - U_data (np.ndarray): Input windows, shape (num_signals, num_windows_per_signal, window_size).
        - U_time (np.ndarray): Plotting purposes, shape (num_signals, num_windows_per_signal, window_size)
        - Y_data (np.ndarray): Trunk coordinates, shape (num_signals, num_windows_per_signal, 1).
        - G_data (np.ndarray): Target, shape (num_signals, num_windows_per_signal, 1).
    """
    window_size = config.memory_window_size
    horizon = config.prediction_horizon
    num_signals, signal_len = signals.shape

    # Calculate how many windows can be extracted from each signal
    num_windows_per_signal = signal_len - window_size - horizon + 1
    if num_windows_per_signal <= 0:
        raise ValueError("Signal length is too short to create any windows with the given settings.")

    total_windows = num_signals * num_windows_per_signal
    logging.info(
        f"Extracting {num_windows_per_signal} windows from each of the {num_signals} signals, for a total of {total_windows} windows.")

    # Pre-allocate arrays
    U_data = np.zeros((num_signals, num_windows_per_signal, window_size), dtype=np.float32)
    # For plotting
    U_time = np.zeros((num_signals, num_windows_per_signal, window_size), dtype=np.float32)
    Y_data = np.zeros((num_signals, num_windows_per_signal, 1), dtype=np.float32)
    G_data = np.zeros((num_signals, num_windows_per_signal, 1), dtype=np.float32)

    # Fill the arrays
    # current_window_idx = 0
    for i in tqdm(range(num_signals), desc="Creating sliding windows"):
        for j in range(num_windows_per_signal):
            end_idx = j + window_size
            U_data[i, j] = signals[i, j:end_idx]
            U_time[i, j] = sliced_time_grid[j:end_idx]
            Y_data[i, j] = sliced_time_grid[end_idx]
            G_data[i, j] = signals[i, end_idx]

    # Plotting stuffs
    if config.verbose:
        for i in range(3):
            # Plot one example to verify the logic
            plt.figure(figsize=(12, 6))
            sample_signal_to_plot = np.random.randint(0, num_signals)
            sample_window_to_plot = np.random.randint(0, num_windows_per_signal)

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

    return U_data, U_time, Y_data, G_data


# --- Filtering and plotting ---

def _filter_by_frequency(
        u_data: np.ndarray, u_time, g_data: np.ndarray, config: Config, y_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the dataset based on the spectral centroid of the output signals.

    The spectral centroid represents the "center of mass" of the signal's
    power spectrum. Signals with lower centroids are dominated by low
    frequencies, while signals with higher centroids have more high-frequency
    content. This provides a continuous metric for effective filtering.

    Args:
        u_data (np.ndarray): Branch data.
        u_time (np.ndarray): Time coordinates for the branch
        g_data (np.ndarray): Output data.
        config (Config): The configuration object.
        y_data (np.ndarray): Trunk data.

    Returns:
        The filtered u_data and g_data.
    """
    if not (0.0 < config.frequency_filter_percentile < 1.0):
        logging.info("No frequency filtering applied (percentile is not between 0.0 and 1.0).")
        return u_data, u_time, y_data, g_data

    # Calculate the dominant frequency for each signal
    num_signals, n_locs, _ = g_data.shape

    # Calculate sampling interval from the trunk coordinates
    unique_coords = np.unique(y_data, axis=0)
    # (num_signals. n_locs, 1)
    sampling_interval = unique_coords[0, 1, 0] - unique_coords[0, 0, 0]

    # Pre-calculate frequency bins and Hann window
    frequencies = np.fft.fftfreq(n_locs, d=sampling_interval)[:n_locs // 2]
    hann_window = np.hanning(n_locs)
    spectral_centroids = np.zeros(num_signals)

    logging.info("Calculating dominant frequency for each signal...")
    for i in range(num_signals):
        signal = g_data[i, :, 0]
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
    filtered_u_time = u_time[keep_mask]
    filtered_g = g_data[keep_mask]
    filtered_y = y_data[keep_mask]
    logging.info(f"Filtering complete. Kept {filtered_u.shape[0]} of {original_count} samples.")
    return filtered_u, filtered_u_time, filtered_y, filtered_g


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
        plt.plot(y_data[idx, :, 0], g_unfiltered[idx, :, 0], '.-g')
        plt.title(f"KEPT Sample (Index: {idx}) | Spectral Centroid metric: {metric_values[idx]:.2f}")
        plt.xlabel("Time")
        plt.ylabel("Voltage")
        plt.grid(True)
        plt.show()
        plt.close()

    for i in range(min(3, len(discarded_indices))):
        idx = discarded_indices[i]
        plt.figure(figsize=(10, 6))
        plt.plot(y_data[idx, :, 0], g_unfiltered[idx, :, 0], '.-r')
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
    sampling_interval = unique_coords[0, 1, 0] - unique_coords[0, 0, 0]

    # Compute Average Power Spectrum with Windowing
    num_signals, n_locs, _ = y_train.shape
    frequencies = np.fft.fftfreq(n_locs, d=sampling_interval)[:n_locs // 2]
    total_power_spectrum = np.zeros(n_locs // 2)

    power_spectra_all_signals = np.zeros((num_signals, n_locs // 2))

    hann_window = np.hanning(n_locs)

    for i in range(num_signals):
        signal = y_train[i, :, 0]

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
        plt.plot(y_data[example_idx, :, 0], y_train[example_idx, :, 0], '.-', label='Original Signal')
        plt.plot(y_data[example_idx, :, 0], y_train[example_idx, :, 0] * hann_window, 'r-',
                 label='Signal after Hann Window')
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


def save_quantum_sim_dataset(config: Config, centralized_voltage: np.ndarray):
    """
    For saving a sampled, small dataset for quantum simulation.
    Uses random sampling for calibration windows to maximize speed.
    Constructs 'pseudo-signals' for calibration to maintain (Signal, Window, Feature) shape.
    """
    logging.info("Starting targeted quantum simulation dataset generation...")

    # Shuffle signals to ensure random selection of source signals
    np.random.shuffle(centralized_voltage)

    # Slice signals to the specified time domain
    original_time_len = centralized_voltage.shape[1]
    full_time_grid = np.linspace(-0.1, 12.0, num=original_time_len)

    limits = config.time_domain_limits
    time_mask = (full_time_grid >= limits[0]) & (full_time_grid <= limits[1])
    centralized_voltage = centralized_voltage[:, time_mask]
    original_signal_time = full_time_grid[time_mask]

    """
    # --- CHANGE START: Global High-Variance Filter ---
    # 1. Calculate variance for ALL signals in the pool
    signal_variances = np.var(centralized_voltage, axis=1)

    # 2. Keep only the Top 10% (90th percentile)
    variance_threshold = np.percentile(signal_variances, 90)
    high_var_indices = np.where(signal_variances >= variance_threshold)[0]

    # 3. Overwrite the main array with ONLY high-variance signals
    # centralized_voltage = centralized_voltage[high_var_indices]

    logging.info(f"Filtered pool to top 10% variance. New pool size: {len(centralized_voltage)}")

    # 4. Now shuffle this "hard" pool so Train/Cal/Test are random selections from it
    np.random.shuffle(centralized_voltage)
    """
    # --- SIMULATION DATASET PARAMETERS ---

    # Constraint: All splits must have the same number of windows per signal (dim 1)
    WINDOWS_PER_SIGNAL = 60

    # 1. Training Parameters (Required to maintain dataset structure)
    NUM_TRAIN_SIGNALS = 10

    # 2. Test Parameters (Real, contiguous signal for plotting)
    NUM_TEST_SIGNALS = 7

    # 3. Calibration Parameters (Randomly sampled windows)
    # We target ~700 samples for robust Conformal Prediction
    CAL_TARGET_TOTAL_SAMPLES = 420

    # Calculate how many "Pseudo-Signals" we need to hold these samples
    # e.g., if we need 100 samples and windows_per_signal is 20, we need 5 pseudo-signals.
    NUM_CAL_PSEUDO_SIGNALS = int(np.ceil(CAL_TARGET_TOTAL_SAMPLES / WINDOWS_PER_SIGNAL))
    CAL_ACTUAL_TOTAL_SAMPLES = NUM_CAL_PSEUDO_SIGNALS * WINDOWS_PER_SIGNAL

    # Reserve pools of raw data
    # We reserve a large pool for calibration to ensure good random diversity
    POOL_TRAIN_SIZE = NUM_TRAIN_SIGNALS
    POOL_CAL_SIZE = 50

    TRAIN_END_IDX = POOL_TRAIN_SIZE
    CAL_END_IDX = TRAIN_END_IDX + POOL_CAL_SIZE

    # Slice the arrays sequentially.
    # Because we filtered the whole pool, these are ALL high-variance.
    train_signals_raw = centralized_voltage[0:TRAIN_END_IDX]
    cal_signals_pool_raw = centralized_voltage[TRAIN_END_IDX:CAL_END_IDX]

    # Take the test signal immediately following the calibration pool
    test_signals_raw = centralized_voltage[CAL_END_IDX: CAL_END_IDX + NUM_TEST_SIGNALS]

    # Store the original test signal voltage for plotting
    original_test_voltage = test_signals_raw[0, :]

    logging.info(f"(QSIM) Configuration:")
    logging.info(f"  - Windows Per Signal (Fixed): {WINDOWS_PER_SIGNAL}")
    logging.info(f"  - Training: {NUM_TRAIN_SIGNALS} real signals")
    logging.info(
        f"  - Calibration: {CAL_ACTUAL_TOTAL_SAMPLES} random windows (grouped into {NUM_CAL_PSEUDO_SIGNALS} pseudo-signals)")
    logging.info(f"  - Test: {NUM_TEST_SIGNALS} real signal")

    # --- 1. PROCESSING TRAINING SET (Real Signals) ---
    logging.info("(QSIM) Processing training signals...")
    U_train, U_time_train, Y_train, G_train = _create_sliding_window_dataset(
        train_signals_raw, original_signal_time, config
    )

    # Slice to fixed window count
    U_train = U_train[:, :WINDOWS_PER_SIGNAL, :]
    U_time_train = U_time_train[:, :WINDOWS_PER_SIGNAL, :]
    Y_train = Y_train[:, :WINDOWS_PER_SIGNAL, :]
    G_train = G_train[:, :WINDOWS_PER_SIGNAL, :]

    train_save_path = OUTPUT_DIR / 'train.npz'
    np.savez_compressed(
        train_save_path,
        X0=U_train.astype(np.float32),
        X1=Y_train.astype(np.float32),
        y=G_train.astype(np.float32),
        X0_plot=U_time_train.astype(np.float32)
    )
    logging.info(f"(QSIM) Training set saved. Shape: {U_train.shape}")

    # --- 2. PROCESSING CALIBRATION SET (Randomly Sampled Pseudo-Signals) ---
    logging.info(f"(QSIM) Generating calibration pool from {POOL_CAL_SIZE} raw signals...")

    # Generate ALL windows from the pool
    U_cal_pool, U_time_cal_pool, Y_cal_pool, G_cal_pool = _create_sliding_window_dataset(
        cal_signals_pool_raw, original_signal_time, config
    )

    # Flatten the pool: (Signals, Windows, Features) -> (Total_Windows, Features)
    win_size = U_cal_pool.shape[2]
    U_cal_flat = U_cal_pool.reshape(-1, win_size)
    U_time_cal_flat = U_time_cal_pool.reshape(-1, win_size)
    Y_cal_flat = Y_cal_pool.reshape(-1, 1)
    G_cal_flat = G_cal_pool.reshape(-1, 1)

    logging.info(f"(QSIM) Sampling {CAL_ACTUAL_TOTAL_SAMPLES} windows from pool of {U_cal_flat.shape[0]} windows...")

    if U_cal_flat.shape[0] < CAL_ACTUAL_TOTAL_SAMPLES:
        raise ValueError("Not enough windows in calibration pool to satisfy sampling requirement.")

    # Randomly select indices
    sample_indices = np.random.choice(U_cal_flat.shape[0], size=CAL_ACTUAL_TOTAL_SAMPLES, replace=False)

    # Reshape into Pseudo-Signals to match (Signals, Windows, Features) structure
    # This creates "signals" that are just collections of random independent windows
    U_cal_sampled = U_cal_flat[sample_indices].reshape(NUM_CAL_PSEUDO_SIGNALS, WINDOWS_PER_SIGNAL, -1)
    U_time_cal_sampled = U_time_cal_flat[sample_indices].reshape(NUM_CAL_PSEUDO_SIGNALS, WINDOWS_PER_SIGNAL, -1)
    Y_cal_sampled = Y_cal_flat[sample_indices].reshape(NUM_CAL_PSEUDO_SIGNALS, WINDOWS_PER_SIGNAL, -1)
    G_cal_sampled = G_cal_flat[sample_indices].reshape(NUM_CAL_PSEUDO_SIGNALS, WINDOWS_PER_SIGNAL, -1)

    cal_save_path = OUTPUT_DIR / 'calibration.npz'
    np.savez_compressed(
        cal_save_path,
        X0=U_cal_sampled.astype(np.float32),
        X1=Y_cal_sampled.astype(np.float32),
        y=G_cal_sampled.astype(np.float32),
        X0_plot=U_time_cal_sampled.astype(np.float32)
    )
    logging.info(f"(QSIM) Calibration set saved. Shape: {U_cal_sampled.shape}")

    # --- 3. PROCESSING TEST SET (Real Signal) ---
    logging.info("(QSIM) Processing test signal...")
    U_test, U_time_test, Y_test, G_test = _create_sliding_window_dataset(
        test_signals_raw, original_signal_time, config
    )

    # Slice to fixed window count
    U_test = U_test[:, :WINDOWS_PER_SIGNAL, :]
    U_time_test = U_time_test[:, :WINDOWS_PER_SIGNAL, :]
    Y_test = Y_test[:, :WINDOWS_PER_SIGNAL, :]
    G_test = G_test[:, :WINDOWS_PER_SIGNAL, :]

    test_save_path = OUTPUT_DIR / 'test.npz'
    np.savez_compressed(
        test_save_path,
        X0=U_test.astype(np.float32),
        X1=Y_test.astype(np.float32),
        y=G_test.astype(np.float32),
        X0_plot=U_time_test.astype(np.float32),
        original_time=original_signal_time.astype(np.float32),
        original_voltage=original_test_voltage.astype(np.float32)
    )
    logging.info(f"(QSIM) Test set saved to {test_save_path}")
    logging.info(f"(QSIM)  - X0 shape: {U_test.shape}")

    logging.info("--- Data for Single Test Signal Plotting (Terminal Output) ---")

    # Y_test and G_test have shape (1, 10, 1) from the single test signal
    # We extract the data by indexing [0, :, 0] to get a 1D array
    trunk_coordinates = Y_test[0, :, 0]
    ground_truth_targets = G_test[0, :, 0]

    print("\n\n================== SINGLE TEST SIGNAL DATA (COPY BELOW) ==================")

    print("\n--- Plotting Data (Window Targets) ---")
    print("\n# Trunk Coordinates (Y_test - time points for each target)")
    # Use repr() for a clean, copy-pasteable array format
    print(f"plot_time = {repr(trunk_coordinates)}")

    print("\n# Ground Truth Targets (G_test - voltage value at each time point)")
    print(f"plot_voltage_truth = {repr(ground_truth_targets)}")

    print("\n\n--- Original Full Signal (for context) ---")
    print("\n# Original Time Coords (Full)")
    print(f"original_time = {repr(original_signal_time)}")

    print("\n# Original Voltage Values (Full)")
    print(f"original_voltage = {repr(original_test_voltage)}")

    print("\n======================= END OF SINGLE TEST SIGNAL DATA =====================\n")
    logging.info("Data for single test signal printed to console.")


# --- Main workflow ---
def run_workflow(config: Config):
    """
    Executes the full data generation and processing pipeline.

    Args:
        config (Config): The configuration object.
    """

    logging.info("Starting sliding window data generation workflow...")
    np.random.seed(config.seed)

    # Load and centralize raw data
    centralized_voltage = _load_and_centralize_raw_data(config)
    # Save a small part of the dataset
    save_quantum_sim_dataset(config, centralized_voltage)

    np.random.shuffle(centralized_voltage)  # Shuffle signals before processing

    # Slice signals to the specified time domain
    original_time_len = centralized_voltage.shape[1]
    full_time_grid = np.linspace(-0.1, 12.0, num=original_time_len)

    limits = config.time_domain_limits
    time_mask = (full_time_grid >= limits[0]) & (full_time_grid <= limits[1])
    centralized_voltage = centralized_voltage[:, time_mask]
    logging.info(f"Slicing to time domain {limits}s. New signal shape: {centralized_voltage.shape}")

    # Create the sliding window dataset
    # Pass time mask to keep track of limited time when creating windows
    U_data, U_time, Y_data, G_data = _create_sliding_window_dataset(centralized_voltage, full_time_grid[time_mask],
                                                                    config)

    # Filter by frequency
    U_data, U_time, Y_data, G_data = _filter_by_frequency(U_data, U_time, G_data, config, Y_data)

    if config.verbose:
        _perform_fourier_analysis(G_data, Y_data)

    # Split data into train, calibration, and test sets
    logging.info(f"Splitting and saving the data to {OUTPUT_DIR}")

    num_signals = U_data.shape[0]
    indices = np.random.permutation(num_signals)

    train_end = int(num_signals * config.train)
    cal_end = train_end + int(num_signals * config.calibration)

    splits: Dict[str, np.ndarray] = {
        "train": indices[:train_end],
        "calibration": indices[train_end:cal_end],
        "test": indices[cal_end:],
    }

    # Plot an entire signal
    if config.verbose:
        test_branch = U_data[splits["test"]]
        test_trunk = Y_data[splits["test"]]
        test_coords = U_time[splits["test"]]
        test_target = G_data[splits["test"]]

        random_indices = np.random.randint(0, len(test_branch), size=3)
        for i in random_indices:
            plt.figure(figsize=(12, 6))
            plt.plot(test_coords[i, :], test_branch[i, :], 'bo-', alpha=0.9, label="Input windows concatenated")
            plt.plot(test_trunk[i, :], test_target[i, :], 'r*', markersize=12, label="Target values")
            plt.title("Windows across 1 signal")

            # Remove duplicates
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.grid(True)
            plt.show()
            plt.close()

    for name, idx in splits.items():
        logging.info(f"Processing '{name}' split with {U_data[idx].shape[0] * U_data[idx].shape[1]} windows.")
        save_path = OUTPUT_DIR / f'{name}_classical.npz'
        np.savez_compressed(
            save_path,
            X0=U_data[idx].astype(np.float32),
            X1=Y_data[idx].astype(np.float32),
            y=G_data[idx].astype(np.float32),
            X0_plot=U_time[idx].astype(np.float32)
        )
    logging.info("All files saved successfully!")


# --- Entry Point ---
def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Generate voltage prediction data for DeepONet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="default_online_voltage",
                        help="Name of the config file in 'configs/data_generation'")
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config_path = Path("configs/data_generation") / f"{args.config}.yaml"
    config = load_config(str(config_path)) if args.config else Config()

    if args.override:
        apply_overrides(config, args.override)

    # Set seeds for reproducibility
    np.random.seed(config.seed)
    run_workflow(config)


if __name__ == "__main__":
    main()
