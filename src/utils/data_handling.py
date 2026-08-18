import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


def transform_input(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Normalizes input data to [-1, 1] and projects it onto a hypersphere.

    This transformation is often used to prepare classical data for encoding
    into quantum states.

    Args:
        x (np.ndarray): The input data array.
        min_val (float): The minimum value for normalization.
        max_val (float): The maximum value for normalization.

    Returns:
        np.ndarray: The transformed data, with an added dimension.
    """
    # Adding a small epsilon for numerical stability if max_val equals min_val
    epsilon = 1e-8
    d = x.shape[-1]

    # Clip evaluation inputs to the training range, then normalize to [-1, 1].
    x_clipped = np.clip(x, min_val, max_val)
    x_normalized = 2 * (x_clipped - min_val) / ((max_val - min_val) + epsilon) - 1

    # Scale by sqrt(d)
    x_scaled = x_normalized / np.sqrt(d)

    # Calculate the new dimension to project onto the hypersphere
    sum_sq = np.sum(x_scaled ** 2, axis=-1, keepdims=True)
    x_d1 = np.sqrt(1 - sum_sq)

    return np.concatenate((x_scaled, x_d1), axis=-1).astype(np.float32)


def _calculate_bounds(x_train):
    branch_min, branch_max = np.min(x_train[0]), np.max(x_train[0])
    trunk_min, trunk_max = np.min(x_train[1]), np.max(x_train[1])

    return {
        "branch_min": branch_min,
        "branch_max": branch_max,
        "trunk_min": trunk_min,
        "trunk_max": trunk_max,
    }


class DataHandler:
    """
    A class to handle loading, preprocessing, and transforming datasets.

    This class follows a two-step initialization:
    1. `__init__`: Sets up configuration.
    2. `load_and_process_data()`: Loads and processes all data, preparing it for use.
    """

    def __init__(self, data_dir: str, fourier_features: bool, online: bool):
        """
        Initializes the DataHandler with a configuration.

        Args:
            fourier_features (bool): If True, trunk input is augmented with Fourier features
            online (bool): If True, data processing is adapted for online 3D datasets
        """
        self.data_path = Path("data/processed_data") / data_dir
        self.fourier_features = fourier_features
        self.online = online

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        self.datasets: Dict[str, Dict] = {}
        self.bounds: Dict[str, float] = {}
        self.dominant_freqs: Optional[np.ndarray] = None

    def load_and_process_data(self):
        """Loads all data splits and applies all preprocessing steps."""
        self._load_datasets()
        if self.fourier_features:
            self._apply_fourier_features()
        self._normalize_and_transform()
        logging.info("Data loading and processing complete.")

    def _load_datasets(self):
        """
        Loads the train, calibration, and test datasets from .npz files.
        """
        logging.info(f"Loading datasets from {self.data_path}...")
        for split in ["train", "calibration", "test"]:
            data = np.load(self.data_path / f'{split}.npz')
            self.datasets[split] = {
                'X': (data['X0'].astype(np.float32), data['X1'].astype(np.float32)),
                'y': data['y'].astype(np.float32),
                'X0_plot': data.get('X0_plot', None),  # Use .get for optional keys
            }

    def _apply_fourier_features(self):
        """Calculates and adds Fourier features to the trunk inputs."""
        logging.info("Applying Fourier features...")
        sampling_interval = self._calculate_sampling_interval()
        self.dominant_freqs = self._compute_dominant_frequencies(sampling_interval)

        for split in self.datasets:
            branch, trunk = self.datasets[split]['X']
            augmented_trunk = self._add_fourier_features(trunk)
            self.datasets[split]['X'] = (branch, augmented_trunk)

    def _calculate_sampling_interval(self) -> float:
        """
        Calculates the sampling interval from the training set's trunk coordinates.

        Returns:
            float: The calculated sampling interval.
        """
        trunk_coords = self.datasets['train']['X'][1]

        # `np.unique` is efficient and returns a sorted array.
        unique_coords = np.unique(trunk_coords, axis=0)

        if self.online:
            # For 3D online data (num_signals. n_locs, 1)
            sampling_interval = unique_coords[0, 1, 0] - unique_coords[0, 0, 0]
        else:
            # For 2D offline data (n_locs, 1)
            sampling_interval = unique_coords[1, 0] - unique_coords[0, 0]

        return sampling_interval

    def _compute_dominant_frequencies(self, sampling_interval: float) -> np.ndarray:
        """
        Performs FFT on the training output signals to find dominant frequencies.

        Args:
            sampling_interval (float): The time step between trunk data points.

        Returns:
            np.ndarray: An array of the most dominant frequencies.
        """
        y_train = self.datasets['train']['y']
        n_locs = y_train.shape[1]
        frequencies = np.fft.fftfreq(n_locs, d=sampling_interval)[:n_locs // 2]
        total_power_spectrum = np.zeros(n_locs // 2)

        # Hanning window to handle discontinuity at the ends
        hann_window = np.hanning(n_locs)

        for i in range(y_train.shape[0]):
            # Online dataset is (num_signals, n_locs, 1), offline dataset is (num_signals, n_locs)
            signal = y_train[i, :, 0] if self.online else y_train[i, :]

            windowed_signal = signal * hann_window
            fft_values = np.fft.fft(windowed_signal)
            power = np.abs(fft_values[:n_locs // 2]) ** 2
            total_power_spectrum += power

        avg_power_spectrum = total_power_spectrum / y_train.shape[0]

        # Get top 5 frequencies (excluding DC component at index 0)
        top_indices = np.argsort(avg_power_spectrum[1:])[-5:][::-1] + 1
        dominant_freqs = frequencies[top_indices]

        logging.info(f"Top 5 identified frequencies: {np.round(dominant_freqs, 2)}")
        return dominant_freqs

    def _add_fourier_features(self, trunk_input: np.ndarray) -> np.ndarray:
        """
        Augments trunk coordinates with sine and cosine features for each dominant frequency.

        Args:
            trunk_input (np.ndarray): The original trunk input data.

        Returns:
            np.ndarray: The trunk data with added Fourier features.
        """
        feature_list = [trunk_input]

        for f in self.dominant_freqs:
            omega_t = 2 * np.pi * f * trunk_input
            feature_list.append(np.cos(omega_t))
            feature_list.append(np.sin(omega_t))

        return np.concatenate(feature_list, axis=-1).astype(np.float32)

    def _normalize_and_transform(self):
        """Calculates normalization bounds and applies the hypersphere transformation."""
        logging.info("Normalizing and transforming datasets...")
        self.bounds = _calculate_bounds(self.datasets['train']['X'])

        for split in self.datasets:
            branch, trunk = self.datasets[split]['X']
            branch_transformed = transform_input(branch, self.bounds["branch_min"], self.bounds["branch_max"])
            trunk_transformed = transform_input(trunk, self.bounds["trunk_min"], self.bounds["trunk_max"])
            self.datasets[split]['X'] = (branch_transformed, trunk_transformed)

    def get_split(self, split_name: str) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Returns the processed data for a specific split.

        Args:
            split_name (str): The name of the split ('train', 'test', 'calibration').

        Returns:
            A tuple containing the processed (X0, X1) inputs and the y targets.
        """
        if not self.datasets:
            raise RuntimeError("Data not loaded. Call `load_and_process_data()` first.")
        return self.datasets[split_name]['X'], self.datasets[split_name]['y']


