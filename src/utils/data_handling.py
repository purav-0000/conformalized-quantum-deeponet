# utils/data_handling.py
import logging
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple

from matplotlib import pyplot as plt


def transform_input(x, min_val, max_val):
    d = x.shape[-1]

    # Adding 1e-8 for stability
    x = 2 * (x - min_val) / ((max_val - min_val) + 1e-8) - 1
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=-1, keepdims=True))

    # Ensure float32, perhaps np.sqrt(d) is float64
    return np.concatenate((x, x_d1), axis=-1, dtype=np.float32)


def normalize_bounds(x_train, x_test, x_cal):
    def get_min_max(idx):
        arrays = [x_train[idx], x_test[idx], x_cal[idx]]
        concatenated = np.concatenate(arrays, axis=0)
        return np.min(concatenated), np.max(concatenated)

    branch_min, branch_max = get_min_max(0)
    trunk_min, trunk_max = get_min_max(1)

    return {
        "branch_min": branch_min,
        "branch_max": branch_max,
        "trunk_min": trunk_min,
        "trunk_max": trunk_max,
    }


class DataHandler:
    """A class to handle loading and preprocessing of simulation datasets."""

    def __init__(self, data_dir: str, fourier_features: bool, online: bool):
        self.data_path = Path("data/processed_data") / data_dir
        self.fourier_features = fourier_features
        self.online = online

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        # Load all datasets at initialization
        (self.x_train, self.y_train, self.x_cal, self.y_cal, self.x_test, self.y_test,
         self.x_train_plot, _, self.x_test_plot) = self._load_dataset()

        # Add Fourier features to the trunk inputs
        if self.fourier_features:
            self.dominant_freqs = self.fourier_decomposition()
            self.x_train = (self.x_train[0], self._add_fourier_features(self.x_train[1]))
            self.x_test = (self.x_test[0], self._add_fourier_features(self.x_test[1]))
            self.x_cal = (self.x_cal[0], self._add_fourier_features(self.x_cal[1]))

        # Normalize and transform the datasets
        self._normalize_and_transform()

    def _load_dataset(self):
        train = np.load((self.data_path / 'train.npz'), allow_pickle=True)
        cal = np.load((self.data_path / 'calibration.npz'), allow_pickle=True)
        test = np.load((self.data_path / 'test.npz'), allow_pickle=True)

        return (train['X0'].astype(np.float32), train['X1'].astype(np.float32)), train['y'].astype(np.float32), \
            (cal['X0'].astype(np.float32), cal['X1'].astype(np.float32)), cal['y'].astype(np.float32), \
            (test['X0'].astype(np.float32), test['X1'].astype(np.float32)), test['y'].astype(np.float32), \
            train['X0_plot'].astype(np.float32), cal['X0_plot'].astype(np.float32), test['X0_plot'].astype(np.float32)

    def _normalize_and_transform(self):
        """Calculates bounds and applies transformations."""
        self.bounds = normalize_bounds(self.x_train, self.x_test, self.x_cal)

        self.x_train = self._transform_split_input(self.x_train)
        self.x_test = self._transform_split_input(self.x_test)
        self.x_cal = self._transform_split_input(self.x_cal)

    def _transform_split_input(self, x_split: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Applies transformation to a split (branch, trunk) input."""
        branch_transformed = transform_input(x_split[0], self.bounds["branch_min"], self.bounds["branch_max"])
        trunk_transformed = transform_input(x_split[1], self.bounds["trunk_min"], self.bounds["trunk_max"])
        return branch_transformed, trunk_transformed

    def fourier_decomposition(self):

        # Online dataset is 3D
        if not self.online:
            num_signals, n_locs = self.y_train.shape
        else:
            num_signals, n_locs, _ = self.y_train.shape

        # Determine frequencies from training set
        # Ensure shuffling does not mess with calculating sampling interval
        sorted_trunk = sorted(self.x_train[1], key=lambda row: row[-1])
        # Duplicates exist in online problems
        unique_coords = np.unique(sorted_trunk, axis=0)

        # Online dataset has 3D shape
        if not self.online:
            sampling_interval = unique_coords[1, 0] - unique_coords[0, 0]
        else:
            sampling_interval = unique_coords[0, 1, 0] - unique_coords[0, 0, 0]

        # Calculate the frequencies corresponding to the FFT output
        # We only need the positive frequencies for the one-sided power spectrum
        frequencies = np.fft.fftfreq(n_locs, d=sampling_interval)[:n_locs // 2]

        # Accumulate power spectra from all signals
        total_power_spectrum = np.zeros(n_locs // 2)
        for i in range(num_signals):
            signal = self.y_train[i, :] if not self.online else self.y_train[i, :, 0]
            fft_values = np.fft.fft(signal)
            # Compute power (squared magnitude) for positive frequencies
            power = np.abs(fft_values[0:n_locs // 2]) ** 2
            total_power_spectrum += power

        # Average the power spectrum across all signals
        average_power_spectrum = total_power_spectrum / num_signals

        # Identify the top 5 dominant frequencies (excluding the DC component)
        top_indices = np.argsort(average_power_spectrum[1:])[-5:][::-1] + 1
        dominant_freqs = frequencies[top_indices]

        logging.info(f"Top identified frequencues: {dominant_freqs}")

        return dominant_freqs

    def _add_fourier_features(self, trunk_input: np.ndarray) -> np.ndarray:
        """Adds Fourier features to the trunk input coordinates."""

        # Start with the original coordinate as the base feature
        feature_list = [trunk_input]

        # Add sine and cosine pairs for each dominant frequency
        for f in self.dominant_freqs:
            omega_t = 2 * np.pi * f * trunk_input
            feature_list.append(np.cos(omega_t))
            feature_list.append(np.sin(omega_t))

        # Concatenate all features into a single array
        augmented_trunk_input = np.concatenate(feature_list, axis=1 if not self.online else 2)

        return augmented_trunk_input.astype(np.float32)

