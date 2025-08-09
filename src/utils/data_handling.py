# utils/data_handling.py

import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple


def transform_input(x, min_val, max_val):
    d = x.shape[1]

    # Adding 1e-8 for stability
    x = 2 * (x - min_val) / ((max_val - min_val) + 1e-8) - 1
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))

    # Ensure float32, perhaps np.sqrt(d) is float64
    return np.concatenate((x, x_d1), axis=1, dtype=np.float32)


def normalize_bounds(x_train, x_test, x_val, x_cal):
    def get_min_max(idx):
        arrays = [x_train[idx], x_test[idx], x_val[idx], x_cal[idx]]
        concatenated = np.concatenate(arrays, axis=0)
        return np.min(concatenated, axis=0), np.max(concatenated, axis=0)

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

    def __init__(self, data_dir: str):
        self.data_path = Path("data") / data_dir
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        # Load all datasets at initialization
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, self.x_test_plot = self._load_dataset()
        self.x_cal, self.y_cal = self._load_calibration_dataset()

        # Add Fourier features to the trunk inputs
        self.x_train = (self.x_train[0], self._add_fourier_features(self.x_train[1]))
        self.x_val = (self.x_val[0], self._add_fourier_features(self.x_val[1]))
        self.x_test = (self.x_test[0], self._add_fourier_features(self.x_test[1]))
        self.x_cal = (self.x_cal[0], self._add_fourier_features(self.x_cal[1]))

        # Normalize and transform the datasets
        # self._normalize_and_transform()

    def _load_dataset(self):
        train = np.load((self.data_path / 'picked_aligned_train.npz'), allow_pickle=True)
        val = np.load((self.data_path / 'picked_aligned_val.npz'), allow_pickle=True)
        test = np.load((self.data_path / 'picked_aligned_test.npz'), allow_pickle=True)

        return (train['X0'].astype(np.float32), train['X1'].astype(np.float32)), train['y'].astype(np.float32), \
            (val['X0'].astype(np.float32), val['X1'].astype(np.float32)), val['y'].astype(np.float32), \
            (test['X0'].astype(np.float32), test['X1'].astype(np.float32)), test['y'].astype(np.float32), \
            test['X0_plot'].astype(np.float32)

    def _load_calibration_dataset(self):
        cal = np.load((self.data_path / 'picked_aligned_calibration.npz'), allow_pickle=True)

        return (cal['X0'].astype(np.float32), cal['X1'].astype(np.float32)), cal['y'].astype(np.float32)

    def _normalize_and_transform(self):
        """Calculates bounds and applies transformations."""
        self.bounds = normalize_bounds(self.x_train, self.x_test, self.x_val, self.x_cal)

        self.x_train = self._transform_split_input(self.x_train)
        self.x_val = self._transform_split_input(self.x_val)
        self.x_test = self._transform_split_input(self.x_test)
        self.x_cal = self._transform_split_input(self.x_cal)

    def _transform_split_input(self, x_split: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Applies transformation to a split (branch, trunk) input."""
        branch_transformed = transform_input(x_split[0], self.bounds["branch_min"], self.bounds["branch_max"])
        trunk_transformed = transform_input(x_split[1], self.bounds["trunk_min"], self.bounds["trunk_max"])
        return (branch_transformed, trunk_transformed)

    def _add_fourier_features(self, trunk_input: np.ndarray) -> np.ndarray:
        """Adds Fourier features to the trunk input coordinates."""
        """
        Adds data-driven Fourier features to the trunk input coordinates.
        """
        # Frequencies identified from your FFT analysis of G_train
        dominant_freqs = [0.16, 0.64, 1.61, 1.12, 0.32, 1.77, 0.96, 1.45]

        # Start with the original coordinate as the base feature
        feature_list = [trunk_input]
        # feature_list = []

        # Add sine and cosine pairs for each dominant frequency
        for f in dominant_freqs:
            # The argument for the trig functions is omega*t = 2*pi*f*t
            omega_t = 2 * np.pi * f * trunk_input
            feature_list.append(np.cos(omega_t))
            feature_list.append(np.sin(omega_t))

        # Concatenate all features into a single array
        # The final shape will be (nLocs, 1 + 2 * len(dominant_freqs))
        augmented_trunk_input = np.concatenate(feature_list, axis=1)

        return augmented_trunk_input.astype(np.float32)


