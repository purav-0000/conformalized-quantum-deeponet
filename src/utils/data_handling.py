# utils/data_handling.py

import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple


def transform_input(x, min_val, max_val):
    d = x.shape[1]
    x = 2 * (x - min_val) / (max_val - min_val) - 1
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)


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

        # Normalize and transform the datasets
        self._normalize_and_transform()

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