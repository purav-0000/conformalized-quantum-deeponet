"""Numerically stable helpers for adaptive conformal intervals."""

import numpy as np


DEFAULT_EPSILON = 1e-8


def _ensemble_statistics(ensemble_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(ensemble_predictions)
    if predictions.ndim < 2:
        raise ValueError("ensemble_predictions must include a model axis")
    return predictions.mean(axis=0), predictions.std(axis=0)


def adaptive_nonconformity(
    truth: np.ndarray,
    ensemble_predictions: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Return finite adaptive scores, including where ensemble spread is zero."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean, spread = _ensemble_statistics(ensemble_predictions)
    truth = np.asarray(truth)
    if mean.shape != truth.shape:
        raise ValueError(f"prediction/truth shapes differ: {mean.shape} != {truth.shape}")
    return np.abs(truth - mean) / (spread + epsilon)


def conformal_interval(
    ensemble_predictions: np.ndarray,
    q_hat: float,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the interval using the same stabilized scale as calibration."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean, spread = _ensemble_statistics(ensemble_predictions)
    radius = q_hat * (spread + epsilon)
    return mean - radius, mean + radius
