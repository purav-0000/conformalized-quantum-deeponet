"""Finite-sample conformal calibration with explicit exchangeability units."""

from __future__ import annotations

import math

import numpy as np


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    """Return r_(ceil((n+1)coverage)), including the +infinity edge case."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between zero and one")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite array")
    rank = math.ceil((values.size + 1) * coverage)
    if rank > values.size:
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


def adaptive_nonconformity(
    truth: np.ndarray,
    ensemble_predictions: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Coordinate scores before grouping into exchangeable sampling units."""
    predictions = np.asarray(ensemble_predictions)
    if predictions.ndim < 2:
        raise ValueError("ensemble_predictions must include a model axis")
    mean = predictions.mean(axis=0)
    spread = predictions.std(axis=0)
    truth = np.asarray(truth)
    if mean.shape != truth.shape:
        raise ValueError(f"prediction/truth shapes differ: {mean.shape} != {truth.shape}")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return np.abs(truth - mean) / (spread + epsilon)


def grouped_conformal_quantile(
    truth: np.ndarray,
    ensemble_predictions: np.ndarray,
    coverage: float,
    *,
    num_units: int,
    unit: str = "trajectory",
    epsilon: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Calibrate by trajectory maxima or preserve legacy coordinate pooling."""
    coordinate_scores = adaptive_nonconformity(truth, ensemble_predictions, epsilon)
    if unit == "trajectory":
        if coordinate_scores.size % num_units:
            raise ValueError("scores cannot be reshaped into the requested trajectories")
        calibration_scores = coordinate_scores.reshape(num_units, -1).max(axis=1)
    elif unit == "coordinate":
        calibration_scores = coordinate_scores.reshape(-1)
    else:
        raise ValueError("unit must be 'trajectory' or 'coordinate'")
    return finite_sample_quantile(calibration_scores, coverage), calibration_scores


def conformal_metrics(
    truth: np.ndarray,
    ensemble_predictions: np.ndarray,
    q_hat: float,
    *,
    num_units: int,
    epsilon: float = 1e-8,
) -> dict[str, float | int]:
    """Measure marginal and simultaneous trajectory coverage of a band."""
    predictions = np.asarray(ensemble_predictions)
    mean = predictions.mean(axis=0)
    spread = predictions.std(axis=0)
    truth = np.asarray(truth)
    if mean.shape != truth.shape:
        raise ValueError(f"prediction/truth shapes differ: {mean.shape} != {truth.shape}")
    radius = q_hat * (spread + epsilon)
    covered = np.abs(truth - mean) <= radius
    if covered.size % num_units:
        raise ValueError("coverage indicators cannot be grouped into trajectories")
    trajectory_covered = covered.reshape(num_units, -1).all(axis=1)
    width = 2.0 * radius
    return {
        "num_trajectories": int(num_units),
        "num_coordinates": int(covered.size),
        "marginal_coverage": float(covered.mean()),
        "simultaneous_trajectory_coverage": float(trajectory_covered.mean()),
        "average_interval_width": float(width.mean()),
        "maximum_interval_width": float(width.max()),
    }
