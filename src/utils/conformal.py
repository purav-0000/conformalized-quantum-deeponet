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


def repeated_trajectory_resplits(
    calibration_truth: np.ndarray,
    calibration_predictions: np.ndarray,
    test_truth: np.ndarray,
    test_predictions: np.ndarray,
    coverage: float,
    *,
    trials: int = 100,
    seed: int = 0,
    epsilon: float = 1e-8,
) -> dict:
    """Estimate calibration variability by resplitting whole trajectories.

    Inputs must already have trajectories on axis zero (axis one for ensemble
    predictions).  The original calibration size is preserved in every split;
    coordinates belonging to a trajectory are never separated.
    """
    calibration_truth = np.asarray(calibration_truth)
    test_truth = np.asarray(test_truth)
    calibration_predictions = np.asarray(calibration_predictions)
    test_predictions = np.asarray(test_predictions)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if calibration_truth.ndim < 2 or test_truth.ndim < 2:
        raise ValueError("truth arrays must have a trajectory axis")
    if calibration_predictions.shape[1:] != calibration_truth.shape:
        raise ValueError("calibration prediction/truth shapes differ")
    if test_predictions.shape[1:] != test_truth.shape:
        raise ValueError("test prediction/truth shapes differ")
    if calibration_predictions.shape[0] != test_predictions.shape[0]:
        raise ValueError("calibration and test ensemble sizes differ")
    if calibration_truth.shape[1:] != test_truth.shape[1:]:
        raise ValueError("calibration and test trajectory shapes differ")

    n_calibration = calibration_truth.shape[0]
    truth = np.concatenate((calibration_truth, test_truth), axis=0)
    predictions = np.concatenate(
        (calibration_predictions, test_predictions), axis=1
    )
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(trials):
        indices = rng.permutation(truth.shape[0])
        calibration_indices = indices[:n_calibration]
        test_indices = indices[n_calibration:]
        q_hat, _ = grouped_conformal_quantile(
            truth[calibration_indices],
            predictions[:, calibration_indices],
            coverage,
            num_units=n_calibration,
            unit="trajectory",
            epsilon=epsilon,
        )
        record = conformal_metrics(
            truth[test_indices],
            predictions[:, test_indices],
            q_hat,
            num_units=test_indices.size,
            epsilon=epsilon,
        )
        mean_prediction = predictions[:, test_indices].mean(axis=0)
        record["relative_l2_error"] = float(
            np.linalg.norm(truth[test_indices] - mean_prediction)
            / np.linalg.norm(truth[test_indices])
        )
        record["q_hat"] = q_hat
        records.append(record)

    metric_names = (
        "marginal_coverage",
        "simultaneous_trajectory_coverage",
        "average_interval_width",
        "maximum_interval_width",
        "relative_l2_error",
        "q_hat",
    )
    summaries = {}
    for name in metric_names:
        values = np.asarray([record[name] for record in records], dtype=np.float64)
        summaries[name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)) if trials > 1 else 0.0,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return {
        "trials": trials,
        "seed": seed,
        "num_calibration_trajectories": int(n_calibration),
        "num_test_trajectories": int(test_truth.shape[0]),
        "metrics": summaries,
    }
