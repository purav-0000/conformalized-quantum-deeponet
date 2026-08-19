# utils/simulation.py

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend

from src.model_definition.quantum_layer_ideal import custom_tomo_fast


def silu(x: np.ndarray) -> np.ndarray:
    """
    Computes the Sigmoid Linear Unit (SiLU) activation function.

    Args:
        x: Input numpy array.

    Returns:
        The element-wise SiLU activation of the input array.
    """
    return x / (1 + np.exp(-x))


def load_weights(directory: Path, layer: int) -> Dict[str, np.ndarray]:
    """
    Loads the weights for a specific layer of the DeepONet model.

    This function assumes weights are stored as text files with a specific
    naming convention within the model directory.

    Args:
        directory: The pathlib.Path object pointing to the model's directory.
        layer: The integer index of the layer for which to load weights.

    Returns:
        A dictionary containing the loaded weight and bias arrays for the
        branch and trunk networks, as well as the final bias.
    """
    logging.debug(f"Loading weights for layer {layer} from {directory}")
    try:
        return {
            "branch_hidden_bias": np.loadtxt(directory / f"branch.hidden_layers.{layer}.bias.txt"),
            "branch_hidden_thetas": np.loadtxt(directory / f"branch.hidden_layers.{layer}.thetas.txt"),
            "branch_output_bias": np.loadtxt(directory / f"branch.output_layer.bias.txt"),
            "branch_output_weight": np.loadtxt(directory / f"branch.output_layer.weight.txt"),
            "trunk_hidden_bias": np.loadtxt(directory / f"trunk.hidden_layers.{layer}.bias.txt"),
            "trunk_hidden_thetas": np.loadtxt(directory / f"trunk.hidden_layers.{layer}.thetas.txt"),
            "trunk_output_bias": np.loadtxt(directory / f"trunk.output_layer.bias.txt"),
            "trunk_output_weight": np.loadtxt(directory / f"trunk.output_layer.weight.txt"),
            "final_bias": np.loadtxt(directory / "b.txt")
        }
    except IOError as e:
        logging.error(f"Error loading weight files for layer {layer} in {directory}: {e}")
        raise


# --- Model Evaluation ---

def evaluate_model(y_pred: np.ndarray, y_true: np.ndarray, save_dir: Optional[Path] = None, verbose: bool = True) -> float:
    """
    Calculates the mean relative L2 error between predictions and true values.

    If the prediction array corresponds to an ensemble, the mean prediction is
    used for the error calculation. Results can be optionally saved to a file.

    Args:
        y_pred: A numpy array of model predictions. Can be 3D (ensemble) or 2D (single model).
        y_true: A numpy array of ground truth values.
        save_dir: An optional path to a directory where evaluation results will be saved.
        verbose: If True, logs the calculated error.

    Returns:
        The mean relative L2 error as a float.
    """
    def _save_results(output_dir: Path, prediction: np.ndarray, error: float):
        """Save evaluation outputs to disk with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        error_path = output_dir / f"simulation_error_{timestamp}.txt"
        output_path = output_dir / f"simulation_output_{timestamp}.txt"
        np.savetxt(error_path, [error])
        np.savetxt(output_path, prediction)
        logging.info(f"Saved evaluation results to {output_dir}")

        with open(output_dir / f"ensemble_output_{timestamp}.txt", "w") as f:
            for i, slice_2d in enumerate(y_pred):
                f.write(f"Slice {i}:\n")
                np.savetxt(f, slice_2d, fmt="%.4f")
                f.write("\n")

    # If predictions are from an ensemble, average across the models first.
    y_pred_mean = y_pred.mean(axis=0) if y_pred.ndim == 3 else y_pred

    # Calculate the mean relative L2 error.
    error = np.mean(np.linalg.norm(y_pred_mean - y_true, axis=1) / np.linalg.norm(y_true, axis=1))

    if verbose:
        logging.info("--- Evaluation Stats ---")
        logging.info(f"Mean Relative L2 Error: {error:.6f}")

    if save_dir:
        _save_results(save_dir, y_pred_mean, error)

    return error


def build_circuit(
        x_input: np.ndarray,
        n_in: int,
        n_out: int,
        W_gate,
        loader_gate,
        loader_inv_gate,
        simulator: Backend,
        cost_check: bool = False,
        noisy: bool = False
) -> Optional[QuantumCircuit]:
    """
    Constructs and transpiles a quantum circuit for a single input.

    Args:
        x_input: The input vector.
        n_in: The number of input qubits.
        n_out: The number of output qubits.
        W_gate: The pre-computed weight matrix gate.
        loader_gate: The pre-computed data loader gate.
        simulator: The Qiskit backend for transpilation.
        cost_check: If True, analyzes and logs the circuit cost without returning it.
        noisy: If True, adds a 'save_density_matrix' instruction for noise simulation.

    Returns:
        The transpiled QuantumCircuit, or None if cost_check is True.
    """
    # Add a small epsilon to avoid division by zero or instability with zero inputs.
    x_input_stable = x_input.copy()
    x_input_stable[np.abs(x_input_stable) < 1e-8] = 1e-8

    circuit = custom_tomo_fast(n_in, n_out, x_input_stable, W_gate, loader_gate, loader_inv_gate)

    if cost_check:
        # Heron: 'cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x'
        # Eagle: 'ecr', 'rz', 'sx', 'x', 'i'
        # Transpile for a realistic backend to estimate cost.
        t_qc = transpile(circuit, optimization_level=2, basis_gates=['ecr', 'rz', 'sx', 'x', 'i'])
        logging.info("\n--- Realistic Circuit Cost ---")
        logging.info(f"Depth: {t_qc.depth()}, Gates: {t_qc.count_ops()}")
        return None

    if noisy:
        circuit.save_density_matrix()
    else:
        circuit.save_statevector('state')

    # For realistic simulations with measurements, one would uncomment the following:
    # circuit.measure_all(add_bits=False)

    return transpile(circuit, simulator, optimization_level=0)


def plot_pred(
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Path,
        x_test_plot: np.ndarray,
        q_hat: Optional[float] = None,
        num_samples: int = 1,
        online: bool = False
) -> None:
    """
    Generates and saves plots comparing model predictions to ground truth.

    For ensembles, it plots the mean prediction and the conformal prediction
    interval. For single models, it plots the direct prediction.

    Args:
        x_test_trunk: The trunk network inputs (coordinates for plotting).
        y_test: The ground truth output values.
        y_pred: The model's predicted output values. 3D for ensemble, 2D for single model.
        output_dir: The directory where the plot image will be saved.
        q_hat: The quantile value for calculating conformal prediction intervals.
        num_samples: The number of random test samples to plot.
    """

    if online:
        plot_pred_online(
            x_test,
            y_test,
            y_pred,
            output_dir,
            x_test_plot,
            q_hat,
            num_samples
        )
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    is_ensemble = y_pred.ndim == 3

    # Ensure we don't try to plot more samples than available
    num_samples = min(num_samples, len(y_test))
    if num_samples == 0:
        logging.warning("No samples available to plot.")
        return

    indices = np.random.choice(len(y_test), size=num_samples, replace=False)
    fig, axs = plt.subplots(
        num_samples, 1,
        figsize=(10, 5 * num_samples),
        sharex=True,
        squeeze=False,  # Ensures axs is always 2D
        sharey=True
    )
    axs = axs.flatten()  # Flatten to 1D array for easy iteration

    x_trunk_coords = x_test[1][:, 0]

    # Define a consistent color palette
    color_truth = '#d62728'  # Red
    color_pred = '#1f77b4'  # Blue
    color_interval = '#aec7e8'  # Light Blue

    for i, idx in enumerate(indices):
        ax = axs[i]
        y_true_sample = y_test[idx]

        # Plot Ground Truth
        ax.plot(x_trunk_coords, y_true_sample, color=color_truth, linestyle='-', linewidth=2.5, label="Ground Truth")

        if is_ensemble:
            samples = y_pred[:, idx, :]
            mean_pred = samples.mean(axis=0)
            std_pred = samples.std(axis=0)

            # Plot Mean Prediction
            ax.plot(x_trunk_coords, mean_pred, color=color_pred, linestyle='--', linewidth=2, label="Mean Prediction")

            # Plot Conformal Prediction Interval
            if q_hat is not None:
                lower_bound = mean_pred - q_hat * (std_pred + 1e-8)
                upper_bound = mean_pred + q_hat * (std_pred + 1e-8)
                ax.fill_between(x_trunk_coords, lower_bound, upper_bound, color=color_interval, alpha=0.55,
                                label="90% Conformal Interval")
        else:
            # Plot Single Model Prediction
            ax.plot(x_trunk_coords, y_pred[idx, :], color=color_pred, linestyle='--', linewidth=2, label="Prediction")

        ax.set_title(f"Test Sample Index: {idx}", fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

        if i == num_samples - 1:  # Add labels only to the last plot
            ax.set_xlabel("Coordinate", fontsize=12)
        ax.set_ylabel("Function Value", fontsize=12)
        ax.legend(loc="best")

    # Calculate and Display Overall Metrics
    error = evaluate_model(y_pred, y_test, verbose=False)
    metrics_text = f"Mean Relative L2 Error: {error:.4f}"

    if is_ensemble and q_hat is not None:
        mean_preds_all = y_pred.mean(axis=0)
        std_preds_all = y_pred.std(axis=0)
        lower_all = mean_preds_all - q_hat * (std_preds_all + 1e-8)
        upper_all = mean_preds_all + q_hat * (std_preds_all + 1e-8)

        in_interval = (y_test >= lower_all) & (y_test <= upper_all)
        coverage = np.mean(in_interval) * 100
        avg_width = np.mean(upper_all - lower_all)
        max_width = np.max(upper_all - lower_all)

        metrics_text += f"\nCoverage: {coverage:.2f}%"
        metrics_text += f"\nAverage Interval Width: {avg_width:.4f}"
        metrics_text += f"\nMax Interval Width: {max_width:.4f}"

    # Stats
    logging.info("--- Evaluation Stats ---")
    logging.info(metrics_text)

    # Add a title and a text box with metrics
    fig.suptitle("Model Prediction vs. Ground Truth", fontsize=18, y=1.02)

    # Save Figure
    plt.tight_layout()  # Adjust layout to make room for suptitle

    plot_dir = output_dir / "simulation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = plot_dir / f"predictions_plot_{timestamp}.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Prediction plot saved to {output_path}")


def plot_pred_online(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    x_test_plot: np.ndarray,
    q_hat: Optional[float] = None,
    num_samples: int = 10
):
    """
    Similar to plot_pred except for online dataset.
    """
    # x_test (branch or trunk) of shape (num_signals, n_locs, features)
    # y_test of shape (num_signals, n_locs, 1)
    # y_pred of shape (num_signals * n_locs, 1) if single model else (num_models, num_signals * n_locs, 1)
    # x_test_plot of shape (num_signals, n_locs, 1)

    plt.style.use('seaborn-v0_8-darkgrid')
    is_ensemble = y_pred.ndim == 3
    # Make y_pred 3D (model outputs 2D)
    num_signals = x_test[0].shape[0]
    if is_ensemble:
        num_models = y_pred.shape[0]
        # (num_models, num_signals, n_locs, 1)
        y_pred = y_pred.reshape(num_models, num_signals, -1, 1)
    else:
        # (num_signals, n_locs, 1)
        y_pred = y_pred.reshape(num_signals, -1, 1)

    # Ensure we don't try to plot more samples than available
    num_samples = min(num_samples, len(y_test))
    if num_samples == 0:
        logging.warning("No samples available to plot.")
        return

    indices = np.random.choice(len(y_test), size=num_samples, replace=False)
    fig, axs = plt.subplots(
        num_samples, 1,
        figsize=(10, 5 * num_samples),
        sharex=True,
        squeeze=False,  # Ensures axs is always 2D
        sharey=True
    )
    axs = axs.flatten()  # Flatten to 1D array for easy iteration

    # Define a consistent color palette
    color_truth = '#d62728'  # Red
    color_pred = '#1f77b4'  # Blue
    color_interval = '#aec7e8'  # Light Blue

    for i, idx in enumerate(indices):
        ax = axs[i]
        y_true_sample = y_test[idx]

        x_trunk_coords = x_test[1][idx, :, 0]

        # Plot Ground Truth
        ax.plot(x_trunk_coords, y_true_sample, color=color_truth, linestyle='-', linewidth=2.5, label="Ground Truth")

        if is_ensemble:
            samples = y_pred[:, idx, :, 0]
            mean_pred = samples.mean(axis=0)
            std_pred = samples.std(axis=0)

            # Plot Mean Prediction
            ax.plot(x_trunk_coords, mean_pred, color=color_pred, linestyle='--', linewidth=2, label="Mean Prediction")

            # Plot Conformal Prediction Interval
            if q_hat is not None:
                lower_bound = mean_pred - q_hat * (std_pred + 1e-8)
                upper_bound = mean_pred + q_hat * (std_pred + 1e-8)
                ax.fill_between(x_trunk_coords, lower_bound, upper_bound, color=color_interval, alpha=0.55,
                                label="90% Conformal Interval")
        else:
            # Plot Single Model Prediction
            ax.plot(x_trunk_coords, y_pred[idx, :, 0], color=color_pred, linestyle='--', linewidth=2, label="Prediction")

        ax.set_title(f"Test Sample Index: {idx}", fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

        if i == num_samples - 1:  # Add labels only to the last plot
            ax.set_xlabel("Coordinate", fontsize=12)
        ax.set_ylabel("Function Value", fontsize=12)
        ax.legend(loc="best")

    # Calculate and Display Overall Metrics
    # Reshape for error calculation which considers only 2 dims (excluding num_models)
    # y_pred.shape[0] is the number of models
    error = evaluate_model(
        y_pred.reshape(y_pred.shape[0], -1, 1) if is_ensemble else y_pred.reshape(-1, 1),
        y_test.reshape(-1, 1),
        verbose=False
    )

    metrics_text = f"Mean Relative L2 Error: {error:.4f}"

    if is_ensemble and q_hat is not None:
        mean_preds_all = y_pred.mean(axis=0)
        std_preds_all = y_pred.std(axis=0)
        lower_all = mean_preds_all - q_hat * (std_preds_all + 1e-8)
        upper_all = mean_preds_all + q_hat * (std_preds_all + 1e-8)

        in_interval = (y_test >= lower_all) & (y_test <= upper_all)
        coverage = np.mean(in_interval) * 100
        avg_width = np.mean(upper_all - lower_all)
        max_width = np.max(upper_all - lower_all)

        metrics_text += f"\nCoverage: {coverage:.2f}%"
        metrics_text += f"\nAverage Interval Width: {avg_width:.4f}"
        metrics_text += f"\nMax Interval Width: {max_width:.4f}"

    # Stats
    logging.info("--- Evaluation Stats ---")
    logging.info(metrics_text)

    # Add a title and a text box with metrics
    fig.suptitle("Model Prediction vs. Ground Truth", fontsize=18, y=1.02)

    # Save Figure
    plt.tight_layout()  # Adjust layout to make room for suptitle

    plot_dir = output_dir / "simulation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = plot_dir / f"predictions_plot_{timestamp}.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Prediction plot saved to {output_path}")
