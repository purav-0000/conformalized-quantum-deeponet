# utils/simulation.py

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from qiskit import transpile
from typing import Optional, Tuple

from src.quantum_layer_ideal import custom_tomo_fast


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))

def load_weights(directory: Path, layer) -> dict:

    weights = {}

    return {
        "branch_hidden_bias": np.loadtxt(os.path.join(directory, f"branch.hidden_layers.{layer}.bias.txt")),
        "branch_hidden_thetas": np.loadtxt(os.path.join(directory, f"branch.hidden_layers.{layer}.thetas.txt")),
        "branch_output_bias": np.loadtxt(os.path.join(directory, f"branch.output_layer.bias.txt")),
        "branch_output_weight": np.loadtxt(os.path.join(directory, f"branch.output_layer.weight.txt")),
        "trunk_hidden_bias": np.loadtxt(os.path.join(directory, f"trunk.hidden_layers.{layer}.bias.txt")),
        "trunk_hidden_thetas": np.loadtxt(os.path.join(directory, f"trunk.hidden_layers.{layer}.thetas.txt")),
        "trunk_output_bias": np.loadtxt(os.path.join(directory, f"trunk.output_layer.bias.txt")),
        "trunk_output_weight": np.loadtxt(os.path.join(directory, f"trunk.output_layer.weight.txt")),

        "final_bias": np.loadtxt(os.path.join(directory, "b.txt"))
    }


def evaluate_model(y_pred: np.ndarray, y_true: np.ndarray, save_dir: Path=None, verbose: bool=False):
    def save_evaluation_results(output_dir, y_pred, error, prefix=""):
        """Save evaluation outputs to disk with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        np.savetxt(os.path.join(output_dir, f"{prefix}simulation_error_" + timestamp + ".txt"), [error])
        np.savetxt(os.path.join(output_dir, f"{prefix}simulation_output_" + timestamp + ".txt"), y_pred)

    # If ensemble predictions, take the mean
    if y_pred.ndim > 2:
        y_pred_mean = y_pred.mean(axis=0)
    else:
        y_pred_mean = y_pred

    error = np.mean(np.linalg.norm(y_pred_mean - y_true, axis=1) / np.linalg.norm(y_true, axis=1))

    if verbose:
        print(f"Mean Relative L2 Error: {error:.6f}")

    if save_dir:
        save_evaluation_results(save_dir, y_pred_mean, error)

    return error


def build_circuit(x_input: np.ndarray, n_in: int, n_out: int, W_gate, loader_gate, loader_inv_gate, simulator, cost_check=False, noisy=False):
    x_input_stable = x_input.copy()
    x_input_stable[np.abs(x_input_stable) < 1e-7] += 1e-7

    circuit = custom_tomo_fast(n_in, n_out, x_input_stable, W_gate, loader_gate, loader_inv_gate)

    # Optional: Analyze circuit cost against a realistic backend
    if cost_check:
        import pyzx as zx
        from qiskit import QuantumCircuit
        from qiskit.qasm2 import dump
        from qiskit.transpiler import PassManager, InstructionDurations
        from qiskit.transpiler.passes import ASAPSchedule


        t_qc = transpile(circuit, optimization_level=2, basis_gates=['cz', 'rz', 'rx', 'sx', 'x', 'rzz'])

        """
        # pyZX stuff
        circ = zx.Circuit.from_qasm_file('circuit.qasm')
        # Optimize
        g = circ.to_basic_gates().to_graph()
        zx.simplify.full_reduce(g, quiet=True)
        g.normalize()
        new_circ = zx.extract_circuit(g)

        # Convert directly to Qiskit QuantumCircuit without writing file
        qasm_str = new_circ.to_qasm()
        t_qc = QuantumCircuit.from_qasm_str(qasm_str)

        t_qc = transpile(t_qc, backend=backend, optimization_level=2)

        print(t_qc)
        """

        print(f"\n--- Realistic Circuit Cost ---")
        print(f"Depth: {t_qc.depth()}, Gates: {t_qc.count_ops()}")

        # exit(1)
        """
        instruction_durations = backend.target.durations()

        
        # 6. Create a PassManager with the explicit durations object
        pm = PassManager([ASAPSchedule(instruction_durations)])
        scheduled_qc = pm.run(t_qc)  # Use one of the transpiled circuits

        # 7. Access the duration in 'dt' units
        duration_dt = scheduled_qc.duration

        if duration_dt:
            # 8. Get the value of dt from the backend's TARGET attribute
            dt_in_seconds = backend.target.dt
            duration_us = duration_dt * dt_in_seconds * 1e6  # convert to microseconds

            print("\n--- Realistic Circuit Cost (Duration) ---")
            print(f"Duration in dt: {duration_dt} dt")
            print(f"Backend dt unit: {dt_in_seconds * 1e9:.3f} ns")
            print(f"Total Duration: {duration_us:.2f} µs")
        else:
            print("\nCircuit could not be scheduled.")
        """
        print()
        return

    if noisy:
        circuit.save_density_matrix()
    else:
        circuit.save_statevector('state')
    return transpile(circuit, simulator, optimization_level=1)


def plot_pred(
        x_test: Tuple[np.ndarray, np.ndarray],
        y_test: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Path,
        x_test_plot: np.ndarray,
        q_hat: Optional[float] = None
):
    is_ensemble = y_pred.ndim == 3  # Ensemble will have 3-dimensional output (models, batch index, output)
    num_samples = 10

    indices = np.random.choice(len(y_test), size=num_samples, replace=False)
    fig, axs = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples), sharex=True, sharey=True)

    # Select trunk inputs
    x_trunk_coords = x_test[1][:, 0]
    for ax, idx in zip(axs, indices):

        y = y_test[idx]

        # Plot input function and ground truth
        ax.plot(x_trunk_coords, x_test_plot[idx, :], color='orange', alpha=0.9, label="Input Function")
        ax.plot(x_trunk_coords, y, 'r-', linewidth=2, label="Ground Truth")

        # Check if ensembles or single model
        if is_ensemble:  # Ensemble
            samples = y_pred[:, idx, :]
            mean_pred = samples.mean(axis=0)
            std_pred = samples.std(axis=0)

            # Confidence interval
            ax.plot(x_trunk_coords, mean_pred, 'b-', label="Mean Prediction")
            lower = mean_pred - q_hat * std_pred
            upper = mean_pred + q_hat * std_pred
            ax.fill_between(x_trunk_coords, lower, upper, color='blue', alpha=0.2, label="Conformal Interval")

        else:   # Single model
            ax.plot(x_trunk_coords, y_pred[idx, :], 'b-', label="Prediction")

        ax.set_title(f"Test Sample Index: {idx}")
        ax.grid(True, linestyle='--', alpha=0.6)

    axs[0].legend()
    error = evaluate_model(y_pred, y_test)

    # Calculate coverage
    mean_pred = y_pred.mean(axis=0)
    std_pred = y_pred.std(axis=0)

    lower = mean_pred - q_hat * std_pred
    upper = mean_pred + q_hat * std_pred

    # Element-wise boolean mask
    in_interval = (y_test >= lower) & (y_test <= upper)

    # Fraction or percentage of covered points
    coverage = np.mean(in_interval)  # fraction in [0, 1]
    # Optional: convert to percent
    coverage_percent = 100 * coverage

    # Average width
    average_width = np.mean(upper - lower)

    # Covariance
    # Reshape to (num_models, batch_size * output_dim)
    num_models = y_pred.shape[0]
    cov_matrix = np.cov(y_pred.reshape(num_models, -1))
    num_off_diagonal = num_models * (num_models - 1)
    avg_covariance = (np.sum(cov_matrix) - np.trace(cov_matrix)) / num_off_diagonal

    fig.suptitle(
        f"Prediction with conformal intervals\n"
        f"Error: {error:.6f}\n"
        f"Coverage: {coverage_percent:.6f}\n"
        f"Average width: {average_width:.6f}"
    )
    print(f"Average width: {average_width:.6f}")
    print(f"Max width: {np.max(upper - lower):.6f}")
    print(f"Avg covariance: {avg_covariance:.6f}")
    print(f"Coverage: {coverage_percent:.6f}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(output_dir / f"predictions_plot_{timestamp}.png")
    plt.close()

