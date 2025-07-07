import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import secrets
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from qiskit import transpile
from qiskit_aer import AerSimulator
from src.quantum_layer_ideal import data_loader, W
from src.spqc import create_spqc_circuit
from src.utils.common import apply_overrides
from src.utils.data_handling import DataHandler
from src.utils.simulation import build_circuit, evaluate_model, load_weights, plot_pred, silu

BRANCH_PREFIX = "branch"
TRUNK_PREFIX = "trunk"

# --- Config ---
@dataclass
class Config:
    """Configuration schema for simulation."""
    data_dir: str = "data_ode_simple"
    model: Optional[str] = None
    ensemble: Optional[str] = None
    greedy: bool = False
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    n_jobs: int = 4
    simulator: str = "CPU"  # or "GPU"
    mode: str = "ideal"  # or "shots"
    shots: int = 0
    batch_size: Optional[int] = None
    coverage: float = 0.9
    spqc: bool = False
    analyze_circuit_cost: bool = False


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --- Core Simulation Logic ---

class SimulationRunner:
    """Encapsulates the entire simulation workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.simulator = AerSimulator(device=self.config.simulator)
        self.data_handler = DataHandler(self.config.data_dir)

        run_type = "SPQC" if self.config.spqc else "Sequential"

        if self.config.ensemble:
            self.output_dir = Path("models", "ensembles", self.config.ensemble)
        else:
            self.output_dir = Path("models", self.config.model)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results for this run will be saved in: {self.output_dir}")

    def run(self):
        """Main execution method."""
        if self.config.spqc:
            if not self.config.ensemble:
                raise ValueError("SPQC mode requires an ensemble configuration.")
            print("--- Running in SPQC Ensemble Mode ---")
            self._run_ensemble_spqc()
        elif self.config.ensemble:
            print("--- Running in Sequential Ensemble Mode ---")
            self._run_ensemble_sequential()
        else:
            print("--- Running in Single Model Mode ---")
            self._run_single_model()


    def _run_single_model(self):
        """Executes the simulation for a single model."""
        model_path = Path("models") / self.config.model
        print(f"Running single model simulation: {self.config.model}")

        y_pred = self._run_model(model_path, self.data_handler.x_test)

        evaluate_model(y_pred, self.data_handler.y_test, self.output_dir, verbose=True)
        plot_pred(
            self.data_handler.x_test, self.data_handler.y_test, y_pred,
            self.output_dir, self.data_handler.x_test_plot
        )

    def _run_ensemble_sequential(self):
        """Executes the simulation for an ensemble of models."""
        ensemble_dir = Path("models", "ensembles", self.config.ensemble)
        model_dirs = [d for d in ensemble_dir.iterdir() if d.is_dir()]
        print(f"Found {len(model_dirs)} models in ensemble: {self.config.ensemble}")

        if self.config.greedy:
            selected_models = self._greedy_ensemble_selection(model_dirs)
            print(f"Greedy selection chose {len(selected_models)} models.")
        else:
            selected_models = model_dirs

        # Calculate calibration scores (conformal prediction)
        cal_outputs = [self._run_model(m, self.data_handler.x_cal) for m in
                       tqdm(selected_models, desc="Running Calibration")]
        cal_outputs = np.array(cal_outputs)

        scores = np.abs(self.data_handler.y_cal - cal_outputs.mean(axis=0)) / cal_outputs.std(axis=0)
        q_hat = np.quantile(scores, self.config.coverage)
        print(f"Conformal quantile q_hat at {self.config.coverage * 100}% coverage: {q_hat:.4f}")

        # Evaluate on test set
        test_outputs = [self._run_model(m, self.data_handler.x_test) for m in
                        tqdm(selected_models, desc="Running Test Set")]
        test_outputs = np.array(test_outputs)

        evaluate_model(test_outputs, self.data_handler.y_test, self.output_dir, verbose=True)
        plot_pred(
            self.data_handler.x_test, self.data_handler.y_test, test_outputs,
            self.output_dir, self.data_handler.x_test_plot, q_hat=q_hat
        )

    def _run_model(self, model_path: Path, inputs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Runs the full forward pass of a single DeepONet model."""
        weights = load_weights(model_path)

        def get_layer_params(prefix: str):
            x = inputs[0] if prefix == BRANCH_PREFIX else inputs[1]
            n_in = x.shape[1]
            n_out = weights[f"{prefix}_hidden0_bias"].shape[0]
            return (
                x, n_in, n_out,
                weights[f"{prefix}_hidden0_bias"],
                weights[f"{prefix}_output_weight"],
                weights[f"{prefix}_output_bias"],
                weights[f"{prefix}_hidden0_thetas"]
            )

        branch_outputs = self._run_quantum_layer(*get_layer_params(BRANCH_PREFIX), is_trunk=False)
        trunk_outputs = self._run_quantum_layer(*get_layer_params(TRUNK_PREFIX), is_trunk=True)

        return np.einsum('bi,ni->bn', branch_outputs, trunk_outputs) + weights["final_bias"]

    def _run_quantum_layer(self, inputs, n_in, n_out, bias, weight, output_bias, thetas, is_trunk: bool):
        """Simulates the execution of one quantum layer (branch or trunk)."""
        # Precompute gates
        sqrt_norm = np.sqrt(max(n_in, n_out))
        W_gate = W(n_in, n_out, thetas)
        loader_gate = data_loader(np.full(max(n_in, n_out), 1 / sqrt_norm))
        loader_inv_gate = loader_gate.inverse()

        all_outputs = []
        batch_size = self.config.batch_size or len(inputs)

        if self.config.analyze_circuit_cost:
            print("Branch, " if not is_trunk else "Trunk, ", end='')
            print(f"n_in: {n_in}, n_out: {n_out}, thetas_shape: {thetas.shape}")
            build_circuit(
                inputs[0], n_in, n_out, W_gate, loader_gate, loader_inv_gate, self.simulator, cost_check=True
            )   # Analyze depth using arbitray input

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]

            circuits = Parallel(n_jobs=self.config.n_jobs)(
                delayed(build_circuit)(x, n_in, n_out, W_gate, loader_gate, loader_inv_gate, self.simulator)
                for x in batch
            )

            results = self.simulator.run(circuits, shots=1).result()

            # Parallelization results in massive overhead, so removed it
            batch_outputs = [
                self._process_quantum_output(j, results, n_in, n_out, bias, weight, output_bias, is_trunk)
                for j in range(len(batch))
            ]
            all_outputs.extend(batch_outputs)

        return np.array(all_outputs)

    def _process_quantum_output(self, idx, results, n_in, n_out, hidden0_bias, output_weight, output_bias,
                                is_trunk: bool):
        """Processes the raw statevector from a single circuit run."""
        statevector = np.real(results.data(idx)['state'].data)

        if self.config.mode == 'shots':
            probabilities = statevector ** 2
            counts = np.random.multinomial(self.config.shots, probabilities)
            state_probs = counts / self.config.shots
        else:  # ideal mode
            state_probs = statevector ** 2

        # Bit indexing to extract expectation values
        output = []
        for i in range(n_out):
            pos_vec = ['0'] * n_out
            pos_vec[i] = '1'
            # Qiskit uses a little-endian convention (qubit 0 is the rightmost bit),
            # so we build the string and then reverse it to match the statevector index.
            pos0_str = (''.join(['0'] + ['0'] * (n_in - n_out) + pos_vec))[::-1]
            pos1_str = (''.join(['1'] + ['0'] * (n_in - n_out) + pos_vec))[::-1]

            result0 = state_probs[int(pos0_str, 2)]
            result1 = state_probs[int(pos1_str, 2)]
            output.append(np.sqrt(max(n_in, n_out)) * (result0 - result1))

        # Apply classical post-processing layers
        output = silu(np.array(output) + hidden0_bias)
        output = np.dot(output, output_weight.T) + output_bias

        return silu(output) if is_trunk else output

    def _greedy_ensemble_selection(self, model_paths: List[Path]) -> List[Path]:
        """Performs forward greedy selection of models based on validation set performance."""
        print("Starting greedy ensemble selection...")
        val_preds_all = {m: self._run_model(m, self.data_handler.x_val) for m in
                         tqdm(model_paths, desc="Evaluating all models on validation set")}

        remaining = list(model_paths)

        # Start with the single best model
        best_model = min(remaining,
                         key=lambda m: evaluate_model(val_preds_all[m], self.data_handler.y_val, save_dir=None,
                                                      verbose=False))

        selected = [best_model]
        ensemble_pred = val_preds_all[best_model]
        best_error = evaluate_model(ensemble_pred, self.data_handler.y_val, save_dir=None, verbose=False)
        remaining.remove(best_model)

        while remaining:
            improvements = []
            for candidate in remaining:
                # Calculate error if we add this candidate to the current ensemble
                potential_pred = np.mean([ensemble_pred, val_preds_all[candidate]], axis=0)
                error = evaluate_model(potential_pred, self.data_handler.y_val, save_dir=None, verbose=False)
                improvements.append((error, candidate))

            min_err, best_candidate = min(improvements, key=lambda x: x[0])

            if min_err < best_error:
                ensemble_pred = np.mean([ensemble_pred, val_preds_all[best_candidate]], axis=0)
                best_error = min_err
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # No further improvement possible
                break

        return selected


    # SPQC stuffs
    def _load_ensemble_parameters(self, model_dirs: List[Path]) -> Dict[str, Dict]:
        """Loads and aggregates parameters from all models in an ensemble."""
        all_params = [load_weights(m) for m in model_dirs]

        aggregated = {
            "branch": {
                "hidden0_bias": np.array([p["branch_hidden0_bias"] for p in all_params]),
                "hidden0_thetas": np.array([p["branch_hidden0_thetas"] for p in all_params]),
                "output_bias": np.array([p["branch_output_bias"] for p in all_params]),
                "output_weight": np.array([p["branch_output_weight"] for p in all_params]),
            },
            "trunk": {
                "hidden0_bias": np.array([p["trunk_hidden0_bias"] for p in all_params]),
                "hidden0_thetas": np.array([p["trunk_hidden0_thetas"] for p in all_params]),
                "output_bias": np.array([p["trunk_output_bias"] for p in all_params]),
                "output_weight": np.array([p["trunk_output_weight"] for p in all_params]),
            },
            "final_bias": np.array([p["final_bias"] for p in all_params])
        }
        return aggregated


    def _run_ensemble_spqc(self):
        """Executes the simulation for an entire ensemble using a single SPQC circuit."""
        ensemble_dir = Path("models", "ensembles", self.config.ensemble)
        model_dirs = [d for d in ensemble_dir.iterdir() if d.is_dir()]

        print(f"Loading parameters for {len(model_dirs)} models from {ensemble_dir}")
        params = self._load_ensemble_parameters(model_dirs)

        # Run branch and trunk layers in SPQC mode
        branch_outputs = self._run_spqc_quantum_layer(
            inputs=self.data_handler.x_test[0],
            n_in=self.data_handler.x_test[0].shape[1],
            n_out=params["branch"]["hidden0_bias"][0].shape[0], # Select an arbitray model's hidden0_bias to calculate size of n_out
            ensemble_params=params["branch"],
            is_trunk=False
        )

        trunk_outputs = self._run_spqc_quantum_layer(
            inputs=self.data_handler.x_test[1],
            n_in=self.data_handler.x_test[1].shape[1],
            n_out=params["trunk"]["hidden0_bias"][0].shape[0],
            ensemble_params=params["trunk"],
            is_trunk=True
        )

        # Output shape from spqc_layer: (num_inputs, num_models, n_out)
        # We need to rearrange to (num_models, num_inputs, n_out)
        branch_outputs = np.transpose(branch_outputs, (1, 0, 2))
        trunk_outputs = np.transpose(trunk_outputs, (1, 0, 2))

        final_preds = []
        for i in range(len(model_dirs)):
            pred = np.einsum('bi,ni->bn', branch_outputs[i], trunk_outputs[i]) + params["final_bias"][i]
            final_preds.append(pred)

        final_preds = np.array(final_preds)

        evaluate_model(final_preds, self.data_handler.y_test, self.output_dir, verbose=True)
        """
        plot_pred(
            self.data_handler.x_test, self.data_handler.y_test, final_preds,
            self.output_dir, self.data_handler.x_test_plot
        )
        """

    def _run_spqc_quantum_layer(self, inputs, n_in, n_out, ensemble_params, is_trunk):
        """Runs a quantum layer for all models in parallel using SPQC."""
        # Precompute gates that are static across all inputs
        sqrt_norm = np.sqrt(max(n_in, n_out))
        loader_gate = data_loader(np.full(max(n_in, n_out), 1 / sqrt_norm))
        loader_inv_gate = loader_gate.inverse()

        if self.config.analyze_circuit_cost:
            print("Branch, " if not is_trunk else "Trunk, ", end='')
            print(f"n_in: {n_in}, n_out: {n_out}, thetas_shape: {ensemble_params['hidden0_thetas'][0].shape}")
            self._build_spqc_circuit(
                inputs[0], n_in, n_out, ensemble_params['hidden0_thetas'], loader_gate, loader_inv_gate, cost_check=True
            )


        # Circuit construction is now per-input, as it depends on the data_array
        circuits = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._build_spqc_circuit)(
                x, n_in, n_out, ensemble_params['hidden0_thetas'], loader_gate, loader_inv_gate
            )
            for x in tqdm(inputs, desc=f"Building SPQC Circuits ({'Trunk' if is_trunk else 'Branch'})")
        )

        results = self.simulator.run(circuits, shots=1).result()

        batch_outputs = [
            self._process_spqc_output(j, results, n_in, n_out, ensemble_params, is_trunk)
            for j in tqdm(range(len(inputs)), desc="Processing SPQC Results")
        ]

        return np.array(batch_outputs)


    def _build_spqc_circuit(self, x_input, n_in, n_out, thetas, loader_gate, loader_inv_gate, cost_check=False):
        """Builds one SPQC circuit for a single input vector."""
        x_input_stable = x_input.copy()
        x_input_stable[np.abs(x_input_stable) < 1e-7] += 1e-7

        circ = create_spqc_circuit(
            n_in, n_out, thetas, x_input_stable, loader_inv_gate, loader_gate
        )

        # Optional: Analyze circuit cost against a realistic backend
        if cost_check:
            from qiskit.providers.fake_provider import FakeGuadalupeV2
            from qiskit.transpiler import PassManager, InstructionDurations
            from qiskit.transpiler.passes import ASAPSchedule
            backend = FakeGuadalupeV2()
            t_qc = transpile(circ, backend=backend, optimization_level=3)
            print(f"\n--- Realistic Circuit Cost ---")
            print(f"Depth: {t_qc.depth()}, CNOTs: {t_qc.count_ops().get('cx', 0)}, RZs: {t_qc.count_ops().get('rz', 0)}")
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
            print()
            return

        circ.save_statevector('state')
        return transpile(circ, self.simulator)


    def _process_spqc_output(self, idx, results, n_in, n_out, params, is_trunk):
        """Processes the combined statevector from an SPQC circuit run."""
        statevector = np.real(results.data(idx)['state'].data)
        state_probs = statevector ** 2

        num_models = len(params['hidden0_bias'])
        all_outputs = np.zeros((num_models, n_out))
        addr_format_bits = int(np.ceil(np.log2(num_models))) if num_models > 1 else 0

        # Bit indexing now includes iterating through model addresses
        for i in range(n_out):
            pos = ['0'] * n_out
            pos[i] = '1'
            pos0_str = (''.join(['0'] + ['0'] * (n_in - n_out) + pos))[::-1]
            pos1_str = (''.join(['1'] + ['0'] * (n_in - n_out) + pos))[::-1]

            for j in range(num_models):
                addr_str = format(j, f'0{addr_format_bits}b')
                # Note: Qiskit's endianness means address qubits might be at the high-order end.
                # Assuming statevector format is |tomo⟩|anc⟩|addr⟩
                idx0 = int(pos0_str + addr_str, 2)
                idx1 = int(pos1_str + addr_str, 2)

                result0 = state_probs[idx0]
                result1 = state_probs[idx1]
                all_outputs[j, i] = np.sqrt(max(n_in, n_out)) * (result0 - result1)

        # Apply classical post-processing layers for each model
        for i in range(num_models):
            # The SPQC output is an expectation value; scaling by num_models approximates the sum
            # that would have occurred from the address qubit superposition.
            output_i = all_outputs[i] * num_models
            output_i = silu(output_i + params['hidden0_bias'][i])
            output_i = np.dot(output_i, params['output_weight'][i].T) + params['output_bias'][i]
            all_outputs[i] = silu(output_i) if is_trunk else output_i

        return all_outputs


# --- Main Entry Point ---

def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description="Quantum DeepONet Simulation")
    parser.add_argument('--config', type=str, default="default", help="Config file name in configs/simulation")
    parser.add_argument("--override", nargs='*', help="Overrides in key=value format (e.g., n_jobs=8 seed=42)")
    args = parser.parse_args()

    config_path = Path("configs/simulation") / (args.config + ".yaml")
    config = load_config(str(config_path))
    apply_overrides(config, args.override)

    set_seeds(config.seed)

    # REFACTOR: The main logic is now encapsulated in the runner
    runner = SimulationRunner(config)
    runner.run()
    print("Simulation finished.")


if __name__ == "__main__":
    main()