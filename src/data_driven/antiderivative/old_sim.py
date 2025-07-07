import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import secrets
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from qiskit_aer import AerSimulator
from src.quantum_layer_ideal import data_loader, W
from src.utils.common import apply_overrides
from src.utils.data_handling import DataHandler
from src.utils.simulation import build_circuit, evaluate_model, load_weights, plot_pred, silu

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


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)

def process_state(idx, results, shots, n_in, n_out, hidden0_bias, output_weight, output_bias, mode='ideal', trunk=False):
    state = np.real(results.data(idx)['state'].data)

    """
    if not trunk:
        # 1. Determine the number of qubits from the length of the state vector
        num_qubits = int(np.log2(len(state)))

        print(f"State vector for a {num_qubits}-qubit system.\n")
        print("First 20 amplitudes and their corresponding bitstrings:")
        print("-" * 50)
        print(f"{'Bitstring':<12} | {'Statevector Amplitude'}")
        print("-" * 50)

        # 2. Loop through the first 20 indices of the state vector
        for i in range(20):
            # Get the amplitude
            amplitude = state[i]

            # 3. Format the index 'i' as a binary string, padded with leading zeros
            #    to match the number of qubits. This is the bitstring label.
            bitstring = f'{i:0{num_qubits}b}'

            print(f"|{bitstring}>       | {amplitude:+.8f}")
    """

    # Sample shots, else calculate probabilities
    if mode == 'shots':
        prob = state ** 2
        counts = np.random.multinomial(shots, prob)
        all_sampled = counts / shots
        state = all_sampled
    else:
        state = state ** 2

    # Bit indexing
    output = []
    for i in range(n_out):
        pos = ['0'] * n_out
        pos[i] = '1'
        pos0 = ['0'] + ['0'] * (n_in - n_out) + pos
        pos1 = ['1'] + ['0'] * (n_in - n_out) + pos
        pos0 = ''.join(pos0)[::-1]
        pos1 = ''.join(pos1)[::-1]
        result0 = state[int(pos0, 2)]
        result1 = state[int(pos1, 2)]
        output.append(np.sqrt(max(n_in, n_out)) * (result0 - result1))

    output = silu(np.array(output) + hidden0_bias)
    output = np.dot(output, output_weight.T) + output_bias

    return silu(output) if trunk else output


def run_quantum_layer(inputs, n_in, n_out, bias, weight, output_bias, thetas, simulator, config: Config, trunk=False):

    # Precompute gates
    sqrt_norm = np.sqrt(max(n_in, n_out))
    W_gate = W(n_in, n_out, thetas)
    loader = data_loader(np.full(max(n_in, n_out), 1 / sqrt_norm))
    loader_inv = loader.inverse()

    outputs = []
    batch_size = config.batch_size or len(inputs)
    disable_tqdm = batch_size == len(inputs)

    for i in tqdm(range(0, len(inputs), batch_size), desc="Running in batches", disable=disable_tqdm):

        # Batching
        batch = inputs[i:i + batch_size]

        # Circuit construction
        circuits = Parallel(n_jobs=config.n_jobs)(
            delayed(build_circuit)(x, n_in, n_out, W_gate, loader, loader_inv, simulator)
            for x in tqdm(batch, desc="Building circuits", disable=not disable_tqdm)
        )

        # Execution
        results = simulator.run(circuits, shots=1).result()

        # process_state executes faster if n_jobs = 1
        batch_outputs = Parallel(n_jobs=1)(
            delayed(process_state)(j, results, config.shots, n_in, n_out, bias, weight, output_bias, config.mode, trunk)
            for j in tqdm(range(len(batch)), desc="Processing", disable=not disable_tqdm)
        )
        outputs.extend(batch_outputs)

    return np.array(outputs)


def run_model(model_path, inputs, simulator, config: Config):
    weights = load_weights(model_path)

    def extract_config(prefix):
        x = inputs[0 if prefix == "branch" else 1]
        n_in = x.shape[1]
        n_out = weights[f"{prefix}_hidden0_bias"].shape[0]
        return x, n_in, n_out, weights[f"{prefix}_hidden0_bias"], weights[f"{prefix}_output_weight"], \
            weights[f"{prefix}_output_bias"], weights[f"{prefix}_hidden0_thetas"]



    branch_outputs = run_quantum_layer(*extract_config("branch"), simulator, config, trunk=False)
    trunk_outputs = run_quantum_layer(*extract_config("trunk"), simulator, config, trunk=True)

    return np.einsum('bi,ni->bn', branch_outputs, trunk_outputs) + weights["final_bias"]


def greedy_ensemble(model_paths, x_val, y_val, simulator, config: Config):

    # Evaluate all models on validation set
    selected = []
    remaining = list(model_paths)
    val_preds_all = {m: run_model(m, x_val, simulator, config) for m in remaining}

    # Select current best performing model
    best_model = min(remaining, key=lambda m: evaluate_model(val_preds_all[m], y_val))
    selected.append(best_model)
    ensemble_pred = val_preds_all[best_model]
    best_error = evaluate_model(ensemble_pred, y_val)
    remaining.remove(best_model)

    while remaining:
        # Mean with each remaining model
        improvements = [
            (evaluate_model(np.mean([ensemble_pred, val_preds_all[c]], axis=0), y_val), c)
            for c in remaining
        ]

        # Select best candidate
        min_err, best_candidate = min(improvements, key=lambda x: x[0])

        # Add to ensemble if error drops
        if min_err < best_error:
            ensemble_pred = np.mean([ensemble_pred, val_preds_all[best_candidate]], axis=0)
            best_error = min_err
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return selected


# --- Main Entry point ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="default", help="Config file name")
    parser.add_argument("--override", nargs='*', help="Optional overrides in key=value format")
    args = parser.parse_args()
    args.config = os.path.join("configs/simulation", args.config + ".yaml")
    config = load_config(args.config)
    apply_overrides(config, args.override)

    set_seeds(config.seed)
    simulator = AerSimulator(device=config.simulator)
    data_path = os.path.join("data", config.data_dir)

    # Normalizing and transforming
    x_train, y_train, x_val, y_val, x_test, y_test, x_test_plot = load_dataset(data_path)
    x_cal, y_cal = load_calibration_dataset(data_path)

    bounds = normalize_bounds(x_train, x_test, x_val, x_cal)
    x_val = (
        transform_input(x_val[0], bounds["branch_min"], bounds["branch_max"]),
        transform_input(x_val[1], bounds["trunk_min"], bounds["trunk_max"]),
    )
    x_test = (
        transform_input(x_test[0], bounds["branch_min"], bounds["branch_max"]),
        transform_input(x_test[1], bounds["trunk_min"], bounds["trunk_max"]),
    )
    x_cal = (
        transform_input(x_cal[0], bounds["branch_min"], bounds["branch_max"]),
        transform_input(x_cal[1], bounds["trunk_min"], bounds["trunk_max"]),
    )

    if config.ensemble:
        ensemble_dir = os.path.join("models", "ensembles", config.ensemble)
        model_dirs = [os.path.join(ensemble_dir, m) for m in os.listdir(ensemble_dir) if os.path.isdir(os.path.join(ensemble_dir, m))]

        selected_models = greedy_ensemble(model_dirs, x_val, y_val, simulator, config) if config.greedy else model_dirs

        # Calculate calibration scores
        outputs = [run_model(m, x_cal, simulator, config) for m in selected_models]
        scores = np.abs(y_cal - np.mean(outputs, axis=0)) / np.std(outputs, axis=0)

        q_hat = np.quantile(scores, config.coverage)

        outputs = [run_model(m, x_test, simulator, config) for m in selected_models]
        evaluate_model(np.array(outputs), y_test, verbose=True, save_dir=ensemble_dir)
        plot_pred(x_test, y_test, np.array(outputs), ensemble_dir, x_test_plot, q_hat)
    else:
        model_path = os.path.join("models", config.model)
        y_pred = run_model(model_path, x_test, simulator, config)
        evaluate_model(y_pred, y_test, verbose=True, save_dir=model_path, model_name=os.path.basename(model_path))
        plot_pred(x_test, y_test, y_pred, model_path, x_test_plot)


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

# --- Modularization ---
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(self):
    train = np.load((self + '/picked_aligned_train.npz'), allow_pickle=True)
    val = np.load((self + '/picked_aligned_val.npz'), allow_pickle=True)
    test = np.load((self + '/picked_aligned_test.npz'), allow_pickle=True)

    return (train['X0'].astype(np.float32), train['X1'].astype(np.float32)), train['y'].astype(np.float32), \
        (val['X0'].astype(np.float32), val['X1'].astype(np.float32)), val['y'].astype(np.float32), \
        (test['X0'].astype(np.float32), test['X1'].astype(np.float32)), test['y'].astype(np.float32), \
        test['X0_plot'].astype(np.float32)

def load_calibration_dataset(self):
    cal = np.load((self + '/picked_aligned_calibration.npz'), allow_pickle=True)

    return (cal['X0'].astype(np.float32), cal['X1'].astype(np.float32)), cal['y'].astype(np.float32)

if __name__ == "__main__":
    main()
