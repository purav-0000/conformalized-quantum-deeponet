import argparse
import logging
import os
import random
import secrets
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeGuadalupe # 16-qubit device
from src.quantum_layer_ideal import data_loader, W
from src.spqc import SPQC_circuit
from src.utils.common import apply_overrides, load_dataset, load_calibration_dataset, normalize_bounds, transform_input
from src.utils.simulation import evaluate_model, load_weights, plot_pred, silu


def build_circuit(x_input0, n_in, n_out, thetas, loader_special_gate, loader_inv_gate, simulator):
    x_input = x_input0.copy()
    x_input += (np.abs(x_input) < 1e-7) * 1e-7
    circ = SPQC_circuit(n_in, n_out, thetas, x_input, loader_inv_gate, loader_special_gate)
    circ.save_statevector('state')
    return transpile(circ, simulator)


def process_state(idx, results, num_models, n_in, n_out, hidden0_bias, output_weight, output_bias, mode='ideal', trunk=False):
    state = np.real(results.data(idx)['state'].data)

    """
    if not trunk:
        # 1. Determine the number of qubits from the length of the state vector
        num_qubits = int(np.log2(len(state)))

        print(f"State vector for a {num_qubits}-qubit system.\n")
        print("First 30 amplitudes and their corresponding bitstrings:")
        print("-" * 50)
        print(f"{'Bitstring':<12} | {'Statevector Amplitude'}")
        print("-" * 50)

        # 2. Loop through the first 20 indices of the state vector
        for i in range(0, 30):
            # Get the amplitude
            amplitude = state[i]

            # 3. Format the index 'i' as a binary string, padded with leading zeros
            bitstring = f'{i:0{num_qubits}b}'

            print(f"|{bitstring}>       | {amplitude:+.8f}")
    """
    state = state ** 2

    all_outputs = np.zeros((num_models, n_out))
    formatting = int(np.ceil(np.log2(num_models)))

    # Bit indexing
    for i in range(n_out):
        pos = ['0'] * n_out
        pos[i] = '1'
        pos0 = ['0'] + ['0'] * (n_in - n_out) + pos
        pos1 = ['1'] + ['0'] * (n_in - n_out) + pos
        pos0 = ''.join(pos0)[::-1]
        pos1 = ''.join(pos1)[::-1]
        for j in range(num_models):
            result0 = state[int(pos0 + format(j, f'0{formatting}b'), 2)]
            result1 = state[int(pos1 + format(j, f'0{formatting}b'), 2)]
            all_outputs[j][i] = (np.sqrt(max(n_in, n_out)) * (result0 - result1))


    for i in range(num_models):
        all_outputs[i] = all_outputs[i] * num_models


    for i in range(num_models):
        # Reasoning for this?
        all_outputs[i] = silu(all_outputs[i] + hidden0_bias[i])
        all_outputs[i] = np.dot(all_outputs[i], output_weight[i].T) + output_bias[i]


    if trunk:
        for i in range(num_models):
            all_outputs[i] = silu(all_outputs[i])
    return all_outputs


def run_quantum_layer(inputs, n_in, n_out, ensemble_params, simulator, trunk=False):
    # bias, weight, output_bias, thetas,
    # Precompute gates
    sqrt_norm = np.sqrt(max(n_in, n_out))
    model_Ws = []
    for i in range(len(ensemble_params['hidden0_thetas'])):
        model_Ws.append(W(n_in, n_out, ensemble_params['hidden0_thetas'][i]))  # Select hidden thetas

    loader = data_loader(np.full(max(n_in, n_out), 1 / sqrt_norm))
    loader_inv = loader.inverse()

    outputs = []
    # Add batch size if memory cannot handle inputs
    batch_size = len(inputs)
    disable_tqdm = batch_size == len(inputs)

    for i in tqdm(range(0, len(inputs), batch_size), desc="Running in batches", disable=disable_tqdm):
        # Batching
        batch = inputs[i:i + batch_size]

        # Circuit construction
        circuits = Parallel(n_jobs=-1)(
            delayed(build_circuit)(x, n_in, n_out, ensemble_params['hidden0_thetas'], loader, loader_inv, simulator)
            for x in tqdm(batch, desc="Building circuits", disable=not disable_tqdm)
        )

        # print(circuits[0].depth())
        # Execution
        results = simulator.run(circuits, shots=1).result()

        # process_state executes faster if n_jobs = 1
        batch_outputs = Parallel(n_jobs=1)(
            delayed(process_state)(j, results, len(ensemble_params['hidden0_bias']), n_in, n_out, ensemble_params['hidden0_bias'], ensemble_params['output_weight'], ensemble_params['output_bias'], 'ideal', trunk)
            for j in tqdm(range(len(batch)), desc="Processing", disable=not disable_tqdm)
        )
        outputs.extend(batch_outputs)

    return np.array(outputs)


def run_ensemble_SPQC(ensemble_path, inputs, simulator):
    ensemble_dir = os.path.join("models", "ensembles", ensemble_path)
    model_dirs = [os.path.join(ensemble_dir, m) for m in os.listdir(ensemble_dir) if
                  os.path.isdir(os.path.join(ensemble_dir, m))]

    all_params = [load_weights(m) for m in model_dirs]


    branch_params = {
        "hidden0_bias": [all_params[i]["branch_hidden0_bias"] for i in range(len(all_params))],
        "hidden0_thetas": [all_params[i]["branch_hidden0_thetas"] for i in range(len(all_params))],
        "output_bias": [all_params[i]["branch_output_bias"] for i in range(len(all_params))],
        "output_weight": [all_params[i]["branch_output_weight"] for i in range(len(all_params))],
    }

    branch_outputs = run_quantum_layer(
        inputs=inputs[0],  # Branch inputs
        n_in=inputs[0].shape[1],
        n_out=branch_params['hidden0_bias'][0].shape[0],
        ensemble_params=branch_params,
        simulator=simulator,
        trunk=False
    )

    trunk_params = {
        "hidden0_bias": [all_params[i]["trunk_hidden0_bias"] for i in range(len(all_params))],
        "hidden0_thetas": [all_params[i]["trunk_hidden0_thetas"] for i in range(len(all_params))],
        "output_bias": [all_params[i]["trunk_output_bias"] for i in range(len(all_params))],
        "output_weight": [all_params[i]["trunk_output_weight"] for i in range(len(all_params))],
    }

    trunk_outputs = run_quantum_layer(
        inputs=inputs[1],  # Trunk inputs
        n_in=inputs[1].shape[1],
        n_out=trunk_params['hidden0_bias'][0].shape[0],
        ensemble_params=trunk_params,
        simulator=simulator,
        trunk=True
    )

    # print(branch_outputs[0])
    # print(branch_outputs[1])
    # print(branch_outputs[2])

    outputs = []
    for i in range(len(model_dirs)):
        outputs.append(np.einsum('bi,ni->bn', branch_outputs[:, i, :], trunk_outputs[:, i, :]) + all_params[i]['b'])
    return outputs


def main():
    simulator = AerSimulator(device='CPU')

    # Mention this
    data_path = os.path.join("data/data_ode_simple")

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

    outputs = run_ensemble_SPQC("overhaul", x_test, simulator)
    evaluate_model(np.array(outputs), y_test, verbose=True)

if __name__ == "__main__":
    main()

