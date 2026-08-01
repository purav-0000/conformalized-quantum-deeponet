"""Original per-sample circuit construction path retained for benchmarks."""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile

from src.model_definition.quantum_layer_ideal import custom_tomo_fast


def build_circuit_legacy(
    x_input: np.ndarray,
    n_in: int,
    n_out: int,
    W_gate,
    loader_gate,
    loader_inv_gate,
    simulator,
    noisy: bool = False,
) -> Optional[QuantumCircuit]:
    """Build and transpile a new circuit for one input, as before optimization."""
    x_input_stable = x_input.copy()
    x_input_stable[np.abs(x_input_stable) < 1e-8] = 1e-8
    circuit = custom_tomo_fast(
        n_in,
        n_out,
        x_input_stable,
        W_gate,
        loader_gate,
        loader_inv_gate,
    )
    if noisy:
        circuit.save_density_matrix()
    else:
        circuit.save_statevector("state")
    return transpile(circuit, simulator, optimization_level=0)
