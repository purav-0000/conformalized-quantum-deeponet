import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from src.model_definition.quantum_layer_ideal import data_loader, RBS
from qiskit.circuit.library import UCRYGate


def pad_to_power_of_two(x):
    """ Required since controlled rotations only handle angles with len(angles) = power of 2"""
    n = x.shape[0]
    target = 2 ** int(np.ceil(np.log2(n)))
    pad_rows = target - n
    if pad_rows == 0:
        return x

    # Pad with 0s which becomes identity matrix for Ry gate.
    padding = np.zeros((pad_rows, x.shape[1]))
    return np.vstack([x, padding])

def create_spqc_circuit(
        n_in: int,
        n_out: int,
        ensemble_thetas: np.ndarray,
        data_arrays: np.ndarray,
        loader_inv_gate,
        loader_special_gate
) -> QuantumCircuit:
    """
    Creates a single quantum circuit to simulate an entire ensemble in parallel (SPQC).

    Args:
        n_in: Number of input features.
        n_out: Number of output features.
        ensemble_thetas: A 2D array of theta parameters, shape (num_models, num_thetas).
        data_arrays: The classical input data vector.
        loader_inv_gate: The inverse data loader gate.
        loader_special_gate: The special data loader gate for tomography.

    Returns:
        A single, large Qiskit QuantumCircuit.
    """
    num_models = ensemble_thetas.shape[0]
    num_qubits = max(n_in, n_out)

    # --- Register Setup ---
    num_addr_qubits = int(np.ceil(np.log2(num_models))) if num_models > 1 else 0
    addr_qr = QuantumRegister(num_addr_qubits, name='addr')
    anc_qr = QuantumRegister(1, name='anc')
    tomo_qr = QuantumRegister(num_qubits, name='tomo')

    circuit = QuantumCircuit(addr_qr, anc_qr, tomo_qr)

    # --- Qubit Indexing ---
    tomo_q_offset = num_addr_qubits + 1 # Where 1 is for ancilla
    input_qubits = list(range(tomo_q_offset + num_qubits - n_in, tomo_q_offset + num_qubits))
    tomo_qubits = list(range(tomo_q_offset, tomo_q_offset + num_qubits))

    # --- Circuit Construction ---
    # 1. Prepare address and ancilla qubits in superposition
    if num_addr_qubits > 0:
        state = np.zeros(2 ** num_addr_qubits)
        state[:num_models] = 1
        circuit.prepare_state(state, addr_qr[:], normalize=True)

    circuit.h(anc_qr)
    circuit.cx(anc_qr, tomo_qr[num_qubits - n_in])

    # 2. Load classical data (different data for each model)
    # loader_data_gate = data_loader(data_arrays[0])
    # circuit.append(loader_data_gate, input_qubits)

    """
    for i, data_array in enumerate(data_arrays):
        loader_gate = data_loader(data_array)
        control_state = f'{i:0{num_addr_qubits}b}'
        controlled_loader = loader_gate.control(
            num_ctrl_qubits=num_addr_qubits,
            label=f'C-Load_{i}',
            ctrl_state=control_state
        )
        # Append the controlled gate to the circuit
        circuit.append(controlled_loader, addr_qr[:] + input_qubits)
    """
    _apply_parallel_data_loader(circuit, data_arrays, addr_qr, input_qubits)

    # 3. Apply the parallelized trainable unitary (W) using UCRY gates
    # Pad to power of 2 first
    ensemble_thetas = pad_to_power_of_two(ensemble_thetas)
    _apply_parallel_w(circuit, n_in, n_out, ensemble_thetas, addr_qr, tomo_qr)

    # 4. Apply tomography components
    circuit.append(loader_inv_gate, tomo_qubits)
    circuit.x(anc_qr)
    circuit.cx(anc_qr, tomo_qr[0])
    circuit.append(loader_special_gate, tomo_qubits)
    circuit.h(anc_qr)

    return circuit


def _apply_parallel_w(circuit, n_in, n_out, thetas, addr_qr, tomo_qr):
    """Internal function to build the controlled-unitary part of the SPQC circuit."""
    larger_features, smaller_features = max(n_in, n_out), min(n_in, n_out)

    correct_size = int((2 * larger_features - 1 - smaller_features) * (smaller_features / 2))   # Number of free parameters
    if thetas.shape[1] != correct_size:
        raise ValueError(f"Size of parameters should be {correct_size} but it is {thetas.shape[1]}")

    if larger_features == smaller_features: smaller_features -= 1
    q_end_indices = np.concatenate(
        [np.arange(2, larger_features + 1), larger_features + 1 - np.arange(2, smaller_features + 1)])
    q_start_indices = np.concatenate([np.arange(q_end_indices.shape[0] + smaller_features - larger_features) % 2,
                                      np.arange(larger_features - smaller_features)])
    q_slice_sizes = q_end_indices - q_start_indices
    if n_in < n_out:
        q_end_indices, q_start_indices, q_slice_sizes = q_end_indices[::-1], q_start_indices[::-1], q_slice_sizes[::-1]

    theta_start_idx = 0
    for i, q_start_index in enumerate(q_start_indices):
        num_thetas_in_slice = q_slice_sizes[i] // 2
        theta_slice = thetas[:, theta_start_idx: theta_start_idx + num_thetas_in_slice]

        for j in range(theta_slice.shape[1]):
            c_qubit, t_qubit = q_start_index, q_start_index + 1

            # Unconditional scaffolding
            circuit.h([tomo_qr[c_qubit], tomo_qr[t_qubit]])
            circuit.cz(tomo_qr[c_qubit], tomo_qr[t_qubit])

            # Conditional rotations (uniformly controlled)
            ucry_c = UCRYGate(list(theta_slice[:, j].flatten()))
            ucry_t = UCRYGate(list(-theta_slice[:, j].flatten()))

            circuit.append(ucry_c, [tomo_qr[c_qubit]] + addr_qr[:])
            circuit.append(ucry_t, [tomo_qr[t_qubit]] + addr_qr[:])

            # Unconditional scaffolding
            circuit.cz(tomo_qr[c_qubit], tomo_qr[t_qubit])
            circuit.h([tomo_qr[c_qubit], tomo_qr[t_qubit]])

            q_start_index += 2
        theta_start_idx += num_thetas_in_slice


def _apply_parallel_data_loader(circuit, data_arrays, addr_qr, input_qubits):
    """
    Builds a depth-optimized, parallel data loader circuit.

    This function applies the H and CZ gates of the RBS structure unconditionally,
    and only controls the RY rotations via UCRYGates, significantly reducing depth.
    """
    num_models = data_arrays.shape[0]
    num_qubits = data_arrays.shape[1]
    num_params = num_qubits - 1

    # Calculate RBS parameters (mimics quantum_layer_ideal.py's data_loader)
    all_loader_params = np.empty((num_models, num_params), dtype=np.float64)
    for i, data_array in enumerate(data_arrays):
        # Normalize data
        norm = np.linalg.norm(data_array, ord=2)
        if abs(norm - 1) > 1e-8:
            data_array = data_array / norm

        # Compute unary encoding parameters (thetas for the RBS gates)
        sin_product = 1.0
        params = np.empty(num_params, dtype=np.float64)
        for j in range(num_params):
            # Clamp the argument to avoid domain errors from floating point inaccuracies
            arg = np.clip(data_array[j] * sin_product, -1.0, 1.0)
            params[j] = np.arccos(arg)

            # Avoid division by zero if sin is ~0
            sin_val = np.sin(params[j])
            sin_product /= sin_val if abs(sin_val) > 1e-9 else 1e-9

        if data_array[-1] < 0:
            params[-1] *= -1

        all_loader_params[i, :] = params

    # Padding to ensure power of 2
    all_loader_params_padded = pad_to_power_of_two(all_loader_params)

    # Build circuit layer by layer
    for i in range(num_params):
        c_qubit, t_qubit = input_qubits[i], input_qubits[i + 1]

        # Unconditional scaffolding
        circuit.h([c_qubit, t_qubit])
        circuit.cz(c_qubit, t_qubit)

        # Conditional rotations (uniformly controlled)
        thetas_for_this_step = all_loader_params_padded[:, i]
        ucry_pos = UCRYGate(list(thetas_for_this_step))
        ucry_neg = UCRYGate(list(-thetas_for_this_step))

        circuit.append(ucry_pos, [c_qubit] + addr_qr[:])
        circuit.append(ucry_neg, [t_qubit] + addr_qr[:])

        # Unconditional scaffolding
        circuit.cz(c_qubit, t_qubit)
        circuit.h([c_qubit, t_qubit])

"""
for j in range(num_models):
    q_start_index_per_model = q_start_index
    theta_slice = thetas[j][theta_start_index:theta_start_index+q_slice_sizes[i]//2]
    print(theta_slice.shape)
    for theta in theta_slice:
        controlled_RBS = RBS(theta).control(num_addr_qubits, ctrl_state=j)
        tomo_circuit.compose(controlled_RBS, qubits=addr_qr[:] + [tomo_qr[q_start_index_per_model]] + [tomo_qr[q_start_index_per_model + 1]], inplace=True)
        q_start_index_per_model += 2
theta_start_index += q_slice_sizes[i]//2
"""

"""
for i, W_gate in enumerate(model_Ws):
controlled_W = W_gate.control(num_addr_qubits, ctrl_state=i)
tomo_circuit.append(controlled_W, addr_qr[:] + tomo_qr[:])
"""

"""
def SPQC_circuit(n_in, n_out, thetas, data_array, loader_inv_gate, loader_special_gate):

    num_qubits = max(n_in, n_out)
    num_models = len(thetas)

    # Address qubits to control which model (W) is applied
    num_addr_qubits = int(np.ceil(np.log2(num_models)))
    # print("Addr qubits: ", num_addr_qubits)
    addr_qr = QuantumRegister(num_addr_qubits, name='addr')
    addr_cr = ClassicalRegister(num_addr_qubits, name='addr_c')


    # Ancilla for extracting output
    anc_qr = QuantumRegister(1, name='anc')
    anc_cr = ClassicalRegister(1, name='anc_c')

    # Main data processing register
    tomo_qr = QuantumRegister(num_qubits, name='tomo')
    tomo_cr = ClassicalRegister(n_out, name='tomo_c')

    tomo_circuit = QuantumCircuit(addr_qr, anc_qr, tomo_qr, addr_cr, anc_cr, tomo_cr)


    input_qubits = list(range(num_qubits - n_in + 1 + num_addr_qubits, num_qubits + 1 + num_addr_qubits))
    tomo_qubits = list(range(1 + num_addr_qubits, num_qubits + 1 + num_addr_qubits))


    if num_addr_qubits != 0:
        state = np.zeros(2 ** num_addr_qubits)
        state[:num_models] = 1
        tomo_circuit.prepare_state(state, normalize=True, qubits=addr_qr)
        # tomo_circuit.h(addr_qr)


    tomo_circuit.h(anc_qr)
    tomo_circuit.cx(anc_qr, tomo_qr[num_qubits - n_in])

    loader_data_gate = data_loader(data_array)
    tomo_circuit.append(loader_data_gate, input_qubits)



    for i, W_gate in enumerate(model_Ws):
        controlled_W = W_gate.control(num_addr_qubits, ctrl_state=i)
        tomo_circuit.append(controlled_W, addr_qr[:] + tomo_qr[:])


    # ----------------------------------------------------

    larger_features = max(n_in, n_out)
    smaller_features = min(n_in, n_out)

    correct_size = int((2 * larger_features - 1 - smaller_features) * (smaller_features / 2))
    if len(thetas[0]) != correct_size:
        raise Exception("Size of parameter should be {:d} but now it is {:d}".format(correct_size, len(thetas)))

    if larger_features == smaller_features:
        smaller_features -= 1  # 6-6 6-5 have the same pyramid
    q_end_indices = np.concatenate([
        np.arange(2, larger_features + 1),
        larger_features + 1 - np.arange(2, smaller_features + 1)
    ])
    q_start_indices = np.concatenate([
        np.arange(q_end_indices.shape[0] + smaller_features - larger_features) % 2,  # [0, 1, 0, 1, ...]
        np.arange(larger_features - smaller_features)
    ])

    q_slice_sizes = q_end_indices - q_start_indices

    if n_in < n_out:  # generate the pyramid for in_features < out_features case
        q_end_indices = q_end_indices[::-1]
        q_start_indices = q_start_indices[::-1]
        q_slice_sizes = q_slice_sizes[::-1]
        # pad x fist if in_features < out_features case

    theta_start_index = 0
    #q_end_indices += num_addr_qubits + 1
    #q_start_indices += num_addr_qubits + 1

    thetas = np.array(thetas)
    for i, q_start_index in enumerate(q_start_indices):
        theta_slice = thetas[:, theta_start_index:theta_start_index+q_slice_sizes[i]//2]
        # import pdb; pdb.set_trace()
        for j in range(theta_slice.shape[1]):

            c_qubit = q_start_index
            t_qubit = q_start_index + 1

            tomo_circuit.h([tomo_qr[c_qubit], tomo_qr[t_qubit]])
            tomo_circuit.cz(tomo_qr[c_qubit], tomo_qr[t_qubit])

            c_qubit_angles = theta_slice[:, j].flatten()
            t_qubit_angles = [-t for t in theta_slice[:, j].flatten()]

            #print(len(c_qubit_angles))
            ucry_c = UCRYGate(list(c_qubit_angles))
            ucry_t = UCRYGate(t_qubit_angles)

            #print(c_qubit_angles)
            #print([tomo_qr[c_qubit]] + addr_qr[:])
            tomo_circuit.append(ucry_c, [tomo_qr[c_qubit]] + addr_qr[:])
            tomo_circuit.append(ucry_t, [tomo_qr[t_qubit]] + addr_qr[:])

            tomo_circuit.cz(tomo_qr[c_qubit], tomo_qr[t_qubit])
            tomo_circuit.h([tomo_qr[c_qubit], tomo_qr[t_qubit]])

            q_start_index += 2
        theta_start_index += q_slice_sizes[i] // 2


        for j in range(num_models):
            q_start_index_per_model = q_start_index
            theta_slice = thetas[j][theta_start_index:theta_start_index+q_slice_sizes[i]//2]
            print(theta_slice.shape)
            for theta in theta_slice:
                controlled_RBS = RBS(theta).control(num_addr_qubits, ctrl_state=j)
                tomo_circuit.compose(controlled_RBS, qubits=addr_qr[:] + [tomo_qr[q_start_index_per_model]] + [tomo_qr[q_start_index_per_model + 1]], inplace=True)
                q_start_index_per_model += 2
        theta_start_index += q_slice_sizes[i]//2

    # ----------------------------------------------------

    tomo_circuit.append(loader_inv_gate, tomo_qubits)

    tomo_circuit.x(anc_qr)
    tomo_circuit.cx(anc_qr, tomo_qr[0])
    tomo_circuit.append(loader_special_gate, tomo_qubits)

    tomo_circuit.h(anc_qr)

    return tomo_circuit
"""
