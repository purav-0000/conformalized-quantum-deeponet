import numpy as np
import qiskit.circuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector

def RBS(theta): # RBS gate with parameter t
    rbs_q = QuantumRegister(2)
    c_qubit = rbs_q[0]
    t_qubit = rbs_q[1]
    rbs = QuantumCircuit([c_qubit,t_qubit], name='RBS_'+str(theta))
    rbs.h(c_qubit)
    rbs.h(t_qubit)
    rbs.cz(c_qubit, t_qubit)

    rbs.ry(theta, c_qubit)
    rbs.ry(-theta, t_qubit)

    rbs.cz(c_qubit, t_qubit)
    rbs.h(c_qubit)
    rbs.h(t_qubit)
    return rbs.to_gate()

def data_loader(data_array):
    """
    Constructs a quantum gate that prepares a unary-encoded quantum state
    from a given classical input vector.

    The encoding uses a sequence of RBS gates applied to adjacent qubits.
    The input vector is automatically normalized if needed.

    Args:
        data_array (array-like): 1D input array of real numbers.

    Returns:
        qiskit.circuit.Gate: A quantum gate representing the data loading circuit.
    """
    if len(data_array) < 2:
        raise ValueError("Input array must have at least 2 elements.")

    # Normalize data if needed
    norm = np.linalg.norm(data_array, ord=2)
    if abs(norm - 1) > 1e-8:
        data_array = data_array / norm

    num_qubits = len(data_array)
    num_params = num_qubits - 1

    # Compute unary encoding parameters
    sin_product = 1.0
    params = np.empty(num_params, dtype=np.float64)
    for i in range(num_params):
        # Clamp the argument to avoid domain errors from floating point inaccuracies
        arg = np.clip(data_array[i] * sin_product, -1.0, 1.0)
        params[i] = np.arccos(arg)

        # Avoid division by zero if sin is ~0
        sin_val = np.sin(params[i])
        sin_product /= sin_val if abs(sin_val) > 1e-9 else 1e-9

    # Flip the final angle if the last component is negative
    if data_array[-1] < 0:
        params[-1] *= -1

    # Build the loading circuit
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)
    for i in range(num_params):
        qc.compose(RBS(params[i]), qubits=[i, i + 1], inplace=True)

    return qc.to_gate(label="DataLoader")


def data_loader_angles(data_arrays):
    """Vectorized unary-loader angles for one vector or a batch of vectors."""
    values = np.asarray(data_arrays, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("data_arrays must have shape (batch, features >= 2)")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot unary-encode an all-zero vector")
    normalized = values / norms
    params = np.empty((len(normalized), normalized.shape[1] - 1), dtype=np.float64)
    sin_product = np.ones(len(normalized), dtype=np.float64)
    for i in range(params.shape[1]):
        params[:, i] = np.arccos(np.clip(normalized[:, i] * sin_product, -1.0, 1.0))
        sin_value = np.sin(params[:, i])
        sin_product /= np.where(np.abs(sin_value) > 1e-9, sin_value, 1e-9)
    params[normalized[:, -1] < 0, -1] *= -1
    return params[0] if was_vector else params


def parameterized_data_loader(num_features, prefix="x"):
    """Return one reusable unary-loader gate and its bindable parameters."""
    if num_features < 2:
        raise ValueError("num_features must be at least 2")
    parameters = ParameterVector(prefix, num_features - 1)
    circuit = QuantumCircuit(num_features, name="DataLoader")
    for i, theta in enumerate(parameters):
        circuit.compose(RBS(theta), qubits=[i, i + 1], inplace=True)
    return circuit.to_gate(label="DataLoader"), tuple(parameters)


def find_nparams(n,d): # size_in, size_out
    return int((2*n-1-d)*(d/2))


def W(n_in, n_out, thetas): #generate thetas else where
    
    larger_features = max(n_in,n_out)
    smaller_features = min(n_in,n_out)

    correct_size = int((2*larger_features - 1 - smaller_features) * (smaller_features / 2))
    if len(thetas) != correct_size:
        raise Exception("Size of parameter should be {:d} but now it is {:d}".format(correct_size, len(thetas)))
    
    W_qr = QuantumRegister(larger_features)
    W_circuit = QuantumCircuit(W_qr)

    if larger_features == smaller_features:
        smaller_features -= 1 #6-6 6-5 have the same pyramid
    q_end_indices = np.concatenate([
        np.arange(2, larger_features +1 ),
        larger_features + 1 - np.arange(2, smaller_features +1 )
    ]) 
    q_start_indices = np.concatenate([
        np.arange(q_end_indices.shape[0] + smaller_features - larger_features)%2,# [0, 1, 0, 1, ...]
        np.arange( larger_features- smaller_features)
    ])  

    q_slice_sizes = q_end_indices - q_start_indices

    if n_in <n_out:  # generate the pyramid for in_features < out_features case
        q_end_indices = q_end_indices[::-1]
        q_start_indices = q_start_indices[::-1]
        q_slice_sizes =  q_slice_sizes[::-1]
        # pad x fist if in_features < out_features case

    theta_start_index = 0

    for i,q_start_index in enumerate(q_start_indices):
        
        theta_slice = thetas[theta_start_index:theta_start_index+q_slice_sizes[i]//2]

        # import pdb; pdb.set_trace()
        for theta in theta_slice:
            #print('theta',theta)
            W_circuit.compose(RBS(theta), qubits=[W_qr[q_start_index], W_qr[ q_start_index+1]], inplace=True)
            q_start_index += 2
        theta_start_index += q_slice_sizes[i]//2
    # fig = W_circuit.draw(output='mpl')
    # plt.show()
    return W_circuit.to_gate()


def custom_tomo_fast(n_in, n_out, data_array, W_gate, loader_special_gate, loader_inv_gate):
    num_qubits = max(n_in, n_out)

    anc_qr = QuantumRegister(1)
    anc_cr = ClassicalRegister(1)
    tomo_qr = QuantumRegister(num_qubits)

    # EDIT?
    tomo_cr = ClassicalRegister(num_qubits)
    tomo_circuit = QuantumCircuit(anc_qr, tomo_qr, anc_cr, tomo_cr)

    input_qubits = list(range(num_qubits - n_in + 1, num_qubits + 1))
    tomo_qubits = list(range(1, num_qubits + 1))

    tomo_circuit.h(anc_qr)
    tomo_circuit.cx(anc_qr, tomo_qr[num_qubits - n_in])

    # These are the only dynamic parts:
    loader_data_gate = data_loader(data_array)

    tomo_circuit.append(loader_data_gate, input_qubits)
    tomo_circuit.append(W_gate, tomo_qr)
    tomo_circuit.append(loader_inv_gate, tomo_qubits)

    tomo_circuit.barrier()
    tomo_circuit.x(anc_qr)
    tomo_circuit.cx(anc_qr, tomo_qr[0])
    tomo_circuit.append(loader_special_gate, tomo_qubits)

    tomo_circuit.barrier()
    tomo_circuit.h(anc_qr)

    return tomo_circuit


def custom_tomo_template(n_in, n_out, loader_data_gate, W_gate,
                         loader_special_gate, loader_inv_gate):
    """Build the static circuit topology used by all inputs in a layer."""
    num_qubits = max(n_in, n_out)
    anc_qr = QuantumRegister(1, "anc")
    tomo_qr = QuantumRegister(num_qubits, "tomo")
    circuit = QuantumCircuit(anc_qr, tomo_qr)
    input_qubits = list(range(num_qubits - n_in + 1, num_qubits + 1))
    tomo_qubits = list(range(1, num_qubits + 1))

    circuit.h(anc_qr)
    circuit.cx(anc_qr, tomo_qr[num_qubits - n_in])
    circuit.append(loader_data_gate, input_qubits)
    circuit.append(W_gate, tomo_qr)
    circuit.append(loader_inv_gate, tomo_qubits)
    circuit.x(anc_qr)
    circuit.cx(anc_qr, tomo_qr[0])
    circuit.append(loader_special_gate, tomo_qubits)
    circuit.h(anc_qr)
    return circuit


def tomo_output_fast(n_in, n_out, data_array, simulator,
                     W_gate, loader_special_gate, loader_inv_gate):
    data_array = data_array + (np.abs(data_array) < 1e-7) * 1e-7
    tomo_circuit = custom_tomo_fast(n_in, n_out, data_array,
                                    W_gate, loader_special_gate, loader_inv_gate)

    tomo_circuit.save_statevector('state')
    state = simulator.run(transpile(tomo_circuit, simulator), shots=1).result()
    result = np.real(state.data()['state'].data)

    output = []
    for i in range(n_out):
        pos = ['0'] * n_out
        pos[i] = '1'
        pos0 = ['0'] + ['0'] * (n_in - n_out) + pos
        pos1 = ['1'] + ['0'] * (n_in - n_out) + pos
        result0 = result[int(''.join(pos0)[::-1], 2)]
        result1 = result[int(''.join(pos1)[::-1], 2)]
        output.append(np.sqrt(max(n_in, n_out)) * (result0 ** 2 - result1 ** 2))

    return np.array(output)
