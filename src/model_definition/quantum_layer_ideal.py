import numpy as np
import qiskit.circuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

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
