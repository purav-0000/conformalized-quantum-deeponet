###############################################################################
# NOTE: Noise simulation on fake backends is not included in this script.
# ---------------------------------------------------------------------------
# If you wish to simulate depolarizing channels, readout errors, and thermal
# relaxation errors on a fake backend you must modify the code manually:
#
# 1) Edit `_setup_simulator()` to construct an Aer simulator with noise
#    derived from a fake backend.
# 2) Replace the probability-distribution extraction in
#    `_process_quantum_output()` with the alternative code provided
#    at the bottom of this script.
# 3) Set the number of shots in `_run_quantum_layer()` to
#    `self.config.shots`.
# 4) In `src/utils/simulation.py`, uncomment `.measure_all()` and
#    comment out any saves of density matrices or statevectors.
###############################################################################

import argparse
import logging
import os
import random
import re
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from tqdm import tqdm

# Local application imports
from src.model_definition.classical_orthogonal_NN import OrthoNN
from src.model_definition.classical_res_ortho_deeponet import ResOrthoNN
from src.model_definition.quantum_layer_ideal import W, data_loader
from src.model_definition.spqc import create_spqc_circuit
from src.utils.common import apply_overrides
from src.utils.data_handling import DataHandler
from src.utils.simulation import (build_circuit, evaluate_model, load_weights,
                                  plot_pred, silu)

# No qiskit verbose prints
logging.getLogger('qiskit').setLevel(logging.WARNING)

# --- Constants ---
BRANCH_PREFIX = "branch"
TRUNK_PREFIX = "trunk"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# GPU Temperature Management Configuration (optional)
MAX_GPU_TEMP = 70  # Celsius
GPU_CHECK_INTERVAL = 5  # Seconds


# --- Config ---

@dataclass
class Config:

    # Data options
    data_dir: str = "antiderivative"
    fourier_features: bool = False

    # Pick model/ensemble
    model: str = None
    ensemble: str = None

    # Simulation parameters
    batch_size: int = None
    coverage: float = 0.9
    spqc: bool = False
    target_gpu: int = 0
    residual: bool = False
    online: bool = False
    n_jobs: int = 4
    simulator: str = "CPU"  # or "GPU"

    # Quantum circuit parameters
    mode: str = "ideal"     # or "shots"
    shots: int = 0
    classical_branch: bool = False
    classical_trunk: bool = False
    noise: float = 0.0

    # Seed and debugging
    analyze_circuit_cost: bool = False
    seed: int = field(default_factory=lambda: secrets.randbits(32))


# --- Utilities ---

def get_gpu_temp(gpu_id=0):
    """
        Retrieves the current temperature of a specified NVIDIA GPU.

        Args:
            gpu_id: The integer ID of the GPU.

        Returns:
            The GPU temperature in Celsius, or None if it cannot be retrieved.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logging.warning(f"Could not get GPU temperature for GPU {gpu_id}: {e}")
        return None


def load_config(path: str) -> Config:
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
        """
        Initializes the SimulationRunner with a given configuration.

        Args:
            config: The simulation configuration object.
        """
        self.config = config
        set_seeds(config.seed)

        self.simulator = self._setup_simulator()
        self.data_handler = self._setup_data_handler()

        if self.config.ensemble:
            self.output_dir = Path("models", "ensembles", self.config.ensemble)
        elif self.config.model:
            self.output_dir = Path("models", self.config.model)
        else:
            raise ValueError("Configuration must specify either a 'model' or an 'ensemble'.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Results will be saved in: {self.output_dir}")

    def _setup_data_handler(self) -> DataHandler:
        """Initializes the DataHandler and loads data."""
        handler = DataHandler(
            data_dir=self.config.data_dir,
            fourier_features=self.config.fourier_features,
            online=self.config.online
        )
        handler.load_and_process_data()
        return handler

    def _setup_simulator(self) -> AerSimulator:
        """
        Configures and returns the Qiskit AerSimulator based on the config.

        Returns:
            An instance of AerSimulator.
        """
        noise_model = None
        method = 'statevector'

        if self.config.noise > 0.0:
            logging.info(f"Setting up depolarizing noise model with level: {self.config.noise}")
            # For basis gates of Eagle processor
            noise_model = NoiseModel(basis_gates=['ecr', 'id', 'rz', 'sx', 'x'])
            error_1q = depolarizing_error(self.config.noise, 1)
            error_2q = depolarizing_error(0.8 * self.config.noise, 2)  # Different error for 2-qubit gates
            noise_model.add_all_qubit_quantum_error(error_1q, ['id', 'rz', 'sx', 'x'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['ecr'])
            method = 'density_matrix'

        # SET METHOD TO AUTOMATIC IF REALISTIC NOISE SIMULATION
        return AerSimulator(device=self.config.simulator, method=method, noise_model=noise_model)

    def run(self) -> None:
        """
        The main execution method that triggers the appropriate simulation workflow.
        """
        if self.config.spqc:
            if not self.config.ensemble:
                raise ValueError("SPQC mode requires an ensemble configuration.")

            # Implementation checks
            """
            if self.config.noise > 0.0:
                raise ValueError("Noisy simulations for SPQC not implemented yet."
                                 "Use SPQC for vanilla antiderivative experiments ideally.")
            """
            if self.config.residual:
                raise ValueError("Residual connections for SPQC not implemented yet."
                                 "Use SPQC for vanilla antiderivative experiments ideally.")
            logging.info("--- Running in SPQC Ensemble Mode ---")
            self._run_ensemble_spqc()
        elif self.config.ensemble:
            logging.info("--- Running in Sequential Ensemble Mode ---")
            self._run_ensemble_sequential()
        elif self.config.model:
            logging.info("--- Running in Single Model Mode ---")
            self._run_single_model()

    def _get_dataset(self, split: str):
        x_split, y_split = self.data_handler.get_split(split)

        # Online dataset is 3D (num_signals, num_locs, features)
        # The 3D shape is ideal for plotting, but model still takes 2D shapes
        if self.config.online:
            x_split = (x_split[0].reshape(-1, x_split[0].shape[-1]), x_split[1].reshape(-1, x_split[1].shape[-1]))
            y_split = y_split.reshape(-1, y_split.shape[-1])

        return x_split, y_split


    def _run_single_model(self) -> None:
        """Executes the simulation for a single model and saves the results."""
        model_path = Path("models") / self.config.model
        logging.info(f"Running single model simulation: {self.config.model}")

        x_test, y_test = self._get_dataset('test')

        y_pred = self._run_deeponet_forward_pass(model_path,
                                                 x_test)

        # Evaluate and save results
        evaluate_model(y_pred, y_test, self.output_dir, verbose=False)

        # data_handler used directly to avoid reshaping for online datasets in _get_dataset
        plot_pred(
            self.data_handler.datasets['test']['X'], self.data_handler.datasets['test']['y'], y_pred,
            self.output_dir, self.data_handler.datasets['test']['X0_plot'], online=self.config.online
        )

    def _run_ensemble_sequential(self) -> None:
        """Executes the simulation for an ensemble of models and applies conformal prediction."""
        ensemble_dir = Path("models", "ensembles", self.config.ensemble)
        model_dirs = sorted([d for d in ensemble_dir.iterdir() if d.is_dir() and d.name != 'simulation_plots'])
        logging.info(f"Found {len(model_dirs)} models in ensemble: {self.config.ensemble}")

        x_cal, y_cal = self._get_dataset('calibration')

        # Run calibration set to calculate conformal scores
        cal_outputs = [self._run_deeponet_forward_pass(m, x_cal) for m in
                       tqdm(model_dirs, desc="Running Calibration")]
        cal_outputs = np.array(cal_outputs)

        # Calculate quantile for conformal prediction
        scores = np.abs(y_cal - cal_outputs.mean(axis=0)) / (cal_outputs.std(axis=0) + 1e-8)
        q_hat = np.quantile(scores, self.config.coverage)
        logging.info(f"Conformal quantile q_hat at {self.config.coverage * 100:.1f}% coverage: {q_hat:.4f}")

        # Run test set
        x_test, y_test = self._get_dataset('test')

        test_outputs = [self._run_deeponet_forward_pass(m, x_test) for m in
                        tqdm(model_dirs, desc="Running Test Set")]
        test_outputs = np.array(test_outputs)

        # Evaluate and save results
        evaluate_model(test_outputs, y_test, self.output_dir, verbose=False)

        # data_handler used directly to avoid reshaping for online datasets in _get_dataset
        plot_pred(
            self.data_handler.datasets['test']['X'], self.data_handler.datasets['test']['y'], test_outputs,
            self.output_dir, self.data_handler.datasets['test']['X0_plot'], q_hat=q_hat, online=self.config.online
        )

    def _get_layer_params(self, prefix: str, inputs: Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, None] |
                          Tuple[None, np.ndarray], weights: Dict) -> Tuple:
        """
        Extracts parameters for a specific layer (branch or trunk) from the weights dictionary.

        Args:
            prefix: The prefix for the layer ('branch' or 'trunk').
            inputs: A tuple containing the branch and trunk input arrays.
            weights: A dictionary of weights for the current layer.

        Returns:
            A tuple of parameters needed to run the layer.
        """
        x = inputs[0] if prefix == BRANCH_PREFIX else inputs[1]
        n_in = x.shape[1]
        n_out = weights[f"{prefix}_hidden_bias"].shape[0]
        return (
            x, n_in, n_out,
            weights[f"{prefix}_hidden_bias"],
            weights[f"{prefix}_output_weight"],
            weights[f"{prefix}_output_bias"],
            weights[f"{prefix}_hidden_thetas"]
        )

    def _run_deeponet_forward_pass(self, model_path: Path, inputs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Runs the full forward pass of a single DeepONet model, layer by layer.

        Args:
            model_path: Path to the directory containing the model's weights.
            inputs: A tuple containing the (branch_inputs, trunk_inputs).

        Returns:
            The final prediction array from the model.
        """
        # Find the number of hidden layers from the saved weight files
        pattern = re.compile(r'\.hidden_layers\.(\d+)\..*\.txt$')

        # THERE COULD BE AN ERROR HERE
        layer_nums = [int(pattern.search(f.name).group(1)) for f in model_path.iterdir() if pattern.search(f.name)]
        if not layer_nums:
            raise FileNotFoundError(f"No valid layer weight files found in {model_path}")

        # NUM LAYERS IMPLEMENTATION SEEMS A LITTLE SHAKY
        num_layers = max(layer_nums) + 1

        branch_outputs, trunk_outputs = inputs

        # Run classical layer if specified
        if self.config.classical_branch:
            branch_outputs = self._run_classical_layer(
                inputs=inputs[0],
                all_weights=[load_weights(model_path, layer=i) for i in range(num_layers)],
                is_trunk=False
            )

        if self.config.classical_trunk:
            trunk_outputs = self._run_classical_layer(
                inputs=inputs[1],
                all_weights=[load_weights(model_path, layer=i) for i in range(num_layers)],
                is_trunk=True
            )

        # Run layer by layer
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            is_residual_layer = (i > 0 and self.config.residual)
            weights = load_weights(model_path, layer=i)

            branch_inputs_current_layer = branch_outputs
            if not self.config.classical_branch:
                branch_params = self._get_layer_params(BRANCH_PREFIX, (branch_inputs_current_layer, None), weights)
                branch_outputs = self._run_quantum_layer(
                    *branch_params,
                    last_layer=is_last_layer,
                    is_trunk=False,
                    residual=is_residual_layer
                )

            trunk_inputs_current_layer = trunk_outputs
            if not self.config.classical_trunk:
                trunk_params = self._get_layer_params(TRUNK_PREFIX, (None, trunk_inputs_current_layer), weights)
                trunk_outputs = self._run_quantum_layer(
                    *trunk_params,
                    last_layer=is_last_layer,
                    is_trunk=True,
                    residual=is_residual_layer
                )

        # Early exit if cost analysis
        if self.config.analyze_circuit_cost:
            logging.info("Exiting")
            exit(0)

        # Final dot product and bias
        final_bias = load_weights(model_path, layer=num_layers - 1)["final_bias"]

        # Online dataset is not Cartesian
        subscripts = 'bi,ni->bn' if not self.config.online else 'bi,bi->b'
        final_output =np.einsum(subscripts, branch_outputs, trunk_outputs) + final_bias
        return final_output if not self.config.online else final_output.reshape(-1, 1)

    def _run_quantum_layer(self, inputs: np.ndarray, n_in: int, n_out: int, bias: np.ndarray,
                           weight: np.ndarray, output_bias: np.ndarray, thetas: np.ndarray,
                           last_layer: bool, is_trunk: bool, residual: bool) -> np.ndarray:
        """
        Simulates the execution of one quantum layer (branch or trunk).

        Args:
            inputs: Input data for the layer.
            n_in, n_out: Input and output dimensions.
            bias, weight, output_bias, thetas: Layer parameters.
            last_layer: Flag indicating if this is the final layer.
            is_trunk: Flag indicating if this is the trunk network.
            residual: Flag indicating if there is a skip connection

        Returns:
            The output of the quantum layer.
        """

        # Precompute gates for efficiency
        sqrt_norm = np.sqrt(max(n_in, n_out))
        W_gate = W(n_in, n_out, thetas)
        loader_gate = data_loader(np.full(max(n_in, n_out), 1 / sqrt_norm))
        loader_inv_gate = loader_gate.inverse()

        all_outputs = []
        batch_size = self.config.batch_size or len(inputs)
        is_noisy = self.config.noise > 0.0

        if self.config.analyze_circuit_cost:
            logging.info("Branch: " if not is_trunk else "Trunk: ")
            logging.info(f"n_in: {n_in}, n_out: {n_out}, thetas_shape: {thetas.shape}")
            build_circuit(
                inputs[0], n_in, n_out, W_gate, loader_gate, loader_inv_gate, self.simulator, cost_check=True
            )   # Analyze depth using arbitray input
            # Dummy output
            return np.zeros((len(inputs), n_out))

        desc = "Running Trunk Layer" if is_trunk else "Running Branch Layer"

        for i in tqdm(range(0, len(inputs), batch_size), desc=desc):
            batch_inputs = inputs[i:i + batch_size]

            # Optional: GPU Temperature Management
            # self._wait_for_gpu_cooldown()

            # Build circuits in parallel
            circuits = Parallel(n_jobs=self.config.n_jobs)(
                delayed(build_circuit)(x, n_in, n_out, W_gate, loader_gate, loader_inv_gate, self.simulator,
                                       noisy=is_noisy)
                for x in batch_inputs
            )

            # SET SHOTS TO ACTUAL SHOTS FOR REALISTIC SIMULATION
            results = self.simulator.run(circuits, shots=1, target_gpus=[self.config.target_gpu]).result()

            # Process results sequentially (faster than parallel because of parallelism overhead I assume)
            batch_outputs = [
                self._process_quantum_output(j, results, n_in, n_out, bias, weight, output_bias, last_layer, is_trunk,
                                             batch_inputs[j] if residual else None)
                for j in range(len(batch_inputs))
            ]
            all_outputs.extend(batch_outputs)

        return np.array(all_outputs)

    def _process_quantum_output(self, idx: int, results, n_in: int, n_out: int, hidden_bias: np.ndarray,
                                output_weight: np.ndarray, output_bias: np.ndarray, last_layer: bool,
                                is_trunk: bool, residual_term: Optional[np.ndarray]) -> np.ndarray:
        """
        Processes the raw result from a single circuit run to compute the layer output.

        Args:
            results: The Qiskit Result object from the simulation.
            idx: The index of the circuit within the batch.
            n_in, n_out: Input and output dimensions.
            hidden_bias, output_weight, output_bias: Layer parameters.
            last_layer, is_trunk: Flags for network structure.
            residual_term: The residual term to add, if any.

        Returns:
            The processed output vector for one input.
        """

        # It is ok to get only real values. RBS gates only operate in the real domain.
        # Get Probabilities from Simulation Result
        if self.config.noise > 0.0:
            # For noisy simulations, we get the diagonal of the density matrix
            probabilities = results.data(idx)['density_matrix'].data.diagonal().real
        else:
            # For ideal simulations, we square the statevector amplitudes
            statevector = np.real(results.data(idx)['state'].data)
            probabilities = statevector ** 2

        # Handle Simulation Mode (Shots vs. Ideal)
        if self.config.mode == 'shots':
            # Sample from the probability distribution to simulate measurement shots
            counts = np.random.multinomial(self.config.shots, probabilities)

            if self.config.noise > 0.0:
                # Mitigate errors by zeroing out probabilities of invalid states
                valid_indices = self._get_valid_measurement_indices(n_in, n_out)
                all_indices = np.arange(len(counts))
                invalid_indices = np.setdiff1d(all_indices, valid_indices)
                counts[invalid_indices] = 0

            state_probs = counts / np.sum(counts)
        else:
            # In 'ideal' mode, we use the raw probabilities directly
            state_probs = probabilities

        # This reconstructs the output vector from the probabilities of specific basis states.
        # It relies on the circuit design where the sign of the output is encoded in an ancilla qubit.
        output = []
        for i in range(n_out):
            # Qiskit uses a little-endian convention (qubit 0 is rightmost).
            # We construct the bitstring for the i-th output neuron being 'on'.
            # 'pos_vec' is a unary vector, e.g., '0010' for i=2, n_out=4.
            pos_vec = ['0'] * n_out
            pos_vec[i] = '1'
            padding = ['0'] * (max(n_in, n_out) - n_out)

            # Bitstring for ancilla=0 and ancilla=1
            pos0_str = (''.join(['0'] + padding + pos_vec))[::-1]
            pos1_str = (''.join(['1'] + padding + pos_vec))[::-1]

            # Get probabilities for these two states
            prob_ancilla0 = state_probs[int(pos0_str, 2)]
            prob_ancilla1 = state_probs[int(pos1_str, 2)]

            # Extract amplitude with sign inferred from ancilla
            amplitude = np.sqrt(max(n_in, n_out)) * (prob_ancilla0 - prob_ancilla1)
            output.append(amplitude)

        # Apply Classical Post-Processing
        output = silu(np.array(output) + hidden_bias)

        if residual_term is not None:
            # Add normalized residual connection
            norm_res = residual_term / np.linalg.norm(residual_term)
            output += norm_res

        if last_layer:
            # Apply final linear layer
            output = np.dot(output, output_weight.T) + output_bias
            if is_trunk:
                output = silu(output)

        return output

    def _get_valid_measurement_indices(self, n_in: int, n_out: int) -> List[int]:
        """
        Calculates the integer indices of the valid basis states for error mitigation.
        A valid state has a unary vector in the output register.
        """
        valid_indices = []
        padding = ['0'] * (max(n_in, n_out) - n_out)
        for i in range(n_out):
            pos_vec = ['0'] * n_out
            pos_vec[i] = '1'
            # Ancilla=0 and Ancilla=1 cases
            pos0_str = (''.join(['0'] + padding + pos_vec))[::-1]
            pos1_str = (''.join(['1'] + padding + pos_vec))[::-1]
            valid_indices.extend([int(pos0_str, 2), int(pos1_str, 2)])
        return valid_indices

    def _run_classical_layer(self, inputs: np.ndarray, all_weights: List[Dict], is_trunk: bool) -> np.ndarray:
        """
        Performs a forward pass through a classical network (OrthoNN or ResOrthoNN).
        Note: This function runs the entire multi-layer network, not just one layer.

        Args:
            inputs: Input data for the network.
            all_weights: A list of weight dictionaries, one for each layer.
            is_trunk: Flag indicating if this is the trunk network.

        Returns:
            The output of the classical network.
        """
        prefix = TRUNK_PREFIX if is_trunk else BRANCH_PREFIX

        # Define network architecture from weights
        layer_dims = [inputs.shape[1]]
        for weights in all_weights:
            layer_dims.append(weights[f"{prefix}_hidden_bias"].shape[0])

        # Doesn't matter which layer you get it from, -1 is arbitrary
        layer_dims.append(all_weights[-1][f"{prefix}_output_bias"].shape[0])

        # Instantiate the correct network type
        if self.config.residual:
            net = ResOrthoNN(layer_dims, activation='silu')
        else:
            net = OrthoNN(layer_dims, activation='silu')

        # Load weights into the network
        for i, weights in enumerate(all_weights):
            net.hidden_layers[i].thetas.data = torch.from_numpy(weights[f"{prefix}_hidden_thetas"]).float()
            net.hidden_layers[i].bias.data = torch.from_numpy(weights[f"{prefix}_hidden_bias"]).float()

        # Again, layer index does not matter
        net.output_layer.weight.data = torch.from_numpy(all_weights[-1][f"{prefix}_output_weight"]).float()
        net.output_layer.bias.data = torch.from_numpy(all_weights[-1][f"{prefix}_output_bias"]).float()

        # Perform forward pass
        with torch.no_grad():
            output = net(torch.from_numpy(inputs).float()).cpu().numpy()

        return silu(output) if is_trunk else output

    # -Optional helper for GPU temp management
    def _wait_for_gpu_cooldown(self):
        """
        Monitors GPU temperature and pauses execution if it exceeds MAX_GPU_TEMP.
        This is an optional utility that can be called within processing loops.
        """
        gpu_temp = get_gpu_temp(self.config.target_gpu)
        while gpu_temp is not None and gpu_temp > MAX_GPU_TEMP:
            # The following lock file logic can be used for multi-process coordination
            # lock_file = f"/tmp/gpu{self.config.target_gpu}.lock"
            # with open(lock_file, "w") as f:
            #     f.write(str(os.getpid()))

            logging.warning(
                f"GPU temp {gpu_temp}°C exceeds max {MAX_GPU_TEMP}°C. Pausing for {GPU_CHECK_INTERVAL}s...")
            time.sleep(GPU_CHECK_INTERVAL)
            gpu_temp = get_gpu_temp(self.config.target_gpu)

            # if os.path.exists(lock_file):
            #     os.remove(lock_file)

    ###############################################################################
    ### SPQC stuffs below
    ###############################################################################

    def _load_ensemble_parameters(self, model_dirs: List[Path], layer: int) -> Dict[str, Dict]:
        """Loads and aggregates parameters from all models in an ensemble."""
        all_params = [load_weights(m, layer=layer) for m in model_dirs]

        aggregated = {
            "branch": {
                "hidden_bias": np.array([p["branch_hidden_bias"] for p in all_params]),
                "hidden_thetas": np.array([p["branch_hidden_thetas"] for p in all_params]),
                "output_bias": np.array([p["branch_output_bias"] for p in all_params]),
                "output_weight": np.array([p["branch_output_weight"] for p in all_params]),
            },
            "trunk": {
                "hidden_bias": np.array([p["trunk_hidden_bias"] for p in all_params]),
                "hidden_thetas": np.array([p["trunk_hidden_thetas"] for p in all_params]),
                "output_bias": np.array([p["trunk_output_bias"] for p in all_params]),
                "output_weight": np.array([p["trunk_output_weight"] for p in all_params]),
            },
            "final_bias": np.array([p["final_bias"] for p in all_params])
        }
        return aggregated

    def _run_ensemble_spqc(self):
        """Executes the simulation for an entire ensemble using a single SPQC circuit."""
        ensemble_dir = Path("models", "ensembles", self.config.ensemble)
        model_dirs = [d for d in ensemble_dir.iterdir() if d.is_dir() and d.name != 'simulation_plots']
        logging.info(f"Found {len(model_dirs)} models in ensemble: {self.config.ensemble}")

        # Find the number of hidden layers from the saved file weights
        pattern = re.compile(r'\.hidden_layers\.(\d+)\..*\.txt$')

        # Select the first model to figure out number of layers (arbitrary, but all models must have same shape)
        model_path = Path(model_dirs[0])

        # Collect layer numbers
        layer_nums = [int(pattern.search(f.name).group(1)) for f in model_path.iterdir() if pattern.search(f.name)]
        if not layer_nums:
            raise FileNotFoundError(f"No valid layer weight files found in {model_path}")

        num_layers = max(layer_nums) + 1

        def execute_spqc(inputs: Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, None], num_layers: int):

            # Repeat data
            inputs = (
                np.tile(inputs[0], (len(model_dirs), 1, 1)),
                np.tile(inputs[1], (len(model_dirs), 1, 1))
            )

            branch_outputs_classic, trunk_outputs_classic = [], []

            if self.config.classical_branch:
                for model in model_dirs:
                    branch_outputs_classic.append(
                        self._run_classical_layer(
                            inputs[0][0],   # Classical layers require only 1 copy
                            [load_weights(model, layer=i) for i in range(num_layers)],
                            is_trunk=False)
                    )

            if self.config.classical_trunk:
                for model in model_dirs:
                    trunk_outputs_classic.append(
                        self._run_classical_layer(
                            inputs[1][0],   # Classical layers require only 1 copy
                            [load_weights(model, layer=i) for i in range(num_layers)],
                            is_trunk=True)
                    )

            # Initialize
            branch_outputs, trunk_outputs, params = None, None, None
            for i in range(num_layers):
                is_last_layer = (i == num_layers - 1)

                params = self._load_ensemble_parameters(model_dirs, layer=i)

                # Run branch and trunk layers in SPQC mode
                if not self.config.classical_branch:
                    branch_outputs = self._run_spqc_quantum_layer(
                        inputs=inputs[0],
                        n_in=inputs[0][0].shape[1],
                        # Select an arbitray model's hidden0_bias to calculate size of n_out
                        n_out=params["branch"]["hidden_bias"][0].shape[0],
                        ensemble_params=params["branch"],
                        last_layer=is_last_layer,
                        is_trunk=False
                    )

                    # Output shape from spqc_layer: (num_inputs, num_models, n_out)
                    # We need to rearrange to (num_models, num_inputs, n_out)
                    branch_outputs = np.transpose(branch_outputs, (1, 0, 2))

                if not self.config.classical_trunk:
                    trunk_outputs = self._run_spqc_quantum_layer(
                        inputs=inputs[1],
                        n_in=inputs[1][0].shape[1],
                        # Select an arbitray model's hidden0_bias to calculate size of n_out
                        n_out=params["trunk"]["hidden_bias"][0].shape[0],
                        ensemble_params=params["trunk"],
                        last_layer=is_last_layer,
                        is_trunk=True
                    )

                    # Output shape from spqc_layer: (num_inputs, num_models, n_out)
                    # We need to rearrange to (num_models, num_inputs, n_out)
                    trunk_outputs = np.transpose(trunk_outputs, (1, 0, 2))

                # Set up for next iteration
                inputs = (branch_outputs, trunk_outputs)

            # Early exit for circuit cost
            if self.config.analyze_circuit_cost:
                logging.info("Exiting")
                exit(0)

            if self.config.classical_branch:
                branch_outputs = np.array(branch_outputs_classic)

            if self.config.classical_trunk:
                trunk_outputs = np.array(trunk_outputs_classic)

            final_preds = []
            for i in range(len(model_dirs)):
                pred = np.einsum('bi,ni->bn', branch_outputs[i], trunk_outputs[i]) + params["final_bias"][i]
                final_preds.append(pred)

            return np.array(final_preds)

        x_cal, y_cal = self._get_dataset('calibration')
        cal_outputs = execute_spqc(inputs=x_cal, num_layers=num_layers)

        scores = np.abs(y_cal - cal_outputs.mean(axis=0)) / (cal_outputs.std(axis=0) + 1e-8)
        q_hat = np.quantile(scores, self.config.coverage)
        logging.info(f"Conformal quantile q_hat at {self.config.coverage * 100}% coverage: {q_hat:.4f}")

        # Run test set
        x_test, y_test = self._get_dataset('test')

        test_outputs = execute_spqc(x_test, num_layers=num_layers)
        test_outputs = np.array(test_outputs)

        # Evaluate and save
        evaluate_model(test_outputs, y_test, self.output_dir, verbose=False)

        # Plot and save
        plot_pred(
            self.data_handler.datasets['test']['X'], self.data_handler.datasets['test']['y'], test_outputs,
            self.output_dir, self.data_handler.datasets['test']['X0_plot'], q_hat=q_hat, online=False   # No online
        )

    def _run_spqc_quantum_layer(self, inputs, n_in, n_out, ensemble_params, last_layer, is_trunk):
        """Runs a quantum layer for all models in parallel using SPQC."""
        # Precompute gates that are static across all inputs
        sqrt_norm = np.sqrt(max(n_in, n_out))
        loader_gate = data_loader(np.full(max(n_in, n_out), 1 / sqrt_norm))
        loader_inv_gate = loader_gate.inverse()

        all_outputs = []
        batch_size = self.config.batch_size or len(inputs[0])

        if self.config.analyze_circuit_cost:
            logging.info("---Branch--- " if not is_trunk else "---Trunk--- ")
            logging.info(f"n_in: {n_in}, n_out: {n_out}, thetas_shape: {ensemble_params['hidden_thetas'][0].shape}")
            self._build_spqc_circuit(
                inputs[:, 0], n_in, n_out, ensemble_params['hidden_thetas'], loader_gate, loader_inv_gate, cost_check=True
            )
            # DUMMY OUTPUT
            return np.zeros((inputs.shape[1], len(ensemble_params['hidden_thetas']), n_out))

        desc = "Running trunk layer" if is_trunk else "Running branch layer"

        for i in tqdm(range(0, len(inputs), batch_size), desc=desc):
            batch = inputs[:, i:i + batch_size]
            # Optional: GPU Temperature Management
            # self._wait_for_gpu_cooldown()

            # Circuit construction is now per-input, as it depends on the data_array
            circuits = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._build_spqc_circuit)(
                    batch[:, j], n_in, n_out, ensemble_params['hidden_thetas'], loader_gate, loader_inv_gate
                )
                for j in range(batch.shape[1])
            )

            results = self.simulator.run(circuits, shots=1, target_gpus=[self.config.target_gpu]).result()

            batch_outputs = [
                self._process_spqc_output(j, results, n_in, n_out, ensemble_params, last_layer, is_trunk)
                for j in range(batch.shape[1])
            ]

            all_outputs.extend(batch_outputs)

        return np.array(all_outputs)

    def _process_spqc_output(self, idx, results, n_in, n_out, params, last_layer, is_trunk):
        """Processes the combined statevector from an SPQC circuit run."""
        """
        # For ideal simulations, we square the statevector amplitudes
        statevector = np.real(results.data(idx)['state'].data)
        probabilities = statevector ** 2

        # Now we have address bits
        num_models = len(params['hidden_bias'])
        addr_format_bits = int(np.ceil(np.log2(num_models))) if num_models > 1 else 0

        if self.config.mode == 'shots':
            counts = np.random.multinomial(self.config.shots, probabilities)
            state_probs = counts / self.config.shots
        else:  # ideal mode
            state_probs = probabilities

        all_outputs = np.zeros((num_models, n_out))

        # Bit indexing now includes iterating through model addresses
        for i in range(n_out):
            pos_vec = ['0'] * n_out
            pos_vec[i] = '1'
            padding = ['0'] * (max(n_in, n_out) - n_out)

            # Bitstring for ancilla=0 and ancilla=1
            pos0_str = (''.join(['0'] + padding + pos_vec))[::-1]
            pos1_str = (''.join(['1'] + padding + pos_vec))[::-1]

            for j in range(num_models):
                addr_str = format(j, f'0{addr_format_bits}b')
                # Note: Qiskit's endianness means address qubits might be at the high-order end.
                # Assuming statevector format is |tomo⟩|anc⟩|addr⟩
                idx0 = int(pos0_str + addr_str, 2)
                idx1 = int(pos1_str + addr_str, 2)

                # Get probabilities for these two states
                prob_ancilla0 = state_probs[idx0]
                prob_ancilla1 = state_probs[idx1]
                all_outputs[j, i] = np.sqrt(max(n_in, n_out)) * (prob_ancilla0 - prob_ancilla1)

        # Apply classical post-processing layers for each model
        ret_val = []
        for i in range(num_models):
            # The SPQC output is an expectation value; scaling by num_models approximates the sum
            # that would have occurred from the address qubit superposition.
            output_i = all_outputs[i] * num_models
            output_i = silu(output_i + params['hidden_bias'][i])

            # Different variable called ret_val to account for shape mismatches
            if last_layer:
                output_i = np.dot(output_i, params['output_weight'][i].T) + params['output_bias'][i]
                ret_val.append(silu(output_i) if is_trunk else output_i)
            else:
                ret_val.append(output_i)

        return np.array(ret_val)
        """
        # It is ok to get only real values. RBS gates only operate in the real domain.
        # Get Probabilities from Simulation Result
        if self.config.noise > 0.0:
            # For noisy simulations, we get the diagonal of the density matrix
            probabilities = results.data(idx)['density_matrix'].data.diagonal().real
        else:
            # For ideal simulations, we square the statevector amplitudes
            statevector = np.real(results.data(idx)['state'].data)
            probabilities = statevector ** 2

        # Now we have address bits
        num_models = len(params['hidden_bias'])
        addr_format_bits = int(np.ceil(np.log2(num_models))) if num_models > 1 else 0

        # Handle Simulation Mode (Shots vs. Ideal)
        if self.config.mode == 'shots':
            # Sample from the probability distribution to simulate measurement shots
            counts = np.random.multinomial(self.config.shots, probabilities)

            if self.config.noise > 0.0:
                # Mitigate errors by zeroing out probabilities of invalid states
                valid_indices = self._get_valid_measurement_indices_spqc(n_in, n_out, num_models, addr_format_bits)
                all_indices = np.arange(len(counts))
                invalid_indices = np.setdiff1d(all_indices, valid_indices)
                counts[invalid_indices] = 0

            state_probs = counts / np.sum(counts)
        else:
            # In 'ideal' mode, we use the raw probabilities directly
            state_probs = probabilities

        all_outputs = np.zeros((num_models, n_out))

        # Bit indexing now includes iterating through model addresses
        for i in range(n_out):
            pos_vec = ['0'] * n_out
            pos_vec[i] = '1'
            padding = ['0'] * (max(n_in, n_out) - n_out)

            # Bitstring for ancilla=0 and ancilla=1
            pos0_str = (''.join(['0'] + padding + pos_vec))[::-1]
            pos1_str = (''.join(['1'] + padding + pos_vec))[::-1]

            for j in range(num_models):
                addr_str = format(j, f'0{addr_format_bits}b')
                # Note: Qiskit's endianness means address qubits might be at the high-order end.
                # Assuming statevector format is |tomo⟩|anc⟩|addr⟩
                idx0 = int(pos0_str + addr_str, 2)
                idx1 = int(pos1_str + addr_str, 2)

                # Get probabilities for these two states
                prob_ancilla0 = state_probs[idx0]
                prob_ancilla1 = state_probs[idx1]
                all_outputs[j, i] = np.sqrt(max(n_in, n_out)) * (prob_ancilla0 - prob_ancilla1)

        # Apply classical post-processing layers for each model
        ret_val = []
        for i in range(num_models):
            # The SPQC output is an expectation value; scaling by num_models approximates the sum
            # that would have occurred from the address qubit superposition.
            output_i = all_outputs[i] * num_models
            output_i = silu(output_i + params['hidden_bias'][i])

            # Different variable called ret_val to account for shape mismatches
            if last_layer:
                output_i = np.dot(output_i, params['output_weight'][i].T) + params['output_bias'][i]
                ret_val.append(silu(output_i) if is_trunk else output_i)
            else:
                ret_val.append(output_i)

        return np.array(ret_val)

    def _get_valid_measurement_indices_spqc(self, n_in: int, n_out: int, num_models: int, addr_format_bits: int) -> List[int]:
        """
        Calculates the integer indices of the valid basis states for error mitigation.
        A valid state has a unary vector in the output register.
        """
        valid_indices = []
        padding = ['0'] * (max(n_in, n_out) - n_out)
        for i in range(n_out):
            pos_vec = ['0'] * n_out
            pos_vec[i] = '1'
            # Ancilla=0 and Ancilla=1 cases
            pos0_str = (''.join(['0'] + padding + pos_vec))[::-1]
            pos1_str = (''.join(['1'] + padding + pos_vec))[::-1]

            for j in range(num_models):
                addr_str = format(j, f'0{addr_format_bits}b')
                valid_indices.extend([int(pos0_str + addr_str, 2), int(pos1_str + addr_str, 2)])

        return valid_indices

    def _build_spqc_circuit(self, x_input, n_in, n_out, thetas, loader_gate, loader_inv_gate, cost_check=False):
        """Builds one SPQC circuit for a single input vector."""
        x_input_stable = x_input.copy()
        x_input_stable[np.abs(x_input_stable) < 1e-8] += 1e-7

        circ = create_spqc_circuit(
            n_in, n_out, thetas, x_input_stable, loader_inv_gate, loader_gate
        )

        # Optional: Analyze circuit cost against a realistic backend
        if cost_check:
            # Heron: 'cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x'
            # Eagle: 'ecr', 'r_z', 'sx', 'x', 'i'
            t_qc = transpile(circ, optimization_level=2, basis_gates=['ecr', 'r_z', 'sx', 'x', 'i'])
            logging.info(f"Depth: {t_qc.depth()}, Gates: {t_qc.count_ops()}\n")
            # logging.info("Exiting.")
            # exit(1)

        if self.config.noise > 0.0:
            circ.save_density_matrix()
        else:
            circ.save_statevector('state')

        return transpile(circ, self.simulator, optimization_level=1)

# --- Main Entry Point ---

def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description="Quantum DeepONet Simulation")
    parser.add_argument('--config', type=str, default="default_antiderivative", help="Config file name in configs/simulation")
    parser.add_argument("--override", nargs='*', help="Overrides in key=value format (e.g., n_jobs=8 seed=42)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config_path = Path("configs/simulation") / (args.config + ".yaml")
    config = load_config(str(config_path))

    if args.override:
        apply_overrides(config, args.override)

    set_seeds(config.seed)

    runner = SimulationRunner(config)
    runner.run()

if __name__ == "__main__":
    main()


# USEFUL CODE

# PROCESSING FOR REALISTIC SIMULATION
"""
if self.config.mode == 'shots':

    # Get measurement counts directly (from repeated noisy shots)
    counts_dict = results.get_counts(idx)
    counts = np.zeros(2 ** (max(n_in, n_out) + 1))  # Total number of basis states

    # Convert bitstrings to integer indices
    for bitstr, count in counts_dict.items():
        index = int(bitstr.replace(' ', ''), 2)
        counts[index] = count

    if self.config.noise > 0.0:

        # Error mitigation
        valid_indices = []
        for i in range(n_out):
            pos_vec = ['0'] * n_out
            pos_vec[i] = '1'
            pos0_str = (''.join(['0'] + ['0'] * (n_in - n_out) + pos_vec))[::-1]
            pos1_str = (''.join(['1'] + ['0'] * (n_in - n_out) + pos_vec))[::-1]
            valid_indices.extend([int(pos0_str, 2), int(pos1_str, 2)])

        invalid_indices = np.setdiff1d(np.arange(len(counts)), valid_indices)
        counts[invalid_indices] = 0

        state_probs = counts / np.sum(counts)

    else:
        # Ideal + shots: sample from ideal probabilities
        state_probs = counts / self.config.shots

else:  # ideal + analytic
    if self.config.noise > 0.0:
        probabilities = results.data(idx)['density_matrix'].data.diagonal().real
    else:
        statevector = np.real(results.data(idx)['state'].data)
        probabilities = statevector ** 2

    state_probs = probabilities
"""
