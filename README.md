> [!WARNING]
> 🚧 **This repository is currently under construction.** 🚧

# Conformalized Quantum DeepONet Ensembles

This repository contains the implementation of the framework proposed in the paper: **"Conformalized Quantum DeepONet Ensembles for Scalable Operator Learning with Distribution-Free Uncertainty"**.

## Abstract

Operator learning enables fast surrogate modeling of high-dimensional dynamical systems, but existing approaches face two fundamental limitations: quadratic inference complexity and unreliable uncertainty quantification in safety-critical settings. We propose **Conformalized Quantum DeepONet Ensembles**, a framework that addresses both challenges simultaneously. 

By leveraging Quantum Orthogonal Neural Networks (QOrthoNNs), we reduce operator inference complexity from $\mathcal{O}(n^2)$ to $\tilde{\mathcal{O}}(n)$, enabling scalable evaluation over fine discretizations. To provide rigorous uncertainty quantification, we combine ensemble-based epistemic modeling with adaptive conformal prediction, yielding distribution-free coverage guarantees. 

A key challenge in ensembling is that naïve parallelism scales hardware resources linearly with the number of models. We resolve this by using Superposed Parameterized Quantum Circuits (SPQCs), which compress multiple ensemble members into a single circuit and enable simultaneous multi-model execution. 

Experiments on synthetic partial differential equations and real-world power system dynamics demonstrate that our approach achieves accurate predictions while maintaining calibrated uncertainty under realistic quantum noise.

## Project Structure

- `configs/`: Configuration files for running experiments.
- `data/`: Datasets used for training and evaluation.
- `logs/`: Experiment logs and outputs.
- `src/`: Source code including network architectures, conformal prediction logic, and quantum circuit generation.
- `requirements.txt`: Python dependencies required to run the project.

## Requirements

For CPU development, install the tested portable dependencies:

```bash
python -m pip install -r requirements.txt
```

Fleet experiments use the reproducible micromamba specification in
`tools/fleet/environment.yaml`. It creates an environment named `qiskit` with
Python 3.12, PyTorch 2.11.0 + CUDA 12.8, Qiskit 2.5.0, Qiskit Aer 0.17.2, and
DeepXDE 1.15.0. See `tools/fleet/README.md` for GPU placement and immutable-run
instructions.

Run the regression suite with:

```bash
python -m pytest -q
```

## Authors

- **Purav Matlia** - Purdue University
- **Christian Moya** - Purdue University
- **Guang Lin** - Purdue University
