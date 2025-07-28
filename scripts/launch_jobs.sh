#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qiskit-deeponet

# Navigate to project root
cd "$(dirname "$0")/.."  # Go one level up from scripts directory

# Launch job launcher in background using nohup
nohup python scripts/job_launcher.py > job_launcher.log 2>&1 &
