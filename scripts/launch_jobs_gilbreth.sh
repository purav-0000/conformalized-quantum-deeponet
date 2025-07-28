#!/bin/bash
#SBATCH -A standby
#SBATCH --job-name=job-launcher
#SBATCH --output=slurm_job_launcher.out
#SBATCH --error=slurm_job_launcher.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00           # Set as needed
#SBATCH --mem=10G                 # Set as needed

# Load conda environment
module load conda
conda activate qiskit-deeponet

# Go to project root
cd $RCAC_SCRATCH/quantum-deeponet-SURF/

# Make sure logs/ exists
mkdir -p logs

# Run the Python job launcher directly (no nohup needed)
python scripts/job_launcher_gilbreth.py