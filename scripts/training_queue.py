import subprocess
import time
import os

# Max parallel jobs
MAX_PROCESSES = 6

# Command prefix
BASE_CMD = [
    "python", "-m", "src.data_driven.antiderivative.training",
    "--config", "ensemble_run", "--override"
]

# Job queue
jobs = [
    "ensemble_name=size16_3 ensemble=16 iterations=30000 lr=0.001",
    "ensemble_name=size16_4 ensemble=16 iterations=30000 lr=0.001",
    "ensemble_name=size16_5 ensemble=16 iterations=30000 lr=0.001",
    "ensemble_name=size20_1 ensemble=20 iterations=30000 lr=0.001",
    "ensemble_name=size20_2 ensemble=20 iterations=30000 lr=0.001",
    "ensemble_name=size20_3 ensemble=20 iterations=30000 lr=0.001",
    "ensemble_name=size20_4 ensemble=20 iterations=30000 lr=0.001",
    "ensemble_name=size20_5 ensemble=20 iterations=30000 lr=0.001",
]

# Track running processes
processes = []

while jobs or processes:
    # Clean up finished processes
    processes = [p for p in processes if p.poll() is None]

    # Launch new jobs if under limit
    while len(processes) < MAX_PROCESSES and jobs:
        job_str = jobs.pop(0)
        cmd = BASE_CMD + job_str.split()
        print(f"Launching: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        processes.append(p)

    time.sleep(5)  # avoid busy waiting
