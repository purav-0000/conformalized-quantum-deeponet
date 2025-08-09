import subprocess
import time
import os
import logging

from datetime import datetime
# Max 1 process per GPU
MAX_PROCESSES_PER_GPU = 1

# Command prefix
BASE_CMD = [
    "python", "-m", "src.data_driven.antiderivative.simulation",
    "--config", "ensemble_run", "--override"
]

# Two job queues: one for each GPU
jobs_gpu_0 = [
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000 noise=0.0002",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000 noise=0.0002",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000 noise=0.0008",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000 noise=0.0008",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000 noise=0.0002",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000 noise=0.0002",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000 noise=0.0008",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000 noise=0.0008",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000 noise=0.0002",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000 noise=0.0002",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000 noise=0.0008",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000 noise=0.0008",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000000 noise=0.0002",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000000 noise=0.0002",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000000 noise=0.0008",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000000 noise=0.0008",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000000 noise=0.0002",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000000 noise=0.0002",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000000 noise=0.0008",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000000 noise=0.0008",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000000 noise=0.0002",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000000 noise=0.0002",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000000 noise=0.0008",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000000 noise=0.0008",
]

jobs_gpu_1 = [
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=1000 noise=0.0004",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=1000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=1000 noise=0.0006",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=1000 noise=0.0006",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=10000 noise=0.0004",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=10000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=10000 noise=0.0006",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=10000 noise=0.0006",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=100000 noise=0.0004",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=100000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=100000 noise=0.0006",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=100000 noise=0.0006",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=1000000 noise=0.0004",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=1000000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=1000000 noise=0.0006",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=1000000 noise=0.0006",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=10000000 noise=0.0004",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=10000000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=10000000 noise=0.0006",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=10000000 noise=0.0006",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=100000000 noise=0.0004",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=100000000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=100000000 noise=0.0006",
    "ensemble=verification_8 simulator=GPU mode=shots batch_size=5 target_gpu=1 shots=100000000 noise=0.0006",
]

# Process trackers for each GPU
processes = {0: [], 1: []}
job_queues = {0: jobs_gpu_0, 1: jobs_gpu_1}

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

while job_queues[0] or job_queues[1] or processes[0] or processes[1]:
    for gpu_id in [0, 1]:
        # Clean up finished processes
        processes[gpu_id] = [p for p in processes[gpu_id] if p.poll() is None]

        lock_file = f"/tmp/gpu{gpu_id}.lock"

        # Launch next job if available
        if len(processes[gpu_id]) < MAX_PROCESSES_PER_GPU and job_queues[gpu_id] and not os.path.exists(lock_file):
            job_str = job_queues[gpu_id].pop(0)
            cmd = BASE_CMD + job_str.split()

            # Log file
            safe_job_name = job_str.replace(' ', '_').replace('/', '_')
            log_file_path = os.path.join(log_dir, f"gpu{gpu_id}_job_{safe_job_name}.out")
            log_f = open(log_file_path, "w")

            # Launch job
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            processes[gpu_id].append(p)

    time.sleep(5)  # avoid busy loop

