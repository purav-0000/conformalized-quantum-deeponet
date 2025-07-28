import subprocess
import time
import os

# Only 1 GPU now
MAX_PROCESSES_PER_GPU = 1
GPU_ID = 0

# Command prefix
BASE_CMD = [
    "python", "-m", "src.data_driven.antiderivative.simulation",
    "--config", "ensemble_run", "--override"
]

# Job list for single GPU
jobs = [
    # (Your jobs from jobs_gpu_0 or _1 go here)
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=1000000 noise=0.0008",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000000 noise=0.0002",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000000 noise=0.0006",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=10000000 noise=0.0008",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000000 noise=0.0002",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000000 noise=0.0004",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000000 noise=0.0006",
    "ensemble=verification_4 simulator=GPU mode=shots batch_size=5 target_gpu=0 shots=100000000 noise=0.0008",
]

# Process tracking
processes = []
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

while jobs or processes:
    # Clean up finished processes
    processes = [p for p in processes if p.poll() is None]

    # Launch new job if GPU is free
    if len(processes) < MAX_PROCESSES_PER_GPU and jobs:
        job_str = jobs.pop(0)
        cmd = BASE_CMD + job_str.split()
        print(f"Launching on GPU {GPU_ID}: {' '.join(cmd)}")

        log_file = os.path.join(log_dir, f"gpu{GPU_ID}_job_{job_str.replace(' ', '_')}.out")
        f = open(log_file, "w")

        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        processes.append(p)

    time.sleep(5)
