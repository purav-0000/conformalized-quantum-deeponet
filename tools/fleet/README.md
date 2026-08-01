# Fleet execution

Fleet runs use immutable commits from the `codex/accelerate-fleet` branch and
the micromamba environment in `tools/fleet/environment.yaml`. The environment is named
`qiskit` and is installed beneath `/home/pmatlia/qiskit-fleet/micromamba` on
each host, independently of other projects.

The current GPU allowlist deliberately excludes `apex:0` and `vector:0`, which
had foreign processes during the August 2026 preflight. Recheck occupancy with
`nvidia-smi` before every submission.

The equivalence/performance benchmark is launched from a detached checkout:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /home/pmatlia/qiskit-fleet/micromamba/envs/qiskit/bin/python \
  scripts/run_performance_benchmark.py \
  --config configs/benchmark/fleet_performance.yaml \
  --output results/performance/benchmark_results.json
```

Results, logs, the exact Git commit, package versions, and GPU metadata belong
under `/home/pmatlia/qiskit-fleet/runs/<run-id>/`. Do not run experiments from
a mutable shared checkout.

## Fleet scheduling

Training one ensemble member per process makes members independently retryable
and allows them to be distributed across hosts:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  /home/pmatlia/qiskit-fleet/micromamba/envs/qiskit/bin/python \
  -m src.runners.training \
  --config default_advection \
  --override ensemble_size=8 ensemble_name=advection_trajectory_v2 \
  ensemble_member=0 seed=20260801 orthogonal_implementation=optimized \
  verbose=false
```

The orthogonal models are kernel-launch-bound and use little GPU memory. On the
RTX 4090/3090 fleet, one process used roughly 17--48% SM utilization and about
610 MiB. Two independent processes on the same GPU used roughly 84--97% SM and
under 1.3 GiB. Check this again for every architecture with `nvidia-smi pmon`;
do not assume that two-process placement is safe for larger models.

`launch_training_after.sh` waits for an existing process and a concrete
completion marker before starting a member from another experiment.
`chain_training_member.sh` provides the shorter same-ensemble form. Both retain
the process ID across the handoff so another follow-up can watch the chain.

From the repository root, `python monitor.py` prints the current running,
queued, stopped, failed, and completed jobs. Use `python monitor.py --watch 30`
for a refreshing display or `python monitor.py --json` for machine-readable
status. The monitor is read-only and is not required for queued chains to run.
