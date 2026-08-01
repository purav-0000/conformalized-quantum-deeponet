# Fleet execution

Fleet runs use immutable commits from the `codex/accelerate-fleet` branch and
the micromamba environment in `environment.yaml`. The environment is named
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
