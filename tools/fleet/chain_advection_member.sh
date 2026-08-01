#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 WATCH_PID GPU COMPLETED_MEMBER NEXT_MEMBER WORKTREE" >&2
  exit 2
fi

watch_pid="$1"
gpu="$2"
completed_member="$3"
next_member="$4"
worktree="$5"

while kill -0 "${watch_pid}" 2>/dev/null; do
  sleep 30
done

completion="${worktree}/models/ensembles/advection_trajectory_v2/model_${completed_member}/timing.json"
if [[ ! -f "${completion}" ]]; then
  echo "watched member ${completed_member} did not complete successfully" >&2
  exit 1
fi

cd "${worktree}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export LD_LIBRARY_PATH="/home/pmatlia/qiskit-fleet/micromamba/envs/qiskit/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DDE_BACKEND=pytorch
export PYTHONPATH=.

exec /home/pmatlia/qiskit-fleet/micromamba/envs/qiskit/bin/python \
  -m src.runners.training \
  --config default_advection \
  --override \
  ensemble_size=8 \
  ensemble_name=advection_trajectory_v2 \
  "ensemble_member=${next_member}" \
  seed=20260801 \
  orthogonal_implementation=optimized \
  verbose=false
