#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 ]]; then
  echo "usage: $0 WATCH_PID GPU COMPLETION_FILE CONFIG ENSEMBLE SIZE NEXT_MEMBER SEED WORKTREE [OVERRIDE ...]" >&2
  exit 2
fi

watch_pid="$1"
gpu="$2"
completion_file="$3"
config="$4"
ensemble="$5"
ensemble_size="$6"
next_member="$7"
seed="$8"
worktree="$9"
shift 9

while kill -0 "${watch_pid}" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "${completion_file}" ]]; then
  echo "watched process ended without completion marker: ${completion_file}" >&2
  exit 1
fi

cd "${worktree}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export LD_LIBRARY_PATH="/home/pmatlia/qiskit-fleet/micromamba/envs/qiskit/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DDE_BACKEND=pytorch
export PYTHONPATH=.

exec /home/pmatlia/qiskit-fleet/micromamba/envs/qiskit/bin/python \
  -m src.runners.training \
  --config "${config}" \
  --override \
  "ensemble_size=${ensemble_size}" \
  "ensemble_name=${ensemble}" \
  "ensemble_member=${next_member}" \
  "seed=${seed}" \
  orthogonal_implementation=optimized \
  verbose=false \
  "$@"
