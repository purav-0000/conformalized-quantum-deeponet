#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 ]]; then
  echo "usage: $0 WATCH_PID GPU CONFIG ENSEMBLE SIZE COMPLETED_MEMBER NEXT_MEMBER SEED WORKTREE [OVERRIDE ...]" >&2
  exit 2
fi

watch_pid="$1"
gpu="$2"
config="$3"
ensemble="$4"
ensemble_size="$5"
completed_member="$6"
next_member="$7"
seed="$8"
worktree="$9"
shift 9

while kill -0 "${watch_pid}" 2>/dev/null; do
  sleep 30
done

completion="${worktree}/models/ensembles/${ensemble}/model_${completed_member}/timing.json"
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
  --config "${config}" \
  --override \
  "ensemble_size=${ensemble_size}" \
  "ensemble_name=${ensemble}" \
  "ensemble_member=${next_member}" \
  "seed=${seed}" \
  orthogonal_implementation=optimized \
  verbose=false \
  "$@"
