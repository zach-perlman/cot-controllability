#!/bin/bash
# exp17 queue up to check-in 2 (results/exp17_jspace_keys/manifest_draft.json, "run_plan"): per model, alone on the
# GPU, Qwen then Gemma, in one process each (the model loads once):
#   baseline:dev check prep posctl dev select
# A failed check stops that model before any trial (exp17.py raises). The test stages run only after the lock.
# Usage: run_exp17.sh "<qwen stages>" "<gemma stages>" (default: all stages for both).
set -uo pipefail
cd "$(dirname "$0")"
PY=/venv/main/bin/python
LOG=../log/exp17
ALL="baseline:dev check prep posctl dev select"
declare -A STAGES=([qwen]="${1:-$ALL}" [gemma]="${2:-$ALL}")
mkdir -p "$LOG"
for model in qwen gemma; do
  echo "[$(date -u +%H:%M:%S)] start $model: ${STAGES[$model]}"
  # shellcheck disable=SC2086
  if EXP17_MODEL=$model $PY exp17.py ${STAGES[$model]} >> "$LOG/$model.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] done  $model"
  else
    echo "[$(date -u +%H:%M:%S)] FAILED $model (log/exp17/$model.log)"
  fi
done
echo "[$(date -u +%H:%M:%S)] queue finished"
