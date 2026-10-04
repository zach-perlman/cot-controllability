#!/usr/bin/env bash
# Upload exp09's partial generations, grades and logs to the private HF dataset every INTERVAL seconds, so an
# instance shutdown loses at most one interval (the generation engine resumes from finished rows).
# Usage: setsid nohup scripts/backup_loop.sh [interval_seconds] > log/exp09/backup_loop.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ID=zachperlman20/cot-controllability-cache
INTERVAL="${1:-1200}"
HF=/venv/main/bin/hf

while true; do
  private=$(/venv/main/bin/python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('${REPO_ID}').private)")
  if [[ "$private" != "True" ]]; then
    echo "$(date -u +%FT%TZ) dataset is not private; not uploading"
  else
    for d in cache/exp09 results/exp09_final_test log/exp09; do
      $HF upload "$REPO_ID" "$d" "$d" --repo-type dataset --commit-message "exp09 B200 backup $(date -u +%FT%TZ)" \
        > /dev/null 2>&1 && echo "$(date -u +%FT%TZ) uploaded $d" || echo "$(date -u +%FT%TZ) FAILED $d"
    done
  fi
  sleep "$INTERVAL"
done
