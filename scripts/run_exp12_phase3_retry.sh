#!/bin/bash
# Runs scripts/run_exp12_phase3.sh up to 3 times; each rerun skips finished stages and resumes fits
# from their checkpoints, so only the failed groups do work again.
cd "$(dirname "$0")/.."
for attempt in 1 2 3; do
  echo "=== $(date -u +%FT%TZ) phase 3 attempt $attempt"
  bash scripts/run_exp12_phase3.sh && { echo "=== $(date -u +%FT%TZ) phase 3 finished"; exit 0; }
  sleep 120
done
echo "=== $(date -u +%FT%TZ) phase 3 gave up after 3 attempts"
exit 1
