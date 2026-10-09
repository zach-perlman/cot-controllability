#!/bin/bash
# Runs scripts/run_exp12_phase2.sh up to 3 times; each rerun skips finished stages and resumes the
# long fit from its checkpoint, so a transient failure (network, a crash mid-fit) costs one stage.
cd "$(dirname "$0")/.."
for attempt in 1 2 3; do
  echo "=== $(date -u +%FT%TZ) phase 2 attempt $attempt"
  bash scripts/run_exp12_phase2.sh && { echo "=== $(date -u +%FT%TZ) phase 2 finished"; exit 0; }
  sleep 120
done
echo "=== $(date -u +%FT%TZ) phase 2 gave up after 3 attempts"
exit 1
