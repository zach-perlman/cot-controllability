#!/bin/bash
# Runs scripts/run_exp12_phase4.sh up to 3 times; each rerun skips finished stages (the Aya fit has no
# checkpoint, so an interrupted fit starts over).
cd "$(dirname "$0")/.."
for attempt in 1 2 3; do
  echo "=== $(date -u +%FT%TZ) phase 4 attempt $attempt"
  bash scripts/run_exp12_phase4.sh && { echo "=== $(date -u +%FT%TZ) phase 4 finished"; exit 0; }
  sleep 120
done
echo "=== $(date -u +%FT%TZ) phase 4 gave up after 3 attempts"
exit 1
