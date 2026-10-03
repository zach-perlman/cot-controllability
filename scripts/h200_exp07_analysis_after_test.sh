#!/bin/bash
# Waits for scripts/h200_exp07_test.sh to log "done", then runs the exp07 test analysis (models the test driver
# skipped after repeated failures are reported as missing). Usage: scripts/h200_exp07_analysis_after_test.sh RUN
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?analysis run name}
TEST_LOG=log/exp07/h200_exp07_test.log
until grep -qE "Z done$" "$TEST_LOG"; do sleep 60; done
echo "$(date -u +%FT%TZ) test done; analysing as $RUN"
/venv/main/bin/python scripts/cc_exp07_analysis.py --run "$RUN" --skip-missing
echo "$(date -u +%FT%TZ) analysis done"
