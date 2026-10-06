#!/bin/bash
# exp10 overnight chain: wait for brew's test runner to exit, run brew's test analysis (primary reading, then the
# mapped and strict sensitivity readings), then the chain and mhn dev pilot (run_exp10_tasks_pilot.sh). Stops there:
# chain and mhn test generation waits for their manifest.
#   setsid nohup scripts/run_exp10_overnight.sh <test runner pid> > log/exp10/overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
runner_pid=$1
while kill -0 "$runner_pid" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) brew test runner exited"
if ! grep -aq "exp10 test done" ../log/exp10/test.log; then
  echo "brew test did not finish (no 'exp10 test done' in log/exp10/test.log); stopping"
  exit 1
fi
for reading in hidden_correct hidden_correct_mapped hidden_correct_strict; do
  /venv/main/bin/python cc_exp10_analysis.py --stage test --reading "$reading" \
    || echo "$(date -u +%FT%TZ) analysis ($reading) failed; continuing"
done
echo "$(date -u +%FT%TZ) brew analysis done; starting chain and mhn dev pilot"
./run_exp10_tasks_pilot.sh > ../log/exp10/tasks_pilot.log 2>&1 \
  && echo "$(date -u +%FT%TZ) tasks pilot done" || echo "$(date -u +%FT%TZ) tasks pilot failed; see log/exp10/tasks_pilot.log"
