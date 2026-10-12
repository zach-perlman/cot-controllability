#!/bin/bash
# exp15 secondary GPU jobs (exploratory). Usage: run_exp15_secondary.sh <pid of the main run_exp15.sh queue>
# Waits for that queue to exit (the test text reads need the GPU alone), then:
#   two lanes of edit trials (~56 GB each, run side by side):
#     ar_edit dev -> ar_edit test        count_repair test -> read_then_write test
#   then, one at a time: selfreport dev, selfreport test (with the thinking-on baseline),
#   nla_iter300 dev and best_of_16 test (NLA generation at batch 384: alone on the GPU)
# A failed step is logged and the queue moves on; logs go to log/exp15/<step>.log.
set -uo pipefail
cd "$(dirname "$0")"
PY=/venv/main/bin/python
LOG=../log/exp15

sec() {             # sec <log name> <exp15_secondary.py args...>
  local name=$1; shift
  echo "[$(date -u +%H:%M:%S)] $name"
  $PY exp15_secondary.py "$@" >> "$LOG/$name.log" 2>&1 || echo "[$(date -u +%H:%M:%S)] FAILED $name"
}

echo "[$(date -u +%H:%M:%S)] waiting for pid $1"
tail --pid="$1" -f /dev/null
( sec ar_edit_dev ar_edit --bank single_test; sec ar_edit_test ar_edit --bank test ) &
( sec count_repair_test count_repair --bank test; sec read_then_write_test read_then_write --bank test ) &
wait
sec selfreport_dev selfreport --bank single_test
sec selfreport_test selfreport --bank test
sec nla_iter300_dev nla_iter300 --bank single_test
sec best_of_16_test best_of_16 --bank test
echo "[$(date -u +%H:%M:%S)] secondary done"
