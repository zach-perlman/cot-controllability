#!/bin/bash
# exp11 test stage 1 (test addendum, commit 85b1226): generate the derived and single-table test banks, write their
# requests, and capture all four test banks once.
set -euo pipefail
cd "$(dirname "$0")"
PY=/venv/main/bin/python
$PY exp11.py twins --bank test_twins
$PY exp11.py fmt_bank --bank single_test
$PY exp11.py twins --bank single_test_twins
for b in test_twins single_test single_test_twins; do $PY exp11.py requests --bank $b; done
for b in test test_twins single_test single_test_twins; do $PY -u exp11.py capture --bank $b; done
echo DONE
