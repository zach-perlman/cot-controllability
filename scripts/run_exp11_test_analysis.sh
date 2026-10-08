#!/bin/bash
# exp11 test stage 2 (test addendum, commit 85b1226; reading-rule code committed before this runs): Q1, Q2 and Q3 on
# the captured test banks, each analysis once.
set -euo pipefail
cd "$(dirname "$0")"
PY=/venv/main/bin/python
PROBE=../cache/exp11/q23_probe_test_k64_wd0.001.pt
echo "=== Q1"
$PY exp11_explore.py --bank test --no-lens
$PY exp11_report.py q1_single --bank single_test
echo "=== Q2 three ingredients"
$PY -u exp11_q23.py probe --bank test --k-pca 64 --wd 0.001 --positions final stir2 question_end
for pos in final stir2 question_end; do $PY exp11_report.py q2_three --probe-file $PROBE --position $pos; done
$PY exp11_report.py q2_three --probe-file $PROBE --confident
$PY exp11_explore.py --bank test --paired
echo "=== Q2 single table"
$PY exp11_report.py q2_single --bank single_test --lens j-lens
$PY exp11_report.py q2_single --bank single_test --lens r-lens
echo "=== Q3"
$PY -u exp11_q23.py backpatch --bank test
$PY -u exp11_q23.py twinpatch --bank test --site final --format three_first
$PY -u exp11_q23.py modepatch --bank test --format three_first --h 2
$PY -u exp11_q23.py twinpatch --bank single_test --site count --format single_first
$PY -u exp11_q23.py twinpatch --bank single_test --site final --format single_first
$PY -u exp11_q23.py twinpatch --bank single_test --site count --format single_first --reverse
$PY -u exp11_q23.py twinpatch --bank single_test --site final --format single_first --reverse
$PY -u exp11_q23.py modepatch --bank single_test --format single_first --h 3
$PY exp11_report.py q3 --bank test
$PY exp11_report.py q3 --bank single_test
echo DONE
