#!/bin/bash
# exp11 dev patching pilots (exploratory; they fix the test addendum's back-patching pair and twin-patch layers).
set -euo pipefail
cd "$(dirname "$0")"
PY=/venv/main/bin/python
$PY -u exp11_q23.py backpatch --bank dev
$PY -u exp11_q23.py backpatch --bank fmt_dev
$PY -u exp11_q23.py twinpatch --bank dev --site final --format three_first
$PY -u exp11_q23.py twinpatch --bank fmt_dev --site count --format single_first
$PY -u exp11_q23.py twinpatch --bank fmt_dev --site final --format single_first
$PY -u exp11_q23.py twinpatch --bank fmt_dev --site count --format single_first --reverse
$PY -u exp11_q23.py twinpatch --bank fmt_dev --site final --format single_first --reverse
echo DONE
