#!/bin/bash
# exp11 J++ re-read (manifest jpp_addendum; addendum and code committed before this runs): the loader gate, then the
# three-column and single-table readouts, each once. The gate exits non-zero on failure, which stops the script.
set -euo pipefail
cd "$(dirname "$0")"
PY=/venv/main/bin/python
OUT=../results/exp11_wrong_intermediates/analysis/jpp_v1
LOG=../log/exp11/jpp.log
mkdir "$OUT"
for stage in gate three single; do
  echo "=== $stage" >> "$LOG"
  $PY -u exp11_jpp.py "$stage" 2>>"$LOG" | tee "$OUT/$stage.txt"
done
echo DONE >> "$LOG"
