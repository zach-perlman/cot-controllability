#!/bin/bash
# exp14 (results/exp14_brew_swaps/manifest.json): capture, Gate 0 readouts, Gate 1 / Stage 2 swaps on h=2 items,
# Stage 3 repair swaps on h=3 errors, then the report. Every stage skips when its output exists.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/.hf_home} HF_HUB_OFFLINE=1
PY=/venv/main/bin/python
C=../cache/exp14
R=../results/exp14_brew_swaps

stage() { echo "=== $(date -u +%FT%TZ) $*"; }

[ -f $C/capture.jsonl ] || { stage capture; $PY exp14_brew_swaps.py capture; }
[ -f $C/readouts.pt ] || { stage readouts; $PY exp14_brew_swaps.py readouts; }
[ -f $C/swaps_h2.pt ] || { stage swaps; $PY exp14_brew_swaps.py swaps; }
[ -f $C/swaps_h3.pt ] || { stage repair; $PY exp14_brew_swaps.py repair; }
[ -f $R/report_v1/report.txt ] || { stage report; $PY exp14_brew_swaps.py report --run report_v1; }
stage done
