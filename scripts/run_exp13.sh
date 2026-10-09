#!/bin/bash
# exp13 (results/exp13_lookup_layers/manifest.json): passage, requests, the mask check, the Gemma padding run, the
# Qwen block ablation, then the report. Every stage skips when its output exists; the Gemma run needs the mask check.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
PY=/venv/main/bin/python
C=../cache/exp13
R=../results/exp13_lookup_layers

stage() { echo "=== $(date -u +%FT%TZ) $*"; }

[ -f $C/passage.txt ] || { stage passage; $PY exp13_lookup_layers.py passage; }
export HF_HUB_OFFLINE=1
[ -f $C/requests_gemma.jsonl ] || { stage requests; $PY exp13_lookup_layers.py requests; }
[ -f $C/mask_check.txt ] || { stage mask-check; $PY exp13_lookup_layers.py mask-check; }
grep -q "MASK CHECK PASSED" $C/mask_check.txt || { echo "mask check did not pass; stopping"; exit 1; }
[ -f $C/gemma_rows.jsonl ] || { stage gemma; $PY exp13_lookup_layers.py gemma; }
[ -f $C/qwen_ablation.pt ] || { stage qwen; $PY exp13_lookup_layers.py qwen; }
[ -f $R/report_v1/report.txt ] || { stage report; $PY exp13_lookup_layers.py report --run report_v1; }
stage done
