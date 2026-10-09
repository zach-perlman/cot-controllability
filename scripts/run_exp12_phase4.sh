#!/bin/bash
# exp12 phase 4 (results/exp12_gemma_lenses/manifest.json, key "phase4"): the Aya fit-text arm. Waits for
# phase 3 to exit, then runs each stage once (every stage skips when its output exists); exits non-zero
# on a failure so scripts/run_exp12_phase4_retry.sh reruns it. The fit has no checkpoint: a rerun
# restarts it.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

PY=/venv/main/bin/python
JPP=third_party/jpp_lens
CLI="env PYTHONPATH=$JPP/src $PY $JPP/scripts/jpp_cli.py"
W="$PY scripts/exp12_lenses.py"
W2="$PY scripts/exp12_phase2.py"
W4="$PY scripts/exp12_phase4.py"
MODEL=google/gemma-4-31B-it
C=cache/exp12
L=$C/lenses
R=results/exp12_gemma_lenses
E=$R/eval_aya_v1
CORRECT=$R/model_correctness.csv
READOUT=8,15,22,30,38,45,52
ITEMS="--eval-data-dir $JPP/data/jlens/evaluations --correctness-csv $CORRECT"

stage() { echo "=== $(date -u +%FT%TZ) $*"; }
need_gb() {  # need_gb <GB> <what>: stop rather than fill the disk mid-write
  local free; free=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  if (( free < $1 )); then echo "STOP: ${free} GB free, ${1} GB needed for $2"; exit 1; fi
}
offload() {  # offload <file>: upload to the private dataset, check the remote sha256, delete; 3 attempts
  for attempt in 1 2 3; do
    [ -f "$1" ] || return 0
    stage offload "$1, attempt $attempt (log/exp12/offload.log)"
    $W2 offload --delete "$1" >> log/exp12/offload.log 2>&1 || sleep 300
  done
  [ ! -f "$1" ] || { echo "STOP: $1 is still on disk"; exit 1; }
}
report() {  # report <reference lens> <file stem>
  [ -f $E/$2.txt ] || $W report --ranks $E/ranks.csv --reference $1 --correctness-csv $CORRECT \
    --out $E/$2.txt || echo "FAILED: report $2 (continuing)"
}
existing() { for f in "$@"; do [ "$f" = logit ] || [ -f "$f" ] && echo "$f"; done; }

while pgrep -f "run_exp12_phase3_retry.sh" > /dev/null; do sleep 60; done

[ -f $C/aya_prompts.json ] || { stage aya prompts; $W4 aya-prompts --out $C/aya_prompts.json; }

# phase 3's short-replicate raw sums, once merged, free the disk the fit needs
if [ -f $C/jpp_short_b_checkpoint_dir.txt ] && [ -f $L/experts_64p_b.pt ] && [ -f $L/rlens_64p_b.pt ]; then
  offload "$(cat $C/jpp_short_b_checkpoint_dir.txt)/experts_K8_checkpoint.pt"
fi

[ -f $C/router_aya_K8.pt ] || { stage aya router; need_gb 1 "the Aya router"
  $W4 fit-router --prompts-json $C/aya_prompts.json --num-prompts 1000 --layers $READOUT --out $C/router_aya_K8.pt; }

[ -f $L/rlens_aya_64p.pt ] || { stage aya fit; need_gb 7 "the Aya experts, pooled lens and weights"
  $W4 fit --prompts-json $C/aya_prompts.json --num-prompts 64 --router-path $C/router_aya_K8.pt --layers $READOUT \
    --experts-out $L/experts_aya_64p.pt --pooled-out $L/rlens_aya_64p.pt --artifacts-dir $C/fits; }

[ -f $L/jpp_aya_64p.pt ] || { stage weights aya_64p; need_gb 2 "the aya_64p weights"
  $CLI fit-weights --hf-model-name $MODEL --layers $READOUT --experts $L/experts_aya_64p.pt $ITEMS --items fit:0 \
    --cache-path $C/residuals_readout_layers.pt --checkpoint-name jpp_aya_64p --out $L/jpp_aya_64p.pt; }

[ -f $E/ranks.csv ] || { stage evaluate aya; mkdir -p $E
  $CLI evaluate --hf-model-name $MODEL --layers $READOUT --items held-out:0 $ITEMS --out $E/ranks.csv \
    --lens $(existing logit $L/rlens_64p.pt $L/jpp_64p.pt $L/rlens_64p_b.pt $L/jpp_64p_b.pt \
      $L/rlens_aya_64p.pt $L/jpp_aya_64p.pt); }
report jpp_64p report_vs_jpp
report rlens_64p report_vs_rlens
[ -f $E/per_eval.txt ] || $W4 per-eval-report --ranks $E/ranks.csv --correctness-csv $CORRECT \
  --out $E/per_eval.txt || echo "FAILED: per-eval report (continuing)"
stage "phase 4 done"
