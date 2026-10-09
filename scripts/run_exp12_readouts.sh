#!/bin/bash
# exp12 readouts for the notebook (results/exp12_gemma_lenses/manifest.json, key "readouts"): Qwen readouts,
# the Gemma J-lens fit, then Gemma readouts. Every stage skips when its output exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

PY=/venv/main/bin/python
JPP=third_party/jpp_lens
W="$PY scripts/exp12_readouts.py"
C=cache/exp12
L=$C/lenses
Q=$C/qwen_check
R=results/exp12_gemma_lenses

stage() { echo "=== $(date -u +%FT%TZ) $*"; }

[ -f $R/readouts_qwen/ranks.csv ] || { stage readouts qwen
  HF_HUB_OFFLINE=1 $W readouts --hf-model-name Qwen/Qwen3.6-27B --layers 8,16,24,32,40,48,56 \
    --lens logit $Q/jlens_camila.pt $Q/rlens_camila.pt $Q/jpp_released.pt \
    --correctness-csv $JPP/data/jlens/evaluations/model_correctness.csv \
    --reference-ranks $R/eval_qwen_check/ranks.csv --out-dir $R/readouts_qwen; }

[ -f $L/jlens_16p.pt ] || { stage fit jlens_16p
  $W fit-jlens --num-prompts 16 --router-path $C/router_K8.pt --artifacts-dir $C/fits --out $L/jlens_16p.pt; }

[ -f $R/readouts_gemma/ranks.csv ] || { stage readouts gemma
  $W readouts --hf-model-name google/gemma-4-31B-it --layers 8,15,22,30,38,45,52 \
    --lens logit $L/jlens_16p.pt $C/snapshots/jpp/rlens_16p.pt $L/rlens_64p.pt $L/jpp_64p.pt \
    --correctness-csv $R/model_correctness.csv \
    --reference-ranks $R/eval_v1/ranks.csv --out-dir $R/readouts_gemma; }
stage readouts done
