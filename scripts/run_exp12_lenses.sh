#!/bin/bash
# exp12: J++, R- and J-lenses for google/gemma-4-31B-it (results/exp12_gemma_lenses/manifest.json).
# Every stage skips when its output exists, so a rerun continues where a stop left off; the
# fits resume from their own checkpoints. Large files go to cache/exp12, small ones to results.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

PY=/venv/main/bin/python
JPP=third_party/jpp_lens
CLI="env PYTHONPATH=$JPP/src $PY $JPP/scripts/jpp_cli.py"
W="$PY scripts/exp12_lenses.py"
MODEL=google/gemma-4-31B-it
C=cache/exp12
L=$C/lenses
R=results/exp12_gemma_lenses
CORRECT=$R/model_correctness.csv
FIT_LAYERS=0,2,4,6,8,10,12,14,15,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,45,46,48,50,52,54,56,58
READOUT=8,15,22,30,38,45,52
ITEMS="--eval-data-dir $JPP/data/jlens/evaluations --correctness-csv $CORRECT"

stage() { echo "=== $(date -u +%FT%TZ) $*"; }
need_gb() {  # need_gb <GB> <what>: stop rather than fill the disk mid-write
  local free; free=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  if (( free < $1 )); then echo "STOP: ${free} GB free, ${1} GB needed for $2"; exit 1; fi
}
fit_started() { compgen -G "$C/fits/*/$1/shard0of1/experts_K$2_checkpoint.pt" > /dev/null; }

mkdir -p "$L" "$R"

# 1. which eval items the model answers correctly (the correctness filter)
[ -f "$CORRECT" ] || { stage correctness; $W correctness --out "$CORRECT"; }

# 2. J++ router: per-layer PCA-64 + k-means K=8 over 1000 WikiText prompts
[ -f $C/router_K8.pt ] || { stage router K8
  $CLI fit-router --hf-model-name $MODEL --layers $FIT_LAYERS --num-prompts 1000 --num-clusters 8 \
    --out $C/router_K8.pt; }

# 3. the LRP fit (J++ experts; R-lens = their pooled map), 64 prompts, snapshots at 8/16/32
if [ ! -f $C/jpp_checkpoint_dir.txt ]; then
  stage fit jpp
  if fit_started jpp 8; then need_gb 45 "the J++ fit (resume)"; else need_gb 75 "the J++ fit"; fi
  $W fit --name jpp --router-path $C/router_K8.pt --lrp-mode rlens --layers $FIT_LAYERS \
    --num-prompts 64 --artifacts-dir $C/fits --snapshot-dir $C/snapshots/jpp --snapshot-at 8,16,32 \
    --snapshot-layers $READOUT --pooled-name rlens --checkpoint-dir-out $C/jpp_checkpoint_dir.txt
fi

# 4. experts and the R-lens from the 64-prompt sums
[ -f $L/rlens_64p.pt ] || { stage merge jpp; need_gb 20 "the merge"
  $CLI merge-experts --checkpoint-dirs "$(cat $C/jpp_checkpoint_dir.txt)" --num-clusters 8 \
    --checkpoint-name jpp_experts --out $L/experts_64p.pt --pooled-lens-out $L/rlens_64p.pt; }

# 5. J++ expert weights on the fit:0 items: the full fit at every fit layer, snapshots at the readout layers
[ -f $L/jpp_64p.pt ] || { stage weights 64p
  $CLI fit-weights --hf-model-name $MODEL --layers $FIT_LAYERS --experts $L/experts_64p.pt $ITEMS \
    --items fit:0 --cache-path $C/residuals_fit_layers.pt --checkpoint-name jpp_64p --out $L/jpp_64p.pt; }
for n in 08 16 32; do
  [ -f $L/jpp_${n}p.pt ] || { stage weights ${n}p
    $CLI fit-weights --hf-model-name $MODEL --layers $READOUT --experts $C/snapshots/jpp/experts_${n}p.pt \
      $ITEMS --items fit:0 --cache-path $C/residuals_readout_layers.pt --checkpoint-name jpp_${n}p \
      --out $L/jpp_${n}p.pt; }
done

# 6. score every lens on the held-out:0 items at the readout layers (Readout Filtering on)
E=$R/eval_v1
[ -f $E/ranks.csv ] || { stage evaluate v1; mkdir -p $E
  $CLI evaluate --hf-model-name $MODEL --layers $READOUT --items held-out:0 $ITEMS --out $E/ranks.csv \
    --lens logit $C/snapshots/jpp/rlens_{08,16,32}p.pt $L/rlens_64p.pt $L/jpp_{08,16,32}p.pt $L/jpp_64p.pt; }
[ -f $E/report_vs_jpp.txt ] || $W report --ranks $E/ranks.csv --reference jpp_64p \
  --correctness-csv "$CORRECT" --out $E/report_vs_jpp.txt
[ -f $E/report_vs_rlens.txt ] || $W report --ranks $E/ranks.csv --reference rlens_64p \
  --correctness-csv "$CORRECT" --out $E/report_vs_rlens.txt

# 7. plain J-lens (gradient fit, K=1): prompt count by the manifest's rule, self-check snapshot at half
NJ=$($W choose-n --report-json $E/report_vs_rlens.json)
HALF=$((NJ / 2))
NJP=$(printf %02d "$NJ"); HALFP=$(printf %02d "$HALF")
echo "J-lens prompts: $NJ (rule: smallest R-lens snapshot within 1 point of rlens_64p)" | tee $E/jlens_prompt_count.txt
[ -f $C/router_K1.pt ] || { stage router K1
  $CLI fit-router --hf-model-name $MODEL --layers $FIT_LAYERS --num-prompts 64 --num-clusters 1 \
    --out $C/router_K1.pt; }
if [ ! -f $C/jlens_checkpoint_dir.txt ]; then
  stage fit jlens "$NJ prompts"; need_gb 12 "the J-lens fit"
  $W fit --name jlens --router-path $C/router_K1.pt --lrp-mode none --layers $FIT_LAYERS \
    --num-prompts "$NJ" --artifacts-dir $C/fits --snapshot-dir $C/snapshots/jlens --snapshot-at "$HALF" \
    --snapshot-layers $READOUT --pooled-name jlens --checkpoint-dir-out $C/jlens_checkpoint_dir.txt
fi
[ -f $L/jlens_${NJP}p.pt ] || { stage merge jlens; need_gb 8 "the J-lens merge"
  $CLI merge-experts --checkpoint-dirs "$(cat $C/jlens_checkpoint_dir.txt)" --num-clusters 1 \
    --checkpoint-name jlens_experts --out $C/jlens_experts.pt --pooled-lens-out $L/jlens_${NJP}p.pt; }

E2=$R/eval_jlens_v1
[ -f $E2/ranks.csv ] || { stage evaluate jlens; mkdir -p $E2
  $CLI evaluate --hf-model-name $MODEL --layers $READOUT --items held-out:0 $ITEMS --out $E2/ranks.csv \
    --lens logit $C/snapshots/jlens/jlens_${HALFP}p.pt $L/jlens_${NJP}p.pt $L/rlens_64p.pt $L/jpp_64p.pt; }
[ -f $E2/report_vs_jlens.txt ] || $W report --ranks $E2/ranks.csv --reference jlens_${NJP}p \
  --correctness-csv "$CORRECT" --out $E2/report_vs_jlens.txt
stage done
