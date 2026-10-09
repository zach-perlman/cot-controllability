#!/bin/bash
# exp12 phase 2 (results/exp12_gemma_lenses/manifest.json, key "phase2"): phase 1's stages 4-6
# as pre-registered, then the base-lens transfer check (B1), the offload of the 64-prompt raw
# sums, and the long-sequence J++ fit (L1-L4). run_exp12_lenses.sh was stopped during its fit,
# which carried on, so its J-lens stage never runs. Every stage skips when its output exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

PY=/venv/main/bin/python
JPP=third_party/jpp_lens
CLI="env PYTHONPATH=$JPP/src $PY $JPP/scripts/jpp_cli.py"
W="$PY scripts/exp12_lenses.py"
W2="$PY scripts/exp12_phase2.py"
MODEL=google/gemma-4-31B-it
C=cache/exp12
L=$C/lenses
R=results/exp12_gemma_lenses
CORRECT=$R/model_correctness.csv
FIT_LAYERS=0,2,4,6,8,10,12,14,15,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,45,46,48,50,52,54,56,58
READOUT=8,15,22,30,38,45,52
ITEMS="--eval-data-dir $JPP/data/jlens/evaluations --correctness-csv $CORRECT"
BASE_LENS=$HF_HOME/hub/models--koayon--jpp-lenses/snapshots/6c96867a91c43c14cec17cf0f6d157b933c5fee5/gemma-4-31b/lens.pt

stage() { echo "=== $(date -u +%FT%TZ) $*"; }
need_gb() {  # need_gb <GB> <what>: stop rather than fill the disk mid-write
  local free; free=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  if (( free < $1 )); then echo "STOP: ${free} GB free, ${1} GB needed for $2"; exit 1; fi
}
report() {  # report <eval dir> <reference lens> <file stem>
  [ -f $1/$3.txt ] || $W report --ranks $1/ranks.csv --reference $2 --correctness-csv $CORRECT --out $1/$3.txt
}

# 0. wait for the 64-prompt fit to finish and free the GPU
while [ ! -f $C/jpp_checkpoint_dir.txt ]; do
  pgrep -f "exp12_lenses.py fit --name jpp " > /dev/null || { echo "STOP: the 64-prompt fit is neither running nor finished"; exit 1; }
  sleep 60
done
while pgrep -f "exp12_lenses.py fit --name jpp " > /dev/null; do sleep 10; done
CKPT=$(cat $C/jpp_checkpoint_dir.txt)
CKPT_FILE=$CKPT/experts_K8_checkpoint.pt

# 4. (phase 1) experts and the R-lens from the 64-prompt sums
[ -f $L/rlens_64p.pt ] || { stage merge jpp; need_gb 20 "the merge"
  $CLI merge-experts --checkpoint-dirs "$CKPT" --num-clusters 8 \
    --checkpoint-name jpp_experts --out $L/experts_64p.pt --pooled-lens-out $L/rlens_64p.pt; }

# offload the 64-prompt raw sums: upload, check the remote sha256, delete (in the background)
OFFLOAD_PID=
if [ -f "$CKPT_FILE" ]; then
  stage offload "$CKPT_FILE (background; log/exp12/offload.log)"
  $W2 offload --delete "$CKPT_FILE" > log/exp12/offload.log 2>&1 &
  OFFLOAD_PID=$!
fi

# 5. (phase 1) J++ expert weights on the fit:0 items
[ -f $L/jpp_64p.pt ] || { stage weights 64p
  $CLI fit-weights --hf-model-name $MODEL --layers $FIT_LAYERS --experts $L/experts_64p.pt $ITEMS \
    --items fit:0 --cache-path $C/residuals_fit_layers.pt --checkpoint-name jpp_64p --out $L/jpp_64p.pt; }
for n in 08 16 32; do
  [ -f $L/jpp_${n}p.pt ] || { stage weights ${n}p
    $CLI fit-weights --hf-model-name $MODEL --layers $READOUT --experts $C/snapshots/jpp/experts_${n}p.pt \
      $ITEMS --items fit:0 --cache-path $C/residuals_readout_layers.pt --checkpoint-name jpp_${n}p \
      --out $L/jpp_${n}p.pt; }
done

# 6. (phase 1) score every lens on the held-out:0 items at the readout layers
E=$R/eval_v1
[ -f $E/ranks.csv ] || { stage evaluate v1; mkdir -p $E
  $CLI evaluate --hf-model-name $MODEL --layers $READOUT --items held-out:0 $ITEMS --out $E/ranks.csv \
    --lens logit $C/snapshots/jpp/rlens_{08,16,32}p.pt $L/rlens_64p.pt $L/jpp_{08,16,32}p.pt $L/jpp_64p.pt; }
report $E jpp_64p report_vs_jpp
report $E rlens_64p report_vs_rlens

# 7. B1: the released J++ lens of the base model, scored on gemma-4-31B-it beside ours
E=$R/eval_transfer_v1
[ -f $E/ranks.csv ] || { stage evaluate transfer
  $W2 evaluate-transfer --lens logit $L/rlens_64p.pt $L/jpp_64p.pt --transferred jpp_base=$BASE_LENS \
    --correctness-csv $CORRECT --out $E/ranks.csv; }
report $E jpp_64p report_vs_jpp

# 8. a K=16 router on the same 1000 short prompts as the K=8 one, at the readout layers
[ -f $C/router_K16.pt ] || { stage router K16
  $CLI fit-router --hf-model-name $MODEL --layers $READOUT --num-prompts 1000 --num-clusters 16 \
    --out $C/router_K16.pt; }

# 9. the long-sequence fit: 14 records x 512 tokens, routers K=8 and K=16, snapshot at 7
if [ -n "$OFFLOAD_PID" ]; then stage wait for the offload; wait $OFFLOAD_PID; fi
[ ! -f "$CKPT_FILE" ] || { echo "STOP: $CKPT_FILE is still on disk (log/exp12/offload.log)"; exit 1; }
if [ ! -f $C/jpp_long_checkpoint_dir.txt ]; then
  stage fit jpp_long
  if compgen -G "$C/fits/*/jpp_long/shard0of1/experts_K8_checkpoint.pt" > /dev/null; then
    need_gb 22 "the long fit (resume)"; else need_gb 38 "the long fit"; fi
  # judged by its output file: the WikiText streaming thread can crash the interpreter at exit
  $W2 fit-long --name jpp_long --router-paths $C/router_K8.pt $C/router_K16.pt --layers $READOUT \
    --num-prompts 14 --min-chars 3000 --max-seq-len 512 --rows-per-pass 4 \
    --prompts-json $C/long_prompts.json --artifacts-dir $C/fits --snapshot-dir $C/snapshots/jpp_long \
    --snapshot-at 7 --snapshot-k 8 --pooled-name rlens_long --checkpoint-dir-out $C/jpp_long_checkpoint_dir.txt \
    || [ -f $C/jpp_long_checkpoint_dir.txt ]
fi
LCKPT=$(cat $C/jpp_long_checkpoint_dir.txt)

# 10. experts (K=8 with the long R-lens, and K=16), then expert weights on fit:0 at the readout layers
[ -f $L/rlens_long_14p.pt ] || { stage merge long K8; need_gb 6 "the long K8 merge"
  $CLI merge-experts --checkpoint-dirs "$LCKPT" --num-clusters 8 --checkpoint-name jpp_long_k8_experts \
    --out $L/experts_long_k8_14p.pt --pooled-lens-out $L/rlens_long_14p.pt; }
[ -f $L/experts_long_k16_14p.pt ] || { stage merge long K16; need_gb 9 "the long K16 merge"
  $CLI merge-experts --checkpoint-dirs "$LCKPT" --num-clusters 16 --checkpoint-name jpp_long_k16_experts \
    --out $L/experts_long_k16_14p.pt; }
for spec in long_k8_14p:$L/experts_long_k8_14p.pt long_k16_14p:$L/experts_long_k16_14p.pt \
            long_k8_07p:$C/snapshots/jpp_long/experts_k8_07p.pt; do
  name=${spec%%:*}; experts=${spec#*:}
  [ -f $L/jpp_$name.pt ] || { stage weights $name; need_gb 2 "the $name weights"
    $CLI fit-weights --hf-model-name $MODEL --layers $READOUT --experts $experts $ITEMS --items fit:0 \
      --cache-path $C/residuals_readout_layers.pt --checkpoint-name jpp_$name --out $L/jpp_$name.pt; }
done

# 11. score the long lenses beside the 64-prompt ones on held-out:0
E=$R/eval_long_v1
[ -f $E/ranks.csv ] || { stage evaluate long; mkdir -p $E
  $CLI evaluate --hf-model-name $MODEL --layers $READOUT --items held-out:0 $ITEMS --out $E/ranks.csv \
    --lens logit $L/rlens_64p.pt $L/jpp_64p.pt $C/snapshots/jpp_long/rlens_long_07p.pt $L/rlens_long_14p.pt \
      $L/jpp_long_k8_07p.pt $L/jpp_long_k8_14p.pt $L/jpp_long_k16_14p.pt; }
report $E jpp_64p report_vs_jpp
report $E rlens_64p report_vs_rlens
report $E jpp_long_k8_14p report_vs_long_k8
stage done
