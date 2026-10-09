#!/bin/bash
# exp12 phase 3 (results/exp12_gemma_lenses/manifest.json, key "phase3"). With no argument: wait for
# phase 2 to exit, then run the four groups in order, each as its own process (bash $0 <group>), so
# one failing (logged "FAILED") does not stop the others; exits non-zero if any failed, so
# scripts/run_exp12_phase3_retry.sh reruns it. Every stage skips when its output exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

PY=/venv/main/bin/python
JPP=third_party/jpp_lens
CLI="env PYTHONPATH=$JPP/src $PY $JPP/scripts/jpp_cli.py"
W="$PY scripts/exp12_lenses.py"
W2="$PY scripts/exp12_phase2.py"
W3="$PY scripts/exp12_phase3.py"
MODEL=google/gemma-4-31B-it
C=cache/exp12
L=$C/lenses
R=results/exp12_gemma_lenses
CORRECT=$R/model_correctness.csv
READOUT=8,15,22,30,38,45,52
ITEMS="--eval-data-dir $JPP/data/jlens/evaluations --correctness-csv $CORRECT"
HUB=$HF_HOME/hub

stage() { echo "=== $(date -u +%FT%TZ) $*"; }
need_gb() {  # need_gb <GB> <what>: stop rather than fill the disk mid-write
  local free; free=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  if (( free < $1 )); then echo "STOP: ${free} GB free, ${1} GB needed for $2"; exit 1; fi
}
report() {  # report <eval dir> <reference lens> <file stem>
  [ -f $1/$3.txt ] || $W report --ranks $1/ranks.csv --reference $2 --correctness-csv $CORRECT \
    --out $1/$3.txt || echo "FAILED: report $1/$3 (continuing)"
}
offload() {  # offload <file>: upload to the private dataset, check the remote sha256, delete; 3 attempts
  for attempt in 1 2 3; do
    [ -f "$1" ] || return 0
    stage offload "$1, attempt $attempt (log/exp12/offload.log)"
    $W2 offload --delete "$1" >> log/exp12/offload.log 2>&1 || sleep 300
  done
  [ ! -f "$1" ] || { echo "STOP: $1 is still on disk"; exit 1; }
}
weights() {  # weights <name> <experts file>: J++ expert weights on fit:0 at the readout layers
  [ -f $L/jpp_$1.pt ] || { stage weights $1; need_gb 2 "the $1 weights"
    $CLI fit-weights --hf-model-name $MODEL --layers $READOUT --experts $2 $ITEMS --items fit:0 \
      --cache-path $C/residuals_readout_layers.pt --checkpoint-name jpp_$1 --out $L/jpp_$1.pt; }
}
existing() { for f in "$@"; do [ "$f" = logit ] || [ -f "$f" ] && echo "$f"; done; }

# P1: the paper's Qwen3.6-27B row with its released lenses, through the same evaluate stage
qwen_check() {
  local E=$R/eval_qwen_check Q=$C/qwen_check
  mkdir -p $Q $E
  ln -sfn $HUB/models--koayon--jpp-lenses/snapshots/6c96867a91c43c14cec17cf0f6d157b933c5fee5/qwen3.6-27b/lens.pt $Q/jpp_released.pt
  ln -sfn $HUB/models--camilablank--workspace-lenses/snapshots/d740106d1e0f95456dc8718fba2895e9c8ffd6ef/qwen3.6-27b/j-lens/lens.pt $Q/jlens_camila.pt
  ln -sfn $HUB/models--camilablank--workspace-lenses/snapshots/d740106d1e0f95456dc8718fba2895e9c8ffd6ef/qwen3.6-27b/r-lens/lens.pt $Q/rlens_camila.pt
  [ -f $E/ranks.csv ] || { stage qwen check
    HF_HUB_OFFLINE=1 $CLI evaluate --hf-model-name Qwen/Qwen3.6-27B --items held-out:0 \
      --eval-data-dir $JPP/data/jlens/evaluations --correctness-csv $JPP/data/jlens/evaluations/model_correctness.csv \
      --out $E/ranks.csv --lens logit $Q/jpp_released.pt $Q/jlens_camila.pt $Q/rlens_camila.pt; }
  [ -f $E/summary.txt ] || $W3 qwen-summary --pass-csv $E/ranks_pass_at_k.csv --out $E/summary.txt
}

# G1: Gemma's no-CoT brew behaviour on exp11's test banks
brew_gate() {
  for bank in test single_test; do
    [ -f $C/brew_gate/$bank.jsonl ] || { stage brew gate $bank; $W3 brew-gate --bank $bank; }
  done
  [ -f $R/brew_gate/gate.txt ] || $W3 brew-gate-report --out $R/brew_gate/gate.txt
}

# R: the long fit again on records 15-28, then records 1-28 from the two fits' summed sums
long_replicate() {
  [ -f $C/jpp_long_checkpoint_dir.txt ] || { echo "STOP: needs the phase 2 long fit"; exit 1; }
  [ -f $L/experts_long_k16_14p.pt ] || { echo "STOP: the long K=16 merge must precede its offload"; exit 1; }
  local LCKPT; LCKPT=$(cat $C/jpp_long_checkpoint_dir.txt)
  offload $LCKPT/experts_K16_checkpoint.pt
  if [ ! -f $C/jpp_long_b_checkpoint_dir.txt ]; then
    stage fit jpp_long_b; need_gb 15 "the long replicate"
    # judged by its output file: the WikiText streaming thread can crash the interpreter at exit
    $W3 fit-replicate --name jpp_long_b --router-paths $C/router_K8.pt --layers $READOUT --skip 14 \
      --num-prompts 14 --min-chars 3000 --min-tokens 512 --max-seq-len 512 --rows-per-pass 4 \
      --disjoint-from $C/long_prompts.json --prompts-json $C/long_prompts_b.json --artifacts-dir $C/fits \
      --checkpoint-dir-out $C/jpp_long_b_checkpoint_dir.txt || [ -f $C/jpp_long_b_checkpoint_dir.txt ]
  fi
  local LBCKPT; LBCKPT=$(cat $C/jpp_long_b_checkpoint_dir.txt)
  [ -f $L/rlens_long_b_14p.pt ] || { stage merge long B; need_gb 5 "the long B merge"
    $CLI merge-experts --checkpoint-dirs "$LBCKPT" --num-clusters 8 --checkpoint-name jpp_long_b_k8_experts \
      --out $L/experts_long_b_k8_14p.pt --pooled-lens-out $L/rlens_long_b_14p.pt; }
  [ -f $L/rlens_long_28p.pt ] || { stage merge long A+B; need_gb 5 "the long A+B merge"
    $CLI merge-experts --checkpoint-dirs "$LCKPT" "$LBCKPT" --num-clusters 8 --checkpoint-name jpp_long_k8_28p_experts \
      --out $L/experts_long_k8_28p.pt --pooled-lens-out $L/rlens_long_28p.pt; }
  weights long_b_k8_14p $L/experts_long_b_k8_14p.pt
  weights long_k8_28p $L/experts_long_k8_28p.pt
  offload $LCKPT/experts_K8_checkpoint.pt
  offload $LBCKPT/experts_K8_checkpoint.pt
  local E=$R/eval_rep_long_v1
  [ -f $E/ranks.csv ] || { stage evaluate replicate long; mkdir -p $E
    $CLI evaluate --hf-model-name $MODEL --layers $READOUT --items held-out:0 $ITEMS --out $E/ranks.csv \
      --lens logit $L/rlens_64p.pt $L/jpp_64p.pt $L/rlens_long_14p.pt $L/rlens_long_b_14p.pt $L/rlens_long_28p.pt \
        $L/jpp_long_k8_14p.pt $L/jpp_long_b_k8_14p.pt $L/jpp_long_k8_28p.pt; }
  report $E jpp_long_k8_14p report_vs_long_k8
  report $E rlens_long_14p report_vs_rlens_long
  report $E jpp_64p report_vs_jpp
}

# R: the 64 x 128 fit again on prompts 65-128, at the readout layers
short_replicate() {
  [ -f $C/short_prompts.json ] || $W3 save-prompts --num-prompts 64 --min-chars 600 --out $C/short_prompts.json
  if [ ! -f $C/jpp_short_b_checkpoint_dir.txt ]; then
    stage fit jpp_short_b; need_gb 15 "the short replicate"
    $W3 fit-replicate --name jpp_short_b --router-paths $C/router_K8.pt --layers $READOUT --skip 64 \
      --num-prompts 64 --min-chars 600 --min-tokens 0 --max-seq-len 128 --rows-per-pass 16 --checkpoint-every 4 \
      --disjoint-from $C/short_prompts.json --prompts-json $C/short_prompts_b.json --artifacts-dir $C/fits \
      --checkpoint-dir-out $C/jpp_short_b_checkpoint_dir.txt || [ -f $C/jpp_short_b_checkpoint_dir.txt ]
  fi
  [ -f $L/rlens_64p_b.pt ] || { stage merge short B; need_gb 5 "the short B merge"
    $CLI merge-experts --checkpoint-dirs "$(cat $C/jpp_short_b_checkpoint_dir.txt)" --num-clusters 8 \
      --checkpoint-name jpp_short_b_experts --out $L/experts_64p_b.pt --pooled-lens-out $L/rlens_64p_b.pt; }
  weights 64p_b $L/experts_64p_b.pt
  local E=$R/eval_rep_short_v1
  [ -f $E/ranks.csv ] || { stage evaluate replicate short; mkdir -p $E
    $CLI evaluate --hf-model-name $MODEL --layers $READOUT --items held-out:0 $ITEMS --out $E/ranks.csv \
      --lens $(existing logit $L/rlens_64p.pt $L/rlens_64p_b.pt $L/jpp_64p.pt $L/jpp_64p_b.pt \
        $L/rlens_long_14p.pt $L/rlens_long_b_14p.pt $L/jpp_long_k8_14p.pt $L/jpp_long_b_k8_14p.pt); }
  report $E jpp_64p report_vs_jpp
  report $E rlens_64p report_vs_rlens
}

PHASE3_GROUPS="qwen_check brew_gate long_replicate short_replicate"
if [ $# -eq 1 ]; then
  [[ " $PHASE3_GROUPS " == *" $1 "* ]] || { echo "unknown group $1"; exit 2; }
  "$1"; exit 0
fi

while pgrep -f "run_exp12_phase2_retry.sh" > /dev/null; do sleep 60; done
failed=0
for group in $PHASE3_GROUPS; do
  stage group $group
  bash "$0" $group || { echo "FAILED: group $group (continuing)"; failed=1; }
done
stage "phase 3 pass done (failed groups: $failed)"
exit $failed
