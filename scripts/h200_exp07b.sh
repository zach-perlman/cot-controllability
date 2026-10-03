#!/usr/bin/env bash
# exp07b, run once (results/exp07b_prompt_compare/manifest.json). Per model: (Qwen3-32B only) exp07's example-pool
# pre-pass; the prep generations (B's openings, L's guides); the main rows (upgraded + the candidates); grading.
# Then the analysis, and stop for review.
#   setsid nohup scripts/h200_exp07b.sh > log/exp07b/h200_exp07b.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
logs=../log/exp07b
cache=../cache/exp07b
pool_items=../cache/exp07/items.jsonl
main_items=$cache/items.jsonl
models=(Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3-32B)
mkdir -p "$logs"
stamp() { echo "$(date -u +%FT%TZ) $*"; }

# generate STEP MODEL ITEMS: one request file on the GPU (3 attempts); the shard log is renamed per step
generate() {
  local step=$1 model=$2 items=$3
  for attempt in 1 2 3; do
    stamp "$model $step (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp07b --model "$model" \
        --requests "$cache/requests_${step}_$model.jsonl" --items "$items"; then
      mv "$logs/generate_$model.shard0of1.log" "$logs/generate_${step}_$model.log"
      return 0
    fi
    stamp "$model $step: stopped or failed"
  done
  return 1
}

# requests COMMAND STEP MODEL: write requests_<STEP>_<MODEL>.jsonl once (cc_exp07b refuses to overwrite one)
requests() {
  local command=$1 step=$2 model=$3
  [ -f "$cache/requests_${step}_$model.jsonl" ] && return 0
  $py cc_exp07b.py "$command" --model "$model" > "$logs/requests_${step}_$model.log" 2>&1 \
    || { stamp "$model: $step request file failed ($logs/requests_${step}_$model.log)"; return 1; }
}

until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )); do sleep 30; done
for model in "${models[@]}"; do
  if [ "$model" = Qwen3-32B ]; then
    requests prepass prepass "$model" && generate prepass "$model" "$pool_items" \
      || { stamp "$model: giving up"; continue; }
  fi
  requests prep prep "$model" && generate prep "$model" "$pool_items" || { stamp "$model: giving up"; continue; }
  requests requests main "$model" || continue
  generate main "$model" "$main_items" || { stamp "$model: giving up"; continue; }
  $py cc_exp07b.py grade --model "$model" > "$logs/grade_$model.log" 2>&1 \
    && stamp "$model graded" || stamp "$model: grading failed ($logs/grade_$model.log)"
done
$py cc_exp07b_analysis.py --run v1 --skip-missing > "$logs/analysis_v1.log" 2>&1 \
  && stamp "analysis written (results/exp07b_prompt_compare/analysis/v1)" || stamp "analysis failed ($logs/analysis_v1.log)"
stamp "done"
