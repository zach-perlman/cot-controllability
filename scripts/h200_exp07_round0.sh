#!/usr/bin/env bash
# exp07 round 0 (results/exp07_prompt_search/manifest.json), after the exp06 off-prefill run frees the GPU.
#   1) search models: pre-pass (own traces on the example pool), then round 0 on the search split, graded and
#      scored into search_log.jsonl
#   2) held-out models: pre-pass only (their few-shot examples for the test run)
#   setsid nohup scripts/h200_exp07_round0.sh > log/exp07/h200_exp07_round0.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
logs=../log/exp07
items=../cache/exp07/items.jsonl
search_models=(Qwen3.8-27B-FP8 Gemma-4-31B-FP8)
heldout_models=(Qwen3.6-27B-FP8 GLM-4.7-Flash Qwen3.6-35B-A3B-FP8 Gemma-4-12B)
round=round0
candidates=baseline,upgraded,opening,six_examples,long_examples,guide_in_system
stamp() { echo "$(date -u +%FT%TZ) $*"; }
mkdir -p "$logs"

until grep -q "generation and grading done" ../log/exp06/h200_exp06_off_prefill.log; do sleep 60; done
until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )); do sleep 30; done
stamp "GPU free"

generate() {  # model, request-file name
  for attempt in 1 2 3; do
    stamp "$1 $2 (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp07 --model "$1" \
        --requests "../cache/exp07/requests_$2_$1.jsonl" --items "$items"; then
      mv "../log/exp07/generate_$1.shard0of1.log" "$logs/generate_$2_$1.log" 2>/dev/null
      return 0
    fi
    stamp "$1 $2: stopped or failed"
  done
  return 1
}

for model in "${search_models[@]}"; do
  generate "$model" prepass || exit 1
  $py cc_exp07.py requests --model "$model" --name "$round" --split search --candidates "$candidates" \
    > "$logs/requests_${round}_$model.log" 2>&1 \
    || { stamp "$model: round requests failed ($logs/requests_${round}_$model.log)"; exit 1; }
  generate "$model" "$round" || exit 1
  $py cc_exp07.py grade --model "$model" --name "$round" > "$logs/grade_${round}_$model.log" 2>&1 \
    || { stamp "$model: grading failed ($logs/grade_${round}_$model.log)"; exit 1; }
done
$py cc_exp07.py score --name "$round" --log > "$logs/score_$round.log" 2>&1 \
  && stamp "round scored ($logs/score_$round.log)" || stamp "scoring failed ($logs/score_$round.log)"

for model in "${heldout_models[@]}"; do
  generate "$model" prepass || stamp "$model: pre-pass failed"
done
stamp "done"
