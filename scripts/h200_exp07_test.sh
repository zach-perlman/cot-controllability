#!/usr/bin/env bash
# exp07 test, run once (results/exp07_prompt_search/manifest.json): baseline, upgraded and the validation winner on
# the test split, all 9 rules, all 6 models; held-out models first (the primary contrast is theirs).
#   setsid nohup scripts/h200_exp07_test.sh WINNER > log/exp07/h200_exp07_test.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
winner=$1
py=/venv/main/bin/python
logs=../log/exp07
items=../cache/exp07/items.jsonl
models=(Qwen3.6-27B-FP8 GLM-4.7-Flash Qwen3.6-35B-A3B-FP8 Gemma-4-12B Qwen3.8-27B-FP8 Gemma-4-31B-FP8)
arms=baseline,upgraded
[ "$winner" != upgraded ] && arms="$arms,$winner"
stamp() { echo "$(date -u +%FT%TZ) $*"; }

for model in "${models[@]}"; do
  $py cc_exp07.py requests --model "$model" --name test --split test --candidates "$arms" \
    > "$logs/requests_test_$model.log" 2>&1 \
    || { stamp "$model: request file failed ($logs/requests_test_$model.log)"; exit 1; }
done
stamp "request files written (arms $arms)"
until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )); do sleep 30; done

for model in "${models[@]}"; do
  ok=0
  for attempt in 1 2 3; do
    stamp "$model test (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp07 --model "$model" \
        --requests "../cache/exp07/requests_test_$model.jsonl" --items "$items"; then
      mv "$logs/generate_$model.shard0of1.log" "$logs/generate_test_$model.log"
      ok=1
      break
    fi
    stamp "$model test: stopped or failed"
  done
  [ "$ok" = 1 ] || { stamp "$model: giving up"; continue; }
  $py cc_exp07.py grade --model "$model" --name test > "$logs/grade_test_$model.log" 2>&1 \
    && stamp "$model graded" || stamp "$model: grading failed ($logs/grade_test_$model.log)"
done
stamp "done"
