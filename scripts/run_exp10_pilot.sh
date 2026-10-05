#!/bin/bash
# exp10 pilot on the dev bank, one engine at a time: every condition but C1b for the pilot models, then C1b (its
# filler length comes from the model's pilot C4 grades). Requests of "pilot" are written before this runs.
#   nohup scripts/run_exp10_pilot.sh > log/exp10/pilot.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
run() {  # set model
  local set=$1 model=$2
  for step in "--shard 0/1" "--merge-shards 1"; do
    ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" \
      --requests "../cache/exp10/requests_${set}_${model}.jsonl" --items ../cache/exp10/items_dev.jsonl $step
  done
  /venv/main/bin/python cc_exp10.py grade --set "$set" --model "$model"
}
for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8; do
  run pilot "$model"
  /venv/main/bin/python cc_exp10.py requests --set pilot_c1b --model "$model"
  run pilot_c1b "$model"
done
echo "exp10 pilot done"
