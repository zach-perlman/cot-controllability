#!/bin/bash
# exp10 pilot 2 on the dev bank (results/exp10_hide_what_you_need/pilot_notes.md): the rule-bearing conditions with
# pilot 1's fixes, then C1b (filler length from pilot 2's C4 grades). Requests of "pilot2" are written before this.
#   nohup scripts/run_exp10_pilot2.sh > log/exp10/pilot2.log 2>&1 &
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
  run pilot2 "$model"
  /venv/main/bin/python cc_exp10.py requests --set pilot2_c1b --model "$model"
  run pilot2_c1b "$model"
done
echo "exp10 pilot2 done"
