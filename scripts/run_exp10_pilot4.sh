#!/bin/bash
# exp10 pilot 4 on the dev bank (results/exp10_hide_what_you_need/pilot_notes.md): Pcode with its code words explicitly allowed (pilot 3 left that unsaid). The instrument check with
# the problem itself in code words. Requests of "pilot4" are written before this runs.
#   nohup scripts/run_exp10_pilot4.sh > log/exp10/pilot4.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
run() {  # set model
  local set=$1 model=$2 requests=../cache/exp10/requests_${1}_${2}.jsonl
  # A shard that finished can still exit nonzero in exp03_exit's teardown (pilot 2); the merge checks every row.
  ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items ../cache/exp10/items_dev.jsonl --shard 0/1 || echo "shard exited nonzero; the merge checks completeness"
  ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items ../cache/exp10/items_dev.jsonl --merge-shards 1
  /venv/main/bin/python cc_exp10.py grade --set "$set" --model "$model"
}
for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8; do
  run pilot4 "$model"
done
echo "exp10 pilot4 done"
