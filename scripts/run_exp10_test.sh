#!/bin/bash
# exp10 test (results/exp10_hide_what_you_need/manifest.json), one engine at a time: the dev C2 pass of the models
# without one (their C4 example turns), then per model the test requests (every condition but C1b), generation and
# grading, then C1b (filler length from the model's test C4 grades).
#   nohup scripts/run_exp10_test.sh > log/exp10/test.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
run() {  # set model items
  local set=$1 model=$2 items=$3 requests=../cache/exp10/requests_${1}_${2}.jsonl
  # A shard that finished can still exit nonzero in exp03_exit's teardown (pilot 2); the merge checks every row.
  ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items "../cache/exp10/items_${items}.jsonl" --shard 0/1 || echo "shard exited nonzero; the merge checks completeness"
  ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items "../cache/exp10/items_${items}.jsonl" --merge-shards 1
  /venv/main/bin/python cc_exp10.py grade --set "$set" --model "$model"
}
for model in Qwen3-32B Qwen3.6-27B-FP8 Qwen3.6-35B-A3B-FP8 GLM-4.7-Flash-FP8 Gemma-4-12B-FP8; do
  [[ -e ../cache/exp10/requests_dev_c2_${model}.jsonl ]] || /venv/main/bin/python cc_exp10.py requests --set dev_c2 --model "$model"
  run dev_c2 "$model" dev
done
for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3-32B Qwen3.6-27B-FP8 Qwen3.6-35B-A3B-FP8 GLM-4.7-Flash-FP8 \
             Gemma-4-12B-FP8; do
  [[ -e ../cache/exp10/requests_test_${model}.jsonl ]] || /venv/main/bin/python cc_exp10.py requests --set test --model "$model"
  run test "$model" test
  [[ -e ../cache/exp10/requests_test_c1b_${model}.jsonl ]] || /venv/main/bin/python cc_exp10.py requests --set test_c1b --model "$model"
  run test_c1b "$model" test
done
echo "exp10 test done"
