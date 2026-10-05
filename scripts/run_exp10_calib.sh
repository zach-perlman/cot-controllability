#!/bin/bash
# exp10 dev runs, one engine at a time: the greedy no-CoT calibration on nocot-bench's shipped brew bank (every
# model), then the dev-bank C2 pass of the pilot models (their C4 example source). Each file is graded when merged.
#   nohup scripts/run_exp10_calib.sh > log/exp10/calib.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
run() {  # set model sampling items
  local set=$1 model=$2 sampling=$3 items=$4
  local requests=../cache/exp10/requests_${set}_${model}.jsonl
  for step in "--shard 0/1" "--merge-shards 1"; do
    ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
      --items "../cache/exp10/items_${items}.jsonl" --sampling "$sampling" $step
  done
  /venv/main/bin/python cc_exp10.py grade --set "$set" --model "$model" --sampling "$sampling"
}
for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3.6-27B-FP8 Gemma-4-12B-FP8 Qwen3-32B Qwen3.6-35B-A3B-FP8 \
             GLM-4.7-Flash-FP8; do
  run calib "$model" greedy shipped
done
for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8; do
  run dev_c2 "$model" card dev
done
echo "exp10 calib and dev_c2 done"
