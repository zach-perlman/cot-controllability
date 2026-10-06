#!/bin/bash
# exp10 on chain and mhn (cc_exp10_tasks), dev stage, one engine at a time; run after brew's test (one GPU):
#   per pilot model: chain's greedy no-CoT calibration (shipped bank), chain dev C2, mhn's hop controls, mhn dev C2
#   (C2 is the source of the model's C4 example turns); then each task's pilot (requests written after C2 is graded)
#   and its C1b (filler length from the model's pilot C4 grades).
#   nohup scripts/run_exp10_tasks_pilot.sh > log/exp10/tasks_pilot.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
run() {  # task set model sampling items
  local task=$1 set=$2 model=$3 sampling=$4 items=$5
  local dir=../cache/exp10/$task requests=../cache/exp10/$task/requests_${set}_${model}.jsonl
  [[ -e $requests ]] || /venv/main/bin/python cc_exp10_tasks.py requests --task "$task" --set "$set" --model "$model"
  # A shard that finished can still exit nonzero in exp03_exit's teardown (brew's pilot 2); the merge checks every row.
  ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items "$dir/items_${items}.jsonl" --sampling "$sampling" --shard 0/1 \
    || echo "shard exited nonzero; the merge checks completeness"
  ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items "$dir/items_${items}.jsonl" --sampling "$sampling" --merge-shards 1
  /venv/main/bin/python cc_exp10_tasks.py grade --task "$task" --set "$set" --model "$model" --sampling "$sampling"
}
for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8; do
  run chain calib "$model" greedy shipped
  run chain dev_c2 "$model" card dev
  run mhn ctl "$model" card ctl
  run mhn dev_c2 "$model" card dev
done
for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8; do
  for task in chain mhn; do
    run "$task" pilot "$model" card dev
    run "$task" pilot_c1b "$model" card dev
  done
done
echo "exp10 tasks pilot done"
