#!/bin/bash
# exp10's depth stage, dev runs (cc_exp10_tasks: dev_c2, depth_pilot), one engine at a time, fastest models first:
#   per model: chain and arithmetic dev C2 (the example turns; for a loop variant, C2 at that loop count), then each
#   task's depth pilot (requests written after the full-loop model's C2 is graded). A model that fails is logged and
#   skipped. Ouro runs under vLLM 0.25.1 (cc_config._OURO_SERVING).
#   nohup scripts/run_exp10_depth_dev.sh > log/exp10/depth_dev.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
run() {  # task set model
  local task=$1 set=$2 model=$3
  local dir=../cache/exp10/$task requests=../cache/exp10/$task/requests_${set}_${model}.jsonl venv=/workspace/.venv-vllm
  [[ $model == Ouro-* ]] && venv=/workspace/.venv-vllm025
  [[ -e $requests ]] || /venv/main/bin/python cc_exp10_tasks.py requests --task "$task" --set "$set" --model "$model" \
    || return 1
  # A shard that finished can still exit nonzero in exp03_exit's teardown; the merge checks every row.
  VLLM_VENV=$venv ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items "$dir/items_dev.jsonl" --shard 0/1 || echo "shard exited nonzero; the merge checks completeness"
  VLLM_VENV=$venv ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items "$dir/items_dev.jsonl" --merge-shards 1 || return 1
  local grades
  grades=$(ls "$dir"/grades/"${model}"__card__stream_abort_"${set}"_"${model}"__*.jsonl 2>/dev/null | head -1)
  [[ -n $grades ]] || /venv/main/bin/python cc_exp10_tasks.py grade --task "$task" --set "$set" --model "$model"
}
model_runs() {  # model
  local model=$1
  for task in chain arithmetic; do run "$task" dev_c2 "$model" || return 1; done
  for task in chain arithmetic; do run "$task" depth_pilot "$model" || return 1; done
}
#   Arguments, if any: the models to run instead of the full list (finished sets are skipped either way).
default_models=(Ouro-2.6B-Thinking Ouro-2.6B-Thinking-loop{1,2,3} Ouro-1.4B-Thinking Ouro-1.4B-Thinking-loop{1,2,3}
                Nanbeige4.2-3B Nanbeige4.2-3B-loop1 Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3-32B IQuest-40B-Loop-Thinking)
models=("$@")
(( ${#models[@]} )) || models=("${default_models[@]}")
for model in "${models[@]}"; do
  echo "=== $model $(date +%H:%M)"
  model_runs "$model" || echo "FAILED: $model (skipped)"
done
echo "exp10 depth dev done $(date +%H:%M)"
