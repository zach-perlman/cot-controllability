#!/usr/bin/env bash
# exp06a round 2 as decided in deviations_round2_decision.json, on 1x H200. Replaces h200_exp06a_round2.sh (stopped).
#   1) per model: requests_<model>_round2.jsonl (resumes Qwen3.8's partial rows), then _round2_added; grade each
#   2) per model: _round2_thinking_off_half; grade. Each generation gets up to 3 attempts (finished rows resume)
#   setsid nohup scripts/h200_exp06a_round2_decided.sh > log/exp06a/h200_exp06a_round2_decided.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp06a/generations
logs=../log/exp06a
models=(Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3-32B)
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

run_part() {  # run_part <model> <part>: generate (up to 3 attempts), then grade in the background
  local requests="../cache/exp06a/requests_$1$2.jsonl"
  for attempt in 1 2 3; do
    stamp "$1$2 (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06a --model "$1" --requests "$requests"; then
      mv "$logs/generate_$1.shard0of1.log" "$logs/generate_$1$2.log"
      $py cc_grade.py --exp exp06a --generations "$gens/$1__card__stream_abort_$1$2__"*.jsonl \
        > "$logs/grade_$1$2.log" 2>&1 &
      grading+=($!)
      return 0
    fi
    stamp "$1$2: stopped or failed ($logs/generate_$1.shard0of1.log)"
  done
  return 1
}

for model in "${models[@]}"; do
  run_part "$model" _round2
  run_part "$model" _round2_added
done
stamp "round 2 thinking-on rows done"
for model in "${models[@]}"; do
  run_part "$model" _round2_thinking_off_half
done

for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed ($logs/grade_*_round2*.log)"; done
stamp "exp06a round 2 done"
