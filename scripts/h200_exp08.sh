#!/usr/bin/env bash
# exp08, run once (results/exp08_openings_channels/manifest.json): per model, its request file, generation and
# grading; then the analysis.
#   setsid nohup scripts/h200_exp08.sh > log/exp08/h200_exp08.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
logs=../log/exp08
cache=../cache/exp08
items=../cache/exp07/items.jsonl
models=(Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3.6-27B-FP8 Qwen3-32B)
mkdir -p "$logs"
stamp() { echo "$(date -u +%FT%TZ) $*"; }

until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )); do sleep 30; done
for model in "${models[@]}"; do
  if [ ! -f "$cache/requests_main_$model.jsonl" ]; then
    $py cc_exp08.py requests --model "$model" > "$logs/requests_$model.log" 2>&1 \
      || { stamp "$model: request file failed ($logs/requests_$model.log)"; continue; }
  fi
  ok=0
  for attempt in 1 2 3; do
    stamp "$model main (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp08 --model "$model" \
        --requests "$cache/requests_main_$model.jsonl" --items "$items"; then
      mv "$logs/generate_$model.shard0of1.log" "$logs/generate_main_$model.log"
      ok=1
      break
    fi
    stamp "$model main: stopped or failed"
  done
  [ "$ok" = 1 ] || { stamp "$model: giving up"; continue; }
  $py cc_exp08.py grade --model "$model" > "$logs/grade_$model.log" 2>&1 \
    && stamp "$model graded" || stamp "$model: grading failed ($logs/grade_$model.log)"
done
$py cc_exp08_analysis.py --run v1 --skip-missing > "$logs/analysis_v1.log" 2>&1 \
  && stamp "analysis written (results/exp08_openings_channels/analysis/v1)" || stamp "analysis failed ($logs/analysis_v1.log)"
stamp "done"
