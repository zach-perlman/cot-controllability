#!/usr/bin/env bash
# exp08 extension, run once (results/exp08_openings_channels/manifest_extension_exp06_rules.json): waits for
# h200_exp08.sh (its pid) to exit, then per model the extension's generation and grading (request files are written
# beforehand), then exp08's analysis v2 with the extension.
#   setsid nohup scripts/h200_exp08_ext.sh <pid of h200_exp08.sh> > log/exp08/h200_exp08_ext.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
wait_pid=${1:?usage: h200_exp08_ext.sh <pid of h200_exp08.sh>}
py=/venv/main/bin/python
logs=../log/exp08
cache=../cache/exp08
items=../cache/exp07/items.jsonl
models=(Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3.6-27B-FP8 Qwen3-32B)
stamp() { echo "$(date -u +%FT%TZ) $*"; }

stamp "waiting for h200_exp08.sh (pid $wait_pid)"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )); do sleep 30; done
for model in "${models[@]}"; do
  requests="$cache/requests_exp06rules_$model.jsonl"
  [ -f "$requests" ] || { stamp "$model: no request file $requests"; continue; }
  ok=0
  for attempt in 1 2 3; do
    stamp "$model exp06rules (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp08 --model "$model" \
        --requests "$requests" --items "$items"; then
      mv "$logs/generate_$model.shard0of1.log" "$logs/generate_exp06rules_$model.log"
      ok=1
      break
    fi
    stamp "$model exp06rules: stopped or failed"
  done
  [ "$ok" = 1 ] || { stamp "$model: giving up"; continue; }
  $py cc_exp08_ext.py grade --model "$model" > "$logs/grade_exp06rules_$model.log" 2>&1 \
    && stamp "$model graded" || stamp "$model: grading failed ($logs/grade_exp06rules_$model.log)"
done
$py cc_exp08_analysis.py --run v2 --skip-missing > "$logs/analysis_v2.log" 2>&1 \
  && stamp "analysis written (results/exp08_openings_channels/analysis/v2)" || stamp "analysis failed ($logs/analysis_v2.log)"
stamp "done"
