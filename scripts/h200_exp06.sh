#!/usr/bin/env bash
# exp06 on 1x H200 (results/exp06_prompt_prefill/manifest.json). Every request file is already written.
#   1) thinking-on grid (requests_<model>_main.jsonl) of the 3 screened models, then their thinking-off rows
#   2) the held-out model (Qwen3.6-27B-FP8): main, then thinking off
#   3) when every grade file is written: the analysis (run "overnight"), if cc_exp06_analysis.py exists by then
# Each generation gets up to 3 attempts (finished rows resume); each finished file is graded in the background.
#   setsid nohup scripts/h200_exp06.sh > log/exp06/h200_exp06.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp06/generations
logs=../log/exp06
screened=(Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3-32B)
held_out=Qwen3.6-27B-FP8
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

run_part() {  # run_part <model> <part>: generate (up to 3 attempts), then grade in the background
  local requests="../cache/exp06/requests_$1$2.jsonl"
  for attempt in 1 2 3; do
    stamp "$1$2 (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06 --model "$1" --requests "$requests"; then
      mv "$logs/generate_$1.shard0of1.log" "$logs/generate_$1$2.log"
      $py cc_grade.py --exp exp06 --generations "$gens/$1__card__stream_abort_$1$2__"*.jsonl \
        > "$logs/grade_$1$2.log" 2>&1 &
      grading+=($!)
      return 0
    fi
    stamp "$1$2: stopped or failed ($logs/generate_$1.shard0of1.log)"
  done
  return 1
}

for model in "${screened[@]}"; do run_part "$model" _main; done
stamp "screened models: thinking-on rows done"
for model in "${screened[@]}"; do run_part "$model" _thinking_off; done
stamp "screened models: thinking-off rows done"
run_part "$held_out" _main
run_part "$held_out" _thinking_off

for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed ($logs/grade_*.log)"; done
stamp "exp06 generation and grading done"
if [[ -f cc_exp06_analysis.py ]]; then
  $py cc_exp06_analysis.py --run overnight > "$logs/analysis_overnight.log" 2>&1 \
    && stamp "analysis written (run overnight)" || stamp "analysis failed ($logs/analysis_overnight.log)"
fi
