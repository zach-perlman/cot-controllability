#!/usr/bin/env bash
# exp06 extension: Qwen3.5-9B as a second held-out model (results/exp06_prompt_prefill/manifest_extension_qwen35.json).
#   1) pre-pass (requests_prepass.jsonl, the 8 few-shot source rows), then its request files and rendering checks
#   2) thinking-on grid, then thinking-off rows (up to 3 attempts each; graded in the background)
#   3) analysis run "with_qwen35" (all 5 models) and story figures v2
#   setsid nohup scripts/h200_exp06_qwen35.sh > log/exp06/h200_exp06_qwen35.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp06/generations
logs=../log/exp06
model=Qwen3.5-9B
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

generate() {  # generate <requests file> <log name>: up to 3 attempts
  for attempt in 1 2 3; do
    stamp "$2 (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06 --model "$model" --requests "$1"; then
      mv "$logs/generate_$model.shard0of1.log" "$logs/generate_$2.log"
      return 0
    fi
    stamp "$2: stopped or failed ($logs/generate_$model.shard0of1.log)"
  done
  return 1
}

until $py cc_download.py --check "$model"; do stamp "waiting for $model weights"; sleep 60; done
generate ../cache/exp06/requests_prepass.jsonl "${model}_prepass" || exit 1
$py cc_exp06.py requests --model "$model" > "$logs/requests_$model.log" 2>&1 \
  || { stamp "request files failed ($logs/requests_$model.log)"; exit 1; }
stamp "request files written"
for part in _main _thinking_off; do
  generate "../cache/exp06/requests_$model$part.jsonl" "$model$part" || exit 1
  $py cc_grade.py --exp exp06 --generations "$gens/${model}__card__stream_abort_$model${part}__"*.jsonl \
    > "$logs/grade_$model$part.log" 2>&1 &
  grading+=($!)
done
for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed ($logs/grade_*.log)"; done
stamp "$model generation and grading done"
$py cc_exp06_analysis.py --run with_qwen35 > "$logs/analysis_with_qwen35.log" 2>&1 \
  && stamp "analysis written (run with_qwen35)" || { stamp "analysis failed ($logs/analysis_with_qwen35.log)"; exit 1; }
$py cc_story_figures.py --run v2 --analysis-run with_qwen35 > "$logs/story_v2.log" 2>&1 \
  && stamp "story figures written (v2)" || stamp "story figures failed ($logs/story_v2.log)"
