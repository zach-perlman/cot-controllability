#!/usr/bin/env bash
# exp06 extension: thinking-off rows with an opening sentence (results/exp06_prompt_prefill/
# manifest_extension_off_prefill.json), all 5 models.
#   1) every model's request file and rendering checks (CPU), before any generation
#   2) per model: generate (up to 3 attempts), grade in the background
#   3) analysis run "with_off_prefill" and story figures v3
#   setsid nohup scripts/h200_exp06_off_prefill.sh > log/exp06/h200_exp06_off_prefill.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp06/generations
logs=../log/exp06
part=_thinking_off_prefill
models=(Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3-32B Qwen3.6-27B-FP8 Qwen3.5-9B)
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

for model in "${models[@]}"; do
  [ -e "../results/exp06_prompt_prefill/requests_record_$model$part.json" ] && continue
  $py cc_exp06.py requests-off-prefill --model "$model" > "$logs/requests_$model$part.log" 2>&1 \
    || { stamp "$model: request files failed ($logs/requests_$model$part.log)"; exit 1; }
done
stamp "request files written"

for model in "${models[@]}"; do
  ok=0
  for attempt in 1 2 3; do
    stamp "$model$part (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06 --model "$model" \
        --requests "../cache/exp06/requests_$model$part.jsonl"; then
      mv "$logs/generate_$model.shard0of1.log" "$logs/generate_$model$part.log"
      ok=1
      break
    fi
    stamp "$model$part: stopped or failed ($logs/generate_$model.shard0of1.log)"
  done
  [ "$ok" = 1 ] || exit 1
  $py cc_grade.py --exp exp06 --generations "$gens/${model}__card__stream_abort_$model${part}__"*.jsonl \
    > "$logs/grade_$model$part.log" 2>&1 &
  grading+=($!)
done
for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed ($logs/grade_*.log)"; done
stamp "generation and grading done"
$py cc_exp06_analysis.py --run with_off_prefill > "$logs/analysis_with_off_prefill.log" 2>&1 \
  && stamp "analysis written (run with_off_prefill)" \
  || { stamp "analysis failed ($logs/analysis_with_off_prefill.log)"; exit 1; }
$py cc_story_figures.py --run v3 --analysis-run with_off_prefill > "$logs/story_v3.log" 2>&1 \
  && stamp "story figures written (v3)" || stamp "story figures failed ($logs/story_v3.log)"
