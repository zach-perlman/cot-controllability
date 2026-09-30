#!/usr/bin/env bash
# exp05 on 1x H200 (results/exp05_dose/manifest.json run_order): the dose rows of each model in cfg.EXP05_MODELS,
# then the effort check (Qwen3.8 at xhigh and low on requests_effort.jsonl). Each generation file is graded in the
# background while the next one generates; the analysis runs at the end.
#   nohup scripts/h200_exp05.sh > log/exp05/h200_exp05.log 2>&1 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
[[ -f ../results/exp05_dose/manifest.json ]] || { echo "missing results/exp05_dose/manifest.json" >&2; exit 1; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

run() {  # model, requests file
  local model=$1 requests=$2
  stamp "$model: $(basename "$requests")"
  CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp05 --model "$model" --requests "$requests"
  local suffix; suffix=$(basename "$requests" .jsonl); suffix=${suffix#requests}
  $py cc_grade.py --exp exp05 --generations ../cache/exp05/generations/"$model"__card__stream_abort"$suffix"__*.jsonl \
    > "../log/exp05/grade_${model}${suffix}.log" 2>&1 &
  grading+=($!)
}

for model in Qwen3-32B Qwen3.6-27B-FP8 Qwen3.8-27B-FP8; do
  run "$model" "../cache/exp05/requests_${model}.jsonl"
done
for model in Qwen3.8-27B-FP8-xhigh Qwen3.8-27B-FP8-low; do
  run "$model" ../cache/exp05/requests_effort.jsonl
done

for pid in "${grading[@]}"; do wait "$pid"; done
stamp "all graded"
$py cc_exp05_analysis.py --run main
stamp "exp05 done"
