#!/usr/bin/env bash
# exp04 on 1x H200, models in cfg.EXP04_MODELS order. Per model: the new conditions (cache/exp04/requests.jsonl),
# then the no-prefill rows: requests_repro.jsonl (100 reruns) for the models exp03 generated, requests_none.jsonl
# for the others. Each model's files are graded in the background while the next model generates.
# Cutoff (manifest): if Gemma-4-31B-FP8 finishes after 14:00 UTC (10:00 EDT) on 2026-09-30, Gemma-4-12B is skipped.
#   nohup scripts/h200_exp04.sh > log/exp04/h200_exp04.log 2>&1 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
[[ -f ../results/exp04_prefill/manifest.json ]] || { echo "missing results/exp04_prefill/manifest.json" >&2; exit 1; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }
cutoff=$(date -u -d "2026-09-30T14:00:00Z" +%s)
reuses_exp03=" Qwen3-32B Qwen3.6-27B-FP8 Qwen3.5-9B "
grading=()

generate() {  # model, requests file
  CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp04 --model "$1" --requests "$2"
}

for model in Qwen3-32B Qwen3.6-27B-FP8 Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Gemma-4-12B Qwen3.5-9B; do
  if [[ $model == Gemma-4-12B && $(date -u +%s) -gt $cutoff ]]; then
    stamp "skipping Gemma-4-12B: past the 14:00 UTC cutoff"
    continue
  fi
  if [[ $reuses_exp03 == *" $model "* ]]; then extra=../cache/exp04/requests_repro.jsonl
  else extra=../cache/exp04/requests_none.jsonl; fi
  stamp "$model: new conditions"
  generate "$model" ../cache/exp04/requests.jsonl
  stamp "$model: $(basename "$extra")"
  generate "$model" "$extra"
  stamp "$model generated"
  files=(../cache/exp04/generations/"$model"__card__stream_abort__*.jsonl
         ../cache/exp04/generations/"$model"__card__stream_abort_*__*.jsonl)
  $py cc_grade.py --exp exp04 --generations "${files[@]}" > "../log/exp04/grade_${model}.log" 2>&1 &
  grading+=($!)
done

for pid in "${grading[@]}"; do wait "$pid"; done
stamp "all graded"
$py cc_exp04_analysis.py --run main
stamp "exp04 done"
