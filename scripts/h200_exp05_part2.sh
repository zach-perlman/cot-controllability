#!/usr/bin/env bash
# exp05 part 2 on 1x H200: the dose rows of cfg.EXP05_ADDED_MODELS (results/exp05_dose/manifest_extension.json
# run_order), after part 1 (h200_exp05.sh: cfg.EXP05_MODELS + the effort check). Waits until part 1's last
# generation file exists and the GPU is free, then per model
#   models with exp04 rows:  requests_<model>.jsonl
#   new models:              requests_base_<model>.jsonl, the base gate + dose requests (cc_exp05.py requests-new),
#                            then requests_<model>.jsonl
# Each generation file is graded in the background. A model that fails is reported and skipped; the next one runs.
# Analysis (deviations_one_analysis.json): 'interim_without_glm' after Qwen3.6-35B-A3B-FP8, then 'main' over every
# exp05 model once part 1 has exited and all grading is done. (Part 1's own closing 'main' call exits without
# writing, since the added models are not generated yet.)
#   nohup scripts/h200_exp05_part2.sh > log/exp05/h200_exp05_part2.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp05/generations
[[ -f ../results/exp05_dose/deviations_one_analysis.json ]] || { echo "missing deviations_one_analysis.json" >&2; exit 1; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

stamp "waiting for part 1's last generation (Qwen3.8-27B-FP8-low) and a free GPU"
until compgen -G "$gens/Qwen3.8-27B-FP8-low__card__stream_abort_effort__*.jsonl" > /dev/null; do sleep 60; done
until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 4000 )); do sleep 15; done
stamp "GPU free"

generate() {  # model, requests file; grades it in the background
  local model=$1 requests=$2 suffix
  suffix=$(basename "$requests" .jsonl); suffix=${suffix#requests}
  stamp "$model: $(basename "$requests")"
  if ! CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp05 --model "$model" --requests "$requests"; then
    stamp "$model: generation of $(basename "$requests") FAILED (log/exp05/generate_${model}.shard0of1.log)"
    return 1
  fi
  cp "../log/exp05/generate_${model}.shard0of1.log" "../log/exp05/generate_${model}${suffix}.log"
  $py cc_grade.py --exp exp05 --generations "$gens/${model}__card__stream_abort${suffix}__"*.jsonl \
    > "../log/exp05/grade_${model}${suffix}.log" 2>&1 &
  grading+=($!)
}

wait_grading() {  # this script's grading jobs, then part 1 (whose last step is grading and its no-op analysis)
  for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed (log/exp05/grade_*.log)"; done
  grading=()
  while pgrep -f '^bash scripts/h200_exp05\.sh$' > /dev/null; do sleep 30; done
}

new_model() {  # base rows, gate, dose requests, dose rows
  local model=$1
  generate "$model" "../cache/exp05/requests_base_${model}.jsonl" || return 1
  if ! $py cc_exp05.py requests-new --model "$model"; then
    stamp "$model: base gate failed or dose requests not written; skipping its dose rows"
    return 1
  fi
  generate "$model" "../cache/exp05/requests_${model}.jsonl"
}

for model in Gemma-4-31B-FP8 Qwen3.5-9B Gemma-4-12B; do
  generate "$model" "../cache/exp05/requests_${model}.jsonl"
done
new_model Qwen3.6-35B-A3B-FP8

wait_grading
stamp "analysis interim_without_glm"
$py cc_exp05_analysis.py --run interim_without_glm --skip-models GLM-4.7-Flash --skip-missing \
  > ../log/exp05/analysis_interim_without_glm.log 2>&1 &
analysis_interim=$!

new_model GLM-4.7-Flash
wait_grading
wait "$analysis_interim" || stamp "analysis interim_without_glm FAILED (log/exp05/analysis_interim_without_glm.log)"
stamp "analysis main"
$py cc_exp05_analysis.py --run main --skip-missing > ../log/exp05/analysis_main.log 2>&1 \
  || stamp "analysis main FAILED (log/exp05/analysis_main.log)"
stamp "exp05 done"
