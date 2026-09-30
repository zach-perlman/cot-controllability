#!/usr/bin/env bash
# exp05 extension on 1x H200 (results/exp05_dose/manifest_extension.json run_order), after the pre-registered run
# (h200_exp05.sh): waits until that run's last generation file exists and the GPU is free, then per model in
# cfg.EXP05_EXTENSION_MODELS order
#   models with exp04 rows:  requests_<model>.jsonl
#   new models:              requests_base_<model>.jsonl, the base gate + dose requests (cc_exp05.py requests-new),
#                            then requests_<model>.jsonl
# Each generation file is graded in the background. A model that fails is reported and skipped; the next one runs.
# Analysis 'extension_a' runs after Qwen3.6-35B-A3B-FP8 (all models but GLM-4.7-Flash), 'extension' at the end.
#   nohup scripts/h200_exp05_ext.sh > log/exp05/h200_exp05_ext.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp05/generations
[[ -f ../results/exp05_dose/manifest_extension.json ]] || { echo "missing manifest_extension.json" >&2; exit 1; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

stamp "waiting for the pre-registered run's last generation (Qwen3.8-27B-FP8-low) and a free GPU"
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

wait_grading() {
  for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed (log/exp05/grade_*.log)"; done
  grading=()
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
stamp "analysis extension_a"
$py cc_exp05_analysis.py --run extension_a --models all --skip-models GLM-4.7-Flash \
  > ../log/exp05/analysis_extension_a.log 2>&1 &
analysis_a=$!

new_model GLM-4.7-Flash
wait_grading
wait "$analysis_a" || stamp "analysis extension_a FAILED (log/exp05/analysis_extension_a.log)"
stamp "analysis extension"
$py cc_exp05_analysis.py --run extension --models all --skip-missing > ../log/exp05/analysis_extension.log 2>&1 \
  || stamp "analysis extension FAILED (log/exp05/analysis_extension.log)"
stamp "exp05 extension done"
