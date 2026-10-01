#!/usr/bin/env bash
# exp05 part 3 on 1x H200 (deviations_gate_cap_truncation.json): the dose rows of the two new models whose base gate
# failed only on cap truncation (their requests were written with cc_exp05.py requests-new --gate cap-truncation),
# then analysis 'main_8_models' over every exp05 model. Each generation file is graded right after it is written.
#   nohup scripts/h200_exp05_part3.sh > log/exp05/h200_exp05_part3.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp05/generations
[[ -f ../results/exp05_dose/deviations_gate_cap_truncation.json ]] || { echo "missing deviation file" >&2; exit 1; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

for model in Qwen3.6-35B-A3B-FP8 GLM-4.7-Flash; do
  stamp "$model: requests_${model}.jsonl"
  if ! CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp05 --model "$model" \
      --requests "../cache/exp05/requests_${model}.jsonl"; then
    stamp "$model: generation FAILED (log/exp05/generate_${model}.shard0of1.log)"
    continue
  fi
  cp "../log/exp05/generate_${model}.shard0of1.log" "../log/exp05/generate_${model}_${model}.log"
  $py cc_grade.py --exp exp05 --generations "$gens/${model}__card__stream_abort_${model}__"*.jsonl \
    > "../log/exp05/grade_${model}_${model}.log" 2>&1 &
  grading+=($!)
done

for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed (log/exp05/grade_*.log)"; done
stamp "analysis main_8_models"
$py cc_exp05_analysis.py --run main_8_models --skip-missing > ../log/exp05/analysis_main_8_models.log 2>&1 \
  || stamp "analysis main_8_models FAILED (log/exp05/analysis_main_8_models.log)"
stamp "exp05 part 3 done"
