#!/usr/bin/env bash
# exp06a round 2 (deviations_round2.json) on 1x H200: generate and grade requests_<model>_round2.jsonl per model,
# then release the paused thinking-off backfill of h200_exp06a_split.sh (log/exp06a/hold_thinking_off).
#   touch log/exp06a/hold_thinking_off; <stop the running backfill generation>
#   setsid nohup scripts/h200_exp06a_round2.sh > log/exp06a/h200_exp06a_round2.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp06a/generations
logs=../log/exp06a
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3-32B; do
  stamp "$model: round 2"
  if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06a --model "$model" \
      --requests "../cache/exp06a/requests_${model}_round2.jsonl"; then
    mv "$logs/generate_${model}.shard0of1.log" "$logs/generate_${model}_round2.log"
    $py cc_grade.py --exp exp06a --generations "$gens/${model}__card__stream_abort_${model}_round2__"*.jsonl \
      > "$logs/grade_${model}_round2.log" 2>&1 &
    grading+=($!)
  else
    stamp "$model: round 2 FAILED ($logs/generate_${model}.shard0of1.log)"
  fi
done

rm -f "$logs/hold_thinking_off"
stamp "round 2 generated; thinking-off backfill released"
for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed ($logs/grade_*_round2.log)"; done
stamp "exp06a round 2 done"
