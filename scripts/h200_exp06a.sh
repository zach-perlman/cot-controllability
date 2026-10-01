#!/usr/bin/env bash
# exp06a round 1 on 1x H200 (results/exp06a_prompt_screen/manifest.json). Per model, in cfg.EXP06A_MODELS order:
#   1) generate requests_prepass.jsonl (few-shot source traces, style-guide turn 1)
#   2) write requests_<model>.jsonl from that pre-pass (cc_exp06a.py requests --model)
#   3) generate requests_<model>.jsonl, then grade it (in the background, while the next model runs)
#   nohup scripts/h200_exp06a.sh > log/exp06a/h200_exp06a.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp06a/generations
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

for model in Qwen3.8-27B-FP8 Gemma-4-31B-FP8 Qwen3-32B; do
  stamp "$model: pre-pass"
  if ! CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06a --model "$model" \
      --requests ../cache/exp06a/requests_prepass.jsonl; then
    stamp "$model: pre-pass FAILED (log/exp06a/generate_${model}.shard0of1.log)"; continue
  fi
  mv "../log/exp06a/generate_${model}.shard0of1.log" "../log/exp06a/generate_${model}_prepass.log"
  if ! $py cc_exp06a.py requests --model "$model" > "../log/exp06a/requests_${model}.log" 2>&1; then
    stamp "$model: writing requests FAILED (log/exp06a/requests_${model}.log)"; continue
  fi
  stamp "$model: round 1"
  if ! CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06a --model "$model" \
      --requests "../cache/exp06a/requests_${model}.jsonl"; then
    stamp "$model: round 1 FAILED (log/exp06a/generate_${model}.shard0of1.log)"; continue
  fi
  mv "../log/exp06a/generate_${model}.shard0of1.log" "../log/exp06a/generate_${model}_round1.log"
  $py cc_grade.py --exp exp06a --generations "$gens/${model}__card__stream_abort_${model}__"*.jsonl \
    > "../log/exp06a/grade_${model}.log" 2>&1 &
  grading+=($!)
done

for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed (log/exp06a/grade_*.log)"; done
stamp "exp06a round 1 done"
