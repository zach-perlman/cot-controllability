#!/usr/bin/env bash
# exp06a round 1 after deviations_thinking_off_half.json, on 1x H200. Replaces h200_exp06a.sh from Qwen3.8's grading on:
#   1) wait for the Qwen3.8 round-1 generation h200_exp06a.sh started (pid $1), then grade it
#   2) Gemma-4-31B-FP8, Qwen3-32B: pre-pass, requests (split), thinking-on rows, grade
#   3) the same two models' thinking-off rows on the half items, grade. Each waits while log/exp06a/hold_thinking_off
#      exists (round 2 goes first); a generation killed for round 2 resumes from its finished rows, up to 3 attempts
#   setsid nohup scripts/h200_exp06a_split.sh <qwen3.8 run_sharded pid> > log/exp06a/h200_exp06a_split.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
gens=../cache/exp06a/generations
logs=../log/exp06a
hold=$logs/hold_thinking_off
stamp() { echo "$(date -u +%FT%TZ) $*"; }
grading=()

generate() {  # generate <model> <requests file> <log suffix>
  CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06a --model "$1" --requests "$2" \
    && mv "$logs/generate_$1.shard0of1.log" "$logs/generate_$1_$3.log"
}
grade() {  # grade <model> <part>, in the background
  $py cc_grade.py --exp exp06a --generations "$gens/$1__card__stream_abort_$1$2__"*.jsonl > "$logs/grade_$1$2.log" 2>&1 &
  grading+=($!)
}

qwen38=Qwen3.8-27B-FP8
stamp "$qwen38: waiting for round 1 (pid $1)"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
if ls "$gens/${qwen38}__card__stream_abort_${qwen38}__"*.jsonl > /dev/null 2>&1; then
  mv "$logs/generate_${qwen38}.shard0of1.log" "$logs/generate_${qwen38}_round1.log"
  stamp "$qwen38: round 1 done, grading"; grade "$qwen38" ""
else
  stamp "$qwen38: round 1 output missing ($logs/generate_${qwen38}.shard0of1.log)"
fi

split_models=(Gemma-4-31B-FP8 Qwen3-32B)
for model in "${split_models[@]}"; do
  stamp "$model: pre-pass"
  if ! CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp06a --model "$model" \
      --requests ../cache/exp06a/requests_prepass.jsonl; then
    stamp "$model: pre-pass FAILED ($logs/generate_${model}.shard0of1.log)"; continue
  fi
  mv "$logs/generate_${model}.shard0of1.log" "$logs/generate_${model}_prepass.log"
  if ! $py cc_exp06a.py requests --model "$model" > "$logs/requests_${model}.log" 2>&1; then
    stamp "$model: writing requests FAILED ($logs/requests_${model}.log)"; continue
  fi
  stamp "$model: thinking on"
  if generate "$model" "../cache/exp06a/requests_${model}_thinking_on.jsonl" thinking_on; then
    grade "$model" _thinking_on
  else
    stamp "$model: thinking on FAILED ($logs/generate_${model}.shard0of1.log)"
  fi
done
stamp "thinking-on rows done"

for model in "${split_models[@]}"; do
  requests="../cache/exp06a/requests_${model}_thinking_off_half.jsonl"
  [ -f "$requests" ] || { stamp "$model: no $requests"; continue; }
  for attempt in 1 2 3; do
    while [ -e "$hold" ]; do sleep 30; done
    stamp "$model: thinking off, half items (attempt $attempt)"
    if generate "$model" "$requests" thinking_off_half; then grade "$model" _thinking_off_half; break; fi
    stamp "$model: thinking off stopped or failed ($logs/generate_${model}.shard0of1.log)"
  done
done

for pid in "${grading[@]}"; do wait "$pid" || stamp "a grading job failed ($logs/grade_*.log)"; done
stamp "exp06a round 1 done"
