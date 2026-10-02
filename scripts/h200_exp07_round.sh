#!/usr/bin/env bash
# One exp07 search or validation round on the search models (results/exp07_prompt_search/manifest.json):
# request files, generation, grading, scoring into search_log.jsonl.
#   setsid nohup scripts/h200_exp07_round.sh ROUND SPLIT CANDIDATES [WAIT_FOR_LOG] > log/exp07/h200_exp07_ROUND.log 2>&1 &
# CANDIDATES: comma-separated exp07_candidates names. WAIT_FOR_LOG: start after that log's last line is "... done".
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
round=$1 split=$2 candidates=$3 wait_for=${4:-}
py=/venv/main/bin/python
logs=../log/exp07
items=../cache/exp07/items.jsonl
search_models=(Qwen3.8-27B-FP8 Gemma-4-31B-FP8)
stamp() { echo "$(date -u +%FT%TZ) $*"; }

if [ -n "$wait_for" ]; then
  until tail -1 "$wait_for" 2>/dev/null | grep -q " done$"; do sleep 30; done
fi
until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )); do sleep 30; done
stamp "GPU free"

for model in "${search_models[@]}"; do
  $py cc_exp07.py requests --model "$model" --name "$round" --split "$split" --candidates "$candidates" \
    > "$logs/requests_${round}_$model.log" 2>&1 \
    || { stamp "$model: request file failed ($logs/requests_${round}_$model.log)"; exit 1; }
done
stamp "request files written"

for model in "${search_models[@]}"; do
  ok=0
  for attempt in 1 2 3; do
    stamp "$model $round (attempt $attempt)"
    if CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp07 --model "$model" \
        --requests "../cache/exp07/requests_${round}_$model.jsonl" --items "$items"; then
      mv "$logs/generate_$model.shard0of1.log" "$logs/generate_${round}_$model.log"
      ok=1
      break
    fi
    stamp "$model $round: stopped or failed"
  done
  [ "$ok" = 1 ] || exit 1
  $py cc_exp07.py grade --model "$model" --name "$round" > "$logs/grade_${round}_$model.log" 2>&1 \
    || { stamp "$model: grading failed ($logs/grade_${round}_$model.log)"; exit 1; }
done
$py cc_exp07.py score --name "$round" --log > "$logs/score_$round.log" 2>&1 \
  && stamp "round scored ($logs/score_$round.log)" || stamp "scoring failed ($logs/score_$round.log)"
stamp "done"
