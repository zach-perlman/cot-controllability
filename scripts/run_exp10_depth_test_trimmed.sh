#!/bin/bash
# exp10 depth stage, test (results/exp10_hide_what_you_need/manifest_depth.json), one engine at a time:
#   1 no-CoT answer log-probabilities (exp10_answer_logprob) on the test bank for every model on disk (Qwen3.5-2B: dev too)
#   2 depth_test_direct for IQuest-40B-Loop-Thinking
#   3 once every IQuest-Loop output exists: delete its weights (human-approved), download IQuest-40B-Thinking and
#     Qwen3.5-4B, then their sets and scores (dev and test)
#   4 depth_test (all 10 conditions) for Qwen3.8 and Nanbeige; depth_test_trimmed (no C4, no Code) for Ouro-2.6B and
#     Gemma-4-31B (manifest_depth.json deviation of 2026-10-07 20:10 UTC)
#   5 depth_test_loops for Ouro-2.6B at 2 and 3 passes
#   6 back up exp10's caches, results and logs to the private HF dataset
# A model that fails is logged and skipped; generation resumes from finished rows if rerun.
#   setsid nohup scripts/run_exp10_depth_test_trimmed.sh <pid to wait for> > log/exp10/depth_test.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
LOOP=IQuest-40B-Loop-Thinking
TWIN=IQuest-40B-Thinking
REPO_ID=zachperlman20/cot-controllability-cache
stamp() { date -u +%FT%TZ; }
venv_of() { [[ $1 == Ouro-* ]] && echo /workspace/.venv-vllm025 || echo /workspace/.venv-vllm; }

gen() {  # task set model
  local task=$1 set=$2 model=$3
  local dir=../cache/exp10/$task requests=../cache/exp10/$task/requests_${set}_${model}.jsonl venv
  venv=$(venv_of "$model")
  [[ -e $requests ]] || /venv/main/bin/python cc_exp10_tasks.py requests --task "$task" --set "$set" --model "$model" \
    || return 1
  # A shard that finished can still exit nonzero in exp03_exit's teardown; the merge checks every row.
  VLLM_VENV=$venv ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items "$dir/items_test.jsonl" --shard 0/1 || echo "shard exited nonzero; the merge checks completeness"
  VLLM_VENV=$venv ./vllm_python.sh cc_generate_abort.py --exp exp10 --model "$model" --requests "$requests" \
    --items "$dir/items_test.jsonl" --merge-shards 1 || return 1
  ls "$dir"/grades/"${model}"__card__stream_abort_"${set}"_"${model}"__*.jsonl > /dev/null 2>&1 \
    || /venv/main/bin/python cc_exp10_tasks.py grade --task "$task" --set "$set" --model "$model"
}
gen_both() {  # set model
  echo "=== gen $1 $2 $(stamp)"
  for task in chain arithmetic; do gen "$task" "$1" "$2" || { echo "FAILED: gen $1 $2 $task"; return 1; }; done
}
score() {  # bank model
  echo "=== score $1 $2 $(stamp)"
  if [[ $2 == Huginn-* ]]; then
    PYTHONPATH=/workspace/.huginn-libs /venv/main/bin/python exp10_answer_logprob.py requests --model "$2" --bank "$1" \
      && PYTHONPATH=/workspace/.huginn-libs /venv/main/bin/python exp10_answer_logprob.py score --model "$2" --bank "$1"
  else
    /venv/main/bin/python exp10_answer_logprob.py requests --model "$2" --bank "$1" \
      && VLLM_VENV=$(venv_of "$2") ./vllm_python.sh exp10_answer_logprob.py score --model "$2" --bank "$1"
  fi || echo "FAILED: score $1 $2"
}
have() { compgen -G "$1" > /dev/null; }

wait_pid=${1:-}
[[ -n $wait_pid ]] && while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
echo "$(stamp) start"

# 1 test-bank scores for every model on disk
score dev Qwen3.5-2B
for model in Huginn-0125-steps4 Huginn-0125-steps8 Huginn-0125-steps16 Huginn-0125 Huginn-0125-steps64 \
             Ouro-2.6B-Thinking Ouro-2.6B-Thinking-loop{1,2,3} Ouro-1.4B-Thinking Ouro-1.4B-Thinking-loop{1,2,3} \
             Nanbeige4.2-3B Nanbeige4.2-3B-loop1 Gemma-4-E2B Gemma-4-E4B Gemma-4-12B Gemma-4-31B-FP8 \
             Qwen3.5-2B Qwen3.8-27B-FP8 "$LOOP"; do
  score test "$model"
done

# 2 the looped IQuest's no-CoT generations
gen_both depth_test_direct "$LOOP"

# 3 swap the IQuest pair's weights (only once every output of the looped one exists)
loop_done=1
for task in chain arithmetic; do
  have "../cache/exp10/$task/grades/${LOOP}__card__stream_abort_depth_test_direct_${LOOP}__*.jsonl" || loop_done=0
  for bank in dev test; do have "../cache/exp10/$task/logprob/${LOOP}__${bank}__*.jsonl" || loop_done=0; done
done
if (( loop_done )); then
  echo "$(stamp) every $LOOP output exists; deleting its weights"
  rm -rf /workspace/.hf_home/hub/models--IQuestLab--IQuest-Coder-V1-40B-Loop-Thinking
  if /venv/main/bin/python cc_download.py "$TWIN" Qwen3.5-4B; then
    gen_both depth_test_direct "$TWIN"
    for bank in dev test; do score "$bank" "$TWIN"; score "$bank" Qwen3.5-4B; done
  else
    echo "FAILED: download $TWIN Qwen3.5-4B"
  fi
else
  echo "$(stamp) NOT deleting $LOOP: some of its outputs are missing; skipping the twin"
fi

# 4 the full depth set for the headline models, fastest first
for model in Qwen3.8-27B-FP8 Nanbeige4.2-3B; do gen_both depth_test "$model"; done
for model in Ouro-2.6B-Thinking Gemma-4-31B-FP8; do gen_both depth_test_trimmed "$model"; done

# 5 Ouro's loop dial on the hiding condition
for model in Ouro-2.6B-Thinking-loop2 Ouro-2.6B-Thinking-loop3; do gen_both depth_test_loops "$model"; done
echo "$(stamp) exp10 depth test done"

# 6 backup (the dataset must be private)
private=$(timeout 120 /venv/main/bin/python -c \
  "from huggingface_hub import HfApi; print(HfApi().dataset_info('${REPO_ID}').private)" 2>/dev/null)
if [[ "$private" != "True" ]]; then
  echo "$(stamp) HF dataset not confirmed private; not uploading"
else
  for d in cache/exp10 results/exp10_hide_what_you_need log/exp10; do
    timeout 3600 /venv/main/bin/hf upload "$REPO_ID" "../$d" "$d" --repo-type dataset \
      --commit-message "exp10 depth test backup $(stamp)" > /dev/null 2>&1 \
      && echo "$(stamp) uploaded $d" || echo "$(stamp) upload FAILED $d"
  done
fi
