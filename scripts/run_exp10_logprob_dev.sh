#!/bin/bash
# exp10 depth stage, dev bank: no-CoT answer log-probabilities (exp10_answer_logprob) for every loop dial and control,
# one engine at a time. Ouro runs under vLLM 0.25.1; Huginn under transformers 4.56 (/workspace/.huginn-libs: its
# custom code predates transformers 5); the rest under vLLM 0.30. A model that fails is logged and skipped.
#   nohup scripts/run_exp10_logprob_dev.sh > log/exp10/logprob_dev.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
run() {  # model
  local model=$1
  if [[ $model == Huginn-* ]]; then
    PYTHONPATH=/workspace/.huginn-libs /venv/main/bin/python exp10_answer_logprob.py requests --model "$model" --bank dev \
      && PYTHONPATH=/workspace/.huginn-libs /venv/main/bin/python exp10_answer_logprob.py score --model "$model" --bank dev
    return
  fi
  local venv=/workspace/.venv-vllm
  [[ $model == Ouro-* ]] && venv=/workspace/.venv-vllm025
  /venv/main/bin/python exp10_answer_logprob.py requests --model "$model" --bank dev \
    && VLLM_VENV=$venv ./vllm_python.sh exp10_answer_logprob.py score --model "$model" --bank dev
}
models=(Huginn-0125-steps4 Huginn-0125-steps8 Huginn-0125-steps16 Huginn-0125 Huginn-0125-steps64
        Ouro-2.6B-Thinking Ouro-2.6B-Thinking-loop{1,2,3} Ouro-1.4B-Thinking Ouro-1.4B-Thinking-loop{1,2,3}
        Nanbeige4.2-3B Nanbeige4.2-3B-loop1 Gemma-4-E2B Gemma-4-E4B Gemma-4-12B Gemma-4-31B-FP8
        IQuest-40B-Loop-Thinking)
for model in "${models[@]}"; do
  echo "=== $model $(date +%H:%M)"
  run "$model" || echo "FAILED: $model (skipped)"
done
echo "exp10 logprob dev done $(date +%H:%M)"
