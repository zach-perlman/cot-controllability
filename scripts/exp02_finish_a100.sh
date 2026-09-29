#!/usr/bin/env bash
# Finish exp02_prompt_grid on the 8x A100-40GB instance (results/exp02_prompt_grid/deviations_a100.json):
# Qwen3.6-27B chunk 2 of 2 as 4 shards x TP 2, the bf16 Qwen3-32B precision check at TP 4, then grade, judge (TP 4),
# analyze and build the verification sheet, as exp02_chain.sh does. Log: log/exp02/finish_a100.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -f ../results/exp02_prompt_grid/deviations_a100.json ]] || { echo "write deviations_a100.json first" >&2; exit 1; }

./run_sharded.sh --tp 2 -- --exp exp02 --model Qwen3.6-27B

# The precision check and the judge need their weights; the downloader may still be running.
until /venv/main/bin/python cc_download.py --check Qwen3-32B-bf16 Qwen3.8-27B; do
  echo "waiting for the bf16 and judge weights"; sleep 120
done

CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_sharded.sh --tp 4 -- --exp exp02 --model Qwen3-32B-bf16 --prompts baseline \
  --modes word_suppression lowercase_thinking \
  --max-model-len "$(/venv/main/bin/python -c 'import cc_config; print(cc_config.PRECISION_CHECK_MAX_MODEL_LEN)')"

/venv/main/bin/python cc_grade.py --exp exp02
CUDA_VISIBLE_DEVICES=0,1,2,3 CC_TENSOR_PARALLEL=4 ./vllm_python.sh cc_judge.py --exp exp02
/venv/main/bin/python cc_analysis.py --exp exp02 --run grid
/venv/main/bin/python cc_verification.py build --exp exp02
echo "exp02 done"
