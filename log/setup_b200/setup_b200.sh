#!/usr/bin/env bash
# One-off setup of this B200 instance for exp09; each step logs to /workspace/setup_logs/.
set -uo pipefail
REPO=/workspace/cot-controllability
LOGS=/workspace/setup_logs
mkdir -p "$LOGS" /workspace/.hf_home/hub

git clone -q --recurse-submodules https://github.com/zach-perlman/cot-controllability.git "$REPO" > "$LOGS/clone.log" 2>&1
cd "$REPO" || exit 1
git config user.name "zach-perlman"
git config user.email "zachperlman20@gmail.com"

# CUDA forward-compat libs (driver 580 supports CUDA 13.0; the vLLM env's torch is cu132). Not a driver package.
(apt-get update -qq && apt-get install -y -qq cuda-compat-13-2) > "$LOGS/cuda_compat.log" 2>&1 &

(uv venv /workspace/.venv-vllm --python 3.12 --managed-python --seed &&
 UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=5 uv pip install --python /workspace/.venv-vllm/bin/python \
   -r log/setup_h200_2/freeze_vllm.txt --torch-backend=auto && echo VLLM_ENV_DONE) > "$LOGS/setup_vllm.log" 2>&1 &

(grep -v -E " @ |^(torch|torchvision|torchaudio|torchcodec|triton|nvidia-[a-z0-9-]+)==" log/setup_h200_2/freeze_main.txt > /tmp/main_pins.txt &&
 UV_HTTP_TIMEOUT=60 uv pip install --python /venv/main/bin/python -r /tmp/main_pins.txt && echo MAIN_ENV_DONE) > "$LOGS/setup_main.log" 2>&1 &

(/venv/main/bin/hf download zachperlman20/cot-controllability-cache --repo-type dataset --local-dir /workspace/hf_restore &&
 for d in cache results log; do cp -r --update=none /workspace/hf_restore/$d/. $d/; done && echo RESTORE_DONE) > "$LOGS/restore.log" 2>&1 &

wait_for_hub() { /venv/main/bin/python -c "import huggingface_hub" 2>/dev/null; }
until wait_for_hub; do sleep 5; done
/venv/main/bin/python scripts/cc_download.py Qwen3.6-27B-FP8 Gemma-4-31B-FP8 Qwen3.8-27B-FP8 Qwen3-32B \
  Gemma-4-12B-FP8 Qwen3.6-35B-A3B-FP8 GLM-4.7-Flash-FP8 > "$LOGS/downloads.log" 2>&1 &

wait
echo SETUP_DONE
