#!/usr/bin/env bash
# One-off setup of this H200 NVL instance (driver 570, CUDA 12.8) after exp10's B200 overnight run; as
# log/setup_b200/setup_b200.sh, minus the clone (repo, submodule and /workspace/nocot-bench at 634d7de done by hand)
# and the HF download (already running into /workspace/hf_restore). Each step logs to /workspace/setup_logs/.
set -uo pipefail
REPO=/workspace/cot-controllability
LOGS=/workspace/setup_logs
mkdir -p "$LOGS" /workspace/.hf_home/hub
cd "$REPO" || exit 1

# CUDA forward-compat libs for the vLLM env's cu132 torch (scripts/vllm_python.sh). Not a driver package.
(apt-get update -qq && apt-get install -y -qq cuda-compat-13-2 && echo COMPAT_DONE) > "$LOGS/cuda_compat.log" 2>&1 &

(uv venv /workspace/.venv-vllm --python 3.12 --managed-python --seed &&
 UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=5 uv pip install --python /workspace/.venv-vllm/bin/python \
   -r log/setup_h200_2/freeze_vllm.txt --torch-backend=auto && echo VLLM_ENV_DONE) > "$LOGS/setup_vllm.log" 2>&1 &

(grep -v -E " @ |^(torch|torchvision|torchaudio|torchcodec|triton|nvidia-[a-z0-9-]+)==" log/setup_h200_2/freeze_main.txt > /tmp/main_pins.txt &&
 UV_HTTP_TIMEOUT=60 uv pip install --python /venv/main/bin/python -r /tmp/main_pins.txt && echo MAIN_ENV_DONE) > "$LOGS/setup_main.log" 2>&1 &

wait_for_hub() { /venv/main/bin/python -c "import huggingface_hub" 2>/dev/null; }
until wait_for_hub; do sleep 5; done
/venv/main/bin/python scripts/cc_download.py Qwen3.6-27B-FP8 Gemma-4-31B-FP8 Qwen3.8-27B-FP8 Qwen3-32B \
  Gemma-4-12B-FP8 Qwen3.6-35B-A3B-FP8 GLM-4.7-Flash-FP8 > "$LOGS/downloads.log" 2>&1 &

wait
echo SETUP_DONE
