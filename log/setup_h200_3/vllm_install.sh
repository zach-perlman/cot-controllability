#!/usr/bin/env bash
# The vLLM env install that worked on this host (driver 570). setup.sh's `--torch-backend=auto` picks cu128 wheels
# from the driver and finds no torch==2.13.0+cu132; `--torch-backend=cu132` then finds no torchaudio==2.11.0 (vLLM
# pins it; the cu132 index has none). As on the 2xH200 instance, torchaudio is the CPU build (unused by text runs);
# torch comes from the cu132 index and runs through cuda-compat-13-2 (scripts/vllm_python.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
sed 's/^torchaudio==2.11.0$/torchaudio==2.11.0+cpu/' log/setup_h200_2/freeze_vllm.txt > /tmp/vllm_pins.txt
UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=5 uv pip install --python /workspace/.venv-vllm/bin/python -r /tmp/vllm_pins.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
