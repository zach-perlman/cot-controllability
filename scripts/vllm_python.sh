#!/usr/bin/env bash
# Python of the vLLM venv; use this instead of /workspace/.venv-vllm/bin/python.
# The venv's torch is built for CUDA 13.2, which needs driver >= 595. On an older
# data-center driver (e.g. 570 on the H100 host), NVIDIA's forward-compat libcuda
# (apt package cuda-compat-13-2) goes first on the library path for this process
# tree only; /venv/main keeps using the host driver.
set -euo pipefail
# VLLM_VENV overrides the venv: /workspace/.venv-vllm025 has vLLM 0.25.1, the last release that serves Ouro.
VLLM_VENV="${VLLM_VENV:-/workspace/.venv-vllm}"
COMPAT_DIR=/usr/local/cuda-13.2/compat
MIN_NATIVE_DRIVER_MAJOR=595

driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if ((driver_major < MIN_NATIVE_DRIVER_MAJOR)); then
  if [[ ! -e "${COMPAT_DIR}/libcuda.so.1" ]]; then
    echo "Driver ${driver_major} < ${MIN_NATIVE_DRIVER_MAJOR} and ${COMPAT_DIR} is missing:" \
      "apt-get install cuda-compat-13-2" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="${COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# FlashInfer JIT-compiles kernels with $CUDA_HOME/bin/nvcc. At boot, Vast's 05-configure-cuda.sh can point
# /usr/local/cuda at /usr/local/cuda-13.2, which holds only the compat libs (no nvcc); then use the newest
# installed toolkit that has nvcc, and its lib64 (the compiled kernels link its libcudart).
if [[ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
  for toolkit in $(ls -d /usr/local/cuda-[0-9]*.[0-9]* 2>/dev/null | sort -V -r); do
    if [[ -x "${toolkit}/bin/nvcc" ]]; then
      export CUDA_HOME="${toolkit}"
      export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${toolkit}/lib64"
      break
    fi
  done
fi
# DeepGEMM's FP8 kernels are JIT-compiled and need nvcc >= 12.9; the system toolkit is 12.8 (Qwen3-32B-FP8 failed
# to start). Without it vLLM uses its precompiled CUTLASS FP8 kernels.
export VLLM_USE_DEEP_GEMM=0
exec "${VLLM_VENV}/bin/python" "$@"
