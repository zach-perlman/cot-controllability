#!/usr/bin/env bash
# Generate traces for several models: one model per GPU in parallel when more than one GPU is visible, one after
# another otherwise. cc_generate.py flags go before "--", models after it. Each GPU works through its share of the
# models in order (model i runs on GPU i mod n). Exits non-zero if any model failed.
#   scripts/generate_models.sh --exp exp02 --engine stream -- Qwen3-8B Qwen3-32B Qwen3.6-27B
# Logs (parallel mode): log/<exp>/generate_<model>.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

flags=()
while [[ $# -gt 0 && $1 != "--" ]]; do flags+=("$1"); shift; done
[[ ${1:-} == "--" && $# -gt 1 ]] || { echo "usage: $0 <cc_generate.py flags> -- <model> [model ...]" >&2; exit 2; }
shift
models=("$@")

exp=""
for ((i = 0; i < ${#flags[@]}; i++)); do [[ ${flags[i]} == "--exp" ]] && exp=${flags[i + 1]}; done
[[ -n $exp ]] || { echo "--exp is required" >&2; exit 2; }

if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

if (( ${#gpus[@]} <= 1 )); then
  for model in "${models[@]}"; do ./vllm_python.sh cc_generate.py "${flags[@]}" --model "$model"; done
  exit 0
fi

mkdir -p "../log/${exp}"
pids=()
for ((g = 0; g < ${#gpus[@]}; g++)); do
  share=()
  for ((m = g; m < ${#models[@]}; m += ${#gpus[@]})); do share+=("${models[m]}"); done
  (( ${#share[@]} )) || continue
  (
    for model in "${share[@]}"; do
      echo "GPU ${gpus[g]}: ${model}"
      CUDA_VISIBLE_DEVICES=${gpus[g]} ./vllm_python.sh cc_generate.py "${flags[@]}" --model "$model" \
        > "../log/${exp}/generate_${model}.log" 2>&1
    done
  ) &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "at least one model failed; see log/${exp}/generate_*.log" >&2; exit 1; }
