#!/usr/bin/env bash
# Run one cc_generate.py call as data-parallel shards, then merge them. The visible GPUs are split into groups of
# --tp consecutive ids (consecutive ids share a PCIe switch / NUMA node on the 8x A100 host), one shard per group.
# The merge runs only if every shard succeeded.
#   scripts/run_sharded.sh --tp 2 -- --exp exp02 --model Qwen3.6-27B
#   CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_sharded.sh --tp 4 -- --exp exp02 --model Qwen3-32B-bf16 ...
# Logs: log/<exp>/generate_<model>.shard<K>of<N>.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

tp=1
while [[ $# -gt 0 && $1 != "--" ]]; do
  case $1 in
    --tp) tp=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ ${1:-} == "--" && $# -gt 1 ]] || { echo "usage: $0 [--tp T] -- <cc_generate.py flags>" >&2; exit 2; }
shift
flags=("$@")

exp="" model=""
for ((i = 0; i < ${#flags[@]}; i++)); do
  [[ ${flags[i]} == "--exp" ]] && exp=${flags[i + 1]}
  [[ ${flags[i]} == "--model" ]] && model=${flags[i + 1]}
done
[[ -n $exp && -n $model ]] || { echo "--exp and --model are required" >&2; exit 2; }

if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
n_shards=$(( ${#gpus[@]} / tp ))
(( n_shards >= 1 )) || { echo "${#gpus[@]} GPUs cannot hold one TP=${tp} shard" >&2; exit 2; }

mkdir -p "../log/${exp}"
pids=()
for ((k = 0; k < n_shards; k++)); do
  group=$(IFS=,; echo "${gpus[*]:k*tp:tp}")
  log="../log/${exp}/generate_${model}.shard${k}of${n_shards}.log"
  echo "shard ${k}/${n_shards}: GPUs ${group} -> ${log}"
  CUDA_VISIBLE_DEVICES=$group CC_TENSOR_PARALLEL=$tp \
    ./vllm_python.sh cc_generate.py "${flags[@]}" --shard "${k}/${n_shards}" > "$log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "a shard failed; see log/${exp}/generate_${model}.shard*.log" >&2; exit 1; }
./vllm_python.sh cc_generate.py "${flags[@]}" --merge-shards "$n_shards"
