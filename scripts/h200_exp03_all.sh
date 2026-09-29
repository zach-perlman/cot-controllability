#!/usr/bin/env bash
# Rest of exp03 on 1x H200: Qwen3.6-27B-FP8 (deviations_h200_fp8.json), then the extension models Qwen3.5-9B and
# Qwen3.5-4B (manifest_extension_qwen35.json), then the exp03 judge on all five models and the survival analysis.
# Replaces scripts/h200_exp03_fp8.sh, which was stopped while it waited. Waits for the Qwen3-32B generation that
# scripts/h200_exp03.sh started (its run_sharded_abort.sh pid is the argument).
#   scripts/h200_exp03_all.sh <pid of the Qwen3-32B run_sharded_abort.sh>
# Logs: log/exp03/h200_exp03_all.log (this script), generate_<model>.shard0of1.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
results=../results/exp03_abort_survival
for f in deviations_h200_fp8.json manifest_extension_qwen35.json; do
  [[ -f $results/$f ]] || { echo "missing $results/$f" >&2; exit 1; }
done
[[ $# -eq 1 ]] || { echo "usage: $0 <pid of the Qwen3-32B run_sharded_abort.sh>" >&2; exit 2; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }

key=$($py -c "import cc_grade; print(cc_grade.code_key())")
generation_of() { ls ../cache/exp03/generations/"$1"__card__stream_abort__*.jsonl; }
grades_of() { echo "../cache/exp03/grades/$(basename "$(generation_of "$1")" .jsonl)__${key}.jsonl"; }
grade() {
  $py cc_grade.py --exp exp03 --generations "$(generation_of "$1")"
  stamp "exp03 $1 graded"
}

stamp "waiting for Qwen3-32B generation (pid $1)"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
grade Qwen3-32B

for model in Qwen3.6-27B-FP8 Qwen3.5-9B Qwen3.5-4B; do
  CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp03 --model "$model"
  stamp "exp03 $model generated"
  grade "$model"
done

all_models=(Qwen3-8B Qwen3-32B Qwen3.6-27B-FP8 Qwen3.5-9B Qwen3.5-4B)
grades=()
for model in "${all_models[@]}"; do grades+=("$(grades_of "$model")"); done
CUDA_VISIBLE_DEVICES=0 CC_TENSOR_PARALLEL=1 ./vllm_python.sh cc_judge.py --exp exp03 --grades "${grades[@]}"
stamp "exp03 judged"
$py cc_survival.py --run main
stamp "exp03 done"
