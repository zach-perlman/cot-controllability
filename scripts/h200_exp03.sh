#!/usr/bin/env bash
# exp03 on 1x H200 (results/exp03_abort_survival/deviations_h200.json): Qwen3-32B and Qwen3.6-27B generated one after
# the other (1 shard x TP 1), each graded as soon as it is generated; then the judge (TP 1) on all three models'
# grades (Qwen3-8B was generated and graded on the A100), then the survival analysis.
# Logs: log/exp03/h200_exp03.log (this script), generate_<model>.shard0of1.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
[[ -f ../results/exp03_abort_survival/manifest.json ]] || { echo "no exp03 manifest" >&2; exit 1; }
[[ -f ../results/exp03_abort_survival/deviations_h200.json ]] || { echo "no H200 deviation record" >&2; exit 1; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }

key=$($py -c "import cc_grade; print(cc_grade.code_key())")
grades=()
for model in Qwen3-8B Qwen3-32B Qwen3.6-27B; do
  if [[ $model != Qwen3-8B ]]; then
    CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp03 --model "$model"
    stamp "exp03 $model generated"
  fi
  gen=$(ls ../cache/exp03/generations/"$model"__card__stream_abort__*.jsonl)
  $py cc_grade.py --exp exp03 --generations "$gen"
  grades+=("../cache/exp03/grades/$(basename "$gen" .jsonl)__${key}.jsonl")
  stamp "exp03 $model graded"
done
CUDA_VISIBLE_DEVICES=0 CC_TENSOR_PARALLEL=1 ./vllm_python.sh cc_judge.py --exp exp03 --grades "${grades[@]}"
stamp "exp03 judged"
$py cc_survival.py --run main
stamp "exp03 done"
