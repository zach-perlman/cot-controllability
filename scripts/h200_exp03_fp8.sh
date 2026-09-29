#!/usr/bin/env bash
# Rest of exp03 on 1x H200 with FP8 weights for the models not yet run (results/exp03_abort_survival/
# deviations_h200_fp8.json). Waits for the Qwen3-32B generation that scripts/h200_exp03.sh started (its
# run_sharded_abort.sh pid is the argument), grades it, generates and grades Qwen3.6-27B-FP8, judges all three
# models with the exp03 judge (cfg.JUDGE_BY_EXP), then runs the survival analysis.
#   scripts/h200_exp03_fp8.sh <pid of the Qwen3-32B run_sharded_abort.sh>
# Logs: log/exp03/h200_exp03_fp8.log (this script), generate_<model>.shard0of1.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
[[ -f ../results/exp03_abort_survival/deviations_h200_fp8.json ]] || { echo "no FP8 deviation record" >&2; exit 1; }
[[ $# -eq 1 ]] || { echo "usage: $0 <pid of the Qwen3-32B run_sharded_abort.sh>" >&2; exit 2; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }

key=$($py -c "import cc_grade; print(cc_grade.code_key())")
grades_of() {  # the current-code grades file of a model's exp03 generation (fails if it was not generated)
  local gen
  gen=$(ls ../cache/exp03/generations/"$1"__card__stream_abort__*.jsonl)
  echo "../cache/exp03/grades/$(basename "$gen" .jsonl)__${key}.jsonl"
}
grade() {
  $py cc_grade.py --exp exp03 --generations "$(ls ../cache/exp03/generations/"$1"__card__stream_abort__*.jsonl)"
  stamp "exp03 $1 graded"
}

stamp "waiting for Qwen3-32B generation (pid $1)"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
grade Qwen3-32B

CUDA_VISIBLE_DEVICES=0 ./run_sharded_abort.sh --tp 1 -- --exp exp03 --model Qwen3.6-27B-FP8
stamp "exp03 Qwen3.6-27B-FP8 generated"
grade Qwen3.6-27B-FP8

CUDA_VISIBLE_DEVICES=0 CC_TENSOR_PARALLEL=1 ./vllm_python.sh cc_judge.py --exp exp03 \
  --grades "$(grades_of Qwen3-8B)" "$(grades_of Qwen3-32B)" "$(grades_of Qwen3.6-27B-FP8)"
stamp "exp03 judged"
$py cc_survival.py --run main
stamp "exp03 done"
