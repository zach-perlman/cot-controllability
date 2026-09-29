#!/usr/bin/env bash
# exp03 generation, grading, judging and analysis, after the engine check and manifest (a100_exp03_resume.sh wrote
# both, then stopped: run_sharded_abort.sh was not executable). Two lanes:
#   GPUs 0-3: Qwen3-8B (4 x TP 1), Qwen3-32B (2 x TP 2) -> grade -> judge (TP 4)
#   GPUs 4-7: Qwen3.6-27B (2 x TP 2) -> grade -> judge (TP 4)
#   Then: exp03 survival analysis.
# Logs: log/exp03/a100_exp03_generate.log (this script), lane_{a,b}_generate.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
logs=../log/exp03
[[ -f ../results/exp03_abort_survival/manifest.json ]] || { echo "no exp03 manifest" >&2; exit 1; }
[[ -x run_sharded_abort.sh ]] || { echo "run_sharded_abort.sh is not executable" >&2; exit 1; }
stamp() { echo "$(date -u +%FT%TZ) $*"; }

# generate_grade_judge GPUS MODEL:TP...: exp03 generation per model (shards of TP GPUs each), then grade and
# judge (TP 4) those files.
generate_grade_judge() {
  local gpus=$1 pair model gen key grades=()
  shift
  for pair in "$@"; do
    model=${pair%%:*}
    CUDA_VISIBLE_DEVICES=$gpus ./run_sharded_abort.sh --tp "${pair##*:}" -- --exp exp03 --model "$model"
    stamp "exp03 $model generated"
  done
  key=$($py -c "import cc_grade; print(cc_grade.code_key())")
  for pair in "$@"; do
    gen=$(ls ../cache/exp03/generations/"${pair%%:*}"__card__stream_abort__*.jsonl)
    $py cc_grade.py --exp exp03 --generations "$gen"
    grades+=("../cache/exp03/grades/$(basename "$gen" .jsonl)__${key}.jsonl")
  done
  CUDA_VISIBLE_DEVICES=$gpus CC_TENSOR_PARALLEL=4 ./vllm_python.sh cc_judge.py --exp exp03 --grades "${grades[@]}"
  stamp "exp03 $* graded and judged"
}

generate_grade_judge 0,1,2,3 Qwen3-8B:1 Qwen3-32B:2 >> "$logs/lane_a_generate.log" 2>&1 &
lane_a=$!
generate_grade_judge 4,5,6,7 Qwen3.6-27B:2 >> "$logs/lane_b_generate.log" 2>&1 &
lane_b=$!
stamp "lanes started (A pid $lane_a, B pid $lane_b)"
failed=0
wait "$lane_a" || { stamp "lane A failed; see $logs/lane_a_generate.log"; failed=1; }
wait "$lane_b" || { stamp "lane B failed; see $logs/lane_b_generate.log"; failed=1; }
(( failed == 0 )) || exit 1
$py cc_survival.py --run main
stamp "exp03 done"
