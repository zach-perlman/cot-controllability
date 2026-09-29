#!/usr/bin/env bash
# exp03_abort_survival on the 8x A100-40GB instance, plus judging and analyzing exp02's finished models (Qwen3-8B,
# Qwen3-32B; results/exp02_prompt_grid/deviations_a100_qwen36_stopped.json). Two GPU lanes:
#   Lane A, GPUs 0-3: exp03 engine check (GPUs 0 and 1; must pass) -> exp03 manifest -> exp03 Qwen3-8B (4 x TP 1)
#                     -> exp03 Qwen3-32B (2 x TP 2) -> grade both -> judge both (TP 4)
#   Lane B, GPUs 4-7: judge exp02 (TP 4) -> exp02 analysis + verification sheet -> (once the manifest exists)
#                     exp03 Qwen3.6-27B (2 x TP 2) -> grade -> judge (TP 4)
#   Then: exp03 survival analysis (pooled with exp02 for the models exp02 completed).
# Logs: log/exp03/a100_exp03.log (this script), lane_{a,b}.log, per-step logs next to them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
logs=../log/exp03
manifest=../results/exp03_abort_survival/manifest.json
mkdir -p "$logs"
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

lane_a() {
  local req gens
  req=$($py check_engine_abort.py prepare | tail -1)
  gens=$(dirname "$req")/generations
  CUDA_VISIBLE_DEVICES=0 ./vllm_python.sh cc_generate.py --exp exp02 --model Qwen3-8B --requests "$req" \
    --sampling greedy >> "$logs/engine_check_v1.log" 2>&1 &
  local v1=$!
  CUDA_VISIBLE_DEVICES=1 ./vllm_python.sh cc_generate_abort.py --exp exp03 --model Qwen3-8B --requests "$req" \
    --items ../cache/exp02/items.jsonl --sampling greedy --shard 0/1 >> "$logs/engine_check_abort.log" 2>&1 &
  local ab=$!
  wait "$v1"
  wait "$ab"
  ./vllm_python.sh cc_generate_abort.py --exp exp03 --model Qwen3-8B --requests "$req" \
    --items ../cache/exp02/items.jsonl --sampling greedy --merge-shards 1
  $py check_engine_abort.py compare "$(ls "$gens"/Qwen3-8B__greedy__*.jsonl | grep -v stream_abort)" \
    "$(ls "$gens"/Qwen3-8B__greedy__stream_abort__*.jsonl)"
  stamp "engine check passed"
  $py cc_exp03.py manifest
  generate_grade_judge 0,1,2,3 Qwen3-8B:1 Qwen3-32B:2
}

lane_b() {
  CUDA_VISIBLE_DEVICES=4,5,6,7 CC_TENSOR_PARALLEL=4 ./vllm_python.sh cc_judge.py --exp exp02
  stamp "exp02 judged"
  $py cc_analysis.py --exp exp02 --run grid
  $py cc_verification.py build --exp exp02
  stamp "exp02 analyzed"
  while [[ ! -f $manifest ]]; do
    kill -0 "$lane_a_pid" 2> /dev/null || { stamp "lane A ended without an exp03 manifest"; return 1; }
    sleep 60
  done
  generate_grade_judge 4,5,6,7 Qwen3.6-27B:2
}

lane_a >> "$logs/lane_a.log" 2>&1 &
lane_a_pid=$!
lane_b >> "$logs/lane_b.log" 2>&1 &
lane_b_pid=$!
stamp "lanes started (A pid $lane_a_pid, B pid $lane_b_pid)"
failed=0
wait "$lane_a_pid" || { stamp "lane A failed; see $logs/lane_a.log"; failed=1; }
wait "$lane_b_pid" || { stamp "lane B failed; see $logs/lane_b.log"; failed=1; }
(( failed == 0 )) || exit 1
$py cc_survival.py --run main
stamp "exp03 done"
