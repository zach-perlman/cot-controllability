#!/usr/bin/env bash
# The rest of exp02 and all of exp03 on the 8x A100-40GB instance, as two GPU lanes, then both judges side by side.
# Replaces exp02_finish_a100.sh after its Qwen3.6-27B step (bf16 check dropped by the human:
# results/exp02_prompt_grid/deviations_a100_bf16_dropped.json) and exp03_chain_a100.sh.
#   Lane A, GPUs 4-7: wait for exp02's Qwen3.6 shards 2-3 -> exp03 engine check (must pass) -> exp03 manifest
#                     -> exp03 Qwen3-8B (4 x TP 1) -> exp03 Qwen3-32B (2 x TP 2)
#   Lane B, GPUs 0-3: wait for exp02's merged Qwen3.6 file -> grade exp02 -> (once the manifest exists)
#                     exp03 Qwen3.6-27B (2 x TP 2)
#   Then: grade exp03; judge exp02 on GPUs 0-3 and exp03 on GPUs 4-7 (TP 4 each); analyses.
# Logs: log/exp03/lanes_a100.log (this script), lane_{a,b}.log, per-step logs next to them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
logs=../log/exp03
mkdir -p "$logs"
stamp() { echo "$(date -u +%FT%TZ) $*"; }

lane_a() {
  while pgrep -f "cc_generate.py --exp exp02 --model Qwen3.6-27B --shard [23]/4" > /dev/null; do sleep 60; done
  stamp "GPUs 4-7 free"
  local req gens
  req=$($py check_engine_abort.py prepare | tail -1)
  gens=$(dirname "$req")/generations
  CUDA_VISIBLE_DEVICES=4 ./vllm_python.sh cc_generate.py --exp exp02 --model Qwen3-8B --requests "$req" \
    --sampling greedy >> "$logs/engine_check_v1.log" 2>&1 &
  local v1=$!
  CUDA_VISIBLE_DEVICES=5 ./vllm_python.sh cc_generate_abort.py --exp exp03 --model Qwen3-8B --requests "$req" \
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
  CUDA_VISIBLE_DEVICES=4,5,6,7 ./run_sharded_abort.sh --tp 1 -- --exp exp03 --model Qwen3-8B
  stamp "exp03 Qwen3-8B done"
  CUDA_VISIBLE_DEVICES=4,5,6,7 ./run_sharded_abort.sh --tp 2 -- --exp exp03 --model Qwen3-32B
  stamp "exp03 Qwen3-32B done"
}

lane_b() {
  until ls ../cache/exp02/generations/Qwen3.6-27B__card__*.jsonl > /dev/null 2>&1; do
    pgrep -f "run_sharded.sh --tp 2 -- --exp exp02 --model Qwen3.6-27B" > /dev/null \
      || { stamp "exp02 Qwen3.6 run ended without a merged file"; return 1; }
    sleep 60
  done
  stamp "exp02 Qwen3.6-27B merged"
  $py cc_grade.py --exp exp02
  stamp "exp02 graded"
  while [[ ! -f ../results/exp03_abort_survival/manifest.json ]]; do
    kill -0 "$lane_a_pid" 2> /dev/null || { stamp "lane A ended without an exp03 manifest"; return 1; }
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_sharded_abort.sh --tp 2 -- --exp exp03 --model Qwen3.6-27B
  stamp "exp03 Qwen3.6-27B done"
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

$py cc_grade.py --exp exp03
stamp "exp03 graded; judging exp02 (GPUs 0-3) and exp03 (GPUs 4-7)"
CUDA_VISIBLE_DEVICES=0,1,2,3 CC_TENSOR_PARALLEL=4 ./vllm_python.sh cc_judge.py --exp exp02 \
  >> "$logs/judge_exp02.log" 2>&1 &
j2=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 CC_TENSOR_PARALLEL=4 ./vllm_python.sh cc_judge.py --exp exp03 \
  >> "$logs/judge_exp03.log" 2>&1 &
j3=$!
wait "$j2"
wait "$j3"
stamp "judges done"
$py cc_analysis.py --exp exp02 --run grid
$py cc_verification.py build --exp exp02
$py cc_survival.py --run main
stamp "exp02 and exp03 done"
