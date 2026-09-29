#!/usr/bin/env bash
# exp03_abort_survival on the 8x A100-40GB instance, in order:
#   1. wait until exp02's Qwen3.6-27B shards release the GPUs (exp02_finish_a100.sh then uses GPUs 0-3 only);
#   2. engine check on GPUs 4 and 5 (greedy Qwen3-8B, smoke requests: batch engine vs streaming abort); must pass;
#   3. manifest (results/exp03_abort_survival/manifest.json), written before any exp03 generation;
#   4. Qwen3-8B on GPUs 4-7 (4 shards x TP 1) while exp02 finishes on 0-3;
#   5. once exp02_finish_a100.sh has exited: Qwen3-32B and Qwen3.6-27B on all 8 GPUs (4 shards x TP 2 each);
#   6. grade, judge (TP 4), survival analysis.
# Log: log/exp03/chain_a100.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=/venv/main/bin/python
mkdir -p ../log/exp03

while pgrep -f "cc_generate.py --exp exp02 --model Qwen3.6-27B" > /dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) GPUs 4-7 free"

req=$($py check_engine_abort.py prepare | tail -1)
gens=$(dirname "$req")/generations
CUDA_VISIBLE_DEVICES=4 ./vllm_python.sh cc_generate.py --exp exp02 --model Qwen3-8B --requests "$req" \
  --sampling greedy >> ../log/exp03/engine_check_v1.log 2>&1 &
v1=$!
CUDA_VISIBLE_DEVICES=5 ./vllm_python.sh cc_generate_abort.py --exp exp03 --model Qwen3-8B --requests "$req" \
  --items ../cache/exp02/items.jsonl --sampling greedy --shard 0/1 >> ../log/exp03/engine_check_abort.log 2>&1 &
ab=$!
wait "$v1"
wait "$ab"
./vllm_python.sh cc_generate_abort.py --exp exp03 --model Qwen3-8B --requests "$req" \
  --items ../cache/exp02/items.jsonl --sampling greedy --merge-shards 1
v1_file=$(ls "$gens"/Qwen3-8B__greedy__*.jsonl | grep -v stream_abort)
abort_file=$(ls "$gens"/Qwen3-8B__greedy__stream_abort__*.jsonl)
$py check_engine_abort.py compare "$v1_file" "$abort_file"
echo "$(date -u +%FT%TZ) engine check passed"

$py cc_exp03.py manifest

CUDA_VISIBLE_DEVICES=4,5,6,7 ./run_sharded_abort.sh --tp 1 -- --exp exp03 --model Qwen3-8B
echo "$(date -u +%FT%TZ) Qwen3-8B done"

while pgrep -f exp02_finish_a100.sh > /dev/null; do sleep 120; done
echo "$(date -u +%FT%TZ) exp02 finished; all GPUs free"
./run_sharded_abort.sh --tp 2 -- --exp exp03 --model Qwen3-32B
echo "$(date -u +%FT%TZ) Qwen3-32B done"
./run_sharded_abort.sh --tp 2 -- --exp exp03 --model Qwen3.6-27B
echo "$(date -u +%FT%TZ) Qwen3.6-27B done"

$py cc_grade.py --exp exp03
CUDA_VISIBLE_DEVICES=0,1,2,3 CC_TENSOR_PARALLEL=4 ./vllm_python.sh cc_judge.py --exp exp03
$py cc_survival.py --run main
echo "$(date -u +%FT%TZ) exp03 done"
