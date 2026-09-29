#!/usr/bin/env bash
# Smoke test: Qwen3-8B on 3 items (one per source) x 9 modes x 4 prompts + no-constraint (120 requests), card
# sampling; then programmatic grading and the judge. Outputs under cache/exp02/smoke/. Log: log/exp02/smoke.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SMOKE=../cache/exp02/smoke

/venv/main/bin/python cc_prompts.py requests --exp exp02 --smoke-items 3
./vllm_python.sh cc_generate.py --exp exp02 --model Qwen3-8B --requests "${SMOKE}/requests.jsonl"
/venv/main/bin/python cc_grade.py --exp exp02 --generations "${SMOKE}"/generations/*.jsonl
./vllm_python.sh cc_judge.py --exp exp02 --grades "${SMOKE}"/grades/*.jsonl
echo "smoke chain done"
