#!/usr/bin/env bash
# exp02_prompt_grid: card sampling, 4 prompts x 9 modes x 50 items + no-constraint reference (50 x 4) on Qwen3-8B,
# Qwen3-32B (FP8), Qwen3.6-27B; bf16 Qwen3-32B precision check on one cell; grade, judge, analyze, verification sheet.
# Starts only if the exp01 gate passed and the exp02 manifest exists. Log: log/exp02/chain.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -f ../results/exp02_prompt_grid/manifest.json ]] || { echo "write the manifest first" >&2; exit 1; }
/venv/main/bin/python - <<'EOF'
import json, sys
g = json.load(open("../results/exp01_paper_gate/analysis/gate/summary.json"))["gate"]
sys.exit(0 if g["pass"] else "exp01 gate did not pass; exp02 does not start")
EOF

for model in Qwen3-8B Qwen3-32B Qwen3.6-27B; do
  ./vllm_python.sh cc_generate.py --exp exp02 --model "$model"
done

# Precision check: same requests and seeds as the FP8 cell (baseline prompt, word_suppression + lowercase).
BF16_DIR=/workspace/.hf_home/hub/models--Qwen--Qwen3-32B
./vllm_python.sh cc_generate.py --exp exp02 --model Qwen3-32B-bf16 --prompts baseline \
  --modes word_suppression lowercase_thinking \
  --max-model-len "$(/venv/main/bin/python -c 'import cc_config; print(cc_config.PRECISION_CHECK_MAX_MODEL_LEN)')"
echo "precision check generated; deleting the bf16 weights (${BF16_DIR})"
rm -rf "${BF16_DIR}"

/venv/main/bin/python cc_grade.py --exp exp02
./vllm_python.sh cc_judge.py --exp exp02
/venv/main/bin/python cc_analysis.py --exp exp02 --run grid
/venv/main/bin/python cc_verification.py build --exp exp02
echo "exp02 done"
