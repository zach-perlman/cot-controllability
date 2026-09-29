#!/usr/bin/env bash
# exp01_paper_gate: greedy, 25k cap, baseline prompt, 50 items x 9 modes on Qwen3-8B and Qwen3-32B (FP8); grade,
# judge, analyze. Exits 1 if the pre-registered gate fails (Qwen3-32B P1 CI must contain 2.3%).
# The manifest must exist before this runs (cc_manifest.py --exp exp01). Log: log/exp01/run.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -f ../results/exp01_paper_gate/manifest.json ]] || { echo "write the manifest first" >&2; exit 1; }

for model in Qwen3-8B Qwen3-32B; do
  ./vllm_python.sh cc_generate.py --exp exp01 --model "$model"
done
/venv/main/bin/python cc_grade.py --exp exp01
./vllm_python.sh cc_judge.py --exp exp01
/venv/main/bin/python cc_analysis.py --exp exp01 --run gate

/venv/main/bin/python - <<'EOF'
import json, sys
g = json.load(open("../results/exp01_paper_gate/analysis/gate/summary.json"))["gate"]
print(f"GATE {'PASS' if g['pass'] else 'FAIL'}: Qwen3-32B P1 = {g['ours']['value']:.1f} "
      f"[{g['ours']['ci'][0]:.1f}, {g['ours']['ci'][1]:.1f}] vs paper {g['target']}")
sys.exit(0 if g["pass"] else 1)
EOF
echo "exp01 done"
