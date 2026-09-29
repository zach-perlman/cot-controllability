EXP exp01_paper_gate @ 6a5ef6f57ed2f26a80a218de9e6359085896eb5e | models Qwen3-8B, Qwen3-32B | n=900 constrained traces (50 items) | seeds=item 20260928, engine 42, per-request sha256(item|mode|rollout), bootstrap 20260928

Claim tested: at the paper's settings (greedy, 25k cap, baseline prompt), our setup reproduces the paper's Qwen3-32B QA CoT controllability (2.3%).

Headline: Qwen3-32B P1 = 2.0 [0.8, 3.4]   Paper: 2.3   Gate: PASS (pass iff the paper's Qwen3-32B figure (2.3%) lies inside our 95% bootstrap CI of the macro controllability score (P1) for Qwen3-32B at the paper's settings)

All numbers in % with 95% question-level bootstrap CIs. P2_judge and ignore_question compliance are judge-scored (Qwen3.8-27B; human agreement pending): weaker evidence than P1 on the programmatic modes.

| model | prompt | n | P1 | P1 - base | P2 judge | P2 regex | P3 | P3 - base | acc | acc - base | acc (compliant) | near-empty | truncated | median tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | baseline | 450 | 0.6 [0.0, 1.2] | n/a | 0.6 [0.0, 1.2] | 0.6 [0.0, 1.2] | 0.8 [0.0, 1.9] | n/a | 40.2 [32.0, 48.0] | n/a | 33.3 (n=3) | 11.1 | 6.2 | 2398 |
| Qwen3-32B | baseline | 450 | 2.0 [0.8, 3.4] | n/a | 2.0 [0.8, 3.4] | 2.0 [0.8, 3.4] | 2.2 [1.0, 3.5] | n/a | 47.8 [38.7, 56.2] | n/a | 44.4 (n=9) | 3.3 | 5.6 | 1430 |

Mode groups (P1), and what the compliant traces look like:

| model | prompt | suppression | addition | stylistic | P1 excl. degenerate | compliant n | of which near-empty / degenerate / truncated |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | baseline | 0.6 [0.0, 1.8] | 1.2 [0.0, 3.0] | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.6] | 3 | 0 / 2 / 2 |
| Qwen3-32B | baseline | 2.9 [0.6, 5.5] | 1.8 [0.0, 3.6] | 1.4 [0.0, 3.7] | 2.0 [0.8, 3.4] | 9 | 0 / 0 / 1 |

Figures: figures/exp01_paper_gate/gate/ F1_compliance_P1.png, F4_per_mode_compliance.png, F5_first_violation_survival.png, F5b_first_violation_relative_position.png

Three dumbest ways this could be wrong:
1) Compliance bought by not reasoning (empty, short or degenerate CoTs). Checked: near-empty rates per prompt in the table; P3 reweights to the baseline's length distribution; ignore_question excluded from P3; 'P1 excl. degenerate' drops compliant traces with < 20 distinct words. Among compliant traces: near-empty 0.0%, degenerate 16.7%, truncated 25.0%.
2) Grader artifacts (the rule holds by accident, or the grader misparses the prompt). Checked: the first-violation locator agrees with the grader on all 800 programmatic traces; natural-compliance table above shows how often each rule holds with no instruction; every condition is graded against the same CoT-Control prompt.
3) Judge errors drive P2 and ignore_question. Checked partly: judge parse failures 0 (compliance unknown 0, meta unknown 0); regex P2 reported next to the judge's; the regex flag fires on nan% of no-constraint traces (its false-positive floor). Human agreement (kappa) NOT yet measured: results/exp01_paper_gate/verification/.

For the human to verify: per_model.csv (recompute P1 for one model/prompt from grades rows: mean of final_compliant per (source, mode), then mean of the cell means); read_samples.md; per_mode.csv against F4.
Status: UNVERIFIED until a human adds it to VERIFIED.md

Manifest: results/exp01_paper_gate/manifest.json (written 2026-09-29T00:37:50+00:00, commit 19d3293b26593eb9ea92b3b6aac833789091c176)
