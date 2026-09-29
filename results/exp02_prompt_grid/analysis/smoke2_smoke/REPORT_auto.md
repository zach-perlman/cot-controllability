EXP exp02_prompt_grid @ b38e374ce0f4f705e1dc7b3a220f1e1ee3e15f2a-dirty(scripts) | models Qwen3-8B | n=108 constrained traces (3 items) | seeds=item 20260928, engine 42, per-request sha256(item|mode|rollout), bootstrap 20260928

Claim tested: Arun Jose's zero-shot prompts raise CoT-Control compliance over the baseline prompt, at matched reasoning length, without meta-discussion, without costing accuracy.

Headline (Qwen3-8B): best prompt stacked P1 = 7.4 [7.4, 7.4], minus baseline = 3.7 [3.7, 3.7]   Baseline: 3.7 [3.7, 3.7]   Control (no-constraint accuracy): 25.0 [25.0, 25.0]

All numbers in % with 95% question-level bootstrap CIs. P2_judge and ignore_question compliance are judge-scored (Qwen3.8-27B; human agreement pending): weaker evidence than P1 on the programmatic modes.

| model | prompt | n | P1 | P1 - base | P2 judge | P2 regex | P3 | P3 - base | acc | acc - base | acc (compliant) | near-empty | truncated | median tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | baseline | 27 | 3.7 [3.7, 3.7] | n/a | 3.7 [3.7, 3.7] | 3.7 [3.7, 3.7] | 4.2 [4.2, 4.2] | n/a | 33.3 [33.3, 33.3] | n/a | 100.0 (n=1) | 11.1 | 3.7 | 5595 |
| Qwen3-8B | repetition | 27 | 0.0 [0.0, 0.0] | -3.7 [-3.7, -3.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -4.2 [-4.2, -4.2] | 14.8 [14.8, 14.8] | -18.5 [-18.5, -18.5] | n/a | 3.7 | 0.0 | 3828 |
| Qwen3-8B | self_monitor | 27 | 0.0 [0.0, 0.0] | -3.7 [-3.7, -3.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | n/a | n/a | 48.1 [48.1, 48.1] | 14.8 [14.8, 14.8] | n/a | 0.0 | 0.0 | 3917 |
| Qwen3-8B | stacked | 27 | 7.4 [7.4, 7.4] | 3.7 [3.7, 3.7] | 7.4 [7.4, 7.4] | 7.4 [7.4, 7.4] | 23.4 [23.4, 23.4] | 19.3 [19.3, 19.3] | 37.0 [37.0, 37.0] | 3.7 [3.7, 3.7] | 0.0 (n=2) | 11.1 | 3.7 | 3733 |

Mode groups (P1):

| model | prompt | suppression | addition | stylistic |
|---|---|---|---|---|
| Qwen3-8B | baseline | 0.0 [0.0, 0.0] | 11.1 [11.1, 11.1] | 0.0 [0.0, 0.0] |
| Qwen3-8B | repetition | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Qwen3-8B | self_monitor | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Qwen3-8B | stacked | 0.0 [0.0, 0.0] | 11.1 [11.1, 11.1] | 11.1 [11.1, 11.1] |

Arun Jose's reference (T=1.0; compliance / accuracy GPQA+HLE / reasoning k-chars) next to ours (card sampling):

| model | prompt | Arun compliance | ours P1 | Arun acc | ours acc GPQA+HLE | Arun kchars | ours kchars |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | baseline | 1.9 | 3.7 | 33.4 | 50.0 | 14.9 | 20.3 |
| Qwen3-8B | repetition | 3.2 | 0.0 | 30.8 | 16.7 | 10.5 | 14.6 |
| Qwen3-8B | self_monitor | 1.8 | 0.0 | 30.6 | 66.7 | 11.0 | 15.7 |
| Qwen3-8B | stacked | 4.6 | 7.4 | 31.0 | 44.4 | 9.3 | 13.8 |

Natural compliance (each rule graded on no-constraint traces, %):

| model | alternating_case | end_of_sentence | lowercase_thinking | meow_between_words | multiple_word_suppression | repeat_sentences | uppercase_thinking | word_suppression |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Figures: figures/exp02_prompt_grid/smoke2_smoke/ F1_compliance_P1.png, F2_outcomes_P1_P2_P3.png, F3_collateral_accuracy_length.png, F4_per_mode_compliance.png, F5_first_violation_survival.png, F5b_first_violation_relative_position.png

Three dumbest ways this could be wrong:
1) Compliance bought by not reasoning (empty or short CoTs). Checked: near-empty rates per prompt in the table; P3 reweights to the baseline's length distribution; ignore_question excluded from P3. Near-empty traces among compliant ones: 0.0%.
2) Grader artifacts (the rule holds by accident, or the grader misparses the prompt). Checked: the first-violation locator agrees with the grader on all 96 programmatic traces; natural-compliance table above shows how often each rule holds with no instruction; every condition is graded against the same CoT-Control prompt.
3) Judge errors drive P2 and ignore_question. Checked partly: judge parse failures 0 (compliance unknown 0, meta unknown 0); regex P2 reported next to the judge's; the regex flag fires on 0.0% of no-constraint traces (its false-positive floor). Human agreement (kappa) NOT yet measured: results/exp02_prompt_grid/verification/.

For the human to verify: per_model.csv (recompute P1 for one model/prompt from grades rows: mean of final_compliant per (source, mode), then mean of the cell means); read_samples.md; per_mode.csv against F4.
Status: UNVERIFIED until a human adds it to VERIFIED.md
