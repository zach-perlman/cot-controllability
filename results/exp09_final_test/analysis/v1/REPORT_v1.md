EXP exp09_final_test @ a7fda33 | models Qwen3.8-27B-FP8, Gemma-4-31B-FP8, Qwen3-32B (design); Qwen3.6-27B-FP8,
Qwen3.6-35B-A3B-FP8, GLM-4.7-Flash-FP8, Gemma-4-12B-FP8 (fresh) | n=120 never-used questions x 12 rules per model
(short rows), 120 x 2 rules (full rows) | seeds=cfg.rollout_seed(item, rule, 0) (short), (item, exp09_full_<rule>, 0)
(full); bootstrap 2000 draws, seed 20261005, questions stratified by source (and rules, for questions_and_rules)

Claim tested: prompt A (stacked rule + failure guide + up to 12 of the model's own rewritten traces) raises CoT-rule
compliance over CoT-Control's prompt on questions, rules and models it was not chosen on (manifest.json, P1-P3).

Headline (all three pre-registered contrasts supported, Holm p < 0.001; bootstrap over questions and rules):
  P1 S(1000), fresh models x 12 rules:      A 11.6 [7.2, 16.9]   Baseline 2.4 [0.8, 4.5]   A - B +9.2 [+4.1, +14.9]
  P2 S(1000), 7 models x 3 new rules:       A 23.4 [19.8, 26.8]  Baseline 6.8 [0.0, 13.9]  A - B +16.7 [+10.0, +22.0]
  P3 whole-trace compliance, 7 x 2 rules:   A 32.4 [23.1, 42.4]  Baseline 3.8 [1.0, 7.0]   A - B +28.7 [+17.0, +41.0]
  Control (accuracy, full rows): A 50.6 [45.5, 55.7], baseline 52.2 [46.6, 57.6], no rule 51.3 [46.0, 56.6];
  A - baseline -1.6 [-4.2, +1.1] (p 0.25).
All metrics are rule graders or answer accuracy (no LLM judge; meta_regex is a regex flag).

Per model (A - baseline, S(1000), 12 rules): Gemma-31B +43.8, Qwen3.8 +25.2, Qwen3-32B +24.0 (design);
Qwen3.6-27B +20.8, Qwen3.6-35B-A3B +10.2, GLM +6.5, Gemma-12B -0.7 [-3.6, +1.9] (fresh). Design gain - fresh gain
+21.8 [13.4, 29.8]: A gains much more on the models it was chosen on. Absolute levels stay low: S(1000) under A is
11.6% on fresh models.

Three dumbest ways this could be wrong:
1) A "complies" by reasoning less. Checked: A's full traces are much shorter (median ~940 tokens vs 2490-3340 for the
   baseline, 3880 with no rule), so P3 favors short traces. Within reasoning-length bins A still complies far more
   (<=500 tokens: 71.0% vs 28.3%; 500-1000: 41.9 vs 6.3; 1000-2000: 28.2 vs 3.2; 2000-4000: 18.0 vs 0.0). Accuracy
   overall is not significantly lower, but within each length bin A is 4-11 points lower (length is post-treatment,
   so this is suggestive, not a clean estimate). Not fully ruled out: shorter reasoning is part of how A works.
2) Empty or degenerate traces. Checked: Gemma-4-12B under A closes its thinking channel at once on 937/1440 short
   rows (65%; 0% under baseline) and 165/240 full rows; no other model does this (<=0.3%). exp07's bf16 Gemma-12B
   did it on 10-16% with 3-example prompts, so it grows with the number of example turns. Empty counts as a violation
   at token 0, so this lowers A: P1 is conservative here (without Gemma-12B, the fresh mean is about +12.5;
   descriptive, not pre-registered).
3) Selection on design models / leakage. Checked: questions never used before (draw excluded every exp01-exp08 item);
   fresh models and the 3 new rules were never used to choose A; the design-minus-fresh gap (+21.8) shows the
   selection effect is real, so fresh-model numbers (P1) are the honest estimate.

For the human to verify: results/exp09_final_test/analysis/v1/primary.json (P1-P3), secondary.json (per_model,
ways_to_look_better, accuracy, full_reasoning_tokens_median); recompute P1 from cells.csv (S_1000, fresh models, mean
over model x rule of A - baseline); Gemma-12B empties: cache/exp09/generations/Gemma-4-12B-FP8__card__stream_abort_
Gemma-4-12B-FP8__*.jsonl, short A rows with reasoning_tokens == 0.
Status: UNVERIFIED until a human adds it to VERIFIED.md
