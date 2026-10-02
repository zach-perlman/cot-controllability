# Deviation (2026-10-02): thinking-off prefill extension stopped after 2 of 5 models

`manifest_extension_off_prefill.json` planned the thinking-off rows with an opening on all 5 models. The run
(`scripts/h200_exp06_off_prefill.sh`) was stopped by the user's decision at 19:43 UTC, during Qwen3-32B, to free the
GPU for exp07.

- Complete and graded: Qwen3.8-27B-FP8, Gemma-4-31B-FP8.
- Qwen3-32B: 1126 of 1900 rows generated, kept in
  `cache/exp06/generations/Qwen3-32B__card__stream_abort_Qwen3-32B_thinking_off_prefill__d862bc6cb725.parts`
  (a rerun of the same request file resumes from them); not graded or analysed.
- Qwen3.6-27B-FP8, Qwen3.5-9B: not run.
- The pipeline's analysis run `with_off_prefill` and story figures `v3` were not made. The analysis of what exists is
  run `partial_off_prefill_2models` (`--skip-missing`).

Why: on the two finished models the extension's question was answered in direction, and a confound was found that
the remaining models would not resolve. With thinking off, most of a compliant opening's gain in `clean_200` (texts
shorter than 200 tokens count as failures) comes from longer texts, not fewer violations: the share of texts with a
violation hardly changes (upgraded prompt, 4 opener rules: Qwen3.8 31.5% -> 32.0%, Gemma 14.3% -> 16.5%), and
Kaplan-Meier S(200) is flat for Gemma (upgraded 95.7 -> 95.5; CoT-Control's prompt 72.9 -> 73.8). The planned exp08
(exp06 repeated with exp07's prompt) is to fix a thinking-off metric that does not reward length before it runs.
These numbers are post hoc and UNVERIFIED.
