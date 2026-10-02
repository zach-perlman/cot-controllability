# Deviation 2026-10-02: post-hoc sensitivity, `</think>` accepted as the closing tag (thinking off)

Status: UNVERIFIED. Written before the figure set `figures/story/v2.2`. Analysis-only: no generation, no change to
stored grades (`cc_grade` outputs) or to any cache key.

## Why

With thinking off, the reasoning is graded on the closed `<output_reasoning>` block; a text with no closed block
counts as empty, so it fails at token 0. Qwen3.5-9B often opens the tag and closes its reasoning with `</think>`
instead (7.4% / 7.4% / 26.4% of rows for the baseline / stacked / upgraded prompts without an opening sentence;
every other model 0-3%). Strict grading then scores a format slip as a rule violation, which is the most likely
confound on its thinking-on vs thinking-off comparison (S5, E4).

## What

- `cc_exp06_analysis.lenient_grades`: for thinking-off rows with no closed block whose text runs from the
  opening tag (or the prefilled one) to a `</think>`, grade the text between them (minus any prefill) with
  `cc_grade.grade_row` + `place_caseless_violations` (the grader's own code; the run fails if the first-violation
  locator disagrees). All other rows keep their strict grade.
- New columns and estimates: `lenient_rescored`, `lenient_tokens`, `lenient_fv_token`; per thinking-off cell
  `lenient_rescored_share`, `clean_200_lenient` (and `_openers`); summary keys `cot_specificity_lenient`,
  `opening_specificity_lenient`; a report section "Sensitivity (post hoc)".
- Strict numbers are unchanged: every cell estimate, primary, secondary and specificity entry of the new analysis
  run equals `with_qwen35`'s.
- Story S5/S5b: hollow diamonds at the lenient values for a model whose thinking-off rows in that comparison are
  over 5% rescored (Qwen3.5-9B only, on current data). The headroom verdict in the subtitle stays strict.

## Runs

- Analysis `results/exp06_prompt_prefill/analysis/with_qwen35_lenient` (`--skip-missing`; the thinking-off prefill
  parts were still generating).
- Story `figures/story/v2.2` from it. The thinking-off prefill pipeline's `with_off_prefill` / `v3` include the
  same sensitivity.

## Result (scratch run, same data)

Qwen3.5-9B upgraded, thinking off: clean at 200 tokens 49.8% strict, 68.6% lenient. Its "on minus off" gain as a
share of headroom goes from -25.3 to -46.6 points: the gain is larger in the answer either way, more so under
lenient grading. Other models move by at most 2.2 points.
