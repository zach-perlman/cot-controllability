# exp07 test analysis: partial_heldout4 (UNVERIFIED until a human adds it to VERIFIED.md)

Winner (validation): `rule_last`. Models: Qwen3.6-27B-FP8, GLM-4.7-Flash, Qwen3.6-35B-A3B-FP8, Gemma-4-12B. Missing: Qwen3.8-27B-FP8, Gemma-4-31B-FP8.

## Contrasts (S(1000) points, 95% CI over questions)

| group | contrast | a | b | difference | p |
|---|---|---|---|---|---|
| held-out rules x held-out models (primary) | rule_last - upgraded | 9.2 | 9.2 | +0.1 [-1.5, +1.6] | 0.954 |
| held-out rules x held-out models (primary) | upgraded - baseline | 9.2 | 3.0 | +6.2 [+4.7, +7.8] | 0.000 |
| search rules x held-out models | rule_last - upgraded | 17.4 | 17.4 | -0.1 [-1.7, +1.5] | 0.922 |
| search rules x held-out models | upgraded - baseline | 17.4 | 2.3 | +15.1 [+13.7, +16.6] | 0.000 |
| multiple_word_suppression x held-out models | rule_last - upgraded | 2.4 | 1.6 | +0.8 [-0.2, +1.7] | 0.102 |
| multiple_word_suppression x held-out models | upgraded - baseline | 1.6 | 2.3 | -0.8 [-2.0, +0.3] | 0.182 |
| rule: lowercase_thinking | rule_last - upgraded | 26.4 | 30.0 | -3.6 [-7.9, +0.8] | 0.108 |
| rule: uppercase_thinking | rule_last - upgraded | 35.2 | 30.5 | +4.7 [+0.6, +9.1] | 0.030 |
| rule: meow_between_words | rule_last - upgraded | 9.8 | 9.7 | +0.1 [-2.9, +3.1] | 0.980 |
| rule: end_of_sentence | rule_last - upgraded | 6.0 | 7.3 | -1.3 [-4.4, +2.0] | 0.437 |
| rule: word_suppression | rule_last - upgraded | 9.3 | 9.7 | -0.3 [-2.7, +2.0] | 0.777 |
| rule: alternating_case | rule_last - upgraded | 0.2 | 0.6 | -0.5 [-1.2, +0.3] | 0.268 |
| rule: multiple_word_suppression | rule_last - upgraded | 2.4 | 1.6 | +0.8 [-0.2, +1.7] | 0.102 |
| rule: no_digits | rule_last - upgraded | 19.3 | 20.9 | -1.6 [-5.0, +2.0] | 0.374 |
| rule: no_first_person | rule_last - upgraded | 8.2 | 6.1 | +2.2 [-0.6, +4.9] | 0.138 |
| model: Qwen3.6-27B-FP8 | rule_last - upgraded | 23.0 | 23.8 | -0.8 [-3.3, +1.8] | 0.533 |
| model: GLM-4.7-Flash | rule_last - upgraded | 8.1 | 6.3 | +1.8 [-0.1, +3.7] | 0.066 |
| model: Qwen3.6-35B-A3B-FP8 | rule_last - upgraded | 12.6 | 13.5 | -0.9 [-3.0, +1.1] | 0.380 |
| model: Gemma-4-12B | rule_last - upgraded | 8.3 | 8.1 | +0.2 [-1.4, +1.7] | 0.811 |
| rule: lowercase_thinking | upgraded - baseline | 30.0 | 0.0 | +30.0 [+26.4, +33.5] | 0.000 |
| rule: uppercase_thinking | upgraded - baseline | 30.5 | 1.5 | +29.0 [+25.1, +32.9] | 0.000 |
| rule: meow_between_words | upgraded - baseline | 9.7 | 0.0 | +9.7 [+7.8, +11.9] | 0.000 |
| rule: end_of_sentence | upgraded - baseline | 7.3 | 0.0 | +7.3 [+4.9, +9.8] | 0.000 |
| rule: word_suppression | upgraded - baseline | 9.7 | 10.0 | -0.3 [-3.4, +2.6] | 0.847 |
| rule: alternating_case | upgraded - baseline | 0.6 | 0.0 | +0.6 [+0.0, +1.4] | 0.119 |
| rule: multiple_word_suppression | upgraded - baseline | 1.6 | 2.3 | -0.8 [-2.0, +0.3] | 0.182 |
| rule: no_digits | upgraded - baseline | 20.9 | 2.7 | +18.2 [+14.3, +22.1] | 0.000 |
| rule: no_first_person | upgraded - baseline | 6.1 | 6.2 | -0.2 [-3.0, +2.6] | 0.919 |
| model: Qwen3.6-27B-FP8 | upgraded - baseline | 23.8 | 2.1 | +21.7 [+19.3, +23.9] | 0.000 |
| model: GLM-4.7-Flash | upgraded - baseline | 6.3 | 1.4 | +4.8 [+3.2, +6.5] | 0.000 |
| model: Qwen3.6-35B-A3B-FP8 | upgraded - baseline | 13.5 | 0.6 | +13.0 [+11.2, +14.7] | 0.000 |
| model: Gemma-4-12B | upgraded - baseline | 8.1 | 6.0 | +2.1 [+0.6, +3.5] | 0.004 |

## Guards (% of texts)

| arm | rules | clean_200 | ended_clean_early | empty | degenerate | meta_regex |
|---|---|---|---|---|---|---|
| baseline | held-out rules | 17.7 | 0.0 | 0.0 | 0.5 | 48.1 |
| baseline | search rules | 5.7 | 0.0 | 0.0 | 0.9 | 28.8 |
| rule_last | held-out rules | 28.0 | 0.5 | 2.2 | 2.6 | 45.7 |
| rule_last | search rules | 37.4 | 0.7 | 5.6 | 6.2 | 36.8 |
| upgraded | held-out rules | 27.5 | 0.5 | 2.9 | 3.3 | 44.1 |
| upgraded | search rules | 34.5 | 0.5 | 2.1 | 2.8 | 38.1 |

## Figures

- [T1_contrasts](../../../../figures/exp07_prompt_search/partial_heldout4/T1_contrasts.html)
- [T2_per_rule](../../../../figures/exp07_prompt_search/partial_heldout4/T2_per_rule.html)
