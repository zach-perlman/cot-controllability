# exp07b analysis: v1 (UNVERIFIED until a human adds it to VERIFIED.md)

Models: Qwen3.8-27B-FP8, Qwen3-32B, Gemma-4-31B-FP8.

**Winner (pre-registered rule): `many_examples`**

## Candidate - upgraded (S(1000) points, 95% CI over questions)

| candidate | cells | difference | p | Holm p | Qwen3.8-27B-FP8 | Qwen3-32B | Gemma-4-31B-FP8 |
|---|---|---|---|---|---|---|---|
| A: many_examples | 27 | +7.2 [+4.7, +9.8] | 0.000 | 0.000 | +12.1 [+7.4, +16.9] | +7.6 [+3.1, +11.9] | +2.0 [-2.8, +6.8] |
| B: own_compliant | 24 | +3.1 [+0.1, +6.0] | 0.041 | 0.123 | -0.3 [-5.6, +4.6] | +4.2 [-1.3, +9.7] | +5.3 [-0.3, +10.2] |
| C: named_once | 12 | +0.7 [-2.5, +3.7] | 0.707 | 0.736 | +4.5 [-1.4, +10.4] | +1.4 [-4.4, +6.5] | -4.0 [-11.1, +3.5] |
| L: own_guide | 27 | -6.4 [-8.9, -3.9] | 0.000 | 0.000 | +4.1 [-0.6, +8.6] | -4.3 [-8.5, +0.1] | -19.0 [-23.8, -14.2] |
| G: monitored | 27 | +1.2 [-1.5, +3.6] | 0.368 | 0.736 | +4.6 [-0.0, +8.9] | -1.5 [-6.6, +3.2] | +0.6 [-3.7, +4.6] |

## Per rule (mean over models where it ran)

| candidate | lowercase_thinking | uppercase_thinking | meow_between_words | end_of_sentence | word_suppression | alternating_case | multiple_word_suppression | no_digits | no_first_person |
|---|---|---|---|---|---|---|---|---|---|
| A | -1.2 | +10.7 | +4.0 | +15.3 | -1.9 | +3.0 | +4.6 | +9.7 | +20.8 |
| B | +2.0 | +16.7 | -1.7 | +5.4 | -2.2 | +7.8 | +2.4 | +3.7 | +1.3 |
| C | - | - | - | - | -6.6 | - | +0.1 | +3.4 | +5.7 |
| L | -8.8 | -5.1 | -14.1 | -7.7 | -15.3 | -1.1 | -1.5 | -10.4 | +6.5 |
| G | -2.9 | +4.9 | +1.7 | +5.7 | -5.8 | -0.2 | +1.4 | +3.7 | +2.4 |

## Guards (% of texts, on the candidate's cells)

| candidate | arm | clean_200 | ended_clean_early | empty | degenerate | meta_regex |
|---|---|---|---|---|---|---|
| many_examples | many_examples | 60.7 | 2.3 | 0.0 | 0.1 | 9.6 |
| many_examples | upgraded | 53.6 | 2.3 | 0.0 | 0.5 | 9.6 |
| own_compliant | own_compliant | 58.9 | 2.8 | 0.0 | 0.4 | 11.4 |
| own_compliant | upgraded | 60.1 | 2.5 | 0.0 | 0.6 | 10.8 |
| named_once | named_once | 45.1 | 2.0 | 0.0 | 0.0 | 15.6 |
| named_once | upgraded | 46.1 | 2.8 | 0.0 | 0.1 | 17.8 |
| own_guide | own_guide | 45.3 | 1.0 | 0.0 | 0.3 | 8.4 |
| own_guide | upgraded | 53.6 | 2.3 | 0.0 | 0.5 | 9.6 |
| monitored | monitored | 52.7 | 1.8 | 0.0 | 0.3 | 10.0 |
| monitored | upgraded | 53.6 | 2.3 | 0.0 | 0.5 | 9.6 |

## Figures

- [B1_differences](../../../../figures/exp07b_prompt_compare/v1/B1_differences.html)
- [B2_per_rule](../../../../figures/exp07b_prompt_compare/v1/B2_per_rule.html)
