# exp09 extension 3 (examples only, openings thinking off, CoT necessity): thinking_on_v1 (UNVERIFIED until a human adds it to VERIFIED.md)

Models: Qwen3.6-27B-FP8, Qwen3.8-27B-FP8, Gemma-4-31B-FP8, Qwen3-32B, Gemma-4-12B-FP8, Qwen3.6-35B-A3B-FP8, GLM-4.7-Flash-FP8.
Not part of exp09's pre-registered test; Holm over E1 and OF1 only. Rule-grader and accuracy metrics (no LLM judge).

**Partial run (--thinking-on-only): no thinking-off rows yet, so only E1, E2 and the thinking-on necessity splits; E1's 'Holm p' is 2 x p (Bonferroni over E1 and OF1, an upper bound of the final Holm p).**

## Contrasts

| contrast | metric | cells | difference | p | Holm p | Qwen3.6-27B-FP8 | Qwen3.8-27B-FP8 | Gemma-4-31B-FP8 | Qwen3-32B | Gemma-4-12B-FP8 | Qwen3.6-35B-A3B-FP8 | GLM-4.7-Flash-FP8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E1 A - examples only (thinking on)** | S_1000 | 84 | +16.0 [+15.0, +17.1] | 0.000 | 0.000 | +9.2 [+6.7, +11.6] | +13.8 [+10.4, +16.9] | +51.4 [+48.0, +54.7] | +23.7 [+21.0, +26.5] | +2.6 [+1.3, +4.0] | +6.4 [+4.3, +8.5] | +5.2 [+3.6, +6.9] |
| E2 examples only - CoT-Control prompt (thinking on) | S_1000 | 84 | +2.5 [+1.7, +3.3] | 0.000 | - | +11.6 [+9.6, +13.7] | +11.4 [+9.3, +13.8] | -7.5 [-9.4, -5.7] | +0.3 [-1.7, +2.4] | -3.4 [-4.7, -2.1] | +3.8 [+2.7, +5.0] | +1.3 [+0.3, +2.4] |

## Openings off on every model with openings-off grades (secondary pool)

| contrast | metric | cells | difference | p | Qwen3.6-27B-FP8 | Qwen3.8-27B-FP8 | Gemma-4-31B-FP8 | Qwen3-32B | Gemma-4-12B-FP8 | Qwen3.6-35B-A3B-FP8 | GLM-4.7-Flash-FP8 |
|---|---|---|---|---|---|---|---|---|---|---|---|

## Per rule set

| contrast | exp06 | held_out_exp07 | new_exp09 |
|---|---|---|---|
| E1 A - examples only (thinking on) | +21.7 [+20.2, +23.2] | +9.2 [+7.5, +10.9] | +15.8 [+13.9, +17.7] |
| E2 examples only - CoT-Control prompt (thinking on) | +4.9 [+3.7, +6.2] | +0.7 [-0.6, +2.1] | +0.9 [-0.4, +2.2] |

## By CoT necessity (5 direct answers per model and question)

Questions per label: {"GLM-4.7-Flash-FP8": {"mixed": 27, "necessary": 76, "unnecessary": 17}, "Gemma-4-12B-FP8": {"mixed": 5, "necessary": 75, "unnecessary": 40}, "Gemma-4-31B-FP8": {"mixed": 5, "necessary": 62, "unnecessary": 53}, "Qwen3-32B": {"mixed": 6, "necessary": 77, "unnecessary": 37}, "Qwen3.6-27B-FP8": {"mixed": 28, "necessary": 57, "unnecessary": 35}, "Qwen3.6-35B-A3B-FP8": {"mixed": 25, "necessary": 60, "unnecessary": 35}, "Qwen3.8-27B-FP8": {"mixed": 28, "necessary": 60, "unnecessary": 32}}

| contrast | necessary | mixed | unnecessary |
|---|---|---|---|
| C1 thinking off - on (CoT-Control prompt) | - | - | - |
| C2 thinking off - on (A) | - | - | - |
| C3 A - CoT-Control prompt, thinking on | +18.2 [+16.9, +19.7] | +15.0 [+12.9, +19.4] | +20.3 [+18.3, +22.7] |
| E1 A - examples only (thinking on) | +16.2 [+14.9, +17.5] | +13.6 [+10.7, +18.8] | +16.2 [+14.1, +18.2] |
