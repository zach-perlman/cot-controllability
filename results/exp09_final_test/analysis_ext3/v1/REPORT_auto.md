# exp09 extension 3 (examples only, openings thinking off, CoT necessity): v1 (UNVERIFIED until a human adds it to VERIFIED.md)

Models: Qwen3.6-27B-FP8, Qwen3.8-27B-FP8, Gemma-4-31B-FP8, Qwen3-32B, Gemma-4-12B-FP8, Qwen3.6-35B-A3B-FP8, GLM-4.7-Flash-FP8.
Not part of exp09's pre-registered test; Holm over E1 and OF1 only. Rule-grader and accuracy metrics (no LLM judge).

## Contrasts

| contrast | metric | cells | difference | p | Holm p | Qwen3.6-27B-FP8 | Qwen3.8-27B-FP8 | Gemma-4-31B-FP8 | Qwen3-32B | Gemma-4-12B-FP8 | Qwen3.6-35B-A3B-FP8 | GLM-4.7-Flash-FP8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E1 A - examples only (thinking on)** | S_1000 | 84 | +16.0 [+15.0, +17.1] | 0.000 | 0.000 | +9.2 [+6.7, +11.6] | +13.8 [+10.4, +16.9] | +51.4 [+48.0, +54.7] | +23.7 [+21.0, +26.5] | +2.6 [+1.3, +4.0] | +6.4 [+4.3, +8.5] | +5.2 [+3.6, +6.9] |
| **OF1 compliant opening - none (A, thinking off)** | S_200 | 35 | -1.7 [-3.2, -0.2] | 0.023 | 0.023 | - | - | -1.0 [-3.3, +1.3] | -3.0 [-6.2, +0.2] | -1.3 [-4.3, +1.9] | -1.7 [-5.4, +2.2] | -1.4 [-5.7, +2.9] |
| E2 examples only - CoT-Control prompt (thinking on) | S_1000 | 84 | +2.5 [+1.7, +3.3] | 0.000 | - | +11.6 [+9.6, +13.7] | +11.4 [+9.3, +13.8] | -7.5 [-9.4, -5.7] | +0.3 [-1.7, +2.4] | -3.4 [-4.7, -2.1] | +3.8 [+2.7, +5.0] | +1.3 [+0.3, +2.4] |
| OF2 rule + opening - opening only (thinking off) | S_200 | 35 | +65.3 [+63.5, +67.2] | 0.000 | - | - | - | +86.5 [+84.1, +88.7] | +60.3 [+56.5, +64.0] | +81.2 [+77.4, +84.6] | +42.9 [+39.3, +46.7] | +55.4 [+52.0, +58.6] |
| OF3 commitment - compliant opening (A, thinking off) | S_200 | 35 | +1.9 [+0.4, +3.6] | 0.014 | - | - | - | -2.6 [-5.0, -0.4] | +3.5 [+0.5, +6.7] | -1.6 [-4.5, +1.5] | +6.8 [+2.8, +11.0] | +3.5 [-0.5, +7.6] |
| OF4 non-compliant opening - none (A, thinking off) | S_200 | 30 | -56.4 [-58.3, -54.4] | 0.000 | - | - | - | -65.3 [-67.9, -62.6] | -59.0 [-62.0, -55.9] | -56.3 [-59.8, -52.9] | -50.5 [-55.1, -45.5] | -50.8 [-55.0, -46.6] |
| OFX opening effect, thinking off - on (A, S(200)) | S_200 | 35 | -18.2 [-20.4, -15.9] | 0.000 | - | - | - | -5.6 [-9.5, -1.9] | -18.8 [-23.4, -14.3] | -54.6 [-59.3, -50.0] | -10.0 [-16.8, -3.5] | -1.9 [-7.8, +4.1] |

## Openings off on every model with openings-off grades (secondary pool)

| contrast | metric | cells | difference | p | Qwen3.6-27B-FP8 | Qwen3.8-27B-FP8 | Gemma-4-31B-FP8 | Qwen3-32B | Gemma-4-12B-FP8 | Qwen3.6-35B-A3B-FP8 | GLM-4.7-Flash-FP8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OF1 compliant opening - none (A, thinking off) | S_200 | 49 | -2.2 [-3.8, -0.7] | 0.002 | -1.1 [-5.2, +3.1] | -6.1 [-10.9, -1.6] | -1.0 [-3.3, +1.3] | -3.0 [-6.2, +0.2] | -1.3 [-4.3, +1.9] | -1.7 [-5.4, +2.2] | -1.4 [-5.7, +2.9] |
| OF2 rule + opening - opening only (thinking off) | S_200 | 49 | +60.0 [+58.3, +61.6] | 0.000 | +49.5 [+45.6, +53.3] | +44.1 [+40.3, +48.0] | +86.5 [+84.1, +88.7] | +60.3 [+56.5, +64.0] | +81.2 [+77.4, +84.6] | +42.9 [+39.3, +46.7] | +55.4 [+52.0, +58.6] |
| OF3 commitment - compliant opening (A, thinking off) | S_200 | 49 | +3.0 [+1.5, +4.6] | 0.000 | +3.9 [+0.4, +7.5] | +7.6 [+4.0, +11.6] | -2.6 [-5.0, -0.4] | +3.5 [+0.5, +6.7] | -1.6 [-4.5, +1.5] | +6.8 [+2.8, +11.0] | +3.5 [-0.5, +7.6] |
| OF4 non-compliant opening - none (A, thinking off) | S_200 | 42 | -58.2 [-59.6, -56.7] | 0.000 | -60.1 [-64.0, -56.0] | -65.2 [-68.5, -61.9] | -65.3 [-67.9, -62.6] | -59.0 [-62.0, -55.9] | -56.3 [-59.8, -52.9] | -50.5 [-55.1, -45.5] | -50.8 [-55.0, -46.6] |
| OFX opening effect, thinking off - on (A, S(200)) | S_200 | 49 | -16.3 [-18.5, -14.1] | 0.000 | -16.0 [-22.4, -9.3] | -7.1 [-13.5, -0.7] | -5.6 [-9.5, -1.9] | -18.8 [-23.4, -14.3] | -54.6 [-59.3, -50.0] | -10.0 [-16.8, -3.5] | -1.9 [-7.8, +4.1] |

## Per rule set

| contrast | exp06 | held_out_exp07 | new_exp09 |
|---|---|---|---|
| E1 A - examples only (thinking on) | +21.7 [+20.2, +23.2] | +9.2 [+7.5, +10.9] | +15.8 [+13.9, +17.7] |
| OF1 compliant opening - none (A, thinking off) | +0.7 [-1.1, +2.5] | -7.7 [-11.2, -4.4] | - |
| E2 examples only - CoT-Control prompt (thinking on) | +4.9 [+3.7, +6.2] | +0.7 [-0.6, +2.1] | +0.9 [-0.4, +2.2] |
| OF2 rule + opening - opening only (thinking off) | +75.4 [+73.3, +77.5] | +39.9 [+35.9, +43.8] | - |
| OF3 commitment - compliant opening (A, thinking off) | +2.2 [+0.3, +4.1] | +1.3 [-1.9, +4.7] | - |
| OF4 non-compliant opening - none (A, thinking off) | -67.8 [-70.2, -65.4] | -33.4 [-36.3, -30.4] | - |

## By CoT necessity (5 direct answers per model and question)

Questions per label: {"GLM-4.7-Flash-FP8": {"mixed": 27, "necessary": 76, "unnecessary": 17}, "Gemma-4-12B-FP8": {"mixed": 5, "necessary": 75, "unnecessary": 40}, "Gemma-4-31B-FP8": {"mixed": 5, "necessary": 62, "unnecessary": 53}, "Qwen3-32B": {"mixed": 6, "necessary": 77, "unnecessary": 37}, "Qwen3.6-27B-FP8": {"mixed": 28, "necessary": 57, "unnecessary": 35}, "Qwen3.6-35B-A3B-FP8": {"mixed": 25, "necessary": 60, "unnecessary": 35}, "Qwen3.8-27B-FP8": {"mixed": 28, "necessary": 60, "unnecessary": 32}}

| contrast | necessary | mixed | unnecessary |
|---|---|---|---|
| C1 thinking off - on (CoT-Control prompt) | +36.1 [+32.3, +39.8] | +33.4 [+26.1, +41.4] | +39.2 [+35.0, +43.4] |
| C2 thinking off - on (A) | +28.6 [+26.9, +30.1] | +28.9 [+18.4, +32.2] | +27.7 [+25.0, +30.8] |
| C3 A - CoT-Control prompt, thinking on | +18.2 [+16.9, +19.7] | +15.0 [+12.9, +19.4] | +20.3 [+18.3, +22.7] |
| E1 A - examples only (thinking on) | +16.2 [+14.9, +17.5] | +13.6 [+10.7, +18.8] | +16.2 [+14.1, +18.2] |
