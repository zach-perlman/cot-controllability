# exp06 analysis: with_qwen35_lenient (UNVERIFIED until a human adds it to VERIFIED.md)

Models: Qwen3.8-27B-FP8, Gemma-4-31B-FP8, Qwen3-32B, Qwen3.6-27B-FP8 (held out), Qwen3.5-9B (held out). Missing parts: ['Qwen3.8-27B-FP8_thinking_off_prefill', 'Gemma-4-31B-FP8_thinking_off_prefill', 'Qwen3-32B_thinking_off_prefill', 'Qwen3.6-27B-FP8_thinking_off_prefill', 'Qwen3.5-9B_thinking_off_prefill'].
100 new questions x 5 rules per cell; thinking on unless stated. S(1000) = KM % of traces with no violation in the first 1000 graded tokens (prefill rows: after the prefill).

## Primary contrasts: S(1000), 5-rule mean (Holm within the screened family and within each held-out model)

| model | contrast | difference (pts) | p | p Holm |
|---|---|---|---|---|
| Qwen3.8-27B-FP8 | C1 upgraded vs stacked (no opening) | 19.9 [14.8, 25.1] | 0.001 | 0.009 |
| Qwen3.8-27B-FP8 | C2 compliant opening (upgraded) | 11.4 [4.9, 17.8] | 0.001 | 0.009 |
| Qwen3.8-27B-FP8 | C3 rule beyond copying (compliant opening) | 39.4 [34.2, 44.1] | 0.001 | 0.009 |
| Gemma-4-31B-FP8 | C1 upgraded vs stacked (no opening) | 32.7 [27.0, 37.6] | 0.001 | 0.009 |
| Gemma-4-31B-FP8 | C2 compliant opening (upgraded) | 14.6 [9.4, 20.2] | 0.001 | 0.009 |
| Gemma-4-31B-FP8 | C3 rule beyond copying (compliant opening) | 67.3 [62.6, 71.6] | 0.001 | 0.009 |
| Qwen3-32B | C1 upgraded vs stacked (no opening) | 33.0 [28.7, 37.5] | 0.001 | 0.009 |
| Qwen3-32B | C2 compliant opening (upgraded) | 8.8 [4.0, 13.5] | 0.001 | 0.009 |
| Qwen3-32B | C3 rule beyond copying (compliant opening) | 44.6 [39.8, 49.5] | 0.001 | 0.009 |
| Qwen3.6-27B-FP8 (held out) | C1 upgraded vs stacked (no opening) | 22.2 [17.5, 26.9] | 0.001 | 0.003 |
| Qwen3.6-27B-FP8 (held out) | C2 compliant opening (upgraded) | 23.5 [18.1, 28.8] | 0.001 | 0.003 |
| Qwen3.6-27B-FP8 (held out) | C3 rule beyond copying (compliant opening) | 43.3 [38.0, 48.4] | 0.001 | 0.003 |
| Qwen3.5-9B (held out) | C1 upgraded vs stacked (no opening) | 2.6 [1.1, 4.2] | 0.001 | 0.003 |
| Qwen3.5-9B (held out) | C2 compliant opening (upgraded) | 8.9 [5.9, 12.3] | 0.001 | 0.003 |
| Qwen3.5-9B (held out) | C3 rule beyond copying (compliant opening) | 6.9 [4.0, 10.2] | 0.001 | 0.003 |

## Cells

| model | prompt | opening | S(1000) 5 rules | S(1000) 4 opener rules | clean through 1000 % | S(1000) word suppression | empty % | median graded tokens |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-FP8 | baseline | none | 4.3 [2.7, 6.2] | 1.0 [0.3, 2.0] | 3.6 [2.0, 5.4] | 17.7 [10.8, 25.5] | 0.0 | 96 |
| Qwen3.8-27B-FP8 | baseline | prefill_compliant | 23.0 [19.6, 26.1] | 24.0 [20.0, 27.8] | 15.4 [12.6, 18.2] | 18.9 [11.4, 26.9] | 0.0 | 256 |
| Qwen3.8-27B-FP8 | baseline | prefill_noncompliant | n/a | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | n/a | 0.0 | 64 |
| Qwen3.8-27B-FP8 | stacked | none | 15.1 [11.9, 18.3] | 12.6 [8.9, 16.3] | 8.6 [6.2, 10.8] | 25.3 [17.1, 34.0] | 0.0 | 160 |
| Qwen3.8-27B-FP8 | stacked | prefill_compliant | 31.7 [27.4, 36.2] | 35.0 [30.0, 40.2] | 16.8 [13.6, 20.0] | 18.9 [11.3, 27.2] | 0.0 | 364 |
| Qwen3.8-27B-FP8 | stacked | prefill_noncompliant | n/a | 1.4 [0.0, 3.6] | 0.2 [0.0, 0.8] | n/a | 0.0 | 96 |
| Qwen3.8-27B-FP8 | upgraded | none | 35.0 [30.3, 39.9] | 37.1 [31.5, 42.6] | 19.2 [15.4, 23.0] | 26.8 [17.2, 36.9] | 0.0 | 548 |
| Qwen3.8-27B-FP8 | upgraded | prefill_compliant | 46.4 [41.6, 50.9] | 50.6 [45.0, 55.5] | 26.6 [22.4, 30.8] | 29.5 [19.6, 39.6] | 0.0 | 528 |
| Qwen3.8-27B-FP8 | upgraded | prefill_noncompliant | n/a | 4.5 [2.3, 7.2] | 2.5 [1.0, 4.8] | n/a | 0.0 | 96 |
| Qwen3.8-27B-FP8 | no_rule | prefill_compliant | 6.9 [4.9, 9.2] | 7.4 [5.1, 9.8] | 4.4 [2.6, 6.4] | 5.0 [1.0, 9.0] | 0.0 | 96 |
| Gemma-4-31B-FP8 | baseline | none | 9.2 [7.2, 11.2] | 0.0 [0.0, 0.0] | 8.6 [6.6, 10.6] | 46.0 [36.2, 55.8] | 0.0 | 96 |
| Gemma-4-31B-FP8 | baseline | prefill_compliant | 21.3 [17.9, 24.7] | 12.4 [8.8, 16.0] | 16.6 [13.8, 19.6] | 57.0 [47.1, 66.4] | 0.0 | 288 |
| Gemma-4-31B-FP8 | baseline | prefill_noncompliant | n/a | 0.6 [0.0, 1.5] | 0.5 [0.0, 1.2] | n/a | 0.0 | 96 |
| Gemma-4-31B-FP8 | stacked | none | 22.2 [19.0, 25.9] | 9.5 [6.4, 13.0] | 10.8 [8.4, 13.4] | 73.0 [63.6, 82.0] | 29.0 | 224 |
| Gemma-4-31B-FP8 | stacked | prefill_compliant | 44.0 [39.6, 48.1] | 35.3 [30.5, 39.9] | 23.8 [19.8, 27.8] | 78.7 [70.4, 86.8] | 0.0 | 514 |
| Gemma-4-31B-FP8 | stacked | prefill_noncompliant | n/a | 3.7 [1.7, 5.9] | 2.0 [0.8, 3.5] | n/a | 0.0 | 96 |
| Gemma-4-31B-FP8 | upgraded | none | 54.9 [50.2, 59.2] | 53.6 [48.0, 58.5] | 30.2 [25.2, 35.2] | 60.2 [49.4, 70.5] | 0.0 | 774 |
| Gemma-4-31B-FP8 | upgraded | prefill_compliant | 69.5 [65.2, 73.4] | 66.9 [61.9, 71.5] | 39.4 [34.0, 45.0] | 79.8 [70.9, 88.3] | 0.0 | 825 |
| Gemma-4-31B-FP8 | upgraded | prefill_noncompliant | n/a | 18.3 [14.5, 22.4] | 9.5 [6.8, 12.5] | n/a | 0.0 | 96 |
| Gemma-4-31B-FP8 | no_rule | prefill_compliant | 2.2 [1.0, 3.6] | 0.2 [0.0, 0.8] | 2.2 [1.0, 3.6] | 10.0 [5.0, 16.0] | 0.0 | 96 |
| Qwen3-32B | baseline | none | 4.7 [2.8, 6.8] | 2.2 [0.5, 4.2] | 2.8 [1.6, 4.2] | 14.7 [8.1, 22.2] | 0.0 | 95 |
| Qwen3-32B | baseline | prefill_compliant | 27.9 [24.4, 31.9] | 31.2 [26.9, 35.6] | 21.0 [18.0, 24.2] | 15.0 [8.2, 22.4] | 0.0 | 224 |
| Qwen3-32B | baseline | prefill_noncompliant | n/a | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | n/a | 0.0 | 64 |
| Qwen3-32B | stacked | none | 8.0 [5.7, 10.2] | 5.7 [3.4, 7.9] | 4.0 [2.4, 5.8] | 17.1 [9.7, 25.0] | 0.0 | 95 |
| Qwen3-32B | stacked | prefill_compliant | 32.5 [28.0, 37.0] | 36.9 [31.6, 42.5] | 20.6 [17.0, 24.2] | 14.8 [7.7, 22.5] | 0.0 | 360 |
| Qwen3-32B | stacked | prefill_noncompliant | n/a | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | n/a | 0.0 | 64 |
| Qwen3-32B | upgraded | none | 40.9 [37.2, 44.7] | 45.7 [41.2, 49.8] | 24.6 [21.2, 28.0] | 22.0 [13.4, 30.9] | 0.0 | 536 |
| Qwen3-32B | upgraded | prefill_compliant | 49.7 [45.5, 53.9] | 58.1 [53.3, 63.1] | 30.4 [27.2, 33.8] | 16.2 [8.2, 24.7] | 0.0 | 581 |
| Qwen3-32B | upgraded | prefill_noncompliant | n/a | 1.2 [0.0, 2.6] | 0.2 [0.0, 0.8] | n/a | 0.0 | 64 |
| Qwen3-32B | no_rule | prefill_compliant | 5.1 [3.3, 7.1] | 5.3 [3.4, 7.5] | 4.2 [2.6, 6.0] | 4.0 [1.0, 8.0] | 0.0 | 96 |
| Qwen3.6-27B-FP8 (held out) | baseline | none | 3.2 [2.0, 4.6] | 1.5 [0.5, 2.5] | 2.8 [1.6, 4.2] | 10.0 [5.0, 16.0] | 0.0 | 96 |
| Qwen3.6-27B-FP8 (held out) | baseline | prefill_compliant | 33.5 [29.3, 37.5] | 37.4 [32.1, 42.3] | 27.0 [23.2, 30.4] | 18.2 [10.9, 26.5] | 0.0 | 503 |
| Qwen3.6-27B-FP8 (held out) | baseline | prefill_noncompliant | n/a | 5.7 [3.5, 8.0] | 4.5 [2.5, 6.5] | n/a | 0.0 | 64 |
| Qwen3.6-27B-FP8 (held out) | stacked | none | 6.5 [4.3, 8.9] | 5.4 [3.0, 7.9] | 4.8 [3.0, 6.8] | 11.0 [5.0, 17.0] | 0.0 | 96 |
| Qwen3.6-27B-FP8 (held out) | stacked | prefill_compliant | 50.4 [45.8, 54.6] | 54.1 [48.8, 59.0] | 36.0 [31.8, 40.0] | 35.6 [26.2, 45.3] | 0.0 | 702 |
| Qwen3.6-27B-FP8 (held out) | stacked | prefill_noncompliant | n/a | 14.1 [10.8, 17.4] | 8.5 [6.0, 11.2] | n/a | 0.0 | 64 |
| Qwen3.6-27B-FP8 (held out) | upgraded | none | 28.7 [24.8, 32.8] | 35.2 [30.3, 40.2] | 24.6 [21.0, 28.2] | 3.0 [0.0, 7.0] | 0.0 | 653 |
| Qwen3.6-27B-FP8 (held out) | upgraded | prefill_compliant | 52.3 [47.9, 56.3] | 59.9 [54.7, 64.8] | 46.6 [42.8, 50.4] | 21.9 [13.0, 31.0] | 0.0 | 994 |
| Qwen3.6-27B-FP8 (held out) | upgraded | prefill_noncompliant | n/a | 2.8 [1.2, 4.3] | 2.5 [1.0, 4.0] | n/a | 0.0 | 64 |
| Qwen3.6-27B-FP8 (held out) | no_rule | prefill_compliant | 8.9 [6.3, 11.7] | 9.7 [6.6, 13.0] | 7.2 [5.0, 9.6] | 6.0 [2.0, 11.0] | 0.0 | 96 |
| Qwen3.5-9B (held out) | baseline | none | 0.6 [0.0, 1.4] | 0.0 [0.0, 0.0] | 0.6 [0.0, 1.4] | 3.0 [0.0, 7.0] | 0.0 | 96 |
| Qwen3.5-9B (held out) | baseline | prefill_compliant | 5.6 [3.5, 7.9] | 5.4 [3.4, 7.8] | 4.4 [2.6, 6.4] | 6.0 [2.0, 11.0] | 0.0 | 128 |
| Qwen3.5-9B (held out) | baseline | prefill_noncompliant | n/a | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | n/a | 0.0 | 64 |
| Qwen3.5-9B (held out) | stacked | none | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 64 |
| Qwen3.5-9B (held out) | stacked | prefill_compliant | 6.3 [4.2, 8.7] | 7.0 [4.5, 10.1] | 4.6 [3.0, 6.2] | 3.3 [0.0, 7.1] | 0.0 | 160 |
| Qwen3.5-9B (held out) | stacked | prefill_noncompliant | n/a | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | n/a | 0.0 | 64 |
| Qwen3.5-9B (held out) | upgraded | none | 2.6 [1.1, 4.2] | 3.2 [1.4, 5.3] | 2.2 [1.0, 3.6] | 0.0 [0.0, 0.0] | 0.0 | 160 |
| Qwen3.5-9B (held out) | upgraded | prefill_compliant | 11.5 [8.8, 14.6] | 13.1 [9.9, 16.8] | 8.0 [5.8, 10.4] | 5.0 [1.0, 10.0] | 0.0 | 256 |
| Qwen3.5-9B (held out) | upgraded | prefill_noncompliant | n/a | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | n/a | 0.0 | 64 |
| Qwen3.5-9B (held out) | no_rule | prefill_compliant | 4.6 [2.9, 6.4] | 3.7 [2.0, 5.6] | 3.8 [2.4, 5.4] | 8.0 [3.0, 14.0] | 0.0 | 96 |

## Thinking-off cells (rule on the <output_reasoning> tag content; openings prefilled after the opening tag)

| model | prompt | opening | clean at 200 % | clean at 200 %, 4 opener rules | clean at 200 %, tagged rows only | no tag content % | of which closed with </think> % of rows | median graded tokens |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-FP8 | baseline | thinking_off | 32.2 [28.4, 36.0] | 31.8 [27.5, 36.0] | 47.1 [42.0, 52.1] | 31.6 | 0.8 | 288 |
| Qwen3.8-27B-FP8 | stacked | thinking_off | 59.8 [55.6, 64.0] | 59.8 [54.8, 65.0] | 66.3 [62.1, 70.6] | 9.8 | 2.6 | 574 |
| Qwen3.8-27B-FP8 | upgraded | thinking_off | 78.6 [74.8, 82.4] | 81.0 [77.2, 84.8] | 84.3 [80.9, 87.6] | 6.8 | 1.6 | 734 |
| Gemma-4-31B-FP8 | baseline | thinking_off | 49.8 [44.4, 55.2] | 46.5 [41.0, 51.8] | 49.8 [44.4, 55.2] | 0.0 | 0.0 | 358 |
| Gemma-4-31B-FP8 | stacked | thinking_off | 60.4 [53.2, 66.6] | 60.0 [52.5, 66.5] | 60.4 [53.2, 66.6] | 0.0 | 0.0 | 332 |
| Gemma-4-31B-FP8 | upgraded | thinking_off | 74.6 [69.6, 79.2] | 80.2 [75.8, 84.5] | 74.6 [69.6, 79.2] | 0.0 | 0.0 | 364 |
| Qwen3-32B | baseline | thinking_off | 38.0 [34.0, 42.0] | 38.8 [34.8, 42.8] | 41.6 [37.4, 45.6] | 8.6 | 0.0 | 279 |
| Qwen3-32B | stacked | thinking_off | 61.4 [56.4, 66.4] | 63.0 [58.0, 68.2] | 62.8 [57.6, 68.0] | 2.2 | 0.0 | 324 |
| Qwen3-32B | upgraded | thinking_off | 80.0 [76.4, 83.4] | 85.5 [82.0, 88.8] | 81.3 [77.8, 84.6] | 1.6 | 0.0 | 391 |
| Qwen3.6-27B-FP8 (held out) | baseline | thinking_off | 43.2 [38.6, 47.4] | 41.0 [36.0, 45.5] | 54.8 [50.0, 59.3] | 21.2 | 0.2 | 592 |
| Qwen3.6-27B-FP8 (held out) | stacked | thinking_off | 65.0 [60.4, 69.6] | 65.0 [59.7, 70.3] | 68.6 [64.1, 73.0] | 5.2 | 0.8 | 948 |
| Qwen3.6-27B-FP8 (held out) | upgraded | thinking_off | 57.0 [52.4, 61.4] | 58.2 [52.8, 63.5] | 74.0 [69.9, 77.9] | 23.0 | 3.0 | 717 |
| Qwen3.5-9B (held out) | baseline | thinking_off | 14.0 [10.8, 17.0] | 11.5 [8.5, 14.5] | 19.2 [15.0, 23.2] | 27.0 | 7.8 | 222 |
| Qwen3.5-9B (held out) | stacked | thinking_off | 34.4 [30.2, 38.6] | 36.5 [31.8, 41.0] | 45.6 [40.2, 51.0] | 24.6 | 8.6 | 232 |
| Qwen3.5-9B (held out) | upgraded | thinking_off | 49.8 [46.0, 53.8] | 52.0 [47.5, 56.8] | 74.1 [70.1, 78.1] | 32.8 | 26.4 | 392 |

## Secondary contrasts (paired; metric per row)

| family | row | model | metric | difference (pts) | p |
|---|---|---|---|---|---|
| compliant opening vs none | CoT-Control prompt | Qwen3.8-27B-FP8 | S_1000 | 18.6 [15.1, 22.0] | 0.001 |
| compliant opening vs none | stacked | Qwen3.8-27B-FP8 | S_1000 | 16.6 [12.0, 21.5] | 0.001 |
| compliant opening vs none | upgraded (stacked + 3-shot + guide) | Qwen3.8-27B-FP8 | S_1000 | 11.4 [4.9, 17.8] | 0.001 |
| compliant opening vs none | CoT-Control prompt | Gemma-4-31B-FP8 | S_1000 | 12.1 [8.6, 15.7] | 0.001 |
| compliant opening vs none | stacked | Gemma-4-31B-FP8 | S_1000 | 21.8 [16.5, 26.3] | 0.001 |
| compliant opening vs none | upgraded (stacked + 3-shot + guide) | Gemma-4-31B-FP8 | S_1000 | 14.6 [9.4, 20.2] | 0.001 |
| compliant opening vs none | CoT-Control prompt | Qwen3-32B | S_1000 | 23.2 [19.1, 27.6] | 0.001 |
| compliant opening vs none | stacked | Qwen3-32B | S_1000 | 24.5 [19.4, 29.3] | 0.001 |
| compliant opening vs none | upgraded (stacked + 3-shot + guide) | Qwen3-32B | S_1000 | 8.8 [4.0, 13.5] | 0.001 |
| compliant opening vs none | CoT-Control prompt | Qwen3.6-27B-FP8 (held out) | S_1000 | 30.3 [25.7, 34.5] | 0.001 |
| compliant opening vs none | stacked | Qwen3.6-27B-FP8 (held out) | S_1000 | 43.9 [38.8, 49.0] | 0.001 |
| compliant opening vs none | upgraded (stacked + 3-shot + guide) | Qwen3.6-27B-FP8 (held out) | S_1000 | 23.5 [18.1, 28.8] | 0.001 |
| compliant opening vs none | CoT-Control prompt | Qwen3.5-9B (held out) | S_1000 | 5.0 [3.0, 7.2] | 0.001 |
| compliant opening vs none | stacked | Qwen3.5-9B (held out) | S_1000 | 6.3 [4.2, 8.7] | 0.001 |
| compliant opening vs none | upgraded (stacked + 3-shot + guide) | Qwen3.5-9B (held out) | S_1000 | 8.9 [5.9, 12.3] | 0.001 |
| rule beyond copying (compliant opening) | CoT-Control prompt | Qwen3.8-27B-FP8 | S_1000 | 16.0 [12.1, 19.9] | 0.001 |
| rule beyond copying (compliant opening) | stacked | Qwen3.8-27B-FP8 | S_1000 | 24.8 [19.6, 29.9] | 0.001 |
| rule beyond copying (compliant opening) | upgraded (stacked + 3-shot + guide) | Qwen3.8-27B-FP8 | S_1000 | 39.4 [34.2, 44.1] | 0.001 |
| rule beyond copying (compliant opening) | CoT-Control prompt | Gemma-4-31B-FP8 | S_1000 | 19.1 [15.6, 22.4] | 0.001 |
| rule beyond copying (compliant opening) | stacked | Gemma-4-31B-FP8 | S_1000 | 41.8 [37.0, 46.1] | 0.001 |
| rule beyond copying (compliant opening) | upgraded (stacked + 3-shot + guide) | Gemma-4-31B-FP8 | S_1000 | 67.3 [62.6, 71.6] | 0.001 |
| rule beyond copying (compliant opening) | CoT-Control prompt | Qwen3-32B | S_1000 | 22.9 [18.9, 27.1] | 0.001 |
| rule beyond copying (compliant opening) | stacked | Qwen3-32B | S_1000 | 27.4 [22.6, 32.2] | 0.001 |
| rule beyond copying (compliant opening) | upgraded (stacked + 3-shot + guide) | Qwen3-32B | S_1000 | 44.6 [39.8, 49.5] | 0.001 |
| rule beyond copying (compliant opening) | CoT-Control prompt | Qwen3.6-27B-FP8 (held out) | S_1000 | 24.6 [19.8, 29.1] | 0.001 |
| rule beyond copying (compliant opening) | stacked | Qwen3.6-27B-FP8 (held out) | S_1000 | 41.4 [36.5, 46.2] | 0.001 |
| rule beyond copying (compliant opening) | upgraded (stacked + 3-shot + guide) | Qwen3.6-27B-FP8 (held out) | S_1000 | 43.3 [38.0, 48.4] | 0.001 |
| rule beyond copying (compliant opening) | CoT-Control prompt | Qwen3.5-9B (held out) | S_1000 | 1.0 [-1.4, 3.6] | 0.419 |
| rule beyond copying (compliant opening) | stacked | Qwen3.5-9B (held out) | S_1000 | 1.7 [-1.1, 4.7] | 0.234 |
| rule beyond copying (compliant opening) | upgraded (stacked + 3-shot + guide) | Qwen3.5-9B (held out) | S_1000 | 6.9 [4.0, 10.2] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | CoT-Control prompt | Qwen3.8-27B-FP8 | S_1000_openers | 24.0 [20.0, 27.8] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | stacked | Qwen3.8-27B-FP8 | S_1000_openers | 33.5 [28.4, 38.6] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | upgraded (stacked + 3-shot + guide) | Qwen3.8-27B-FP8 | S_1000_openers | 46.1 [40.1, 51.5] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | CoT-Control prompt | Gemma-4-31B-FP8 | S_1000_openers | 11.8 [8.5, 15.1] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | stacked | Gemma-4-31B-FP8 | S_1000_openers | 31.6 [26.8, 36.1] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | upgraded (stacked + 3-shot + guide) | Gemma-4-31B-FP8 | S_1000_openers | 48.6 [43.2, 53.7] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | CoT-Control prompt | Qwen3-32B | S_1000_openers | 31.2 [26.9, 35.6] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | stacked | Qwen3-32B | S_1000_openers | 36.9 [31.6, 42.5] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | upgraded (stacked + 3-shot + guide) | Qwen3-32B | S_1000_openers | 56.8 [51.7, 61.9] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | CoT-Control prompt | Qwen3.6-27B-FP8 (held out) | S_1000_openers | 31.7 [26.0, 37.1] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | stacked | Qwen3.6-27B-FP8 (held out) | S_1000_openers | 40.0 [35.0, 44.8] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | upgraded (stacked + 3-shot + guide) | Qwen3.6-27B-FP8 (held out) | S_1000_openers | 57.1 [51.8, 62.5] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | CoT-Control prompt | Qwen3.5-9B (held out) | S_1000_openers | 5.4 [3.4, 7.8] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | stacked | Qwen3.5-9B (held out) | S_1000_openers | 7.0 [4.5, 10.1] | 0.001 |
| compliant vs non-compliant opening (4 opener rules) | upgraded (stacked + 3-shot + guide) | Qwen3.5-9B (held out) | S_1000_openers | 13.1 [9.9, 16.8] | 0.001 |
| upgraded vs stacked prompt | no opening | Qwen3.8-27B-FP8 | S_1000 | 19.9 [14.8, 25.1] | 0.001 |
| upgraded vs stacked prompt | compliant opening | Qwen3.8-27B-FP8 | S_1000 | 14.6 [8.5, 20.5] | 0.001 |
| upgraded vs stacked prompt | non-compliant opening | Qwen3.8-27B-FP8 | S_1000_openers | 3.1 [-0.1, 6.4] | 0.057 |
| upgraded vs stacked prompt | no opening | Gemma-4-31B-FP8 | S_1000 | 32.7 [27.0, 37.6] | 0.001 |
| upgraded vs stacked prompt | compliant opening | Gemma-4-31B-FP8 | S_1000 | 25.5 [20.2, 30.8] | 0.001 |
| upgraded vs stacked prompt | non-compliant opening | Gemma-4-31B-FP8 | S_1000_openers | 14.6 [10.3, 19.0] | 0.001 |
| upgraded vs stacked prompt | no opening | Qwen3-32B | S_1000 | 33.0 [28.7, 37.5] | 0.001 |
| upgraded vs stacked prompt | compliant opening | Qwen3-32B | S_1000 | 17.2 [12.7, 21.7] | 0.001 |
| upgraded vs stacked prompt | non-compliant opening | Qwen3-32B | S_1000_openers | 1.2 [0.0, 2.6] | 0.076 |
| upgraded vs stacked prompt | no opening | Qwen3.6-27B-FP8 (held out) | S_1000 | 22.2 [17.5, 26.9] | 0.001 |
| upgraded vs stacked prompt | compliant opening | Qwen3.6-27B-FP8 (held out) | S_1000 | 1.9 [-3.6, 7.6] | 0.520 |
| upgraded vs stacked prompt | non-compliant opening | Qwen3.6-27B-FP8 (held out) | S_1000_openers | -11.3 [-15.0, -7.6] | 0.001 |
| upgraded vs stacked prompt | no opening | Qwen3.5-9B (held out) | S_1000 | 2.6 [1.1, 4.2] | 0.001 |
| upgraded vs stacked prompt | compliant opening | Qwen3.5-9B (held out) | S_1000 | 5.2 [1.5, 8.6] | 0.006 |
| upgraded vs stacked prompt | non-compliant opening | Qwen3.5-9B (held out) | S_1000_openers | 0.0 [0.0, 0.0] | 1.000 |
| interaction | opening gain under upgraded - under stacked | Qwen3.8-27B-FP8 | S_1000 | -5.3 [-13.4, 2.2] | 0.202 |
| interaction | opening gain under upgraded - under stacked | Gemma-4-31B-FP8 | S_1000 | -7.2 [-14.5, 1.1] | 0.097 |
| interaction | opening gain under upgraded - under stacked | Qwen3-32B | S_1000 | -15.7 [-22.3, -9.1] | 0.001 |
| interaction | opening gain under upgraded - under stacked | Qwen3.6-27B-FP8 (held out) | S_1000 | -20.3 [-27.7, -13.0] | 0.001 |
| interaction | opening gain under upgraded - under stacked | Qwen3.5-9B (held out) | S_1000 | 2.7 [-1.5, 6.3] | 0.188 |

## Opening specificity: compliant opening's gain over none, % of texts reaching 200 tokens clean, thinking on vs off

| model | prompt | on (pts) | off (pts) | on - off (pts) | on (% headroom) | off (% headroom) | on - off (% headroom) |
|---|---|---|---|---|---|---|---|

## CoT specificity: gain over baseline in % of texts reaching 200 tokens clean, thinking on vs off

| model | prompt | on (pts) | off (pts) | on - off (pts) | on - off, tagged rows only | on (% headroom) | off (% headroom) | on - off (% headroom) |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-FP8 | stacked | 16.8 [14.0, 19.6] | 27.6 [23.2, 32.0] | -10.8 [-16.2, -5.6] | -2.4 [-8.1, 3.4] | 18.1 [15.1, 21.1] | 40.7 [35.0, 46.6] | -22.6 [-28.6, -16.6] |
| Qwen3.8-27B-FP8 | upgraded | 57.2 [53.2, 61.4] | 46.4 [41.2, 51.6] | 10.8 [4.0, 17.8] | 19.9 [13.0, 26.8] | 61.6 [57.4, 66.0] | 68.4 [62.4, 74.2] | -6.8 [-13.9, 0.3] |
| Gemma-4-31B-FP8 | stacked | 24.0 [20.4, 27.4] | 10.6 [5.6, 15.6] | 13.4 [6.6, 20.2] | 13.4 [6.6, 20.2] | 28.9 [24.8, 32.6] | 21.1 [11.1, 31.0] | 7.8 [-3.7, 19.3] |
| Gemma-4-31B-FP8 | upgraded | 64.2 [60.2, 68.0] | 24.8 [19.8, 29.8] | 39.4 [32.6, 45.8] | 39.4 [32.6, 45.8] | 77.3 [72.8, 81.6] | 49.4 [41.1, 57.3] | 27.9 [18.7, 37.1] |
| Qwen3-32B | stacked | 3.8 [1.2, 6.4] | 23.4 [18.8, 28.2] | -19.6 [-24.8, -14.4] | -17.4 [-22.9, -12.1] | 4.2 [1.4, 7.1] | 37.7 [30.6, 45.3] | -33.5 [-40.8, -26.2] |
| Qwen3-32B | upgraded | 50.4 [46.8, 54.0] | 42.0 [37.8, 46.6] | 8.4 [3.0, 13.2] | 10.7 [5.2, 16.0] | 56.0 [52.5, 59.3] | 67.7 [62.4, 73.0] | -11.7 [-17.1, -6.2] |
| Qwen3.6-27B-FP8 (held out) | stacked | 4.8 [1.6, 7.8] | 21.8 [16.0, 27.6] | -17.0 [-24.2, -10.0] | -8.9 [-15.7, -2.4] | 5.2 [1.8, 8.4] | 38.4 [29.6, 46.8] | -33.2 [-42.9, -23.3] |
| Qwen3.6-27B-FP8 (held out) | upgraded | 43.4 [39.0, 47.8] | 13.8 [8.8, 18.8] | 29.6 [23.4, 35.8] | 24.2 [17.3, 30.6] | 47.0 [42.7, 51.2] | 24.3 [16.0, 31.7] | 22.7 [14.3, 31.4] |
| Qwen3.5-9B (held out) | stacked | -3.0 [-4.4, -1.6] | 20.4 [16.0, 25.0] | -23.4 [-28.4, -18.8] | -29.4 [-35.8, -23.4] | -3.1 [-4.6, -1.6] | 23.7 [18.8, 28.7] | -26.8 [-32.1, -21.6] |
| Qwen3.5-9B (held out) | upgraded | 15.8 [12.0, 19.4] | 35.8 [31.4, 40.6] | -20.0 [-25.8, -14.6] | -39.1 [-45.5, -32.9] | 16.3 [12.4, 20.0] | 41.6 [37.0, 46.4] | -25.3 [-31.3, -19.8] |

## Sensitivity (post hoc): </think> accepted as the closing tag

Thinking-off rows with no closed <output_reasoning> block whose reasoning runs from the opening tag to a </think> are graded on that text (cc_grade.grade_row); every other row as above.

| model | prompt | opening | clean at 200 % (strict) | clean at 200 % (lenient) | rescored % of rows |
|---|---|---|---|---|---|
| Qwen3.8-27B-FP8 | baseline | thinking_off | 32.2 [28.4, 36.0] | 32.4 [28.4, 36.2] | 0.8 |
| Qwen3.8-27B-FP8 | stacked | thinking_off | 59.8 [55.6, 64.0] | 61.6 [57.4, 65.6] | 2.6 |
| Qwen3.8-27B-FP8 | upgraded | thinking_off | 78.6 [74.8, 82.4] | 79.6 [76.0, 83.2] | 1.2 |
| Gemma-4-31B-FP8 | baseline | thinking_off | 49.8 [44.4, 55.2] | 49.8 [44.4, 55.2] | 0.0 |
| Gemma-4-31B-FP8 | stacked | thinking_off | 60.4 [53.2, 66.6] | 60.4 [53.2, 66.6] | 0.0 |
| Gemma-4-31B-FP8 | upgraded | thinking_off | 74.6 [69.6, 79.2] | 74.6 [69.6, 79.2] | 0.0 |
| Qwen3-32B | baseline | thinking_off | 38.0 [34.0, 42.0] | 38.0 [34.0, 42.0] | 0.0 |
| Qwen3-32B | stacked | thinking_off | 61.4 [56.4, 66.4] | 61.4 [56.4, 66.4] | 0.0 |
| Qwen3-32B | upgraded | thinking_off | 80.0 [76.4, 83.4] | 80.0 [76.4, 83.4] | 0.0 |
| Qwen3.6-27B-FP8 (held out) | baseline | thinking_off | 43.2 [38.6, 47.4] | 43.2 [38.6, 47.4] | 0.2 |
| Qwen3.6-27B-FP8 (held out) | stacked | thinking_off | 65.0 [60.4, 69.6] | 65.0 [60.4, 69.6] | 0.8 |
| Qwen3.6-27B-FP8 (held out) | upgraded | thinking_off | 57.0 [52.4, 61.4] | 59.2 [54.6, 63.6] | 3.0 |
| Qwen3.5-9B (held out) | baseline | thinking_off | 14.0 [10.8, 17.0] | 15.4 [12.2, 18.4] | 7.4 |
| Qwen3.5-9B (held out) | stacked | thinking_off | 34.4 [30.2, 38.6] | 38.4 [34.2, 42.8] | 7.4 |
| Qwen3.5-9B (held out) | upgraded | thinking_off | 49.8 [46.0, 53.8] | 68.6 [64.8, 72.6] | 26.4 |

| model | gain | on - off, % headroom (strict) | on - off, % headroom (lenient) |
|---|---|---|---|
| Qwen3.8-27B-FP8 | upgraded prompt | -6.8 [-13.9, 0.3] | -8.2 [-15.2, -1.1] |
| Gemma-4-31B-FP8 | upgraded prompt | 27.9 [18.7, 37.1] | 27.9 [18.7, 37.1] |
| Qwen3-32B | upgraded prompt | -11.7 [-17.1, -6.2] | -11.7 [-17.1, -6.2] |
| Qwen3.6-27B-FP8 (held out) | upgraded prompt | 22.7 [14.3, 31.4] | 18.8 [10.6, 27.6] |
| Qwen3.5-9B (held out) | upgraded prompt | -25.3 [-31.3, -19.8] | -46.6 [-52.8, -40.9] |

## Accuracy and whole-trace compliance (full-trace cells, no opening): paired differences

| model | a - b | metric | difference (pts) | p |
|---|---|---|---|---|
| Qwen3.8-27B-FP8 | upgraded - stacked | correct | -1.6 [-8.3, 5.5] | 0.723 |
| Qwen3.8-27B-FP8 | upgraded - stacked | whole_trace_compliant | 20.0 [11.4, 29.4] | 0.001 |
| Qwen3.8-27B-FP8 | stacked - baseline | correct | 1.6 [-4.9, 7.1] | 0.694 |
| Qwen3.8-27B-FP8 | stacked - baseline | whole_trace_compliant | 14.4 [8.0, 21.8] | 0.001 |
| Qwen3.8-27B-FP8 | upgraded - baseline | correct | 0.0 [-7.8, 7.7] | 1.000 |
| Qwen3.8-27B-FP8 | upgraded - baseline | whole_trace_compliant | 34.4 [25.0, 43.9] | 0.001 |
| Gemma-4-31B-FP8 | upgraded - stacked | correct | 9.6 [2.7, 16.2] | 0.005 |
| Gemma-4-31B-FP8 | upgraded - stacked | whole_trace_compliant | 29.6 [20.9, 38.5] | 0.001 |
| Gemma-4-31B-FP8 | stacked - baseline | correct | -4.0 [-10.2, 1.9] | 0.265 |
| Gemma-4-31B-FP8 | stacked - baseline | whole_trace_compliant | 13.6 [7.3, 19.7] | 0.001 |
| Gemma-4-31B-FP8 | upgraded - baseline | correct | 5.6 [0.6, 11.5] | 0.050 |
| Gemma-4-31B-FP8 | upgraded - baseline | whole_trace_compliant | 43.2 [33.6, 52.5] | 0.001 |
| Qwen3-32B | upgraded - stacked | correct | -7.2 [-16.7, 2.5] | 0.188 |
| Qwen3-32B | upgraded - stacked | whole_trace_compliant | 33.6 [25.6, 41.4] | 0.001 |
| Qwen3-32B | stacked - baseline | correct | 2.4 [-6.7, 11.7] | 0.672 |
| Qwen3-32B | stacked - baseline | whole_trace_compliant | 4.8 [1.6, 8.6] | 0.006 |
| Qwen3-32B | upgraded - baseline | correct | -4.8 [-11.4, 1.7] | 0.202 |
| Qwen3-32B | upgraded - baseline | whole_trace_compliant | 38.4 [30.8, 45.6] | 0.001 |
| Qwen3.6-27B-FP8 (held out) | upgraded - stacked | correct | 0.8 [-6.7, 8.0] | 0.913 |
| Qwen3.6-27B-FP8 (held out) | upgraded - stacked | whole_trace_compliant | 9.6 [3.4, 15.8] | 0.002 |
| Qwen3.6-27B-FP8 (held out) | stacked - baseline | correct | -4.8 [-10.7, 1.5] | 0.161 |
| Qwen3.6-27B-FP8 (held out) | stacked - baseline | whole_trace_compliant | 1.6 [-1.5, 5.0] | 0.477 |
| Qwen3.6-27B-FP8 (held out) | upgraded - baseline | correct | -4.0 [-11.6, 4.0] | 0.342 |
| Qwen3.6-27B-FP8 (held out) | upgraded - baseline | whole_trace_compliant | 11.2 [5.2, 17.0] | 0.001 |
| Qwen3.5-9B (held out) | upgraded - stacked | correct | -8.0 [-16.7, -0.7] | 0.048 |
| Qwen3.5-9B (held out) | upgraded - stacked | whole_trace_compliant | 1.6 [0.0, 4.0] | 0.243 |
| Qwen3.5-9B (held out) | stacked - baseline | correct | 1.6 [-4.2, 7.7] | 0.684 |
| Qwen3.5-9B (held out) | stacked - baseline | whole_trace_compliant | 0.0 [0.0, 0.0] | 1.000 |
| Qwen3.5-9B (held out) | upgraded - baseline | correct | -6.4 [-13.5, 0.8] | 0.095 |
| Qwen3.5-9B (held out) | upgraded - baseline | whole_trace_compliant | 1.6 [0.0, 4.0] | 0.243 |

## Checks

```
{
 "thinking_on_status_share": {
  "aborted": 73.54042553191489,
  "closed": 16.868085106382978,
  "truncated": 9.548936170212766,
  "no_think_close": 0.0425531914893617
 },
 "thinking_on_empty_share_by_cell": {
  "Gemma-4-31B-FP8|baseline|none": 0.0,
  "Gemma-4-31B-FP8|baseline|prefill_compliant": 0.0,
  "Gemma-4-31B-FP8|baseline|prefill_noncompliant": 0.0,
  "Gemma-4-31B-FP8|no_rule|prefill_compliant": 0.0,
  "Gemma-4-31B-FP8|stacked|none": 28.999999999999996,
  "Gemma-4-31B-FP8|stacked|prefill_compliant": 0.0,
  "Gemma-4-31B-FP8|stacked|prefill_noncompliant": 0.0,
  "Gemma-4-31B-FP8|upgraded|none": 0.0,
  "Gemma-4-31B-FP8|upgraded|prefill_compliant": 0.0,
  "Gemma-4-31B-FP8|upgraded|prefill_noncompliant": 0.0,
  "Qwen3-32B|baseline|none": 0.0,
  "Qwen3-32B|baseline|prefill_compliant": 0.0,
  "Qwen3-32B|baseline|prefill_noncompliant": 0.0,
  "Qwen3-32B|no_rule|prefill_compliant": 0.0,
  "Qwen3-32B|stacked|none": 0.0,
  "Qwen3-32B|stacked|prefill_compliant": 0.0,
  "Qwen3-32B|stacked|prefill_noncompliant": 0.0,
  "Qwen3-32B|upgraded|none": 0.0,
  "Qwen3-32B|upgraded|prefill_compliant": 0.0,
  "Qwen3-32B|upgraded|prefill_noncompliant": 0.0,
  "Qwen3.5-9B|baseline|none": 0.0,
  "Qwen3.5-9B|baseline|prefill_compliant": 0.0,
  "Qwen3.5-9B|baseline|prefill_noncompliant": 0.0,
  "Qwen3.5-9B|no_rule|prefill_compliant": 0.0,
  "Qwen3.5-9B|stacked|none": 0.0,
  "Qwen3.5-9B|stacked|prefill_compliant": 0.0,
  "Qwen3.5-9B|stacked|prefill_noncompliant": 0.0,
  "Qwen3.5-9B|upgraded|none": 0.0,
  "Qwen3.5-9B|upgraded|prefill_compliant": 0.0,
  "Qwen3.5-9B|upgraded|prefill_noncompliant": 0.0,
  "Qwen3.6-27B-FP8|baseline|none": 0.0,
  "Qwen3.6-27B-FP8|baseline|prefill_compliant": 0.0,
  "Qwen3.6-27B-FP8|baseline|prefill_noncompliant": 0.0,
  "Qwen3.6-27B-FP8|no_rule|prefill_compliant": 0.0,
  "Qwen3.6-27B-FP8|stacked|none": 0.0,
  "Qwen3.6-27B-FP8|stacked|prefill_compliant": 0.0,
  "Qwen3.6-27B-FP8|stacked|prefill_noncompliant": 0.0,
  "Qwen3.6-27B-FP8|upgraded|none": 0.0,
  "Qwen3.6-27B-FP8|upgraded|prefill_compliant": 0.0,
  "Qwen3.6-27B-FP8|upgraded|prefill_noncompliant": 0.0,
  "Qwen3.8-27B-FP8|baseline|none": 0.0,
  "Qwen3.8-27B-FP8|baseline|prefill_compliant": 0.0,
  "Qwen3.8-27B-FP8|baseline|prefill_noncompliant": 0.0,
  "Qwen3.8-27B-FP8|no_rule|prefill_compliant": 0.0,
  "Qwen3.8-27B-FP8|stacked|none": 0.0,
  "Qwen3.8-27B-FP8|stacked|prefill_compliant": 0.0,
  "Qwen3.8-27B-FP8|stacked|prefill_noncompliant": 0.0,
  "Qwen3.8-27B-FP8|upgraded|none": 0.0,
  "Qwen3.8-27B-FP8|upgraded|prefill_compliant": 0.0,
  "Qwen3.8-27B-FP8|upgraded|prefill_noncompliant": 0.0
 },
 "thinking_off_no_tag_share": {
  "Gemma-4-31B-FP8|baseline|thinking_off": 0.0,
  "Gemma-4-31B-FP8|stacked|thinking_off": 0.0,
  "Gemma-4-31B-FP8|upgraded|thinking_off": 0.0,
  "Qwen3-32B|baseline|thinking_off": 8.6,
  "Qwen3-32B|stacked|thinking_off": 2.1999999999999997,
  "Qwen3-32B|upgraded|thinking_off": 1.6,
  "Qwen3.5-9B|baseline|thinking_off": 27.0,
  "Qwen3.5-9B|stacked|thinking_off": 24.6,
  "Qwen3.5-9B|upgraded|thinking_off": 32.800000000000004,
  "Qwen3.6-27B-FP8|baseline|thinking_off": 21.2,
  "Qwen3.6-27B-FP8|stacked|thinking_off": 5.2,
  "Qwen3.6-27B-FP8|upgraded|thinking_off": 23.0,
  "Qwen3.8-27B-FP8|baseline|thinking_off": 31.6,
  "Qwen3.8-27B-FP8|stacked|thinking_off": 9.8,
  "Qwen3.8-27B-FP8|upgraded|thinking_off": 6.800000000000001
 },
 "fewshot_items": {
  "Qwen3.8-27B-FP8": [
   "MMLU-Pro:69",
   "MMLU-Pro:103",
   "MMLU-Pro:118"
  ],
  "Gemma-4-31B-FP8": [
   "MMLU-Pro:69",
   "MMLU-Pro:103",
   "MMLU-Pro:118"
  ],
  "Qwen3-32B": [
   "MMLU-Pro:69",
   "MMLU-Pro:103",
   "MMLU-Pro:118"
  ],
  "Qwen3.6-27B-FP8": [
   "MMLU-Pro:69",
   "MMLU-Pro:103",
   "MMLU-Pro:118"
  ],
  "Qwen3.5-9B": [
   "MMLU-Pro:69",
   "MMLU-Pro:103",
   "MMLU-Pro:118"
  ]
 },
 "rendering_checks": {
  "Qwen3.8-27B-FP8": {
   "same_as_exp06a": 163,
   "prefill_appended": 116,
   "longest_prompt_tokens": 9083
  },
  "Gemma-4-31B-FP8": {
   "same_as_exp06a": 163,
   "prefill_appended": 116,
   "longest_prompt_tokens": 10433
  },
  "Qwen3-32B": {
   "same_as_exp06a": 163,
   "prefill_appended": 116,
   "longest_prompt_tokens": 9193
  },
  "Qwen3.6-27B-FP8": {
   "same_as_exp06a": 0,
   "prefill_appended": 116,
   "longest_prompt_tokens": 9899
  },
  "Qwen3.5-9B": {
   "same_as_exp06a": 0,
   "prefill_appended": 116,
   "longest_prompt_tokens": 9815
  }
 },
 "off_prefill_rendering_checks": {
  "Qwen3.8-27B-FP8": {
   "twins_match_request_file": 1500,
   "prefill_appended": 300,
   "graded_text_cut": 300
  },
  "Gemma-4-31B-FP8": {
   "twins_match_request_file": 1500,
   "prefill_appended": 300,
   "graded_text_cut": 300
  },
  "Qwen3-32B": {
   "twins_match_request_file": 1500,
   "prefill_appended": 300,
   "graded_text_cut": 300
  },
  "Qwen3.6-27B-FP8": {
   "twins_match_request_file": 1500,
   "prefill_appended": 300,
   "graded_text_cut": 300
  },
  "Qwen3.5-9B": {
   "twins_match_request_file": 1500,
   "prefill_appended": 300,
   "graded_text_cut": 300
  }
 }
}
```

## Figures

- [E1_prefill_comparison](../../../../figures/exp06_prompt_prefill/with_qwen35_lenient/E1_prefill_comparison.html)
- [E2_what_each_lever_adds](../../../../figures/exp06_prompt_prefill/with_qwen35_lenient/E2_what_each_lever_adds.html)
- [E3_survival_and_ends](../../../../figures/exp06_prompt_prefill/with_qwen35_lenient/E3_survival_and_ends.html)
- [E4_cot_vs_output](../../../../figures/exp06_prompt_prefill/with_qwen35_lenient/E4_cot_vs_output.html)
- [E5_per_rule](../../../../figures/exp06_prompt_prefill/with_qwen35_lenient/E5_per_rule.html)
- [E6_accuracy](../../../../figures/exp06_prompt_prefill/with_qwen35_lenient/E6_accuracy.html)
- [E7_replication](../../../../figures/exp06_prompt_prefill/with_qwen35_lenient/E7_replication.html)
- [E8_openings_both_channels](../../../../figures/exp06_prompt_prefill/with_qwen35_lenient/E8_openings_both_channels.html)
