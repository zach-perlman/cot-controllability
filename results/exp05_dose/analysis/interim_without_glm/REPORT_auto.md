# exp05_dose: automated report (UNVERIFIED)

Run interim_without_glm. Models: Qwen3-32B, Qwen3.6-27B-FP8, Qwen3.8-27B-FP8, Gemma-4-31B-FP8, Qwen3.5-9B, Gemma-4-12B (skipped, no graded rows: Qwen3.6-35B-A3B-FP8). Scoring: empty thinking traces are violations at token 0 (manifest). Every number is grader-scored (no LLM judge).

## Primary: what the rule adds beyond copying (rule arm - no-rule arm, S(1000), 4 opener rules)

Holm over all 18 contrasts (deviations_one_analysis.json).

| model | contrast | difference | p | p (Holm) |
|---|---|---|---|---|
| Qwen3-32B | R_d1 | 31.7 [27.9, 35.9] | 0.001 | 0.018 |
| Qwen3-32B | R_d3 | 17.2 [11.5, 22.6] | 0.001 | 0.018 |
| Qwen3-32B | R_trend | -14.6 [-21.2, -8.2] | 0.001 | 0.018 |
| Qwen3.6-27B-FP8 | R_d1 | 13.8 [10.4, 17.3] | 0.001 | 0.018 |
| Qwen3.6-27B-FP8 | R_d3 | 11.7 [6.7, 16.4] | 0.001 | 0.018 |
| Qwen3.6-27B-FP8 | R_trend | -2.2 [-8.0, 3.7] | 0.482 | 0.964 |
| Qwen3.8-27B-FP8 | R_d1 | 16.3 [11.0, 21.1] | 0.001 | 0.018 |
| Qwen3.8-27B-FP8 | R_d3 | 10.9 [3.7, 18.1] | 0.002 | 0.018 |
| Qwen3.8-27B-FP8 | R_trend | -5.3 [-14.1, 3.6] | 0.244 | 0.896 |
| Gemma-4-31B-FP8 | R_d1 | 4.0 [2.0, 6.7] | 0.001 | 0.018 |
| Gemma-4-31B-FP8 | R_d3 | 3.8 [1.8, 6.2] | 0.001 | 0.018 |
| Gemma-4-31B-FP8 | R_trend | -0.3 [-3.5, 2.5] | 0.750 | 0.964 |
| Qwen3.5-9B | R_d1 | 2.7 [1.0, 4.6] | 0.001 | 0.018 |
| Qwen3.5-9B | R_d3 | 0.9 [0.0, 2.0] | 0.072 | 0.504 |
| Qwen3.5-9B | R_trend | -1.9 [-3.9, 0.2] | 0.079 | 0.504 |
| Gemma-4-12B | R_d1 | 1.0 [-0.2, 2.5] | 0.119 | 0.595 |
| Gemma-4-12B | R_d3 | -0.7 [-1.8, 0.4] | 0.224 | 0.896 |
| Gemma-4-12B | R_trend | -1.7 [-3.7, 0.0] | 0.060 | 0.480 |

## Secondary: d3 - d1 within each arm (no multiplicity correction)

| model | condition | d3 - d1 | p |
|---|---|---|---|
| Qwen3-32B | prefill_compliant | 11.0 [5.4, 16.3] | 0.001 |
| Qwen3-32B | prefill_no_rule | 25.6 [21.3, 30.0] | 0.001 |
| Qwen3.6-27B-FP8 | prefill_compliant | 3.5 [-1.9, 8.7] | 0.208 |
| Qwen3.6-27B-FP8 | prefill_no_rule | 5.6 [2.6, 8.7] | 0.001 |
| Qwen3.8-27B-FP8 | prefill_compliant | 10.2 [2.8, 17.4] | 0.008 |
| Qwen3.8-27B-FP8 | prefill_no_rule | 15.6 [10.7, 20.0] | 0.001 |
| Gemma-4-31B-FP8 | prefill_compliant | -0.0 [-3.1, 2.6] | 0.891 |
| Gemma-4-31B-FP8 | prefill_no_rule | 0.3 [0.0, 0.9] | 0.731 |
| Qwen3.5-9B | prefill_compliant | -2.4 [-4.2, -0.5] | 0.015 |
| Qwen3.5-9B | prefill_no_rule | -0.5 [-1.3, 0.0] | 0.290 |
| Gemma-4-12B | prefill_compliant | -1.2 [-2.6, 0.0] | 0.059 |
| Gemma-4-12B | prefill_no_rule | 0.5 [-0.7, 1.7] | 0.482 |

## Per cell (d0 and generic are exp04 reference rows; the new models' d0 is exp05's base rows)

| model | opening | condition | n | opening tokens (median) | S(1000) | S(200) | P1 | case rules S(1000), lenient | accuracy (full) | continuation tokens (full, median) | empty % | aborted % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | d0 | none | 400 | None | 2.6 [1.2, 4.4] | 4.2 [2.5, 6.2] | 2.8 [1.2, 4.2] | 3.5 [1.5, 6.0] | 53.0 [42.7, 63.6] | 1640.5 | 0.0 | 72.8 |
| Qwen3-32B | generic | prefill_compliant | 400 | None | 35.9 [31.4, 40.3] | 53.0 [49.5, 56.5] | 29.0 [24.8, 33.2] | 64.7 [56.3, 72.1] | 48.0 [37.3, 59.0] | 1586.0 | 0.0 | 51.7 |
| Qwen3-32B | generic | prefill_no_rule | 400 | None | 8.2 [5.7, 11.0] | 11.7 [9.0, 14.7] | 5.5 [3.5, 7.5] | 14.4 [9.3, 19.1] | 50.0 [39.0, 61.8] | 2721.5 | 0.0 | 70.2 |
| Qwen3-32B | d1 | prefill_compliant | 400 | 11.0 | 35.0 [31.4, 38.9] | 51.7 [48.0, 55.5] | 29.8 [25.8, 34.0] | 63.9 [55.7, 71.3] | 51.0 [40.6, 61.6] | 1484.0 | 0.0 | 51.2 |
| Qwen3-32B | d1 | prefill_no_rule | 400 | 11.0 | 3.3 [1.7, 5.0] | 7.0 [4.5, 9.7] | 2.2 [1.0, 3.8] | 7.3 [3.5, 11.7] | 53.0 [42.4, 64.0] | 3075.5 | 0.0 | 73.0 |
| Qwen3-32B | d2 | prefill_compliant | 400 | 69.0 | 37.5 [33.2, 42.1] | 67.5 [63.0, 71.7] | 35.0 [30.0, 40.0] | 75.1 [66.9, 82.4] | 50.0 [39.4, 61.3] | 1743.0 | 0.0 | 46.0 |
| Qwen3-32B | d2 | prefill_no_rule | 400 | 69.0 | 23.6 [19.8, 27.3] | 43.2 [39.5, 46.7] | 16.2 [12.8, 19.8] | 36.6 [26.9, 46.2] | 50.0 [39.1, 61.7] | 3116.0 | 0.0 | 61.5 |
| Qwen3-32B | d3 | prefill_compliant | 400 | 158.0 | 46.0 [41.0, 51.0] | 77.3 [73.5, 81.1] | 41.5 [36.0, 47.0] | 80.7 [72.9, 87.4] | 47.0 [36.8, 58.5] | 1620.0 | 0.0 | 40.0 |
| Qwen3-32B | d3 | prefill_no_rule | 400 | 158.0 | 28.9 [24.4, 33.4] | 54.7 [50.4, 59.0] | 23.2 [18.8, 28.0] | 49.4 [38.5, 59.6] | 50.0 [38.6, 61.8] | 2842.5 | 0.0 | 54.5 |
| Qwen3.6-27B-FP8 | d0 | none | 400 | None | 0.7 [0.0, 1.5] | 2.7 [1.3, 4.2] | 0.2 [0.0, 0.8] | 4.0 [1.4, 6.8] | 54.0 [43.7, 64.9] | 4746.5 | 0.0 | 74.5 |
| Qwen3.6-27B-FP8 | generic | prefill_compliant | 400 | None | 39.2 [33.9, 44.4] | 72.5 [68.2, 76.7] | 29.0 [24.2, 34.0] | 68.4 [61.1, 75.7] | 53.0 [42.1, 64.5] | 3132.5 | 0.0 | 51.0 |
| Qwen3.6-27B-FP8 | generic | prefill_no_rule | 400 | None | 14.5 [11.5, 17.7] | 24.0 [20.2, 28.0] | 8.0 [5.5, 10.8] | 18.5 [13.1, 24.2] | 59.0 [48.0, 70.0] | 4100.5 | 0.0 | 69.0 |
| Qwen3.6-27B-FP8 | d1 | prefill_compliant | 400 | 20.0 | 17.2 [13.6, 21.0] | 36.7 [31.0, 42.5] | 4.8 [3.0, 6.8] | 40.0 [31.7, 48.5] | 49.0 [38.5, 61.1] | 12094.5 | 0.0 | 71.0 |
| Qwen3.6-27B-FP8 | d1 | prefill_no_rule | 400 | 20.0 | 3.3 [1.7, 5.0] | 10.7 [7.7, 14.0] | 1.8 [0.8, 3.0] | 10.4 [5.0, 15.3] | 62.0 [51.7, 72.6] | 12993.0 | 0.0 | 73.8 |
| Qwen3.6-27B-FP8 | d2 | prefill_compliant | 400 | 71.0 | 17.8 [14.3, 21.3] | 54.7 [49.7, 60.0] | 3.8 [2.0, 5.8] | 53.8 [46.1, 61.3] | 55.0 [44.3, 66.3] | 11429.5 | 0.0 | 71.2 |
| Qwen3.6-27B-FP8 | d2 | prefill_no_rule | 400 | 71.0 | 8.3 [5.7, 11.2] | 33.5 [28.7, 38.0] | 3.2 [1.8, 4.8] | 26.6 [17.9, 34.9] | 55.0 [45.2, 66.0] | 13238.5 | 0.0 | 73.0 |
| Qwen3.6-27B-FP8 | d3 | prefill_compliant | 400 | 162.0 | 20.6 [16.5, 24.7] | 61.2 [56.5, 65.7] | 5.2 [3.2, 7.5] | 50.9 [42.4, 58.6] | 57.0 [46.5, 67.5] | 10554.0 | 0.0 | 70.8 |
| Qwen3.6-27B-FP8 | d3 | prefill_no_rule | 400 | 162.0 | 9.0 [6.3, 11.6] | 44.3 [39.6, 48.9] | 4.8 [2.8, 6.8] | 29.9 [21.5, 38.5] | 52.0 [41.0, 63.4] | 11986.0 | 0.0 | 71.2 |
| Qwen3.8-27B-FP8 | d0 | none | 400 | None | 0.0 [0.0, 0.0] | 0.8 [0.0, 1.8] | 0.0 [0.0, 0.0] | 1.5 [0.0, 4.0] | 62.0 [51.5, 72.7] | 1477.0 | 0.0 | 75.0 |
| Qwen3.8-27B-FP8 | generic | prefill_compliant | 400 | None | 24.2 [20.4, 28.2] | 48.7 [43.7, 53.5] | 20.8 [16.7, 25.2] | 68.9 [61.4, 75.8] | 54.0 [43.1, 65.2] | 1810.5 | 0.0 | 58.5 |
| Qwen3.8-27B-FP8 | generic | prefill_no_rule | 400 | None | 7.5 [5.0, 10.0] | 16.0 [13.2, 19.0] | 6.2 [4.5, 8.2] | 28.0 [21.3, 34.5] | 55.0 [44.1, 66.7] | 2719.0 | 0.0 | 69.2 |
| Qwen3.8-27B-FP8 | d1 | prefill_compliant | 400 | 18.5 | 21.7 [17.6, 26.5] | 49.5 [45.0, 54.0] | 21.8 [18.0, 26.0] | 57.9 [49.0, 67.3] | 60.0 [49.5, 70.9] | 1534.0 | 0.0 | 56.2 |
| Qwen3.8-27B-FP8 | d1 | prefill_no_rule | 400 | 18.5 | 5.5 [3.9, 8.0] | 15.9 [12.0, 20.2] | 6.2 [4.0, 8.8] | 16.7 [9.9, 23.6] | 56.0 [44.1, 67.8] | 1787.5 | 0.0 | 68.8 |
| Qwen3.8-27B-FP8 | d2 | prefill_compliant | 400 | 85.5 | 30.9 [25.4, 36.3] | 66.1 [61.4, 71.2] | 28.0 [23.0, 33.0] | 75.4 [67.6, 82.4] | 57.0 [45.6, 68.7] | 1473.5 | 0.0 | 50.7 |
| Qwen3.8-27B-FP8 | d2 | prefill_no_rule | 400 | 85.5 | 22.9 [18.8, 27.2] | 50.0 [45.8, 54.2] | 19.8 [16.0, 23.5] | 69.6 [62.1, 76.5] | 60.0 [48.4, 70.9] | 1529.0 | 0.0 | 55.2 |
| Qwen3.8-27B-FP8 | d3 | prefill_compliant | 400 | 173.5 | 32.0 [26.4, 37.6] | 74.8 [70.5, 79.2] | 30.5 [25.0, 36.2] | 74.5 [66.3, 82.5] | 59.0 [48.5, 69.7] | 1256.0 | 0.0 | 43.2 |
| Qwen3.8-27B-FP8 | d3 | prefill_no_rule | 400 | 173.5 | 21.1 [16.6, 25.8] | 55.7 [50.5, 60.9] | 21.2 [16.5, 26.0] | 63.9 [54.1, 73.2] | 61.0 [49.5, 72.8] | 1425.0 | 0.0 | 51.2 |
| Gemma-4-31B-FP8 | d0 | none | 400 | None | 0.2 [0.0, 0.8] | 0.5 [0.0, 1.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 1.5] | 60.0 [48.6, 71.6] | 3473.5 | 0.0 | 75.0 |
| Gemma-4-31B-FP8 | generic | prefill_compliant | 400 | None | 9.7 [6.6, 12.8] | 44.5 [39.7, 49.7] | 7.2 [4.8, 10.2] | 20.6 [14.6, 27.2] | 64.0 [52.1, 75.8] | 3068.0 | 0.0 | 67.5 |
| Gemma-4-31B-FP8 | generic | prefill_no_rule | 400 | None | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.8] | 0.0 [0.0, 0.0] | 0.5 [0.0, 1.5] | 62.0 [50.5, 73.4] | 4301.0 | 0.0 | 75.0 |
| Gemma-4-31B-FP8 | d1 | prefill_compliant | 400 | 24.0 | 4.0 [2.0, 6.7] | 30.5 [26.0, 35.2] | 5.5 [3.5, 7.8] | 8.4 [3.5, 14.5] | 61.0 [49.6, 72.2] | 2861.5 | 0.0 | 69.8 |
| Gemma-4-31B-FP8 | d1 | prefill_no_rule | 400 | 24.0 | 0.0 [0.0, 0.0] | 2.8 [1.0, 5.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 3.6] | 59.0 [47.9, 71.0] | 4087.0 | 0.0 | 75.0 |
| Gemma-4-31B-FP8 | d2 | prefill_compliant | 400 | 71.0 | 6.7 [3.9, 10.1] | 38.8 [34.0, 43.9] | 6.8 [3.7, 10.2] | 10.6 [4.1, 17.5] | 61.0 [49.5, 72.6] | 2884.5 | 0.0 | 66.5 |
| Gemma-4-31B-FP8 | d2 | prefill_no_rule | 400 | 71.0 | 0.0 [0.0, 0.0] | 6.0 [3.3, 9.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 64.0 [52.5, 75.9] | 4087.5 | 0.0 | 73.8 |
| Gemma-4-31B-FP8 | d3 | prefill_compliant | 400 | 166.0 | 4.0 [2.0, 6.5] | 45.2 [39.9, 50.8] | 7.8 [5.0, 11.0] | 11.3 [5.1, 18.3] | 61.0 [49.5, 72.9] | 2911.0 | 0.0 | 66.8 |
| Gemma-4-31B-FP8 | d3 | prefill_no_rule | 400 | 166.0 | 0.3 [0.0, 0.9] | 16.1 [11.3, 20.8] | 0.5 [0.0, 1.2] | 0.0 [0.0, 0.0] | 51.0 [39.8, 63.2] | 3603.5 | 0.0 | 73.5 |
| Qwen3.5-9B | d0 | none | 400 | None | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 50.0 [40.6, 60.5] | 8527.0 | 0.0 | 75.0 |
| Qwen3.5-9B | generic | prefill_compliant | 400 | None | 6.9 [4.5, 9.2] | 29.7 [26.0, 33.2] | 4.5 [2.8, 6.2] | 19.5 [14.1, 24.9] | 51.0 [40.4, 62.8] | 3204.5 | 0.0 | 72.2 |
| Qwen3.5-9B | generic | prefill_no_rule | 400 | None | 6.5 [4.5, 9.0] | 13.2 [10.5, 16.0] | 3.0 [1.5, 4.8] | 21.1 [15.8, 26.6] | 47.0 [37.0, 58.1] | 4356.5 | 0.0 | 73.0 |
| Qwen3.5-9B | d1 | prefill_compliant | 400 | 25.0 | 3.2 [1.7, 5.0] | 12.8 [10.2, 15.6] | 2.2 [1.0, 3.5] | 1.5 [0.0, 3.7] | 48.0 [36.0, 60.2] | 8472.5 | 0.0 | 73.5 |
| Qwen3.5-9B | d1 | prefill_no_rule | 400 | 25.0 | 0.5 [0.0, 1.3] | 2.1 [0.8, 3.5] | 0.8 [0.0, 1.8] | 0.5 [0.0, 1.5] | 42.0 [31.5, 53.1] | 9933.0 | 0.0 | 74.8 |
| Qwen3.5-9B | d2 | prefill_compliant | 400 | 72.0 | 0.2 [0.0, 0.8] | 10.7 [8.0, 13.5] | 0.2 [0.0, 0.8] | 0.8 [0.0, 2.8] | 43.0 [31.9, 54.5] | 8769.0 | 0.0 | 74.8 |
| Qwen3.5-9B | d2 | prefill_no_rule | 400 | 72.0 | 0.0 [0.0, 0.0] | 3.0 [1.5, 4.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 40.0 [29.7, 51.1] | 10070.0 | 0.0 | 75.0 |
| Qwen3.5-9B | d3 | prefill_compliant | 400 | 160.5 | 0.9 [0.0, 2.0] | 16.2 [12.7, 20.0] | 0.5 [0.0, 1.2] | 1.1 [0.0, 3.4] | 50.0 [38.9, 61.8] | 8631.5 | 0.0 | 74.8 |
| Qwen3.5-9B | d3 | prefill_no_rule | 400 | 160.5 | 0.0 [0.0, 0.0] | 5.0 [3.0, 7.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 42.0 [31.0, 52.7] | 10704.5 | 0.0 | 75.0 |
| Gemma-4-12B | d0 | none | 400 | None | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.0 [32.4, 55.1] | 6165.0 | 0.0 | 75.0 |
| Gemma-4-12B | generic | prefill_compliant | 400 | None | 1.3 [0.3, 2.6] | 16.5 [12.7, 20.7] | 0.2 [0.0, 0.8] | 0.9 [0.0, 2.9] | 44.0 [33.3, 56.0] | 7378.5 | 0.0 | 74.8 |
| Gemma-4-12B | generic | prefill_no_rule | 400 | None | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.0 [37.6, 59.2] | 8513.0 | 0.0 | 75.0 |
| Gemma-4-12B | d1 | prefill_compliant | 400 | 23.5 | 1.5 [0.5, 2.8] | 18.2 [15.2, 21.5] | 0.2 [0.0, 0.8] | 1.1 [0.0, 3.4] | 45.0 [34.2, 56.5] | 7602.0 | 0.0 | 75.0 |
| Gemma-4-12B | d1 | prefill_no_rule | 400 | 23.5 | 0.5 [0.0, 1.2] | 4.0 [2.2, 6.0] | 0.2 [0.0, 0.8] | 0.0 [0.0, 0.0] | 46.0 [35.2, 57.5] | 7041.0 | 0.0 | 74.8 |
| Gemma-4-12B | d2 | prefill_compliant | 400 | 74.0 | 1.2 [0.2, 2.5] | 25.7 [21.5, 30.2] | 0.0 [0.0, 0.0] | 0.8 [0.0, 2.5] | 41.0 [30.8, 52.6] | 8146.0 | 0.0 | 75.0 |
| Gemma-4-12B | d2 | prefill_no_rule | 400 | 74.0 | 1.7 [0.5, 3.0] | 14.2 [11.0, 17.2] | 0.5 [0.0, 1.2] | 0.0 [0.0, 0.0] | 48.0 [37.6, 59.3] | 7445.0 | 0.0 | 74.5 |
| Gemma-4-12B | d3 | prefill_compliant | 400 | 164.5 | 0.3 [0.0, 1.0] | 26.5 [22.2, 30.7] | 0.2 [0.0, 0.8] | 0.7 [0.0, 2.2] | 40.0 [30.6, 50.9] | 9093.5 | 0.0 | 74.8 |
| Gemma-4-12B | d3 | prefill_no_rule | 400 | 164.5 | 1.0 [0.2, 2.0] | 13.2 [10.2, 16.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.0 [32.3, 54.1] | 8808.0 | 0.0 | 75.0 |

## Effort check (Qwen3.8; exp04's requests; Holm over the 4 contrasts)

| arm | effort | n | S(1000) | S(200) | P1 | median reasoning tokens (full) | empty % |
|---|---|---|---|---|---|---|---|
| none / stacked | xhigh | 400 | 0.8 [0.0, 1.7] | 3.7 [2.0, 5.5] | 1.0 [0.2, 2.0] | 6494.5 | 0.0 |
| none / stacked | medium | 400 | 13.8 [10.5, 16.9] | 24.0 [20.7, 27.2] | 14.2 [11.2, 17.0] | 1316.0 | 0.0 |
| none / stacked | low | 400 | 3.0 [1.5, 4.5] | 3.0 [1.5, 4.5] | 2.5 [1.2, 4.0] | 1033.5 | 0.0 |
| prefill_compliant / baseline | xhigh | 400 | 35.7 [31.3, 39.9] | 62.6 [57.7, 67.3] | 28.0 [24.0, 32.0] | 2691.5 | 0.0 |
| prefill_compliant / baseline | medium | 400 | 24.2 [20.4, 28.2] | 48.7 [43.7, 53.5] | 20.8 [16.7, 25.2] | 1810.5 | 0.0 |
| prefill_compliant / baseline | low | 400 | 27.6 [23.0, 32.3] | 47.4 [42.3, 52.4] | 24.8 [20.7, 28.8] | 1643.5 | 0.0 |

| arm | contrast | difference | p | p (Holm) |
|---|---|---|---|---|
| none / stacked | xhigh - medium | -13.1 [-16.3, -9.7] | 0.001 | 0.004 |
| none / stacked | low - medium | -10.8 [-14.2, -7.1] | 0.001 | 0.004 |
| prefill_compliant / baseline | xhigh - medium | 11.5 [6.0, 16.7] | 0.001 | 0.004 |
| prefill_compliant / baseline | low - medium | 3.4 [-2.1, 9.2] | 0.238 | 0.238 |

## Secondary: answer correctness vs rule-keeping (full-trace cells; opening doses pooled)

Only full-trace cells observe correctness without selecting on compliance. Wrong answers are longer, so read the length-matched column (rollouts that reached 1000 tokens). Right − wrong in % points; n/a: fewer than 10 rollouts in a group; 0.0 [0.0, 0.0]: nobody in either group kept the rule (a floor, not evidence of no link).

| model | arm | n right / wrong | median tokens right / wrong | obeyed throughout: right − wrong | no violation in first 200: right − wrong | no violation in first 1000: right − wrong (n right / wrong) |
|---|---|---|---|---|---|---|
| Qwen3-32B | no opening | 53 / 47 | 1115.0 / 2715.0 | 3.8 [0.0, 9.6] | -0.5 [-8.6, 6.6] | 3.3 [0.0, 11.1] (30 / 34) |
| Qwen3-32B | rule in prompt | 148 / 152 | 950.0 / 2753.5 | 16.9 [1.1, 32.3] | -5.1 [-19.0, 8.7] | -13.8 [-28.5, 3.4] (72 / 116) |
| Qwen3-32B | no rule (imitation only) | 153 / 147 | 1531.0 / 4984.0 | -1.1 [-11.9, 9.3] | -5.6 [-17.5, 6.5] | -2.7 [-12.3, 8.0] (96 / 125) |
| Qwen3.6-27B-FP8 | no opening | 54 / 46 | 4239.0 / 6047.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] (54 / 46) |
| Qwen3.6-27B-FP8 | rule in prompt | 161 / 139 | 9839.0 / 15266.0 | 8.0 [1.7, 15.3] | 10.7 [-1.9, 22.8] | 1.6 [-7.6, 11.2] (148 / 138) |
| Qwen3.6-27B-FP8 | no rule (imitation only) | 169 / 131 | 9544.0 / 19417.0 | 6.2 [0.7, 12.4] | 3.1 [-10.5, 16.2] | -2.8 [-7.8, 2.1] (141 / 129) |
| Qwen3.8-27B-FP8 | no opening | 62 / 38 | 1066.0 / 1895.0 | 0.0 [0.0, 0.0] | -2.6 [-9.1, 0.0] | 0.0 [0.0, 0.0] (32 / 27) |
| Qwen3.8-27B-FP8 | rule in prompt | 176 / 124 | 951.0 / 2607.0 | 1.3 [-13.7, 17.0] | -6.8 [-19.2, 5.5] | -5.5 [-20.8, 8.8] (85 / 96) |
| Qwen3.8-27B-FP8 | no rule (imitation only) | 177 / 123 | 1107.0 / 3167.0 | 2.6 [-10.6, 15.8] | -13.2 [-25.4, -0.4] | -8.0 [-21.4, 3.9] (94 / 97) |
| Gemma-4-31B-FP8 | no opening | 60 / 40 | 2906.0 / 4395.5 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] (46 / 36) |
| Gemma-4-31B-FP8 | rule in prompt | 183 / 117 | 2202.0 / 4109.0 | 7.6 [1.2, 14.2] | -3.8 [-18.1, 9.8] | 3.0 [-3.7, 11.0] (128 / 104) |
| Gemma-4-31B-FP8 | no rule (imitation only) | 174 / 126 | 3266.0 / 4733.5 | 0.6 [0.0, 1.9] | 4.9 [-0.5, 10.2] | 0.0 [0.0, 0.0] (129 / 117) |
| Qwen3.5-9B | no opening | 50 / 50 | 7420.0 / 10125.5 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] (49 / 50) |
| Qwen3.5-9B | rule in prompt | 141 / 159 | 7425.0 / 10514.0 | 1.5 [-1.1, 4.4] | 7.4 [-1.0, 16.4] | -0.6 [-2.1, 0.0] (131 / 155) |
| Qwen3.5-9B | no rule (imitation only) | 124 / 176 | 5509.0 / 16795.5 | 0.2 [-1.6, 2.4] | 1.2 [-3.6, 6.5] | -0.6 [-1.9, 0.0] (108 / 176) |
| Gemma-4-12B | no opening | 43 / 57 | 5116.0 / 9652.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] (38 / 56) |
| Gemma-4-12B | rule in prompt | 126 / 174 | 4969.0 / 12399.0 | 0.8 [0.0, 2.9] | 6.5 [-5.0, 18.0] | -0.3 [-3.7, 2.2] (116 / 173) |
| Gemma-4-12B | no rule (imitation only) | 137 / 163 | 5446.0 / 10213.0 | 0.0 [0.0, 0.0] | 0.9 [-6.5, 8.3] | 1.0 [-1.3, 3.6] (124 / 162) |
| all models | no opening | 322 / 278 | 3219.5 / 5895.0 | 0.6 [0.0, 1.6] | -0.5 [-2.0, 0.9] | 0.4 [0.0, 1.3] (249 / 249) |
| all models | rule in prompt | 935 / 865 | 2736.0 / 6571.0 | 6.8 [1.4, 12.2] | 3.7 [-3.4, 10.7] | -2.1 [-6.6, 2.2] (680 / 782) |
| all models | no rule (imitation only) | 934 / 866 | 3016.0 / 7248.0 | 2.1 [-1.7, 5.8] | 0.5 [-6.1, 6.7] | -1.6 [-4.5, 0.9] (692 / 806) |

## Per rule S(1000)

|  | model | dose | condition | mode | n | S_t_star | ci_lo | ci_hi | S_short | first_violation_lt5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Qwen3-32B | d0 | none | lowercase_thinking | 100 | 2.4 | 0.0 | 7.0 | 6.0 | 88.0 |
| 1 | Qwen3-32B | d0 | none | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 2 | Qwen3-32B | d0 | none | meow_between_words | 100 | 8.0 | 3.0 | 13.0 | 11.0 | 76.0 |
| 3 | Qwen3-32B | d0 | none | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 4 | Qwen3-32B | generic | prefill_compliant | lowercase_thinking | 100 | 27.7 | 18.2 | 37.7 | 46.0 | 20.0 |
| 5 | Qwen3-32B | generic | prefill_compliant | uppercase_thinking | 100 | 49.2 | 39.4 | 59.4 | 73.0 | 2.0 |
| 6 | Qwen3-32B | generic | prefill_compliant | meow_between_words | 100 | 66.7 | 56.7 | 75.5 | 93.0 | 0.0 |
| 7 | Qwen3-32B | generic | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 8 | Qwen3-32B | generic | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 9 | Qwen3-32B | generic | prefill_no_rule | uppercase_thinking | 100 | 21.8 | 13.3 | 30.8 | 31.0 | 35.0 |
| 10 | Qwen3-32B | generic | prefill_no_rule | meow_between_words | 100 | 11.0 | 5.0 | 18.0 | 16.0 | 69.0 |
| 11 | Qwen3-32B | generic | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 12 | Qwen3-32B | d1 | prefill_compliant | lowercase_thinking | 100 | 23.4 | 14.2 | 32.7 | 41.0 | 32.0 |
| 13 | Qwen3-32B | d1 | prefill_compliant | uppercase_thinking | 100 | 39.0 | 28.9 | 49.2 | 70.0 | 0.0 |
| 14 | Qwen3-32B | d1 | prefill_compliant | meow_between_words | 100 | 77.8 | 69.2 | 86.1 | 94.0 | 1.0 |
| 15 | Qwen3-32B | d1 | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 3.0 | 2.0 | 0.0 |
| 16 | Qwen3-32B | d1 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 3.0 | 92.0 |
| 17 | Qwen3-32B | d1 | prefill_no_rule | uppercase_thinking | 100 | 3.5 | 0.0 | 7.6 | 9.0 | 76.0 |
| 18 | Qwen3-32B | d1 | prefill_no_rule | meow_between_words | 100 | 9.8 | 4.0 | 15.9 | 16.0 | 75.0 |
| 19 | Qwen3-32B | d1 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20 | Qwen3-32B | d2 | prefill_compliant | lowercase_thinking | 100 | 30.3 | 20.1 | 40.3 | 74.9 | 2.0 |
| 21 | Qwen3-32B | d2 | prefill_compliant | uppercase_thinking | 100 | 42.0 | 32.1 | 52.1 | 77.0 | 0.0 |
| 22 | Qwen3-32B | d2 | prefill_compliant | meow_between_words | 100 | 77.7 | 68.8 | 85.7 | 91.0 | 0.0 |
| 23 | Qwen3-32B | d2 | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 8.4 | 27.0 | 1.0 |
| 24 | Qwen3-32B | d2 | prefill_no_rule | lowercase_thinking | 100 | 9.0 | 3.2 | 15.8 | 35.0 | 17.0 |
| 25 | Qwen3-32B | d2 | prefill_no_rule | uppercase_thinking | 100 | 32.7 | 23.4 | 42.7 | 54.0 | 7.0 |
| 26 | Qwen3-32B | d2 | prefill_no_rule | meow_between_words | 100 | 52.8 | 42.6 | 62.8 | 83.0 | 3.0 |
| 27 | Qwen3-32B | d2 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 28 | Qwen3-32B | d3 | prefill_compliant | lowercase_thinking | 100 | 46.4 | 34.9 | 58.0 | 80.7 | 0.0 |
| 29 | Qwen3-32B | d3 | prefill_compliant | uppercase_thinking | 100 | 53.1 | 42.0 | 63.5 | 81.0 | 0.0 |
| 30 | Qwen3-32B | d3 | prefill_compliant | meow_between_words | 100 | 78.2 | 68.9 | 86.9 | 98.0 | 0.0 |
| 31 | Qwen3-32B | d3 | prefill_compliant | end_of_sentence | 100 | 6.4 | 0.0 | 14.6 | 49.6 | 0.0 |
| 32 | Qwen3-32B | d3 | prefill_no_rule | lowercase_thinking | 100 | 11.4 | 4.6 | 18.8 | 53.9 | 3.0 |
| 33 | Qwen3-32B | d3 | prefill_no_rule | uppercase_thinking | 100 | 38.4 | 27.3 | 48.6 | 69.0 | 0.0 |
| 34 | Qwen3-32B | d3 | prefill_no_rule | meow_between_words | 100 | 65.7 | 55.6 | 75.4 | 86.0 | 1.0 |
| 35 | Qwen3-32B | d3 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 9.9 | 1.0 |
| 36 | Qwen3.6-27B-FP8 | d0 | none | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 97.0 |
| 37 | Qwen3.6-27B-FP8 | d0 | none | uppercase_thinking | 100 | 2.7 | 0.0 | 6.0 | 10.0 | 76.0 |
| 38 | Qwen3.6-27B-FP8 | d0 | none | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 39 | Qwen3.6-27B-FP8 | d0 | none | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 40 | Qwen3.6-27B-FP8 | generic | prefill_compliant | lowercase_thinking | 100 | 44.7 | 35.6 | 53.7 | 60.0 | 25.0 |
| 41 | Qwen3.6-27B-FP8 | generic | prefill_compliant | uppercase_thinking | 100 | 34.1 | 25.5 | 43.3 | 61.0 | 0.0 |
| 42 | Qwen3.6-27B-FP8 | generic | prefill_compliant | meow_between_words | 100 | 58.0 | 47.9 | 67.9 | 93.0 | 1.0 |
| 43 | Qwen3.6-27B-FP8 | generic | prefill_compliant | end_of_sentence | 100 | 19.9 | 9.8 | 30.6 | 76.0 | 0.0 |
| 44 | Qwen3.6-27B-FP8 | generic | prefill_no_rule | lowercase_thinking | 100 | 4.8 | 1.0 | 9.8 | 8.0 | 88.0 |
| 45 | Qwen3.6-27B-FP8 | generic | prefill_no_rule | uppercase_thinking | 100 | 18.3 | 11.4 | 25.4 | 32.0 | 51.0 |
| 46 | Qwen3.6-27B-FP8 | generic | prefill_no_rule | meow_between_words | 100 | 34.9 | 26.0 | 44.0 | 56.0 | 12.0 |
| 47 | Qwen3.6-27B-FP8 | generic | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 19.0 |
| 48 | Qwen3.6-27B-FP8 | d1 | prefill_compliant | lowercase_thinking | 100 | 24.7 | 16.7 | 33.0 | 46.0 | 42.0 |
| 49 | Qwen3.6-27B-FP8 | d1 | prefill_compliant | uppercase_thinking | 100 | 14.3 | 7.9 | 21.4 | 36.0 | 24.0 |
| 50 | Qwen3.6-27B-FP8 | d1 | prefill_compliant | meow_between_words | 100 | 26.6 | 18.3 | 35.5 | 44.0 | 3.0 |
| 51 | Qwen3.6-27B-FP8 | d1 | prefill_compliant | end_of_sentence | 100 | 3.0 | 0.0 | 7.0 | 21.0 | 20.0 |
| 52 | Qwen3.6-27B-FP8 | d1 | prefill_no_rule | lowercase_thinking | 100 | 3.0 | 0.0 | 7.0 | 7.0 | 84.0 |
| 53 | Qwen3.6-27B-FP8 | d1 | prefill_no_rule | uppercase_thinking | 100 | 8.2 | 3.0 | 13.8 | 22.0 | 49.0 |
| 54 | Qwen3.6-27B-FP8 | d1 | prefill_no_rule | meow_between_words | 100 | 2.0 | 0.0 | 5.0 | 12.0 | 38.0 |
| 55 | Qwen3.6-27B-FP8 | d1 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 43.0 |
| 56 | Qwen3.6-27B-FP8 | d2 | prefill_compliant | lowercase_thinking | 100 | 26.7 | 18.4 | 35.0 | 65.0 | 3.0 |
| 57 | Qwen3.6-27B-FP8 | d2 | prefill_compliant | uppercase_thinking | 100 | 18.4 | 11.5 | 25.7 | 61.0 | 0.0 |
| 58 | Qwen3.6-27B-FP8 | d2 | prefill_compliant | meow_between_words | 100 | 22.0 | 15.0 | 30.0 | 58.0 | 2.0 |
| 59 | Qwen3.6-27B-FP8 | d2 | prefill_compliant | end_of_sentence | 100 | 4.0 | 1.0 | 8.0 | 35.0 | 1.0 |
| 60 | Qwen3.6-27B-FP8 | d2 | prefill_no_rule | lowercase_thinking | 100 | 8.0 | 3.0 | 13.0 | 37.0 | 10.0 |
| 61 | Qwen3.6-27B-FP8 | d2 | prefill_no_rule | uppercase_thinking | 100 | 16.3 | 8.7 | 24.1 | 44.0 | 4.0 |
| 62 | Qwen3.6-27B-FP8 | d2 | prefill_no_rule | meow_between_words | 100 | 9.0 | 4.0 | 15.0 | 48.0 | 2.0 |
| 63 | Qwen3.6-27B-FP8 | d2 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 5.0 | 8.0 |
| 64 | Qwen3.6-27B-FP8 | d3 | prefill_compliant | lowercase_thinking | 100 | 27.0 | 19.0 | 35.0 | 78.0 | 1.0 |
| 65 | Qwen3.6-27B-FP8 | d3 | prefill_compliant | uppercase_thinking | 100 | 20.7 | 13.5 | 28.4 | 60.9 | 2.0 |
| 66 | Qwen3.6-27B-FP8 | d3 | prefill_compliant | meow_between_words | 100 | 26.7 | 18.6 | 35.6 | 60.0 | 1.0 |
| 67 | Qwen3.6-27B-FP8 | d3 | prefill_compliant | end_of_sentence | 100 | 8.0 | 3.0 | 14.0 | 46.0 | 4.0 |
| 68 | Qwen3.6-27B-FP8 | d3 | prefill_no_rule | lowercase_thinking | 100 | 12.9 | 6.7 | 19.6 | 50.8 | 8.0 |
| 69 | Qwen3.6-27B-FP8 | d3 | prefill_no_rule | uppercase_thinking | 100 | 13.3 | 6.2 | 21.0 | 56.9 | 2.0 |
| 70 | Qwen3.6-27B-FP8 | d3 | prefill_no_rule | meow_between_words | 100 | 9.7 | 4.6 | 15.9 | 53.5 | 1.0 |
| 71 | Qwen3.6-27B-FP8 | d3 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 16.0 | 3.0 |
| 72 | Qwen3.8-27B-FP8 | d0 | none | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 98.0 |
| 73 | Qwen3.8-27B-FP8 | d0 | none | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 97.0 |
| 74 | Qwen3.8-27B-FP8 | d0 | none | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 75 | Qwen3.8-27B-FP8 | d0 | none | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 76 | Qwen3.8-27B-FP8 | generic | prefill_compliant | lowercase_thinking | 100 | 53.1 | 42.9 | 63.2 | 77.0 | 2.0 |
| 77 | Qwen3.8-27B-FP8 | generic | prefill_compliant | uppercase_thinking | 100 | 27.5 | 18.0 | 37.1 | 56.0 | 1.0 |
| 78 | Qwen3.8-27B-FP8 | generic | prefill_compliant | meow_between_words | 100 | 14.6 | 8.0 | 22.1 | 51.0 | 5.0 |
| 79 | Qwen3.8-27B-FP8 | generic | prefill_compliant | end_of_sentence | 100 | 1.6 | 0.0 | 5.2 | 10.9 | 0.0 |
| 80 | Qwen3.8-27B-FP8 | generic | prefill_no_rule | lowercase_thinking | 100 | 3.0 | 0.0 | 7.0 | 4.0 | 93.0 |
| 81 | Qwen3.8-27B-FP8 | generic | prefill_no_rule | uppercase_thinking | 100 | 23.8 | 15.4 | 33.0 | 46.0 | 1.0 |
| 82 | Qwen3.8-27B-FP8 | generic | prefill_no_rule | meow_between_words | 100 | 3.0 | 0.0 | 6.1 | 14.1 | 36.0 |
| 83 | Qwen3.8-27B-FP8 | generic | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 84 | Qwen3.8-27B-FP8 | d1 | prefill_compliant | lowercase_thinking | 100 | 32.2 | 21.2 | 44.0 | 60.9 | 19.0 |
| 85 | Qwen3.8-27B-FP8 | d1 | prefill_compliant | uppercase_thinking | 100 | 31.2 | 22.3 | 40.8 | 58.6 | 1.0 |
| 86 | Qwen3.8-27B-FP8 | d1 | prefill_compliant | meow_between_words | 100 | 18.2 | 10.3 | 26.3 | 41.6 | 38.0 |
| 87 | Qwen3.8-27B-FP8 | d1 | prefill_compliant | end_of_sentence | 100 | 5.4 | 0.0 | 12.4 | 36.9 | 0.0 |
| 88 | Qwen3.8-27B-FP8 | d1 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 6.1 | 14.7 | 77.0 |
| 89 | Qwen3.8-27B-FP8 | d1 | prefill_no_rule | uppercase_thinking | 100 | 7.1 | 1.8 | 13.1 | 18.4 | 43.0 |
| 90 | Qwen3.8-27B-FP8 | d1 | prefill_no_rule | meow_between_words | 100 | 14.9 | 7.8 | 22.5 | 28.6 | 47.0 |
| 91 | Qwen3.8-27B-FP8 | d1 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 1.0 |
| 92 | Qwen3.8-27B-FP8 | d2 | prefill_compliant | lowercase_thinking | 100 | 46.5 | 35.0 | 58.5 | 74.2 | 0.0 |
| 93 | Qwen3.8-27B-FP8 | d2 | prefill_compliant | uppercase_thinking | 100 | 39.8 | 28.9 | 50.2 | 72.1 | 0.0 |
| 94 | Qwen3.8-27B-FP8 | d2 | prefill_compliant | meow_between_words | 100 | 28.7 | 19.1 | 38.7 | 64.9 | 2.0 |
| 95 | Qwen3.8-27B-FP8 | d2 | prefill_compliant | end_of_sentence | 100 | 8.4 | 2.8 | 15.0 | 53.4 | 1.0 |
| 96 | Qwen3.8-27B-FP8 | d2 | prefill_no_rule | lowercase_thinking | 100 | 34.6 | 23.8 | 45.1 | 61.2 | 3.0 |
| 97 | Qwen3.8-27B-FP8 | d2 | prefill_no_rule | uppercase_thinking | 100 | 36.4 | 26.5 | 46.2 | 67.0 | 0.0 |
| 98 | Qwen3.8-27B-FP8 | d2 | prefill_no_rule | meow_between_words | 100 | 20.5 | 11.9 | 29.4 | 57.5 | 5.0 |
| 99 | Qwen3.8-27B-FP8 | d2 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 2.1 | 14.1 | 2.0 |
| 100 | Qwen3.8-27B-FP8 | d3 | prefill_compliant | lowercase_thinking | 100 | 45.8 | 33.2 | 58.3 | 79.2 | 1.0 |
| 101 | Qwen3.8-27B-FP8 | d3 | prefill_compliant | uppercase_thinking | 100 | 39.4 | 28.2 | 51.0 | 77.3 | 1.0 |
| 102 | Qwen3.8-27B-FP8 | d3 | prefill_compliant | meow_between_words | 100 | 34.3 | 23.7 | 44.4 | 80.9 | 2.0 |
| 103 | Qwen3.8-27B-FP8 | d3 | prefill_compliant | end_of_sentence | 100 | 8.4 | 0.0 | 17.8 | 61.8 | 0.0 |
| 104 | Qwen3.8-27B-FP8 | d3 | prefill_no_rule | lowercase_thinking | 100 | 28.5 | 19.0 | 38.6 | 64.5 | 1.0 |
| 105 | Qwen3.8-27B-FP8 | d3 | prefill_no_rule | uppercase_thinking | 100 | 34.2 | 24.0 | 44.2 | 69.5 | 0.0 |
| 106 | Qwen3.8-27B-FP8 | d3 | prefill_no_rule | meow_between_words | 100 | 21.6 | 13.6 | 30.9 | 58.1 | 5.0 |
| 107 | Qwen3.8-27B-FP8 | d3 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 30.8 | 4.0 |
| 108 | Gemma-4-31B-FP8 | d0 | none | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 99.0 |
| 109 | Gemma-4-31B-FP8 | d0 | none | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 96.0 |
| 110 | Gemma-4-31B-FP8 | d0 | none | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 39.0 |
| 111 | Gemma-4-31B-FP8 | d0 | none | end_of_sentence | 100 | 1.0 | 0.0 | 3.0 | 2.0 | 1.0 |
| 112 | Gemma-4-31B-FP8 | generic | prefill_compliant | lowercase_thinking | 100 | 21.1 | 12.8 | 29.6 | 43.0 | 35.0 |
| 113 | Gemma-4-31B-FP8 | generic | prefill_compliant | uppercase_thinking | 100 | 3.3 | 0.0 | 8.0 | 40.0 | 9.0 |
| 114 | Gemma-4-31B-FP8 | generic | prefill_compliant | meow_between_words | 100 | 1.2 | 0.0 | 4.0 | 28.0 | 1.0 |
| 115 | Gemma-4-31B-FP8 | generic | prefill_compliant | end_of_sentence | 100 | 13.1 | 6.0 | 20.1 | 67.0 | 0.0 |
| 116 | Gemma-4-31B-FP8 | generic | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 98.0 |
| 117 | Gemma-4-31B-FP8 | generic | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 97.0 |
| 118 | Gemma-4-31B-FP8 | generic | prefill_no_rule | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 |
| 119 | Gemma-4-31B-FP8 | generic | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 120 | Gemma-4-31B-FP8 | d1 | prefill_compliant | lowercase_thinking | 100 | 10.3 | 4.3 | 17.1 | 25.0 | 17.0 |
| 121 | Gemma-4-31B-FP8 | d1 | prefill_compliant | uppercase_thinking | 100 | 0.0 | 0.0 | 5.5 | 19.0 | 10.0 |
| 122 | Gemma-4-31B-FP8 | d1 | prefill_compliant | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 24.0 | 3.0 |
| 123 | Gemma-4-31B-FP8 | d1 | prefill_compliant | end_of_sentence | 100 | 5.9 | 1.3 | 11.9 | 54.0 | 2.0 |
| 124 | Gemma-4-31B-FP8 | d1 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 60.0 |
| 125 | Gemma-4-31B-FP8 | d1 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 43.0 |
| 126 | Gemma-4-31B-FP8 | d1 | prefill_no_rule | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 3.0 | 27.0 |
| 127 | Gemma-4-31B-FP8 | d1 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| 128 | Gemma-4-31B-FP8 | d2 | prefill_compliant | lowercase_thinking | 100 | 11.5 | 4.7 | 19.0 | 40.0 | 16.0 |
| 129 | Gemma-4-31B-FP8 | d2 | prefill_compliant | uppercase_thinking | 100 | 2.6 | 0.0 | 7.7 | 32.4 | 10.0 |
| 130 | Gemma-4-31B-FP8 | d2 | prefill_compliant | meow_between_words | 100 | 0.0 | 0.0 | 4.1 | 23.3 | 2.0 |
| 131 | Gemma-4-31B-FP8 | d2 | prefill_compliant | end_of_sentence | 100 | 12.6 | 5.9 | 19.4 | 59.5 | 0.0 |
| 132 | Gemma-4-31B-FP8 | d2 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 47.0 |
| 133 | Gemma-4-31B-FP8 | d2 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 4.0 | 37.0 |
| 134 | Gemma-4-31B-FP8 | d2 | prefill_no_rule | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 10.0 | 7.0 |
| 135 | Gemma-4-31B-FP8 | d2 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 8.0 | 2.0 |
| 136 | Gemma-4-31B-FP8 | d3 | prefill_compliant | lowercase_thinking | 100 | 9.8 | 3.9 | 17.1 | 56.6 | 5.0 |
| 137 | Gemma-4-31B-FP8 | d3 | prefill_compliant | uppercase_thinking | 100 | 2.6 | 0.0 | 8.2 | 36.4 | 9.0 |
| 138 | Gemma-4-31B-FP8 | d3 | prefill_compliant | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 29.2 | 2.0 |
| 139 | Gemma-4-31B-FP8 | d3 | prefill_compliant | end_of_sentence | 100 | 3.8 | 0.0 | 8.2 | 58.4 | 0.0 |
| 140 | Gemma-4-31B-FP8 | d3 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 8.0 | 43.0 |
| 141 | Gemma-4-31B-FP8 | d3 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 12.0 | 31.0 |
| 142 | Gemma-4-31B-FP8 | d3 | prefill_no_rule | meow_between_words | 100 | 1.1 | 0.0 | 3.5 | 26.3 | 4.0 |
| 143 | Gemma-4-31B-FP8 | d3 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 18.0 | 0.0 |
| 144 | Qwen3.5-9B | d0 | none | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 145 | Qwen3.5-9B | d0 | none | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 146 | Qwen3.5-9B | d0 | none | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 147 | Qwen3.5-9B | d0 | none | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 148 | Qwen3.5-9B | generic | prefill_compliant | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 3.0 | 88.0 |
| 149 | Qwen3.5-9B | generic | prefill_compliant | uppercase_thinking | 100 | 19.2 | 11.6 | 27.5 | 57.0 | 11.0 |
| 150 | Qwen3.5-9B | generic | prefill_compliant | meow_between_words | 100 | 8.2 | 3.2 | 14.5 | 55.0 | 2.0 |
| 151 | Qwen3.5-9B | generic | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 4.0 | 0.0 |
| 152 | Qwen3.5-9B | generic | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 153 | Qwen3.5-9B | generic | prefill_no_rule | uppercase_thinking | 100 | 22.0 | 14.0 | 30.0 | 37.0 | 24.0 |
| 154 | Qwen3.5-9B | generic | prefill_no_rule | meow_between_words | 100 | 4.0 | 1.0 | 8.0 | 16.0 | 19.0 |
| 155 | Qwen3.5-9B | generic | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 |
| 156 | Qwen3.5-9B | d1 | prefill_compliant | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 94.0 |
| 157 | Qwen3.5-9B | d1 | prefill_compliant | uppercase_thinking | 100 | 1.0 | 0.0 | 3.0 | 10.0 | 4.0 |
| 158 | Qwen3.5-9B | d1 | prefill_compliant | meow_between_words | 100 | 12.0 | 6.1 | 19.1 | 40.3 | 5.0 |
| 159 | Qwen3.5-9B | d1 | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 33.0 |
| 160 | Qwen3.5-9B | d1 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 99.0 |
| 161 | Qwen3.5-9B | d1 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 60.0 |
| 162 | Qwen3.5-9B | d1 | prefill_no_rule | meow_between_words | 100 | 2.1 | 0.0 | 5.3 | 8.3 | 28.0 |
| 163 | Qwen3.5-9B | d1 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 53.0 |
| 164 | Qwen3.5-9B | d2 | prefill_compliant | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 41.0 |
| 165 | Qwen3.5-9B | d2 | prefill_compliant | uppercase_thinking | 100 | 1.0 | 0.0 | 3.0 | 8.0 | 19.0 |
| 166 | Qwen3.5-9B | d2 | prefill_compliant | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 24.0 | 2.0 |
| 167 | Qwen3.5-9B | d2 | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 11.0 | 4.0 |
| 168 | Qwen3.5-9B | d2 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 48.0 |
| 169 | Qwen3.5-9B | d2 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 39.0 |
| 170 | Qwen3.5-9B | d2 | prefill_no_rule | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 12.0 | 4.0 |
| 171 | Qwen3.5-9B | d2 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 6.0 |
| 172 | Qwen3.5-9B | d3 | prefill_compliant | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 3.0 | 32.0 |
| 173 | Qwen3.5-9B | d3 | prefill_compliant | uppercase_thinking | 100 | 1.5 | 0.0 | 4.8 | 13.0 | 13.0 |
| 174 | Qwen3.5-9B | d3 | prefill_compliant | meow_between_words | 100 | 2.0 | 0.0 | 5.0 | 31.0 | 2.0 |
| 175 | Qwen3.5-9B | d3 | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 18.0 | 3.0 |
| 176 | Qwen3.5-9B | d3 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 46.0 |
| 177 | Qwen3.5-9B | d3 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 28.0 |
| 178 | Qwen3.5-9B | d3 | prefill_no_rule | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 17.0 | 6.0 |
| 179 | Qwen3.5-9B | d3 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| 180 | Gemma-4-12B | d0 | none | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 99.0 |
| 181 | Gemma-4-12B | d0 | none | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 96.0 |
| 182 | Gemma-4-12B | d0 | none | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 29.0 |
| 183 | Gemma-4-12B | d0 | none | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 184 | Gemma-4-12B | generic | prefill_compliant | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 94.0 |
| 185 | Gemma-4-12B | generic | prefill_compliant | uppercase_thinking | 100 | 1.0 | 0.0 | 3.0 | 15.0 | 58.0 |
| 186 | Gemma-4-12B | generic | prefill_compliant | meow_between_words | 100 | 4.4 | 1.0 | 9.0 | 35.0 | 12.0 |
| 187 | Gemma-4-12B | generic | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 16.0 | 0.0 |
| 188 | Gemma-4-12B | generic | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 189 | Gemma-4-12B | generic | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 97.0 |
| 190 | Gemma-4-12B | generic | prefill_no_rule | meow_between_words | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 53.0 |
| 191 | Gemma-4-12B | generic | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 192 | Gemma-4-12B | d1 | prefill_compliant | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 45.0 |
| 193 | Gemma-4-12B | d1 | prefill_compliant | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 6.0 | 20.0 |
| 194 | Gemma-4-12B | d1 | prefill_compliant | meow_between_words | 100 | 3.1 | 0.0 | 7.2 | 44.0 | 2.0 |
| 195 | Gemma-4-12B | d1 | prefill_compliant | end_of_sentence | 100 | 3.0 | 0.0 | 6.0 | 21.0 | 1.0 |
| 196 | Gemma-4-12B | d1 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 0.0 | 65.0 |
| 197 | Gemma-4-12B | d1 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 43.0 |
| 198 | Gemma-4-12B | d1 | prefill_no_rule | meow_between_words | 100 | 2.0 | 0.0 | 5.0 | 13.0 | 24.0 |
| 199 | Gemma-4-12B | d1 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 |
| 200 | Gemma-4-12B | d2 | prefill_compliant | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 5.0 | 36.0 |
| 201 | Gemma-4-12B | d2 | prefill_compliant | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 14.0 | 16.0 |
| 202 | Gemma-4-12B | d2 | prefill_compliant | meow_between_words | 100 | 5.0 | 1.0 | 10.0 | 55.0 | 1.0 |
| 203 | Gemma-4-12B | d2 | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 29.0 | 0.0 |
| 204 | Gemma-4-12B | d2 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 50.0 |
| 205 | Gemma-4-12B | d2 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 4.0 | 31.0 |
| 206 | Gemma-4-12B | d2 | prefill_no_rule | meow_between_words | 100 | 7.0 | 2.0 | 12.0 | 48.0 | 3.0 |
| 207 | Gemma-4-12B | d2 | prefill_no_rule | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 3.0 | 0.0 |
| 208 | Gemma-4-12B | d3 | prefill_compliant | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 4.0 | 34.0 |
| 209 | Gemma-4-12B | d3 | prefill_compliant | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 18.0 | 13.0 |
| 210 | Gemma-4-12B | d3 | prefill_compliant | meow_between_words | 100 | 1.2 | 0.0 | 4.0 | 47.0 | 2.0 |
| 211 | Gemma-4-12B | d3 | prefill_compliant | end_of_sentence | 100 | 0.0 | 0.0 | 0.0 | 37.0 | 0.0 |
| 212 | Gemma-4-12B | d3 | prefill_no_rule | lowercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 1.0 | 50.0 |
| 213 | Gemma-4-12B | d3 | prefill_no_rule | uppercase_thinking | 100 | 0.0 | 0.0 | 0.0 | 2.0 | 33.0 |
| 214 | Gemma-4-12B | d3 | prefill_no_rule | meow_between_words | 100 | 3.0 | 0.0 | 7.0 | 43.0 | 2.0 |
| 215 | Gemma-4-12B | d3 | prefill_no_rule | end_of_sentence | 100 | 1.0 | 0.0 | 3.0 | 7.0 | 0.0 |

## Figures

- [G1_dose_response](../../../../figures/exp05_dose/interim_without_glm/G1_dose_response.html) ![G1_dose_response](../../../../figures/exp05_dose/interim_without_glm/G1_dose_response.png)
- [G1b_dose_response_case_lenient](../../../../figures/exp05_dose/interim_without_glm/G1b_dose_response_case_lenient.html) ![G1b_dose_response_case_lenient](../../../../figures/exp05_dose/interim_without_glm/G1b_dose_response_case_lenient.png)
- [G2_rule_minus_copying](../../../../figures/exp05_dose/interim_without_glm/G2_rule_minus_copying.html) ![G2_rule_minus_copying](../../../../figures/exp05_dose/interim_without_glm/G2_rule_minus_copying.png)
- [G3_per_rule](../../../../figures/exp05_dose/interim_without_glm/G3_per_rule.html) ![G3_per_rule](../../../../figures/exp05_dose/interim_without_glm/G3_per_rule.png)
- [G5_correctness_check](../../../../figures/exp05_dose/interim_without_glm/G5_correctness_check.html) ![G5_correctness_check](../../../../figures/exp05_dose/interim_without_glm/G5_correctness_check.png)
- [G4_effort](../../../../figures/exp05_dose/interim_without_glm/G4_effort.html) ![G4_effort](../../../../figures/exp05_dose/interim_without_glm/G4_effort.png)

## Checks

```
{
 "rows_per_model_and_file": {
  "Gemma-4-12B | Gemma-4-12B__card__stream_abort_Gemma-4-12B__1eb5ba7bb8dd.jsonl": 2400,
  "Gemma-4-31B-FP8 | Gemma-4-31B-FP8__card__stream_abort_Gemma-4-31B-FP8__211b07fd59f9.jsonl": 2400,
  "Qwen3-32B | Qwen3-32B__card__stream_abort_Qwen3-32B__192802ead6f7.jsonl": 2400,
  "Qwen3.5-9B | Qwen3.5-9B__card__stream_abort_Qwen3.5-9B__23e3e86181b2.jsonl": 2400,
  "Qwen3.6-27B-FP8 | Qwen3.6-27B-FP8__card__stream_abort_Qwen3.6-27B-FP8__faf37d246441.jsonl": 2400,
  "Qwen3.6-35B-A3B-FP8 | Qwen3.6-35B-A3B-FP8__card__stream_abort_base_Qwen3.6-35B-A3B-FP8__d3d0589d7344.jsonl": 500,
  "Qwen3.8-27B-FP8 | Qwen3.8-27B-FP8__card__stream_abort_Qwen3.8-27B-FP8__55f0f9d20301.jsonl": 2400,
  "Qwen3.8-27B-FP8-low | Qwen3.8-27B-FP8-low__card__stream_abort_effort__bf0edca26846.jsonl": 800,
  "Qwen3.8-27B-FP8-xhigh | Qwen3.8-27B-FP8-xhigh__card__stream_abort_effort__39ce97faad9f.jsonl": 800
 },
 "aborted_only_where_allowed": true,
 "aborted_all_non_compliant": true,
 "prefill_stored_as_requested": false,
 "empty_trace_rows": 0,
 "truncated_rows": 448,
 "no_think_close_rows": 14
}
```

Status: UNVERIFIED until a human adds it to VERIFIED.md
