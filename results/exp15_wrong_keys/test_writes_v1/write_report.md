# exp15 write report: test (UNVERIFIED)

| method | layer | alpha | gate (best answer-swap) | key swap - null | key swap outcomes | other-item success | answer-swap success |
|---|---|---|---|---|---|---|---|
| logit | 32 | 1.0 | 0.89 (pass) | +0.064 [+0.032, +0.096] n=440 | echo 0.12, other 0.27, start_swap 0.06, success 0.10, unchanged 0.46 | 0.05 | 0.13 |
| jlens | 28 | 1.0 | 0.72 (pass) | +0.139 [+0.104, +0.175] n=440 | echo 0.19, other 0.34, start_swap 0.09, success 0.16, unchanged 0.21 | 0.05 | 0.15 |
| rlens | 28 | 1.0 | 0.78 (pass) | +0.125 [+0.091, +0.159] n=440 | echo 0.14, other 0.35, start_swap 0.12, success 0.15, unchanged 0.24 | 0.07 | 0.14 |
| jpp | 28 | 1.0 | 0.77 (pass) | +0.179 [+0.143, +0.218] n=440 | echo 0.07, other 0.46, start_swap 0.12, success 0.20, unchanged 0.15 | 0.07 | 0.04 |
| template | 36 | 1.0 | 0.40 (pass) | +0.041 [+0.014, +0.070] n=440 | echo 0.24, other 0.33, start_swap 0.03, success 0.07, unchanged 0.33 | 0.04 | 0.20 |
| diffmeans | 40 | 0.5 | 0.51 (pass) | +0.550 [+0.502, +0.598] n=440 | echo 0.09, other 0.21, start_swap 0.11, success 0.57, unchanged 0.02 | 0.03 | 0.11 |
| nla_ar | 42 | 0.5 | 0.38 (pass) | +0.416 [+0.370, +0.461] n=440 | echo 0.15, other 0.24, start_swap 0.08, success 0.44, unchanged 0.09 | 0.05 | 0.11 |

Head-to-head at layer 42 (each method at its best dev alpha there):

| method | alpha | key swap - null | key swap success |
|---|---|---|---|
| logit | 0.5 | +0.034 [+0.016, +0.054] n=440 | 0.05 |
| jlens | 0.5 | +0.045 [+0.023, +0.070] n=440 | 0.07 |
| rlens | 0.25 | +0.025 [+0.009, +0.043] n=440 | 0.05 |
| jpp | 0.5 | +0.025 [+0.004, +0.048] n=440 | 0.05 |
| template | 1.0 | +0.039 [+0.016, +0.064] n=440 | 0.05 |
| diffmeans | 0.5 | +0.486 [+0.439, +0.534] n=440 | 0.51 |
| nla_ar | 0.5 | +0.416 [+0.370, +0.461] n=440 | 0.44 |

P4 repair (consistent off-cycle h=3 errors):

| method | repair (gold) | repair - null | redirect (T(r)) | null gold rate |
|---|---|---|---|---|
| jlens | +0.132 [+0.094, +0.173] n=266 | +0.113 [+0.071, +0.154] n=266 | +0.128 [+0.077, +0.179] n=156 | +0.019 [+0.004, +0.038] n=266 |
| rlens | +0.124 [+0.086, +0.165] n=266 | +0.105 [+0.064, +0.150] n=266 | +0.122 [+0.070, +0.173] n=156 | +0.019 [+0.004, +0.038] n=266 |
| jpp | +0.124 [+0.086, +0.165] n=266 | +0.105 [+0.064, +0.150] n=266 | +0.128 [+0.077, +0.186] n=156 | +0.019 [+0.004, +0.038] n=266 |
| diffmeans | +0.515 [+0.455, +0.575] n=266 | +0.440 [+0.376, +0.504] n=266 | +0.468 [+0.391, +0.545] n=156 | +0.075 [+0.045, +0.109] n=266 |
| nla_ar | +0.365 [+0.308, +0.421] n=266 | +0.312 [+0.248, +0.376] n=266 | +0.333 [+0.263, +0.410] n=156 | +0.053 [+0.026, +0.079] n=266 |
