# exp15 write report: single_test (UNVERIFIED)

| method | layer | alpha | gate (best answer-swap) | key swap - null | key swap outcomes | other-item success | answer-swap success |
|---|---|---|---|---|---|---|---|
| logit | 32 | 1.0 | 0.89 (pass) | +0.069 [+0.032, +0.111] n=189 | echo 0.07, other 0.33, start_swap 0.03, success 0.08, unchanged 0.48 | 0.00 | 0.13 |
| jlens | 28 | 1.0 | 0.72 (pass) | +0.132 [+0.085, +0.180] n=189 | echo 0.22, other 0.30, start_swap 0.14, success 0.15, unchanged 0.19 | 0.00 | 0.22 |
| rlens | 28 | 1.0 | 0.78 (pass) | +0.148 [+0.101, +0.201] n=189 | echo 0.17, other 0.37, start_swap 0.11, success 0.16, unchanged 0.19 | 0.00 | 0.13 |
| jpp | 28 | 1.0 | 0.77 (pass) | +0.212 [+0.153, +0.270] n=189 | echo 0.05, other 0.48, start_swap 0.11, success 0.23, unchanged 0.14 | 0.00 | 0.10 |
| template | 36 | 1.0 | 0.40 (pass) | +0.042 [+0.000, +0.090] n=189 | echo 0.20, other 0.28, start_swap 0.06, success 0.09, unchanged 0.38 | 0.00 | 0.20 |
| diffmeans | 40 | 0.5 | 0.51 (pass) | +0.667 [+0.598, +0.730] n=189 | echo 0.06, other 0.15, start_swap 0.08, success 0.69, unchanged 0.01 | 0.00 | 0.13 |
| nla_ar | 42 | 0.5 | 0.38 (pass) | +0.365 [+0.296, +0.439] n=189 | echo 0.19, other 0.26, start_swap 0.09, success 0.38, unchanged 0.08 | 0.00 | 0.17 |

Head-to-head at layer 42 (each method at its best dev alpha there):

| method | alpha | key swap - null | key swap success |
|---|---|---|---|
| logit | 0.5 | +0.021 [+0.000, +0.048] n=189 | 0.03 |
| jlens | 0.5 | +0.032 [+0.000, +0.064] n=189 | 0.04 |
| rlens | 0.25 | +0.026 [+0.005, +0.053] n=189 | 0.03 |
| jpp | 0.5 | +0.032 [+0.000, +0.064] n=189 | 0.04 |
| template | 1.0 | +0.026 [-0.016, +0.069] n=189 | 0.06 |
| diffmeans | 0.5 | +0.598 [+0.529, +0.667] n=189 | 0.61 |
| nla_ar | 0.5 | +0.365 [+0.296, +0.439] n=189 | 0.38 |

P4 repair (consistent off-cycle h=3 errors):

| method | repair (gold) | repair - null | redirect (T(r)) | null gold rate |
|---|---|---|---|---|
| jlens | +0.118 [+0.000, +0.294] n=17 | +0.059 [-0.118, +0.235] n=17 | +0.091 [+0.000, +0.273] n=11 | +0.059 [+0.000, +0.176] n=17 |
| rlens | +0.059 [+0.000, +0.176] n=17 | +0.000 [-0.176, +0.176] n=17 | +0.000 [+0.000, +0.000] n=11 | +0.059 [+0.000, +0.176] n=17 |
| jpp | +0.235 [+0.059, +0.471] n=17 | +0.176 [+0.000, +0.353] n=17 | +0.273 [+0.000, +0.545] n=11 | +0.059 [+0.000, +0.176] n=17 |
| diffmeans | +0.706 [+0.471, +0.882] n=17 | +0.647 [+0.412, +0.882] n=17 | +0.818 [+0.545, +1.000] n=11 | +0.059 [+0.000, +0.176] n=17 |
| nla_ar | +0.294 [+0.118, +0.529] n=17 | +0.235 [+0.059, +0.471] n=17 | +0.545 [+0.273, +0.818] n=11 | +0.059 [+0.000, +0.176] n=17 |
