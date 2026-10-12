# exp15 wrong-key distribution: test (exploratory, behaviour only, UNVERIFIED)

Consistent off-cycle errors, n = 323. Chi-square p against uniform: colours {'w': 0.0, 'answer': 0.0, 'true_prev': 1e-06}; rule-line positions {'w': 0.0, 'true_prev': 0.0}.

| colour | share as w | share as answer a' | share as true s_{h-1} | off-cycle rate when s_{h-1} is it |
|---|---|---|---|---|
| black | 0.009 | 0.164 | 0.084 | 0.102 |
| blue | 0.121 | 0.093 | 0.080 | 0.107 |
| brown | 0.006 | 0.059 | 0.195 | 0.242 |
| gold | 0.053 | 0.062 | 0.108 | 0.123 |
| gray | 0.204 | 0.217 | 0.050 | 0.058 |
| green | 0.170 | 0.046 | 0.068 | 0.09 |
| pink | 0.207 | 0.031 | 0.087 | 0.112 |
| purple | 0.084 | 0.096 | 0.108 | 0.136 |
| red | 0.059 | 0.053 | 0.111 | 0.137 |
| white | 0.087 | 0.180 | 0.108 | 0.136 |

| rule-line position (0 = printed first) | share as w's line | share as s_{h-1}'s line |
|---|---|---|
| 0 | 0.000 | 0.080 |
| 1 | 0.189 | 0.059 |
| 2 | 0.254 | 0.059 |
| 3 | 0.248 | 0.096 |
| 4 | 0.118 | 0.068 |
| 5 | 0.111 | 0.071 |
| 6 | 0.056 | 0.124 |
| 7 | 0.019 | 0.115 |
| 8 | 0.006 | 0.164 |
| 9 | 0.000 | 0.164 |

w's line minus s_{h-1}'s line (positions): {'-8': 6, '-7': 30, '-6': 38, '-5': 26, '-4': 24, '-3': 44, '-2': 41, '-1': 39, '1': 23, '2': 21, '3': 19, '4': 9, '5': 1, '6': 2}

Most common s_{h-1} -> w pairs: [('brown->pink', 14), ('purple->gray', 13), ('brown->gray', 12), ('white->pink', 12), ('red->pink', 11), ('brown->green', 10), ('green->pink', 9), ('brown->blue', 8)]

Accuracy (all items of that h) by the printed position of each lookup's rule line (lookup k uses the line of s_{k-1}):

| h, lookup | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| h2_lookup1 | 0.89 | 0.77 | 0.65 | 0.72 | 0.82 | 0.76 | 0.80 | 0.72 | 0.67 | 0.62 |
| h2_lookup2 | 0.59 | 0.87 | 0.93 | 0.84 | 0.80 | 0.81 | 0.78 | 0.60 | 0.59 | 0.58 |
| h3_lookup1 | 0.53 | 0.52 | 0.41 | 0.36 | 0.30 | 0.34 | 0.29 | 0.28 | 0.20 | 0.21 |
| h3_lookup2 | 0.42 | 0.31 | 0.34 | 0.29 | 0.24 | 0.34 | 0.39 | 0.41 | 0.33 | 0.40 |
| h3_lookup3 | 0.09 | 0.53 | 0.60 | 0.52 | 0.50 | 0.50 | 0.30 | 0.22 | 0.12 | 0.10 |
