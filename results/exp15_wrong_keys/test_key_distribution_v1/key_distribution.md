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
