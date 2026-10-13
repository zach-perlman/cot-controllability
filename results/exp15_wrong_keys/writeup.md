# Reading and writing the wrong intermediate: on Qwen3.6-27B's no-CoT brew errors, the true intermediate is readable as well as (more weakly) the wrong key, and difference-of-means and NLA directions set the key the last lookup uses

**Status: UNVERIFIED** until a human adds it to `VERIFIED.md`. Model: Qwen3.6-27B (bf16, thinking off,
"Answer:" prefilled; answer = argmax over the 10 colour tokens). Pre-registration: `manifest.json` (locked
2026-10-12T00:40:51Z, before the test bank existed; 19 deviations listed in it) and `write_lock.json` (each write
method's layer and alpha, chosen on dev before any test write trial). Every number below is from the held-out test
bank unless marked dev. Exploratory analyses (not in the plan) are marked as such. No item text appears in this
file or in the figures.

## TL;DR

The question: on single-table brew items the model gets wrong with an answer a' that is not on the start colour's
cycle at all ("off-cycle" errors), the colour a' came from some rule line w -> a', so w = T^-1(a') is the key the
model's last lookup must have used. Do activation readers find w before the answer exists? Can colour directions
built by each method write a key into the model, and does writing the true key repair the error?

1. **Reads (P1, P2).** On 323 consistent off-cycle errors, the pre-registered P2 test (w above controls, and above
   T(a')) passes for the probe (0.80 [0.76, 0.83]; direction 0.70), template lens, J-lens, diffmeans and R-lens
   (0.64-0.69), and fails for logit, J++ and all three text readers (NLA, Oracle lens, AO: 0.52-0.54). But readers
   have colour preferences and w is far from uniform, so these raw AUCs are partly prior. Against a shuffled-item
   null and after subtracting each colour's mean score, the probe keeps a clear w signal (0.77 vs null 0.56),
   diffmeans and the logit, J-, R- and J++ lenses a weak one (0.61-0.67), the text readers 0.54-0.57, and the
   template lens's result is entirely its prior (exploratory controls).
2. **The true intermediate is also there, and the error looks like a retrieval failure** (exploratory). On the same
   errors, the true s_{h-1} ranks top for the probe on 59% of items (shuffled null 23%), w on 28% (null 31%). Before
   the answer forms (layers 36-48, where the lenses rank the answer first on <= 20% of items), the lenses read w and
   the true s_{h-1} about equally. Behaviourally, the last lookup fails when its rule line is printed first or near
   the end (h=3 accuracy 0.09 and 0.10, vs 0.50-0.60 in the middle), and w's line is at printed positions 1-3 in 69%
   of errors. The data fit "right query, wrong line retrieved" better than "a wrong key was computed".

3. **Writes (P3).** On 440 held-out correct h=2 items, swapping the first-stir key s1 -> s1' (success = the model
   now answers T(s1'), the colour the *new* key looks up to) works for the difference-of-means direction
   (+0.55 [+0.50, +0.60] above a random direction of the same size) and the NLA activation reconstructor's direction
   (+0.42 [+0.37, +0.46]). Token-lens directions (logit, J-lens, R-lens, J++, template) reach +0.04 to +0.18. A
   direction built from colours unrelated to the item succeeds 3-7% of the time for every method, so the effect is
   item-specific. In the layer-42 head-to-head: diffmeans +0.49, NLA AR +0.42, every lens +0.03 to +0.05.
4. **The two directions that work write a key; the lens directions write the output.** Written in place of s2
   (the "answer swap"), diffmeans for colour a' makes the model answer T(a') 42% of the time and a' itself 11%
   (+0.31 [+0.25, +0.38]; NLA AR 29% vs 11%): the written colour is looked up once more. The token-lens
   directions (built from the unembedding) do the opposite at layers 40-44 on dev: the model answers a' itself
   (0.33-0.89) and almost never T(a') (<= 0.06). They fail at key swaps because they write an answer, not a key
   (exploratory, `test_write_breakdown_v1/`).
5. **Repair (P4).** On 266 consistent off-cycle h=3 errors, writing w -> s2 gives the gold answer +0.44
   [+0.38, +0.50] above null with diffmeans and +0.31 [+0.25, +0.38] with NLA AR (lenses about +0.11). But writing
   w -> an arbitrary off-path colour r ("redirect") yields T(r) at the same rate (0.47 and 0.33). So repair shows
   that the key the last lookup uses is writable; it does not show the model held the true s2 and lost it.
6. **Instruments.** agu18dec's ptag activation reconstructor failed its retrieval check (0.48 at layer 44, bar 0.8,
   published 0.93) and was dropped from the writes, as pre-registered, after three debugging rounds; the gap grows
   with depth and is most likely their gated whitener. The token lenses miss the literal lens-sanity bar (0.59-0.86
   top-1 at layer 60, bar 0.90; 0.84-0.995 on confident items). Everything else passed (table below).

## Setup

**Task.** A potion starts one colour; one per-item random derangement T of 10 colours is applied h times; answer
the final colour, no CoT (exp11's single-table format: one h=2 example shot, then the item). The true path is
s_0..s_h with s_k = T^k(s_0). An answer a' is *off-cycle* if a' != T^k(s_0) for every k in 0..h+3, and
*consistent* if p(a') >= 0.3 over the colour softmax (pre-lock deviation from 0.5, which left 3 dev errors).
Per item: key w = T^-1(a') (on a correct item, w = s_{h-1}); forward neighbour T(a'); controls = colours off the
true path, off the path implied by a', and not T(a') (about 3-5 per item).

**Banks.** Dev = exp11's `single_test` (250 per h; already seen in exp11): used to choose cells, layers and write
settings and to test the code. Test = fresh seed `exp15_test_20261012`, 100 / 600 / 2000 items at h = 1 / 2 / 3
(enlarged per kill rule K2), generated after the manifest lock, no problem text shared with any existing bank.
Test groups: P1 correct h=2 445, h=3 692; P2 consistent off-cycle 323 (h=2 57, h=3 266; 561 at any p(a'), 67 at
p >= 0.5, 6 at p >= 0.8); miscounts 834. Text readers ran on all off-cycle errors plus seeded samples of 300 correct
items per h and 300 miscounts (1,461 rows).

**Readers** (all scored per item at pre-registered primary cells `question_end`, `nl_asst`, `final`):
- Token lenses at layers 36-48: logit lens, J-lens and R-lens (workspace lenses), J++ (koayon), template lens.
  Score = the colour's rank among the 10 within each (cell, layer), averaged.
- Probe (PCA 64 + logistic regression) and difference of means, trained on correct h=2/h=3 items to predict the
  key s_{h-1}, 5-fold cross-fitted; off-cycle items are scored by the mean of the five fold models.
- Text readers, 4 samples per vector, score = fraction of samples naming the colour (regex matcher, English and
  Chinese colour words, hand-audited): NLA (AV `iter_000800`, layer 42), ceselder's Oracle lens (layer 42),
  agu18dec's activation oracle (AO, layers 36/40/44/48).

**Tests.** P1: AUC of the true key vs controls on correct items (s1 for h=2, s2 for h=3). P2, on consistent
off-cycle errors (h=2 and h=3 pooled): AUC of w vs controls; direction test w vs T(a') (rules out "both colours
on the answer's rule line"); the same with only the (cell, layer) pairs where J++ does not yet rank a' in its top
3 (answer-leak filter); per item, which of {w, true s_{h-1}, controls} ranks top (H_key / H_override / H_slip).
Items bootstrapped (10k), Holm across the 40 primary reader x test comparisons.

**Writes.** Additive edit at one layer, every position from the start-colour token to the final token:
dh = alpha * ||h|| * unit(v_l(c') - v_l(c)), so every method gets the same edit size. v_l(c) per method: logit
(g * W_U[c]), J-lens / R-lens / J++ (g * W_U[c] pulled back through J_l), template lens row, diffmeans (dev, final
cell), NLA AR (mean AR output over 4 fixed explanation templates; layer 42 only). Null: a seeded random unit
direction at the same layer and alpha. Each method's layer (28-44) and alpha (0.1-1.0) maximise dev key swap -
null (`write_lock.json`); all methods passed the answer-swap gate (best dev answer-swap success >= 0.30). Repair
was run for methods with dev key swap - null >= 0.10 (jlens, rlens, jpp, diffmeans, nla_ar).

## Instrument checks (day 0; `checks_day0/`)

| check | result | bar | verdict |
|---|---|---|---|
| behaviour reproduces exp11 (dev) | max per-(h, label) difference 3 items | <= 3 | pass |
| lens sanity: answer top-1 at layer 60, final cell | logit, J, R, J++ 0.76-0.86 (>= 0.99 when p(a') >= 0.8); template 0.59 (0.84) | >= 0.90 | fail, kept (deviation) |
| NLA layer convention / AV->AR round trip | cos 0.9993 with hidden[43] / FVE 0.77 | 0.99 / 0.6 | pass |
| Oracle lens, AO: bullets overlap the next tokens | 0.036 vs 0.004 shuffled; AO 0.027 vs 0.004 | > shuffled | pass |
| ptag AR retrieval@1, layer 44 | 0.48 (three debug rounds: best 0.53) | >= 0.8 | fail, dropped |
| matcher vs hand labels (50 readouts; 30 Chinese AO readouts) | P 1.00 R 1.00; P 0.99 R 1.00 | >= 0.95 | pass |
| edit harness at alpha = 0 reproduces captured answers | 0.992 of 256 trials | >= 0.99 | pass |
| AO on WorkspaceBench's 100 brew items: gold vs off-path | 0.495 vs 0.224 | gold > off-path | pass |

ptag AR debugging (three rounds, `checks_day0/ptag_debug_*.json`; retrieval@1 at layer 44): reading the final hidden
state instead of block L (0.47), chat-formatted assistant-turn targets (0.28), batch-centring (0.50), whitening the
prediction too (0.53) and a diagonal whitener (0.34) all stay below the bar. Within pools of 24 spans our layer-20
retrieval matches their published number (0.93 vs 0.93) but layer 44 is 0.74 (published 0.93) and layer 60 0.67
(published 0.98), and the scalar-optimal FVE is about 0 (published 0.22): the model loads correctly, and the
mismatch grows with depth, which points at their gated pooled whitener (we use our own natural-text stand-in).

## Results

### Reads (P1, P2)

`test_v1/read_report.md`; figures `figures/exp15_wrong_keys/test_v2/fig_reads.png`, `fig_decomposition.png`.
Pre-registered numbers (AUC [95% CI]; P1 h=2 n=445, h=3 n=692, text readers 300 each; P2 n=323, direction n=246):

| reader | P1 h=2: s1 | P1 h=3: s2 | P2: w vs controls | P2: w vs T(a') | P2 leak-filtered |
|---|---|---|---|---|---|
| probe | 0.996 | 0.933 | 0.798 [0.761, 0.834] | 0.695 [0.638, 0.752] | 0.773 |
| template | 0.609 | 0.604 | 0.688 [0.647, 0.729] | 0.691 [0.632, 0.748] | 0.685 |
| J-lens | 0.730 | 0.680 | 0.675 [0.632, 0.716] | 0.695 [0.638, 0.752] | 0.669 |
| diffmeans | 0.740 | 0.772 | 0.666 [0.622, 0.709] | 0.614 [0.553, 0.675] | 0.656 |
| R-lens | 0.801 | 0.661 | 0.635 [0.590, 0.679] | 0.650 [0.589, 0.707] | 0.629 |
| logit | 0.638 | 0.548 | 0.539 [0.494, 0.585] | 0.492 | 0.531 |
| NLA (L42) | 0.608 | 0.574 | 0.538 [0.495, 0.581] | 0.537 | 0.539 |
| Oracle lens (L42) | 0.607 | 0.580 | 0.523 [0.485, 0.561] | 0.500 | 0.528 |
| J++ | 0.735 | 0.639 | 0.517 [0.471, 0.563] | 0.445 | 0.514 |
| AO (L36-48) | 0.649 | 0.609 | 0.517 [0.472, 0.561] | 0.488 | 0.540 |

After Holm over the 40 primary comparisons: every P1 test passes; P2 key AUC and the direction test pass for the
probe, template, J-lens, R-lens and diffmeans (adjusted p = 0.004 each, the floor at 10k resamples); no text reader
passes either P2 test (NLA key AUC adjusted p = 0.43).

**The colour-prior check changes the reading** (exploratory, `test_shuffle_null_v1/shuffle_null.md`). Each reader
has fixed colour preferences, and on these errors w is far from uniform (gray 0.20, pink 0.21, green 0.17; black
0.01, brown 0.01; `test_key_distribution_v2/`). A reader with only colour preferences can then score w above the
controls. Two controls: a shuffled-item null (each P2 item scored with another P2 item's reads; chance for a reader
that ignores the item) and prior-corrected scores (each colour's mean score over the bank subtracted).

| reader | P2 key AUC: raw / shuffled null / prior-corrected [95% CI] | direction: raw / null / prior-corrected |
|---|---|---|
| probe | 0.798 / 0.558 / 0.773 [0.735, 0.810] | 0.695 / 0.564 / 0.642 [0.581, 0.699] |
| diffmeans | 0.666 / 0.594 / 0.672 [0.627, 0.715] | 0.614 / 0.599 / 0.606 [0.545, 0.667] |
| J-lens | 0.675 / 0.626 / 0.633 [0.590, 0.678] | 0.695 / 0.652 / 0.610 [0.549, 0.671] |
| R-lens | 0.635 / 0.586 / 0.619 [0.574, 0.664] | 0.650 / 0.602 / 0.549 [0.488, 0.610] |
| J++ | 0.517 / 0.490 / 0.618 [0.572, 0.662] | 0.445 / 0.451 / 0.532 [0.471, 0.594] |
| logit | 0.539 / 0.512 / 0.608 [0.562, 0.653] | 0.492 / 0.485 / 0.557 [0.496, 0.618] |
| AO | 0.517 / 0.486 / 0.571 [0.526, 0.617] | 0.488 / 0.461 / 0.565 [0.500, 0.626] |
| template | 0.688 / 0.672 / 0.570 [0.522, 0.616] | 0.691 / 0.680 / 0.500 [0.435, 0.561] |
| NLA | 0.538 / 0.536 / 0.550 [0.503, 0.596] | 0.537 / 0.527 / 0.541 [0.476, 0.602] |
| Oracle lens | 0.523 / 0.508 / 0.543 [0.498, 0.589] | 0.500 / 0.494 / 0.516 [0.455, 0.581] |

The template lens's P2 result is its colour preference (raw 0.688, shuffled 0.672; direction corrected to 0.500).
The probe's survives both controls (0.773, against a 0.558 shuffled null), as do diffmeans and, more weakly, the
J-lens and R-lens. So **an item-specific w signal is present before the answer for the probe and diffmeans, weakly
for the lenses, and at most marginally for the text readers**. On correct items the controls barely move P1
(probe 0.933 -> 0.933; lenses rise after correction, e.g. J-lens h=3 0.680 -> 0.836).

**The true intermediate is there too** (exploratory). Share of P2 errors where each candidate ranks top
(w / true s_{h-1} / a control; chance with about 4 controls is about 1/6 / 1/6 / 2/3):

| reader | raw | prior-corrected | shuffled-item null |
|---|---|---|---|
| probe | 0.33 / 0.54 / 0.12 | 0.28 / 0.59 / 0.13 | 0.31 / 0.23 / 0.46 |
| diffmeans | 0.40 / 0.30 / 0.29 | 0.39 / 0.34 / 0.27 | 0.34 / 0.21 / 0.46 |
| J-lens | 0.37 / 0.29 / 0.35 | 0.32 / 0.42 / 0.27 | 0.37 / 0.19 / 0.44 |
| R-lens | 0.36 / 0.30 / 0.34 | 0.30 / 0.40 / 0.30 | 0.33 / 0.21 / 0.46 |
| template | 0.44 / 0.17 / 0.39 | 0.27 / 0.37 / 0.36 | 0.42 / 0.16 / 0.43 |

The w share is at its shuffled null for every reader except diffmeans (0.39 vs 0.34); the true-s_{h-1} share is
above its null for all of them (probe 0.59 vs 0.23). On these errors the probe also separates the true s_{h-1} from
the earlier path colours (0.740 [0.711, 0.769]). The pre-registered raw shares favour w for the lenses and diffmeans,
but those are at or near what colour preferences alone produce; read against the shuffled null and after the prior
correction, the decomposition favours "the true intermediate is computed and a different key wins" over "only the
wrong key is held".

**Layer course** (exploratory, `test_layer_course_v2/`, figure
`figures/exp15_wrong_keys/test_layer_course_v2/fig_layer_course_prior_corrected.png`). With the colour prior
subtracted, every lens curve is at 0.5 up to layer ~32 (the raw curves are not: w already scores 0.6-0.7 at layer
0, which is the prior). In the primary window (layers 36-48) at `nl_asst` and `final`, the J-lens scores w vs
controls 0.58-0.68 and the true s_{h-1} on the same error items 0.60-0.69 (other lenses 0.52-0.72 for both), while
it ranks the model's answer first on only 3-20% of these items; the answer emerges after layer ~50. So the w
signal is not a read-out of an already-formed answer, but in the window the lenses see w and the true intermediate
about equally.

**Where w comes from: the last lookup is position-biased** (exploratory, behaviour only,
`test_key_distribution_v2/key_distribution.md`). The rule lines are printed in a random order per item (each colour
equally often at each position). On P2 errors, w's line is at printed positions 1-3 in 69% of items and never at
position 0 or 9, while the line the model needed (s_{h-1}'s) is at positions 6-9 in 57%. Over all items, accuracy
depends on where the *last* lookup's line is printed: h=3, 0.09 if it is first, 0.50-0.60 at positions 1-5, 0.10 if
it is last (h=2: 0.59, 0.80-0.93, 0.58). The first lookup degrades steadily with position (h=3: 0.53 -> 0.21), the
middle one is flat. All lines are formatted identically. One account consistent with this and with the reads:
the final lookup queries with the right intermediate, fails to retrieve a line outside its preferred span of
positions, and lands on an early line, whose left side is w. This is post hoc and untested by intervention here.

### Writes (P3): key swaps on correct h=2 items

`test_writes_v1/write_report.md`; figure `figures/exp15_wrong_keys/test_v2/fig_writes.png`.

| method | layer, alpha (dev-locked) | key swap - null [95% CI] | key swap success | other-item success | null success |
|---|---|---|---|---|---|
| diffmeans | 40, 0.5 | +0.550 [+0.502, +0.598] | 0.57 | 0.03 | 0.03 |
| NLA AR | 42, 0.5 | +0.416 [+0.370, +0.461] | 0.44 | 0.05 | 0.02 |
| J++ | 28, 1.0 | +0.179 [+0.143, +0.218] | 0.20 | 0.07 | 0.02 |
| J-lens | 28, 1.0 | +0.139 [+0.104, +0.175] | 0.16 | 0.05 | 0.02 |
| R-lens | 28, 1.0 | +0.125 [+0.091, +0.159] | 0.15 | 0.07 | 0.02 |
| logit | 32, 1.0 | +0.064 [+0.032, +0.096] | 0.10 | 0.05 | 0.04 |
| template | 36, 1.0 | +0.041 [+0.014, +0.070] | 0.07 | 0.04 | 0.03 |

n = 440 items per row; every row survives Holm (adjusted p <= 0.002). Key-swap outcomes for diffmeans: success
0.57, start-swap (s1' read as the start colour, answer T^2(s1')) 0.11, echo (answers s1') 0.09, unchanged 0.02,
other 0.21. Success is spread over the 10 written colours (by s1': diffmeans 0.28-0.84, NLA AR 0.10-0.76), so it
is not carried by one or two colours; the logit lens's small effect is mostly one colour (green 0.42, the other
nine 0.00-0.14) (exploratory, `test_write_breakdown_v1/write_breakdown.md`).

**What the directions write** (exploratory, same file). The answer swap writes a' in place of the answer s2. A
direction that encodes an output colour makes the model say a'; one that encodes a key makes it say T(a').

| direction, layer, alpha | answer a' | answer T(a') | T(a') - a' [95% CI] | random null: a' / T(a') |
|---|---|---|---|---|
| diffmeans, 40, 0.5 (locked) | 0.109 | 0.423 | +0.314 [+0.252, +0.375] | 0.025 / 0.021 |
| NLA AR, 42, 0.5 (locked) | 0.111 | 0.293 | +0.182 [+0.125, +0.239] | 0.023 / 0.016 |
| J++, 28, 1.0 (locked) | 0.043 | 0.141 | +0.098 [+0.057, +0.136] | 0.032 / 0.025 |
| J-lens, 28, 1.0 (locked) | 0.148 | 0.127 | -0.021 [-0.070, +0.029] | 0.032 / 0.025 |
| J-lens, 42, 0.5 | 0.157 | 0.050 | -0.107 [-0.148, -0.066] | 0.023 / 0.016 |
| template, 42, 1.0 | 0.402 | 0.036 | -0.366 [-0.418, -0.316] | 0.021 / 0.043 |

On the dev grid (alpha 1.0), the token-lens directions move from neither (layers 28-32) to writing the output
(layers 40-44: a' 0.51-0.89 for logit, J-lens, R-lens and J++, 0.33-0.40 for template; T(a') <= 0.06). diffmeans
writes the key at layers 36-40 (T(a') 0.25-0.28) and the output too when the edit is large (a' 0.33-0.51 at alpha
1.0; at alpha 0.5 on test, T(a') dominates).

Head-to-head at layer 42 (each method at its best dev alpha there): diffmeans +0.486 [+0.439, +0.534], NLA AR
+0.416 [+0.370, +0.461], J-lens +0.045, template +0.039, logit +0.034, R-lens +0.025, J++ +0.025.

The lens directions' dev optimum is at the edge of the grid (layer 28, alpha 1.0, the earliest layer and the
largest edit), so their test numbers are not necessarily their best; within the grid they are far below
diffmeans and NLA AR at every layer and alpha (`figures/exp15_wrong_keys/dev_v1/fig_dev_grid.png`).

### Repair (P4): consistent off-cycle h=3 errors

| method | repair: gold | repair - null [95% CI] | redirect: T(r) | null gold rate |
|---|---|---|---|---|
| diffmeans | 0.515 | +0.440 [+0.376, +0.504] | 0.468 [0.391, 0.545] | 0.075 |
| NLA AR | 0.365 | +0.312 [+0.248, +0.376] | 0.333 [0.263, 0.410] | 0.053 |
| J-lens | 0.132 | +0.113 [+0.071, +0.154] | 0.128 | 0.019 |
| R-lens | 0.124 | +0.105 [+0.064, +0.150] | 0.122 | 0.019 |
| J++ | 0.124 | +0.105 [+0.064, +0.150] | 0.128 | 0.019 |

n = 266 (redirect: the 156 items with a colour r off both paths whose T(r) is also off them). Repair and redirect
succeed at about the same rate for every method: whatever key is written, the last lookup uses it. Repair is
therefore evidence that the key is writable at layers 28-42, not that the true s2 was present and overridden.

### Secondary (exploratory)

All on the test bank unless marked dev; outputs in `test_count_target_v1/`, `test_edits_secondary_v1/`,
`dev_edits_secondary_v1/`, `test_selfreport_v1/` (dev `dev_selfreport_v1/`), `test_nla_variants_v1/` and
`dev_nla_variants_v1/`.

**Count target** (820 miscount errors with a unique implied count k != h; AUC of the count word k vs the other
counts, at the primary cells and layers). The implied count is barely read: logit 0.53, J-lens 0.52, R-lens 0.48,
J++ 0.58, template 0.56, NLA 0.55, Oracle lens 0.50, AO 0.54 (text readers n = 275). The true count h is read
almost perfectly (lenses 1.00, AO 0.996), as expected for a number word printed in the prompt. Miscount errors do
not come with a readable wrong count.

**Count repair** (512 consistent miscount errors; count-word direction k -> h at each method's dev-best setting):
+0.06 to +0.11 above null (logit +0.092, J-lens +0.107, J++ +0.074, template +0.063). On dev the grid's best cells
were about +0.10 with many negative cells, so the test numbers are about what the dev selection promised.

**AR edit** (per-item directions AR(the item's own NLA explanation with s1 replaced by s1') - AR(explanation), on the
206 key-swap items whose NLA explanations name s1). At layer 42, alpha 0.5: +0.257 [+0.194, +0.325] above null, but
the fixed-template NLA AR direction does better on the same items (+0.456): AR edit - template -0.199 [-0.282,
-0.117]. Dev agrees (-0.136, n = 125). The item's own explanation adds nothing over a generic one.

**Read-then-write** (P4 with each reader's top decoded key in place of w; 266 errors). Readers decode w as their top
colour on only 6-21% of items. With diffmeans at its locked setting, repair from a decoded key still reaches
0.40-0.49 gold (with the true w 0.515; differences -0.03 to -0.12), counting the 8-22% of items whose decoded key
is s2 itself, where no edit is made, as failures. The subtracted colour barely matters: the repair comes from adding
the true key, consistent with the redirect control.

**Self-report** (`test_selfreport_v1/`). After its own no-CoT answer, the model is asked (thinking off) for the colour
before the last stir. Share naming w / the true s_{h-1}:

| group | n | names w | names true s_{h-1} |
|---|---|---|---|
| correct h=2 (w = s1) | 445 | 0.989 | 0.989 |
| correct h=3 (w = s2) | 692 | 0.565 | 0.565 |
| consistent off-cycle h=2 | 57 | 0.053 [0.000, 0.123] | 0.825 [0.719, 0.912] |
| consistent off-cycle h=3 | 266 | 0.410 [0.353, 0.470] | 0.056 [0.030, 0.086] |

The report follows the shortest route to an answer, not the key the model used. At h=2 the true s1 is one forward
lookup from the start, and the model names it (0.83) although its answer used w. At h=3 the true s2 needs two
lookups, while w is one *inverse* lookup from the answer it just gave, and it names w (0.41). Prompt-only with
thinking on (100 seeded h=2/h=3 off-cycle error items, all finished, median 896 tokens): the true s_{h-1} in 100 of
100, w in 0. The self-report is not evidence about the internal key in either direction.

**NLA variants** (raw AUCs, no prior correction). *Checkpoint* (dev, `dev_nla_variants_v1/`): iter_000300 and
iter_000800 read the same (P1 h=2 0.62 vs 0.61, h=3 0.54 vs 0.56; P2 0.43 vs 0.41, n = 28). *Best-of-16* (test,
`test_nla_variants_v1/`; 16 samples per vector at the final cell, layer 42, ranked by the AR reconstruction's cosine
with the true activation): the top-ranked sample barely beats a random one (P2 0.542 [0.507, 0.579] vs 0.520
[0.485, 0.554]; P1 h=2 0.537 vs 0.525), and pooling all 16 gives 0.559. The AR cannot tell the samples apart: mean
cosine 0.904, the best sample only 0.012 above the item's mean. NLA explanations name several colours (at least one
control colour on 83-100% of items with the 4-sample reads), which caps their specificity.

## What this does and does not show

Kill rules: K1 did not fire (every reader read the on-path key on dev); K2 fired before the lock and the h=3 bank
was enlarged to 2,000; K3 and K4 did not fire (all methods passed the answer-swap gate; five reached the repair
threshold).

What it shows, on Qwen3.6-27B's single-table brew items without CoT:
- **The key of the last lookup is causally writable**, item-specifically, from layers 36-42 with diffmeans or NLA
  AR directions, and the model then looks the written key up (the answer is T(written colour)). This is the hard
  metric here (rates against a same-size random direction and an unrelated-colour direction, on held-out items
  with settings locked on dev).
- **A wrong key w is readable before the answer on off-cycle errors**, but clearly only with the supervised
  readers (probe 0.77, diffmeans 0.67 after the colour-prior correction); the token lenses give 0.57-0.63 and the
  text readers 0.54-0.57.
- **On those same errors, the true intermediate is readable too**, at least as often as w (probe: top on 59% of
  errors), and the errors track where the needed rule line is printed. The leading account is that the model
  computes the right intermediate and the last lookup retrieves the wrong line. This account is exploratory and
  was not tested by an intervention here.

What it does not show:
- **That the model held the right s2 and something overrode it, causally.** Repair works equally for an arbitrary
  key (redirect), and with keys decoded wrongly by readers, so it tests writability, not what was there.
- **That text readers can read wrong intermediates.** NLA, the Oracle lens and the AO fail P2. They read on-path
  keys on correct items weakly (P1 0.57-0.65 raw, 0.62-0.84 after the prior correction).
- **That lens directions are poor at writing keys in general.** They write the output colour at layers >= 40 and
  were only tried as unembedding-derived directions.
- **Anything beyond one model, one task family and one prompt format** (Gemma not run). The self-report cannot
  arbitrate: it follows whichever lookup is easiest.

## Deviations after the lock

The 19 pre-run deviations are in `manifest.json`. After the lock:
1. ptag AR dropped from the writes (pre-registered rule; three debug rounds, logged).
2. Exploratory analyses not in the plan, each in its own folder: the colour-prior check
   (`test_shuffle_null_v1/`; dev `dev_shuffle_null_v1/`, `_v2/`), the layer course (`test_layer_course_v1/`, raw
   only; `test_layer_course_v2/` adds the prior-corrected curves), what the write directions write
   (`test_write_breakdown_v1/`), the wrong-key distribution and rule-line positions (`test_key_distribution_v1/`;
   `_v2/` adds accuracy by line position), and the lock-order check in the verification package.
3. NLA iter_000300 vs iter_000800 ran on dev (as the code was written), best-of-16 on test.
4. Figures are in `figures/exp15_wrong_keys/<run>/` (AGENTS.md convention) rather than `results/`.
5. The Gemma-4-31B token-lens replication was not run (the plan's first cut when time is short): the GPU was busy
   with the test text reads and the secondary runs. Nothing here is shown to hold beyond Qwen3.6-27B.

## Three dumbest ways this could be wrong

1. **The P2 reads are colour preferences, not item-specific reads.** w is heavily skewed (gray, pink, green) and
   readers prefer some colours. *Checked* (exploratory): shuffled-item null and prior-corrected scores. The template
   lens's result is entirely prior; the probe's survives (0.773 corrected, null 0.558); the other lenses keep a
   weak signal (0.61-0.63).
2. **The readers read the answer's rule line, not a key.** A reader that sees a' and its line "w -> a'" scores w
   high. *Partly checked*: w beats T(a') (whose line also contains a'; probe 0.64 and J-lens 0.61 after the prior
   correction), the leak filter barely moves the AUCs, and in the primary window the lenses rank a' first on <= 20%
   of items. *Not excluded*: if the last lookup retrieves the line w -> a', reading w and reading the retrieved
   line are the same thing. So "w is read" does not mean "w was the query".
3. **The write successes are a generic effect of a large edit, or the edit writes the output directly.** *Checked*:
   a same-size random direction and an unrelated-colour direction succeed 0.02-0.07; success is spread over all
   written colours (diffmeans 0.28-0.84 per colour); with the answer swap, diffmeans and NLA AR produce T(a') more
   than a'. For repair, *checked and it bites*: an arbitrary key works as well as the true one (redirect), so
   repair is not read as evidence that s2 was held.

All metrics here are rates and AUCs computed by code; the text readers are scored by a regex colour matcher
(hand-audited, precision 0.99-1.00). No LLM-judge numbers are used.

## For the human to verify

The package is `results/exp15_wrong_keys/verification_v2/` (gitignored; on the private HF dataset), built by
`scripts/exp15_verify.py build --bank test --read-tag test_v1 --write-tag test_writes_v1 --out verification_v2`.
(`verification_v1/` is the same except that its bank check counted the one-shot example as a 601st h=2 item.)
1. `dumb_checks.json`: every check should read `"pass": true`. It re-derives the groups, keys and controls from the
   item text with independent code, recomputes every primary P2 AUC from the stored reads and compares with
   `test_v1/read_report.json` (to 1e-4), and re-derives every write trial's expected outcome colours from the item
   table. `lock_order` compares the lock times written in `manifest.json` (00:40:51Z) and `write_lock.json`
   (01:18:32Z) with the filesystem creation times of the test bank (00:41:25Z) and of the first test trial file
   (01:19:04Z). The lock times are self-reported by the code that wrote them, so this rules out a mix-up, not
   tampering.
2. `spot_check.md`, part 1 (8 off-cycle items): from the item's rule lines, follow the start colour h stirs by hand,
   then find the rule line ending in the model's answer; its left side is w. Check w, the controls, and that the
   listed reader scores rank w where the report says.
3. `spot_check.md`, part 2 (8 diffmeans key-swap trials): check by hand that "success" is T(s1') for the written
   s1', and that the recorded answer matches the outcome label.
4. `p2_scores_long.csv` (one row per item x reader x colour): recompute one reader's P2 key AUC in a spreadsheet.
5. Sanity of the headline: in `test_writes_v1/write_report.md`, the random null and other-item rows should be
   near 0.03, and the answer-swap row of `test_write_breakdown_v1/write_breakdown.md` should show T(a') > a' for
   diffmeans.

## Report

```
EXP exp15_wrong_keys @ fd05cc4 (code and main results; this write-up and the last secondaries in the next commit)
  | model Qwen3.6-27B (bf16, thinking off) | n = test bank 2,700 items (P2 323 off-cycle errors; P3 440 correct
  h=2 items; P4 266 off-cycle h=3 errors) | seeds = bank exp15_test_20261012, text rows exp15_text_rows, edit
  nulls per-item seeded, bootstrap 10k (seed 0)
Claim tested: on no-CoT off-cycle errors, readers find the wrong key w before the answer (P1/P2), and colour
  directions can set the key of the last lookup (P3) and repair the error (P4).
Headline: P3 diffmeans key swap success - random-direction success = +0.550 [+0.502, +0.598]
  Baseline: random direction 0.03   Control: unrelated-colour direction 0.03; NLA AR +0.416 [+0.370, +0.461];
  lenses +0.04 to +0.18.
  P2 probe w-vs-controls AUC 0.798 [0.761, 0.834]; colour-prior-corrected 0.773 [0.735, 0.810]; shuffled-item null
  0.558. P4 diffmeans repair - null +0.440 [+0.376, +0.504], but redirect to an arbitrary key 0.468.
Figures: figures/exp15_wrong_keys/test_v2/{fig_reads,fig_decomposition,fig_writes,fig_dev_grid}.png,
  figures/exp15_wrong_keys/test_layer_course_v2/fig_layer_course_prior_corrected.png
Three dumbest ways this could be wrong: 1) P2 = colour preferences: checked (template yes, probe no);
  2) readers read the answer's rule line: partly checked, not excluded; 3) writes = generic large edits or output
  writing: checked (random / other-item 0.02-0.07; T(a') > a'); repair = any key: checked, and it bites.
For the human to verify: results/exp15_wrong_keys/verification_v2/ (dumb_checks.json all pass; spot_check.md
  8 items + 8 trials by hand; p2_scores_long.csv to recompute one AUC).
Status: UNVERIFIED until a human adds it to VERIFIED.md
```
