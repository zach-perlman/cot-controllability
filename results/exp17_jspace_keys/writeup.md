# Is the brew lookup key in the J++ "global workspace"? On Qwen3.6-27B the key the lookup uses lies outside J++, and J++ carries the answer only after the lookup; on Gemma-4-31B-it the J++ swap fails the paper's own positive control

**Status: UNVERIFIED** until a human adds it to `VERIFIED.md`. Models: Qwen3.6-27B and Gemma-4-31B-it (bf16, no CoT,
"Answer:" prefilled; answer = argmax over the 10 colour tokens). Pre-registration: `manifest.json` (locked
2026-10-12T22:44:16Z by zach-perlman: design, Qwen's settings; 7 deviations listed) and
`manifest_addendum_patch_and_gemma.json` (locked 2026-10-13T01:16:14Z by zach-perlman, before any test patching or
Gemma test trial: Gemma's settings, the interchange-patching arms of both models, Gemma's swap test made
descriptive; 7 deviations). Every number is from the held-out test bank unless marked dev. No item text appears in
this file or in the figures.

## TL;DR

The paper finds that on two-hop factual prompts the J-space part of an intermediate's direction carries most of its
causal effect. We asked the same of the key of a no-CoT lookup (brew items: s1 = T(s0), answer T(s1)), with the
paper's clamped coordinate swap and, added before the test, dose-free interchange patching.

1. **Qwen: the key the lookup uses is outside J++.** Swapping the key s1 -> s1' with the difference-of-means key
   direction (layers 42-54) makes the model answer T(s1') on 0.360 of 930 test items. Its J++ part alone, at the same
   per-position size, gives 0.024 and its remainder 0.359 (T1 = -0.336 [-0.367, -0.303]; paper on factual prompts:
   0.61 vs 0.28). Clamping the J++ coordinates leaves the remainder's effect intact (T3 = +0.002 [-0.003, +0.009];
   paper: 0.28 -> 0.06). Patching the counterfactual prompt's 10 J++ colour coordinates at layers 42-54 moves 0.4% of
   answers (random 10-d subspace 0.1%; P1 = +0.003 [0.000, +0.007], p_holm 0.13); its key span moves 59.4%
   (P2 = +0.590 [+0.555, +0.627]).
2. **J++ works on Qwen, and carries the answer after the lookup.** The paper's factual control reproduces (0.644 vs
   null 0.000). A J++ answer swap at layers 48-60 succeeds on 0.968 (null 0.001), and patching the same J++ span
   there moves 67% of answers. At 42-54, J++ answer directions flip 41% of answers (dev, alpha 8, relative edit
   0.078), while J++ key directions at a similar size (0.087) flip 0.75% of keys and at twice it 1.1% (test).
3. **Caveat: at layers 24-48 J++ is inert for the answer too.** J++ answer swaps there succeed <= 0.031 even at
   alpha 8 (nulls <= 0.021), so the null key swap at the J++ band (0.023) cannot tell "key not in J++" from "J++
   edits do nothing here". The claim rests on layers 42-54.
4. **Gemma: a methods finding, not a replication.** Raw J++ vectors lie 80-95% outside the subspace Gemma's residual
   stream occupies, and Gemma-it cannot do the raw-text factual control (0/76). Projected onto that subspace, with
   the chat template, the J++ swap fails the positive control (gain 0.077 < 0.15, stop rule K1), so the Gemma swap
   test is descriptive. Gemma also collapses under matched-norm edits along any occupied-subspace direction (random
   ones included), so those comparisons compare broken models. Dose-free patching avoids this. Patching the key
   span moves 0.651 of answers, specifically (specificity +0.59). Patching the J++ span moves 0.122, but that is a
   scramble, not a key write (specificity +0.03; the pre-registered P1 passes on the success rate alone).
5. **Not shown:** anything about J++ on tasks other than this lookup, and what the non-J++ remainder is.

## Question and design

The global-workspace paper (anthropics/jacobian-lens) finds that on two-hop factual prompts, swapping the J-space
part of an intermediate's probe direction with a clamped coordinate swap moves the answer on 61% of items, vs 28% for
the remainder, and the remainder's effect falls to 6% when the J coordinates are clamped. Here the intermediate is
the key of a no-CoT brew lookup. A potion starts s0; one random derangement T of 10 colours is applied twice; the
model answers T(s1) with s1 = T(s0). We ask whether the key the lookup uses (s1) is carried by the J++ subspace.

**Swaps.** Per correct h=2 item, a clamped coordinate swap writes s1 -> s1' (s1' off the true path, exp15's targets)
at every J++ layer of a band and every position from the start colour to the last token. Success = the model now
answers T(s1'), the colour the new key looks up. Other outcomes: echo (answers s1' itself), start_swap (answers
T(T(s1')), as if s1' were the start colour), unchanged, other.

| arm | direction | size |
|---|---|---|
| dm_full | difference-of-means key direction (exp15's best key writer) | its own swap |
| dm_jpart / dm_rem | its J++ part (non-negative gradient pursuit over J++ atoms, k = 25) / the remainder | dm_full's per-position edit norms |
| dm_rem_clamp | dm_rem with the J++ colour pair's coordinates held at their clean values | dm_full's norms |
| jpp_matched | the J++ colour vectors | dm_full's norms (same band, layers, positions) |
| jpp_raw | the J++ colour vectors at their own selected band and alpha | own swap |
| x2, x4 | dm_jpart, jpp_matched and the null at 2x and 4x dm_full's norms | dose-response |
| jpp_answer_ctl | in-task control: J++ answer swap gold -> a' at a later band | own swap |
| other_item | dm_full swap of a colour pair unrelated to the item | dm_full's size |
| null_* | a random unit direction at the reference arm's recorded norms | matched |

**Interchange patching** (`exp17_patch.py`, locked in the addendum). The counterfactual prompt is the same prompt
with start colour s0' = T^-1(s1'), so its answer is T(s1'); one token differs. At every J++ layer of a band, the
item's residual coordinates in one subspace are set to the counterfactual's. The subspaces are: the whole residual;
the 10-d span of the 10 key directions; the span of their J++ parts; the span of their remainders; the 10 J++ colour
vectors; or a random 10-d subspace. No dose is chosen. Primary window: from the token after the start colour, so the
start colour's own position (which names s0') is never patched. Items count only if clean gives gold and the
counterfactual gives T(s1'). Specificity (secondary) is computed per item as 1[success] - (1[echo] + 1[start_swap]) / 2,
i.e. the target colour against two other wrong colours tied to s1'. A patch that only scrambles the answer scores ~0.

**Statistics.** Paired bootstrap over items (10,000 resamples), 95% CIs; Holm within each family (T2; P1-P3). All
metrics are rates computed by code from argmax answers; no LLM judge is involved.

**Locked settings** (chosen on dev by the manifest's rule). Qwen: J++ band 24-36, alpha 8; key direction band 42-54,
alpha 1; NLA activation reconstructor at layer 42, alpha 4; answer control 48-60, alpha 2. Gemma (occupied-subspace
lens): J++ band 34-46, alpha 4; key direction band 40-52, alpha 1; answer control 46-58, alpha 8 (fallback: no band
met the 0.9 rule; dev success 0.565 vs null 0.087).

## Qwen3.6-27B (confirmatory)

**Positive control passes.** On the paper's two-hop factual prompts, the clamped J++ intermediate swap at every
position reproduces the paper's effect: best gain 0.644 (layers 24-36, alpha 4) vs null 0.000, n = 59 (stop rule K1
needs >= 0.15). With the answer's direction removed from the edit it keeps 0.31-0.53 (exploratory).

**Locked swap test (930 correct h=2 test items).**

| arm | success | minus its null |
|---|---|---|
| dm_full (key direction, 42-54) | 0.360 | +0.357 [+0.326, +0.388] |
| dm_rem (its remainder, dm_full's norms) | 0.359 | +0.356 [+0.325, +0.387] |
| dm_rem_clamp (remainder, J++ colour coordinates held) | 0.357 | +0.354 [+0.323, +0.385] |
| dm_jpart (its J++ part, dm_full's norms) | 0.024 | +0.020 [+0.011, +0.031] |
| jpp_matched (J++ colour vectors, dm_full's norms, 42-54) | 0.0075 | +0.004 [-0.001, +0.011], p_holm 0.10 |
| jpp_raw (J++ colour vectors, own band 24-36, alpha 8) | 0.023 | +0.016 [+0.005, +0.027] |
| ar_full / ar_rem / ar_jpart (NLA reconstructor, layer 42) | 0.170 / 0.167 / 0.036 | +0.166 / +0.162 / +0.031 |
| jpp_answer_ctl (J++ answer swap gold -> a', 48-60) | 0.968 | +0.967 [+0.954, +0.979] |
| other_item (unrelated colour pair, dm_full's size) | 0.023 | |
| null_dm (random direction, dm_full's norms) | 0.003 | |

- T1 (J++ part minus remainder): -0.336 [-0.367, -0.303]. The paper's ordering is reversed: J++ part 0.024 vs
  remainder 0.359 here, 0.61 vs 0.28 there. The J++ part's effect is small but above the null (+0.020, p_holm 0.001).
- T3 (clamping the J++ coordinates): +0.002 [-0.003, +0.009]. The remainder's effect does not route through J++.
- T5 (key direction minus J++ vectors at the same norms): +0.353 [+0.322, +0.384].
- Dose-response (secondary): at 2x and 4x the J++ part reaches 0.030 and 0.050 and the J++ vectors 0.0075 and
  0.011, while the random direction reaches 0.010 and 0.028. At 4x the J++ part and the random direction change 0.43
  and 0.42 of answers to something other than T(s1'). Larger edits break the model; they do not write the key.
- Every position edited (secondary, rule table included; 921 items, 9 near-tie items flip under batched numerics):
  key direction 0.166, J++ vectors 0.037, null 0.004.

**Interchange patching (test; 710 of 930 items with clean = gold and counterfactual = T(s1')).** Window from the
token after the start colour.

| subspace patched | 24-36 | 42-54 (primary) | 48-60 (control band) |
|---|---|---|---|
| whole residual | 0.451 | 0.989 | 0.997 |
| key directions (10-d) | 0.079 | 0.594 | |
| their J++ parts (10-d) | 0.001 | 0.020 | |
| their remainders (10-d) | 0.073 | 0.596 | |
| J++ colour vectors (10-d) | 0.000 | 0.004 | 0.673 |
| random 10-d subspace | 0.000 | 0.001 | 0.001 |

- P1 (J++ span minus random span, 42-54): +0.003 [0.000, +0.007], p_holm 0.13.
- P2 (key span minus J++ span): +0.590 [+0.555, +0.627], p_holm 0.0003.
- P3 (remainder span minus J++-part span): +0.576 [+0.539, +0.613], p_holm 0.0003.
- Specificity: key span +0.491 [+0.444, +0.538], J++ span +0.001; patched items move to T(s1'), not to noise.
  Patching from the start colour itself gives the same picture (key span 0.604, J++ span 0.004 at 42-54).
- Dev (153 items, in-sample for the key directions) gave the same numbers (key span 0.575, J++ span 0.013).
- The same J++ span patched at 48-60 moves 67% of answers (specificity +0.65). The J++ lens reads the gold answer
  at > 50% top-1 from layer 51, so what the J++ span carries there is the answer, not the key.

**Where the key is readable (dev, final position).** The key direction's cross-fitted readout ranks s1 first on 0.675
[0.61, 0.74] of items (peak layer 56); the J++ lens peaks at 0.40 [0.34, 0.47] (layer 41) and falls to ~0 once the
answer appears. J++ atoms explain the key direction no better than random atoms do: with k = 25 / 100 / 400 atoms,
0.05 / 0.08 / 0.09 of it at layer 42 vs 0.04 / 0.07 / 0.08 for random atoms (the NLA reconstructor's direction:
0.15 / 0.20 / 0.22). So J++ reads the key weakly, but the copy the lookup uses is outside it.

## Gemma-4-31B-it (methods finding; swap test descriptive)

**The raw J++ lens cannot be used for swaps on Gemma.** At layers 28-58 its J++ colour vectors lie 80-95% outside the
subspace the residual stream occupies. That subspace is the span of the top eigenvectors holding 99.99% of WikiText
residual energy, rank 3,100-4,900 of 5,376; a J++ vector keeps 5-20% of its norm inside it. Raw J++ swaps therefore
move almost nothing. Gemma-it also cannot do the paper's raw-text factual control (0 of 76 prompts answered).
Pre-registered fix (addendum): project the J++ vectors and the random nulls onto the occupied subspace, and run the
factual control inside the chat template.

**The positive control fails, so the swap test is descriptive.** With the chat template, 65 of 76 factual prompts
are answered; the best clamped J++ intermediate swap gains 0.077 (layers 22-34, alpha 4; null 0.000), below the K1
bar of 0.15. The swap test was run anyway, labelled descriptive (approved).

**Gemma breaks under small edits along occupied-subspace directions.** At the key direction's per-position edit
norms, every matched-norm arm, the random null included, leaves colour mass <= 0.051 (the model stops answering
with a colour). Only the key direction itself (0.9997) and the same-size swap of an unrelated colour pair (0.9998)
leave it intact. Descriptive swap test (1,094 items):

| arm | success | echo | start_swap | colour mass |
|---|---|---|---|---|
| dm_full (key direction, 40-52) | 0.314 | 0.080 | 0.003 | 0.9997 |
| other_item (unrelated colour pair, same size) | 0.003 | 0.017 | 0.001 | 0.9998 |
| dm_rem | 0.148 | 0.112 | 0.071 | 0.051 |
| dm_jpart | 0.090 | 0.110 | 0.099 | 0.000 |
| jpp_matched | 0.091 | 0.107 | 0.095 | 0.000 |
| jpp_raw (own band 34-46, alpha 4) | 0.098 | 0.093 | 0.100 | 0.0003 |
| null_dm (random, dm_full's norms) | 0.078 | 0.091 | 0.096 | 0.0009 |
| null_jpp (random, jpp_raw's norms) | 0.001 | 0.006 | 0.001 | 0.656 |
| jpp_answer_ctl (46-58, alpha 8) | 0.554 | | | 0.000 |

The J++ arms' success rates equal their echo and start_swap rates (0.09-0.11, about 1 in 9): they scramble the
answer. Their tests (T1 -0.059 [-0.085, -0.033], T5 +0.222 [+0.189, +0.256]) compare broken models and are not
interpreted. The one clean contrast is dm_full vs other_item, 0.314 vs 0.003: as on Qwen, the key direction writes
the key item-specifically.

**Interchange patching (test, 1,013 of 1,094 items; confirmatory per the addendum).** Patching sets coordinates to
values the model produces itself, so it does not break Gemma (the random 10-d patch leaves 98.8% of answers
unchanged). Window from the token after the start colour; success (specificity).

| subspace patched | 34-46 | 40-52 (primary) | 46-58 (control band) |
|---|---|---|---|
| whole residual | 0.997 | 0.999 | 1.000 |
| key directions (10-d) | 0.674 (+0.615) | 0.651 (+0.592) | |
| their J++ parts (10-d) | 0.093 (+0.007) | 0.092 (+0.003) | |
| their remainders (10-d) | 0.648 (+0.605) | 0.348 (+0.270) | |
| J++ colour vectors (10-d) | 0.098 (-0.006) | 0.122 (+0.029) | 0.227 (+0.137) |
| random 10-d subspace | 0.000 | 0.001 | 0.002 |

- P1 (J++ span minus random span): +0.121 [+0.102, +0.142], p_holm 0.0003. It passes on the locked success rate, but
  the pre-registered specificity shows a scramble. After the J++ patch the model answers T(s1') 12.2%, s1' itself
  10.5%, another colour 57.9%, and keeps its answer 11.2% (specificity +0.029 [+0.004, +0.054]).
- P2 (key span minus J++ span): +0.529 [+0.493, +0.565], p_holm 0.0003.
- P3 (remainder span minus J++-part span): +0.257 [+0.223, +0.290], p_holm 0.0003. The J++ parts carry nothing
  specific at either band. The remainders carry all of the key span's effect at 34-46 and about half at 40-52.
- At the control band 46-58 the J++ span moves 23% (specificity +0.14), much less than on Qwen (67%).

**Where the key is readable (dev).** The key direction's readout peaks at 0.72 [0.66, 0.77] (layer 39); the J++ lens
peaks at 0.23 [0.18, 0.29] (layer 36) and falls to ~0 once the answer appears (J++ reads the gold answer at > 50%
from layer 42).

## Figures

`figures/exp17_jspace_keys/final_v1/`:
- `fig_layers.png`: per model, what is readable at the final position by layer (key: own readout vs J++; answer:
  J++), and what changes the output when swapped by band (key direction, J++ key, J++ answer, random).
- `fig_test.png`: the locked arms on test, dose-response, and the rate of answers changed to something other than
  T(s1').
- `fig_test_patch.png`: interchange patching on test, both windows, both models (`fig_patch.png`: the same on dev).
- `fig_controls.png`: the factual positive control (Qwen raw text; Gemma chat template).
- `fig_lens.png`: how much of each J++ colour vector lies in the occupied subspace, by layer.
- `fig_preview.png`: the locked arms on dev at the dev selection (before the lock).

## Three dumbest ways this could be wrong

1. **The J++ vectors are broken, or the edited band gives J++ no causal handle at all.** *Checked, and it bites at
   24-48.* On Qwen the paper's factual control reproduces (0.644), the J++ answer swap at 48-60 succeeds (0.968), and
   the J++ span patched at 48-60 moves 67% of answers. But at 24-48 J++ answer swaps are as inert as key swaps
   (<= 0.031), so the J++-band null result is uninformative. The claim is    restricted to 42-54, where J++ moves the
   answer (0.41 at alpha 8) and not the key (<= 0.011 at a similar or larger size; patching: 0.004).
2. **The key direction wins because it was selected on dev, or J++ loses because its edit is too small.**
   *Checked.* Test items were held out (manifest locked 22:44:16, test bank written 22:44:29, first test trial
   22:46:06). J++ arms were matched to the key direction's per-position norms and run at 2x and 4x. Patching uses
   no dose at all and replicated from dev to test (key span 0.575 -> 0.594; J++ span 0.013 -> 0.004).
3. **"Success" counts answers that a scrambled model lands on by chance.** *Checked, and it bites on Gemma.*
   Specificity (pre-registered secondary) is +0.49 to +0.62 for the key-span patches at the primary bands of both
   models. Gemma's J++-span
   P1 "pass" (+0.121) has specificity +0.03, and every Gemma matched-norm swap arm, J++ and random alike, succeeds at
   the scramble rate (~0.1) with colour mass ~0. Qwen's J++ arms mostly leave the answer alone (unchanged
   0.83-0.97).

## For the human to verify

No verification package was built for exp17 (exp15 has one). The minimum checks:
1. **Lock order.** `manifest.json` locked 2026-10-12T22:44:16Z; `test_bank.json` created 22:44:29Z; first test swap
   trial file `cache/exp17/qwen/trials_test.jsonl` created 22:46:06Z. Addendum locked 2026-10-13T01:16:14Z; first test
   patching files 01:17:00Z (Qwen) and 01:42:02Z (Gemma); Gemma swap test 02:16:35Z (filesystem birth times,
   `stat -c %w`). Lock times are self-reported by the code that wrote them.
2. **Recompute the patching headline in plain Python** from the trial file, without the report code. This should
   print 710 items and 0.004 / 0.001 / 0.594:
   ```python
   import json
   rows = [json.loads(line) for line in open("cache/exp17/qwen/trials_test_patch_after.jsonl")]
   ok = {r["item_id"] for r in rows if r["arm"] == "clean" and r["outcome"] == "unchanged"}
   ok &= {r["item_id"] for r in rows if r["arm"] == "cf_clean" and r["outcome"] == "success"}
   for arm in ("patch_jpp|b42_54", "patch_random|b42_54", "patch_dm|b42_54"):
       rs = [r for r in rows if r["arm"] == arm and r["item_id"] in ok]
       print(arm, len(rs), round(sum(r["outcome"] == "success" for r in rs) / len(rs), 3))
   ```
3. **Unit tests of the edit hooks**: `python scripts/exp17_swaps.py selftest` (8 tests: swap algebra, clamping,
   norm matching, pursuit, the hook's window and band, occupied-subspace projection, interchange patching).
4. **Spot-check outcome labels by hand** on a few test items. From the item's rule table, compute T(s1') for the
   stored swap target (`outcomes.key.echo` = s1'). Check that the counterfactual's start colour is T^-1(s1') (asserted
   in `exp17_patch.counterfactual_targets`), and that a trial's recorded answer matches its outcome label.
5. **Headline sanity** in `analysis/final_v1/report.txt`: Qwen random rows near 0.00-0.01; Gemma's J++ swap rows have
   success ~ echo ~ start_swap.

## Deviations

Pre-run deviations: 7 in `manifest.json`, 7 in the addendum. The addendum's include: the occupied-subspace lens and
chat-template control for Gemma; K1 triggered so Gemma's swap test is descriptive; the answer-control fallback;
patching added and run on dev before the lock, with dev key-span patching in-sample; and the `exp17_late` early
bands, which showed J++ is inert at Qwen 24-48. After the test runs:
1. The all-positions swap arms (a pre-registered secondary) were missing from the report code; added as
   `test_all_positions` after the test. Nothing was selected on it.
2. Cosmetic figure fixes (legend placement, a truncated title).
3. The raw-lens Gemma dev runs (`results/exp17_jspace_keys/gemma/`) are superseded by the occupied-subspace variant;
   raw Gemma has no test trials.

## Report

```
EXP exp17_jspace_keys @ fe7d751 (code and results; this write-up in the next commit) | models Qwen3.6-27B,
  Gemma-4-31B-it (bf16) | n = 930 (Qwen) / 1,094 (Gemma) correct h=2 test items; patching 710 / 1,013 |
  seeds = test bank exp17_test_20261012, random directions sha256(item, arm, layer), bootstrap 10k
Claim tested: the key the brew lookup uses (s1) is carried by the J++ subspace (the paper's global-workspace claim
  for two-hop intermediates).
Headline (Qwen): key swap, J++ part - remainder = -0.336 [-0.367, -0.303] (0.024 vs 0.359; paper 0.61 vs 0.28).
  Patching at 42-54: J++ span - random span = +0.003 [0.000, +0.007] (p_holm 0.13); key span - J++ span = +0.590
  [+0.555, +0.627].
  Baseline: random direction 0.003; random 10-d patch 0.001.
  Control: factual positive control 0.644 (null 0); J++ answer swap at 48-60 0.968; J++ span patch at 48-60 0.673.
  Gemma: positive control fails (0.077 < 0.15), swap test descriptive (matched-norm arms break the model); patching
  key span 0.651 (specificity +0.59) vs J++ span 0.122 (specificity +0.03, a scramble).
Figures: figures/exp17_jspace_keys/final_v1/{fig_test,fig_test_patch,fig_layers,fig_controls,fig_lens}.png
Three dumbest ways this could be wrong: 1) J++ has no causal handle in the band: checked, bites at 24-48, claim
  restricted to 42-54; 2) dev selection or dose: checked (held-out test, matched + 2x/4x, dose-free patching
  replicates); 3) successes are scrambles: checked, bites on Gemma (J++ P1 specificity +0.03).
For the human to verify: lock order from file times; the plain-Python recompute above (710, 0.004 / 0.001 /
  0.594); exp17_swaps.py selftest; hand spot-check of outcome labels.
Status: UNVERIFIED until a human adds it to VERIFIED.md
```
