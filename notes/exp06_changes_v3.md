# exp06 change list, v3 (after exp06a round 1, thinking on and off)

Draft for the human's choice; not a result. v1: 2026-10-01, before exp06a. v2 (`exp06_changes_v2.md`): after round 1
thinking on. v3 adds round 1's thinking-off rows for all 3 models and, in its 2026-10-01 late update, round 2's
results (point 9; items 1, 10, 21, 26, 29 revised). Same numbering (1-32, P1-P13); each item is **kept**,
**revised**, **done in exp06a**, **evidence against** or **dropped**; new items 33-34.

Sources (all UNVERIFIED):
- `results/exp06a_prompt_screen/analysis/round2_complete_report` (all rows, rounds 1-2) and
  `figures/exp06a_prompt_screen/round2_complete` (R2-1 to R2-4).
- `log/exp06a/round2_tags_check.txt`: round 2's thinking-off gains among rows with tag content only (item 33).
- `log/exp06a/round2_copying_check.txt`: few-shot copying in round 2.
- `results/exp06a_prompt_screen/analysis/round1_on_off_figures`: thinking on on 100 questions, thinking off on the
  fixed 51-question half; S(t) = share with no violation in the first t tokens, 5-rule mean unless stated.
- `log/exp06a/clean_through_200_check.txt`: the length-robust rate, i.e. the share of ALL texts that reach 200 tokens
  with no violation (a short clean text counts as a failure), as gains over the in-environment baseline rerun on the
  51 questions. Below, "clean-200 gain on / off" means this gain with thinking on / off, for Qwen3.8 / Gemma-31B /
  Qwen3-32B.

## What exp06a round 1 changed

1. **Revised by round 2: single prompt pieces are weak, combinations are not.** In round 1 the best single harness's
   S(1000) mean over the 3 models is 12.0 (failure-targeted guide), equal to stacked (12.0), next to exp04's compliant
   prefill + stacked at 31-41. In round 2 the best combination per model reaches 40-71 (point 9), above prefill +
   stacked. The prompt is therefore a main axis for exp06 too, not a minor one. Caveat: chosen on these items.
2. **Harness effects depend on the model.** On Qwen3.8 and Gemma every single harness loses to stacked (Holm p <=
   0.012). On Qwen3-32B, where stacked barely helps, start-with, few-shot and the failure guide reach S(1000) 18.5-21.0.
3. **Means over rules hide most of the signal.** Qwen3-32B's gains sit in meow and lowercase. Uppercase and
   end-with-'safe' stay near 0 under every harness, thinking on, on every model. Gemma's non-zero cells are almost all
   word suppression.
4. **Gemma's thinking ignores instructions about its form.** It starts with the requested sentence 0% of the time
   (Qwen3.8 14%, Qwen3-32B 60%), yet with thinking off it does so 100% of the time.
5. **Revised: the CoT-vs-output gap is robust for the baseline. Harness gains split three ways.**
   - Baseline, clean-200 rate, thinking on vs off: 5 vs 31, 16 vs 47, 12 vs 35. The gap survives the length-robust
     metric.
   - **General instruction following** (gain larger with thinking off): few-shot (on +13 / +8 / +20, off +32 / +20 /
     +32) and start-with (on +4 / -2 / +16, off +15 / +12 / +22).
   - **CoT-specific** (gain with thinking on at least as large as off): the failure guide (on +11 / 0 / +20, off +15 /
     +3 / +9) and stacked on Gemma (on +24, off 0).
   - **Output only** (helps output, not CoT): warm-up (Gemma off +46, on -7). This shows the two channels are steered
     separately.
   - v2's "every harness gains more off than on" was Qwen3.8 only, and was partly a tag artifact (item 33).
6. **Checks that passed:**
   - In-environment reruns match the reused exp03/exp04 rows on all 3 models: thinking on within 1.7 points, thinking
     off within 3.1 points at S(200); every CI includes 0.
   - Qwen3-32B's thinking-on gains are not short traces: the share reaching the 1200-token stop clean is 2.6%
     (stacked) vs 8.6-10.8%.
7. **Nulls with thinking on:** warm-up, style guide, monitor + stealth and meta-allowed. Meta-allowed equals
   baseline, so the no-mention clause is not what fails.
8. **New: thinking-off survival scores flatter short outputs.** KM censors a clean text at its end, and P1 scores a
   short text as compliant. The style guide's thinking-off S(200) of 78-99 is brevity: its tag content has a median of
   61-102 tokens, and only 4-9% of its rows reach 200 clean tokens (baseline 31-47%).
9. **New: round 2's results.** Thinking-on S(1000), 100 questions (stacked rerun 11.7 / 18.9 / 6.8; exp04 prefill +
   stacked 30.8 / 41.1 / 34.8), for Qwen3.8 / Gemma / Qwen3-32B:
   - stacked + few-shot + guide + start-with ("stacked_all"): 39.6 / 71.4 / 42.3.
   - stacked + few-shot + guide: 36.8 / 53.5 / 40.9.
   - stacked + start-with: 16.9 / 14.0 / 51.3 (Gemma: 69% empty traces).
   - stacked + few-shot 24.6 / 19.4 / 29.3; stacked + guide 17.0 / 28.9 / 11.3; few-shot + guide 21.0 / 11.0 / 32.1.
   - Interactions, as stacked_all's gain over the stacked rerun vs the sum of the three single additions to stacked:
     - Gemma: strongly superadditive (+53 vs +6).
     - Qwen3.8: about additive (+28 vs +23).
     - Qwen3-32B: subadditive (+36 vs +71), because start-with alone carries it.
   - Not long-trace or empty-trace artifacts: Gemma stacked_all is 95% clean through 200 tokens, median 825 tokens,
     0.2% empty, 99.8% start with the requested sentence.
   - Little copying: at most 0.4% of 8-grams appear in the examples; at least 98.6% of 8-grams are distinct.
   - **CoT specificity** (clean-200 gain, thinking on minus off, 51 questions) depends on the model:
     - Gemma: CoT-specific on every check. stacked_all +44 points [37, 50]; +3-shot + guide +48 [38, 57]; 92% vs 58%
       of headroom.
     - Qwen3.8: depends on scaling. Raw points: +10 [-1, 20], rising to +23 when thinking off is scored on tagged rows
       only (item 33). As a share of headroom: about equal (61% vs 65-67%).
     - Qwen3-32B: general instruction following. On minus off is +4 to +8 points (CIs include 0), and below 0 as a
       share of headroom (stacked_all 59% vs 74%). The failure guide alone is its only CoT-specific piece (+12).

## A. Changes to exp04/exp05 for exp06

**Models** (human decision, 2026-10-01: exp06 runs exp06a's three models, Qwen3.8-27B-FP8, Gemma-4-31B-FP8 and
Qwen3-32B, and no others for now)

1. **Drop Gemma-4-12B and Qwen3.5-9B. Kept** (covered by the decision).
2. **Add GPT-OSS-120B. Dropped** (human decision). Cost: no fourth model family to test whether a harness
   generalizes, and no direct comparison with Jozdien's GPT-OSS numbers.
3. **GLM-4.7-Flash and Qwen3.6-35B-A3B. Dropped** (covered by the decision).
4. **Qwen3.6-27B. Deferred, not dropped** (human decision: not in exp06 now, may be added later). Its exp05 recovery
   from a non-compliant start (29% under stacked) is the strongest non-copying evidence, so it is the first candidate
   to add, with the caps in 5. Until then item 22 relies on the three models.

**Compute**

5. **Full-trace cells: cap at 8k tokens and/or 10% of cells. Kept.**
6. **Stop cells at about 1200 graded tokens. Done in exp06a; adopt.**
7. **Item count. Revised.**
   - 100 questions for every thinking-on arm. Paired standard errors are 1.4-1.7 points at 100 and 2.1-2.2 at 50.
   - Thinking off on a fixed stratified half. At 51 questions the on-minus-off contrasts have CIs of about +-8-10
     points, enough for the 15-40-point splits in point 5 but not for 5-point ones. If CoT specificity becomes a
     confirmatory claim, use 100.

**Design**

8. **Opening lengths: keep 1 sentence and >= 150 tokens. Kept.**
9. **No-rule twin for every elicitation arm. Kept, with v2's fix.** Report twin contrasts on the opener rules only, or
   use a word-suppression twin that keeps the ban but drops the demonstration.
10. **On-policy few-shot. Done in exp06a.** Alone it is mostly general instruction following (point 5), but it is
    in every top round-2 combination (point 9). It enters exp06 through the carried arms. Out-of-distribution
    few-shot is still untested, and exp06 needs few-shot examples drawn without its own test items.
11. **Prefill that states the rule vs demonstrates it. Kept.**
12. **Periodic reinsertion. Kept.**
13. **Resample at the violation. Kept.**
14. **Lenient grader or programmatic held-out modes. Kept, stronger.** Uppercase and end-with-'safe' are near 0
    thinking on, but with thinking off some arms reach high S(1000) (Qwen3-32B stacked: uppercase 70, end-'safe' 78).
    They are followable in principle, just not in the CoT.

**Cuts**

15. **Drop the CoT-necessity probe. Kept.**
16. **Thinking-off ceiling. Revised again: keep it, it is the cleanest contrast, but score it length-robustly.**
    - The gap survives the clean-200 rate (point 5).
    - It is expensive: 3,317 tokens per thinking-off row vs about 500 thinking on (84% of round 1's tokens). Run it on
      the fixed half, for the arms exp06 carries.
    - Primary thinking-off metric: the clean-200 rate (item 27) next to S(200), not P1 or S(1000).
    - Stopping thinking-off rows at the first violation inside the tags would cut cost, but needs tag-aware streaming
      grading.
17. **Drop the effort check. Revised:** with no GPT-OSS there is nowhere to move it.
18. **Drop correctness figures; keep accuracy. Kept.**

**Diagnostics**

19. **Seam check. Kept.**

## B. Larger additions

20. **Engineered-necessity task. Kept (own mini-project).**
21. **Best prompt crossed with prefill. Kept, still the top addition.** Each model's carried arm (item 26) alone
    beats exp04's prefill + stacked (point 9), so the open question is whether a prefill adds anything on top. Cross
    {stacked, carried arm} with {no prefill, compliant prefill}. "Asked to start compliant" (start-with) vs "forced
    to" (prefill) is in that cross.
22. **Recovery as a primary outcome. Kept.**
23. **Cross-model openings. Kept.** Point 4 sharpens it: Gemma ignores requested openings in the trace but follows them
    in the output, and responds to prefilled ones (exp04).
24. **Monitor outcome. Kept, only with 20.**
25. **Hazard per 100 tokens. Kept.**

## New items from exp06a

26. **Prompt arm per model, chosen by a pre-registered rule. Revised.** Selection rule:
    - Choose on the thinking-on gain over stacked.
    - Report the arm's CoT specificity, i.e. its thinking-on gain minus its thinking-off gain on the clean-200 rate.
    - Flag an arm whose gain is mostly general instruction following.
    - Carry stacked plus the best round-2 arm per model.
    - Optimistic, since it is selected on these items; exp06 confirms on new items.
    - **Applied to round 2 (needs human sign-off):**
      - Qwen3.8 and Gemma: stacked_all. On Qwen3.8 it ties stacked + few-shot + guide (+26.4 vs +23.7 over stacked).
        Flag: specificity depends on the scaling on Qwen3.8 (point 9).
      - Qwen3-32B: stacked + start-with (+46.1 vs stacked_all +37.2, CIs [41.7, 50.4] vs [32.5, 42.0]). Flag: mostly
        general instruction following.
      - Alternative: stacked_all for all 3 (one prompt, simpler to report), costing about 9 points on Qwen3-32B.
27. **Length-robust co-primary: share of texts reaching t tokens with no violation. Done in the exp06a analysis**
    (`clean_short`, t = 200; H2's bottom row). It is required for thinking off (point 8) and useful for thinking on.
    For exp06: report it at 200 and 1000 next to S(t).
28. **An "instruction executed" check for every harness. Kept.** Point 4's Gemma result (0% in the trace, 100% in the
    output) is exactly what it catches.
29. **Empty traces as a reported outcome. Kept, stronger.** Gemma leaves the trace empty in 32% of stacked rows and
    69% of stacked + start-with rows, but 0-0.2% under the few-shot combinations.
30. **Report word suppression separately from the opener rules. Kept.**
31. **Drop warm-up, the style guide, monitor + stealth and meta-allowed. Kept, stronger.** The style guide's thinking-off
    score is brevity (point 8). Warm-up helps the output and not the CoT (point 5).
32. **Generate the frozen harness doc from the config. Kept.** The 6-vs-8 few-shot candidate mismatch is recorded in
    `deviations_fewshot_candidates.json`.
33. **New: report thinking-off format failures separately, and give CoT specificity a tags-only sensitivity.**
    - A thinking-off row with no `<output_reasoning>` content is a violation at token 0.
    - Qwen3.8 leaves out the tags in 32% of baseline rows and 2-25% under harnesses. Gemma almost always uses them
      (at most 0.4% missing); Qwen3-32B leaves them out in 3-23%.
    - So on Qwen3.8 about half of each harness's thinking-off gain is "used the tags". Among tagged rows only:
      few-shot +18 (vs +32), stacked +16 (vs +25), failure guide +9 (vs +15).
    - The point-5 classification holds either way, but the no-tag share must be in every thinking-off table.
34. **New: measure CoT specificity on the arm actually carried, not on its parts.** Pieces split three ways (point 5),
    so a combination's specificity cannot be inferred. Round 2's thinking-off part does this for its 6 arms. Any exp06
    arm without a thinking-off twin is reported as "specificity unknown".

## C. Prompt-only ideas: status after round 1

- **P1. Monitoring awareness / stealth. Tested (V6): null** on, and small off. Drop.
- **P2. Start-With. Tested (V1):** thinking-on gain only on Qwen3-32B; larger gains off; Gemma executes it only in the
  output. Kept through round 2 (stacked + start-with; stacked_all vs stacked + few-shot + guide gives its marginal
  effect).
- **P3. Few-shot. On-policy tested (V3):** helps all 3 vs baseline, about half copying on Qwen3-32B, mostly general
  instruction following. Out-of-distribution untested.
- **P4. Self-generated style guide. Tested (V4): null** on; its thinking-off score is a brevity artifact. Drop.
- **P5. Warm-up. Tested (V2): null on, helps off.** Drop.
- **P6. Channel inversion / short reasoning. Untested.** Item 27's metric is required for it; point 8 shows how
  shortness fakes compliance.
- **P7. Identity framing. Untested.**
- **P8. Rule as a procedure with edge cases. Tested as V5 (failure guide):** best single harness by S(1000) mean and
  the most CoT-specific. Kept, in 4 of round 2's 6 arms.
- **P9. Low effort / brevity. Untested.** Same caveat as P6.
- **P10. Rule placement (system / user / both). Untested.** More relevant: stacked's thinking-on gain on Gemma is
  CoT-specific, and no user-turn addition matches it on Qwen3.8 or Gemma.
- **P11. "Violations will be resampled". Untested.**
- **P12. LLM-driven prompt iteration. Partly done** (rounds 1-2). Further iteration needs new items.
- **P13. Loophole guard. Kept;** items 29 and 33 cover empty traces and missing tags.

## Round 2 (done 2026-10-02 02:12 UTC; deviations_round2_decision.json; results in point 9)

- **The original 5 arms**, thinking on, 100 questions: stacked + failure guide, stacked + few-shot, stacked +
  start-with, stacked + few-shot + guide + start-with, few-shot + failure guide.
- **Added arm:** stacked + few-shot + failure guide (no start-with), thinking on, 100 questions.
- **Thinking off for all 6 arms**, on the 51 questions.

## D. Default selection for exp06 (v3)

Models: Qwen3.8-27B-FP8, Gemma-4-31B-FP8, Qwen3-32B (items 1-4). Items: 5, 6, 7, 9 (with the fix), 11, 13, 14, 15,
16 (revised: length-robust), 17 (drop), 18, 19, 21, 22, 25, 26 (revised rule), 27, 28, 29, 30, 31, 33, 34.

Carried prompt arms (item 26, pending sign-off): stacked for all 3, plus stacked_all (Qwen3.8, Gemma) and stacked +
start-with (Qwen3-32B).

Changes from v2: GPT-OSS (2) leaves and Qwen3.6-27B (4) is deferred, by human decision; 33 and 34 join; 16, 17, 26 and 27
revised. Round 2 revised 1, 10, 21, 26 and 29.
