# exp06 change list, v3 (after exp06a round 1, thinking on and off)

Draft for the human's choice; not a result. v1: 2026-10-01, before exp06a. v2 (`exp06_changes_v2.md`): after round 1
thinking on. v3 adds round 1's thinking-off rows for all 3 models and the round-2 plan. Same numbering (1-32, P1-P13);
each item is **kept**, **revised**, **done in exp06a**, **evidence against** or **dropped**; new items 33-34.

Sources (all UNVERIFIED):
- `results/exp06a_prompt_screen/analysis/round1_on_off_figures`: thinking on on 100 questions, thinking off on the
  fixed 51-question half; S(t) = share with no violation in the first t tokens, 5-rule mean unless stated.
- `log/exp06a/clean_through_200_check.txt`: the length-robust rate, i.e. the share of ALL texts that reach 200 tokens
  with no violation (a short clean text counts as a failure), as gains over the in-environment baseline rerun on the
  51 questions. Below, "clean-200 gain on / off" means this gain with thinking on / off, for Qwen3.8 / Gemma-31B /
  Qwen3-32B.

## What exp06a round 1 changed

1. **Prompt-only elicitation is weak next to text in the trace.** The best harness's S(1000) mean over the 3 models
   is 12.0 (failure-targeted guide), equal to stacked (12.0). exp04's compliant prefill + stacked is 31-41. Prompt
   arms are a minor axis for exp06; prefill and its interaction with the prompt are the main one.
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

## A. Changes to exp04/exp05 for exp06

**Models** (human decision, 2026-10-01: exp06 runs exp06a's three models, Qwen3.8-27B-FP8, Gemma-4-31B-FP8 and
Qwen3-32B, and no others for now)

1. **Drop Gemma-4-12B and Qwen3.5-9B. Kept** (covered by the decision).
2. **Add GPT-OSS-120B. Dropped** (human decision). Cost: no fourth model family to test whether a harness
   generalizes, and no direct comparison with Jozdien's GPT-OSS numbers.
3. **GLM-4.7-Flash and Qwen3.6-35B-A3B. Dropped** (covered by the decision).
4. **Qwen3.6-27B. Dropped** (covered by the decision). Cost: its exp05 recovery from a non-compliant start (29% under
   stacked) was the strongest non-copying evidence; item 22 now relies on the three models.

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
10. **On-policy few-shot. Done in exp06a.** It beats baseline on all 3 models with thinking on, but loses to stacked on
    Qwen3.8 and Gemma, and is the least CoT-specific of the useful harnesses (point 5). It enters exp06 only through
    round 2. Out-of-distribution few-shot is still untested.
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
21. **Best prompt crossed with prefill. Kept, the top addition.** Round 2 now has the prompt-only counterpart
    (stacked + start-with) with thinking on and off, so exp06 can compare "asked to start compliant" and "forced to"
    on both channels.
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
27. **Length-robust co-primary: share of texts reaching t tokens with no violation. Done in the exp06a analysis**
    (`clean_short`, t = 200; H2's bottom row). It is required for thinking off (point 8) and useful for thinking on.
    For exp06: report it at 200 and 1000 next to S(t).
28. **An "instruction executed" check for every harness. Kept.** Point 4's Gemma result (0% in the trace, 100% in the
    output) is exactly what it catches.
29. **Empty traces as a reported outcome. Kept.**
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

## Round 2 (running; deviations_round2_decision.json)

- **The original 5 arms**, thinking on, 100 questions: stacked + failure guide, stacked + few-shot, stacked +
  start-with, stacked + few-shot + guide + start-with, few-shot + failure guide.
- **Added arm:** stacked + few-shot + failure guide (no start-with), thinking on, 100 questions.
- **Thinking off for all 6 arms**, on the 51 questions.

## D. Default selection for exp06 (v3)

Models: Qwen3.8-27B-FP8, Gemma-4-31B-FP8, Qwen3-32B (items 1-4). Items: 5, 6, 7, 9 (with the fix), 11, 13, 14, 15,
16 (revised: length-robust), 17 (drop), 18, 19, 21, 22, 25, 26 (revised rule), 27, 28, 29, 30, 31, 33, 34.

Changes from v2: GPT-OSS (2) and Qwen3.6-27B (4) leave by human decision; 33 and 34 join; 16, 17, 26 and 27
revised. Items 21 and 26 still depend on round 2's result.
