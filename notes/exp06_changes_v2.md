# exp06 change list, v2 (after exp06a round 1)

Draft for the human's choice; not a result. v1 was written on 2026-10-01 before exp06a. This version keeps v1's
numbering (1-25, P1-P13), marks each item **kept**, **revised**, **done in exp06a**, **evidence against** or **dropped**,
and adds items 26-32 from exp06a round 1. Numbers are from `results/exp06a_prompt_screen/analysis/round1_thinking_on`
(UNVERIFIED; thinking on, 100 questions, S(1000) = share with no violation in the first 1000 tokens, 5-rule mean
unless stated). Round 2 and the thinking-off backfill for Gemma and Qwen3-32B are still running; items that depend on
them say so.

## What exp06a round 1 changed

1. **Prompt-only elicitation is weak next to text in the trace.** The best harness mean over the 3 models is 12.0
   (failure-targeted guide), equal to stacked (12.0); exp04's compliant prefill + stacked is 31-41. Prompt arms are a
   minor axis for exp06; prefill and its interaction with the prompt are the main one.
2. **Harness effects depend on the model.** On Qwen3.8 and Gemma-31B every single harness loses to stacked (by 5-15
   points, all Holm p <= 0.012). On Qwen3-32B, where stacked barely helps (5.1 vs baseline 4.2), start-with,
   few-shot and the failure guide reach 18.5-21.0.
3. **Means over rules hide most of the signal.** Qwen3-32B's gains sit in meow (start-with 74, few-shot 69, baseline
   8) and lowercase (failure guide 50); uppercase and end-with-'safe' stay near 0 under every harness on every model.
   Gemma's non-zero cells are almost all word suppression.
4. **Gemma's thinking ignores instructions about its form.** It never starts with the requested sentence (0%, vs
   Qwen3.8 14%, Qwen3-32B 60%); its traces open with markdown bullet notes whatever the prompt says.
5. **The CoT-vs-output gap holds for every harness** (Qwen3.8; Gemma and Qwen3-32B pending): each harness's gain at
   S(200) is 7-34 points larger with the reasoning in output tags than in the thinking trace.
6. **Checks that passed:** in-environment reruns match the reused exp03/exp04 rows on all 3 models (within 1.7
   points, every CI includes 0), so reusing rows across experiments and hosts is safe. Qwen3-32B's gains are not short
   traces: the share reaching the 1200-token stop with no violation rises from 2.6% (stacked) to 8.6-10.8%.
7. **Nulls:** warm-up, self-generated style guide, monitor + stealth and meta-allowed do not help any model.
   Meta-allowed equals baseline (1.3 vs 1.1, 8.0 vs 8.1, 3.0 vs 4.2), so the no-mention clause is not what fails.

## A. Changes to exp04/exp05 for exp06

**Models**

1. **Drop Gemma-4-12B and Qwen3.5-9B. Kept.** Nothing in exp06a bears on it.
2. **Add GPT-OSS-120B. Kept, and more useful.** Round 1 shows prompt effects differ by model, so a fourth family
   (and the one Jozdien's 5.5% -> 15% prompt gain is on) tests whether any harness generalizes. Same adapter risk.
3. **GLM-4.7-Flash optional; Qwen3.6-35B-A3B dropped. Kept.**
4. **Keep Qwen3.6-27B, with the caps in 5. Kept.**

**Compute**

5. **Full-trace cells: cap at 8k tokens and/or 10% of cells. Kept.**
6. **Stop cells at about 1200 graded tokens. Done in exp06a; adopt.** It worked: thinking-on rows averaged about 500
   tokens and S(1000) is unchanged as an estimand.
7. **Item count. Revised:** 100 questions for every thinking-on arm (round 1's paired contrasts have standard errors of
   1.4-1.7 points at 100 vs 2.1-2.2 at 50; harness effects of 5 points matter). Arms that are secondary or expensive
   (thinking off) use a fixed stratified half, as exp06a did.

**Design**

8. **Opening lengths: keep 1 sentence and >= 150 tokens. Kept.**
9. **No-rule twin for every elicitation arm. Kept, with a fix.** Round 1: Qwen3-32B's few-shot twin keeps 8.8 of
   few-shot's 18.5 (41 of 69 on meow), so copying is a large share. But for word suppression the twin also removes the
   ban, so "arm minus twin" there measures "is the rule stated", not copying (Gemma's 8-point twin gaps are almost all
   word suppression). Report twin contrasts on the opener rules only, or define a word-suppression twin that keeps the
   ban but drops the demonstration.
10. **On-policy few-shot. Done in exp06a.** It helps only Qwen3-32B as a single addition (18.5 vs stacked 5.1) and
    loses to stacked elsewhere. Carry it into exp06 only through the round-2 result (stacked + few-shot) if that wins.
    Out-of-distribution few-shot is still untested.
11. **Prefill that states the rule vs demonstrates it. Kept.**
12. **Periodic reinsertion. Kept.** Decay over the trace is still the main failure under every harness.
13. **Resample at the violation. Kept.**
14. **Rules: lenient grader or programmatic held-out modes. Kept, stronger.** Uppercase and end-with-'safe' are near 0
    under every prompt harness on every model (point 3), so they mostly measure prefill and stacked.

**Cuts**

15. **Drop the CoT-necessity probe. Kept.**
16. **Thinking-off ceiling. Revised.** Still the most robust contrast, but it is expensive: in exp06a thinking-off rows
    averaged 3,317 tokens (vs about 500 thinking on) and took about 84% of round 1's tokens. Run it on the fixed
    half of the questions and only for the arms exp06 needs; option: stop thinking-off rows at the first violation
    inside the tags as well (needs tag-aware streaming grading), since S(200) is the length-matched comparison.
17. **Drop or move the effort check. Kept.**
18. **Drop correctness figures; keep accuracy. Kept.** exp06a has no answer phase, so it gives no accuracy evidence
    either way.

**Diagnostics**

19. **Seam check. Kept.**

## B. Larger additions

20. **Engineered-necessity task. Kept (own mini-project).**
21. **Best prompt crossed with prefill. Kept, now the top addition.** Prefill + stacked (31-41) beats every prompt arm,
    and prefill alone (18-31) beats every single harness. Round 2's stacked + start-with is the prompt-only
    counterpart of prefill + stacked: same sentence, asked for instead of written in.
22. **Recovery as a primary outcome. Kept.**
23. **Cross-model openings. Kept.** Point 4 adds a reason: Gemma ignores requested openings but does respond to
    prefilled ones (exp04), so what it is shown matters more than what it is told.
24. **Monitor outcome. Kept, only with 20.**
25. **Hazard per 100 tokens. Kept.** Round 1's early gains that fade by 1000 tokens (Gemma few-shot: S(200) 24 vs 16
    baseline, S(1000) 9 vs 8) are exactly what this separates.

## New items from exp06a

26. **Prompt arm per model, chosen by a pre-registered rule.** One prompt for all models hides point 2. Either pick per
    model on exp06a data and confirm on new items, or carry two prompts (stacked and the best round-2 arm) into exp06.
    Optimistic as selected on these items; exp06 confirms.
27. **Length-independent co-primary: share reaching the stop (or 1000 tokens) with no violation.** S(1000) treats a
    trace that closes early as censored; this share cannot be raised by shortening the trace. Report both.
28. **An "instruction executed" check for every harness.** For example, the share of traces that begin with the
    requested sentence, or contain the warm-up. Without it, a null harness can mean "not followed" (Gemma) or "followed
    and useless" (Qwen3.8's 14% start-with).
29. **Empty traces as a reported outcome.** Gemma skips thinking in 32% of stacked rows (13.6% under the style guide);
    they count as violations at token 0. Report the share per arm and the censored sensitivity variant.
30. **Report word suppression separately from the opener rules.** It dominates Gemma's 5-rule means (about 40% vs about
    0% on the 4 openers) and its twin means something different (item 9).
31. **Drop warm-up, the style guide, monitor + stealth and meta-allowed from exp06** (point 7). If the style guide
    returns, its thinking-off turn 1 needs the tag fix: 65-100% of those examples are wrapped in literal
    `<output_reasoning>` tags, which the turn-1 grader scores (Gemma's uppercase pass rate shows 0.0).
32. **Record fixed generation settings in the frozen harness doc.** HARNESS.md says few-shot examples come from 6
    candidates; the code drew 8. It changed nothing (every model used the first 3), but exp06's frozen doc should be
    generated from the config, not written by hand.

## C. Prompt-only ideas: status after round 1

- **P1. Monitoring awareness / stealth. Tested (V6): null** on all 3 models. Drop.
- **P2. Start-With. Tested (V1):** +16 on Qwen3-32B (mostly meow), nothing on Qwen3.8, not executed by Gemma. Kept
  only through round 2 (stacked + start-with vs prefill + stacked).
- **P3. Few-shot. On-policy tested (V3):** helps Qwen3-32B only, about half copying. Out-of-distribution still untested.
- **P4. Self-generated style guide. Tested (V4): null**, and Gemma's own examples often break the rule (12% of its meow
  examples pass). Drop.
- **P5. Warm-up. Tested (V2): null or harmful** (Gemma 4.0 vs baseline 8.1). Drop.
- **P6. Channel inversion / short reasoning. Untested.** Same length caveat; item 27 is the right metric for it.
- **P7. Identity framing. Untested.**
- **P8. Rule as a procedure with edge cases. Tested as V5 (failure-targeted guide):** best single harness by mean
  (12.0), driven by Qwen3-32B (21.0; lowercase 50). Kept.
- **P9. Low effort / brevity. Untested.**
- **P10. Rule placement (system / user / both). Untested.** More relevant now: stacked's system prompt may be what
  works on Qwen3.8 and Gemma, since no user-turn addition matches it.
- **P11. "Violations will be resampled". Untested.**
- **P12. LLM-driven prompt iteration. Partly done** (round 1 + round 2, fixed budget). A third round would overfit
  these 100 questions; any further iteration needs new items.
- **P13. Loophole guard (caseless scripts, empty traces). Kept;** item 29 covers the empty-trace part.

## D. Default selection for exp06 (v2)

1, 2, 5, 6, 7, 9 (with the fix), 11, 13, 14, 15, 16 (revised), 17, 18, 19, 21, 22, 25, 26, 27, 28, 29, 30, 31.

Changes from v1's default (1, 2, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19): 10 leaves (done; enters only through
round 2); 7, 21, 22, 25 join; the new 26-31 join. Items 21 and 26 depend on round 2's result.
