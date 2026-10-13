"""exp17's manifest. `draft` writes results/exp17_jspace_keys/manifest_draft.json before any run; `lock` writes
manifest.json once, after a human has approved the listed models' dev selections: the draft plus each model's locked
settings (read from dev_selection.json and the in-task control's dev trials, not typed in), the models still pending,
and any deviations recorded since the draft. `addendum` locks later runs (gemma_occupied) and the interchange-patching
arms (exp17_patch) the same way, in a separate write-once file, before they touch test items.

  python exp17_manifest.py draft
  python exp17_manifest.py lock --approved-by NAME [--models qwen] [--pending TEXT ...] [--deviation TEXT ...]
  python exp17_manifest.py addendum NAME --approved-by NAME [--runs RUN ...] [--patch RUN ...] [--deviation TEXT]
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json

import cc_config as cfg

EXP_ID = "exp17_jspace_keys"
RESULTS = cfg.REPO_ROOT / "results" / EXP_ID
CACHE = cfg.REPO_ROOT / "cache" / "exp17"
DRAFT = RESULTS / "manifest_draft.json"
MODELS = ("qwen", "gemma")
CONTROL_BAND_ORDER = {"qwen": ("b48_60", "b54_62"), "gemma": ("b46_58", "b52_58")}   # exp17.SPECS control_bands

DRAFT_CONTENT = {
    "experiment": EXP_ID,
    "question": ("Is the key of a no-CoT brew lookup (the intermediate colour s1 on a correct h=2 item) carried by the "
                 "J++ subspace (the 'global workspace') or outside it?"),
    "background": {
        "paper": ("Global-workspace / Jacobian-lens paper (anthropics/jacobian-lens): on two-hop factual prompts the "
                  "J-space part of an intermediate's probe, swapped by a clamped coordinate swap, moves the answer in "
                  "61% of items vs 28% for the remainder, and the remainder's effect falls to 6% when the J "
                  "coordinates are clamped; the J part explains ~10-15% of the probe's variance."),
        "ours": ("exp15/exp16 (Qwen3.6-27B): token-lens key swaps (J++ included) +0.03-0.13 over the random null, "
                 "difference-of-means +0.56-0.67 at layer 40; lens answer swaps succeed (0.72-0.89). exp14 (Gemma): "
                 "J++ reads s1 at the final position at AUC 0.66 where a probe reads it at 1.00."),
    },
    "models": {
        "qwen": {"model": "Qwen3.6-27B", "n_layers": 64,
                 "jpp": "koayon/jpp-lenses qwen3.6-27b/lens.pt (snapshot 6c96867a), layers 0-62, target block 63",
                 "final_norm_gain": "1 + norm.weight", "unembedding": "lm_head.weight"},
        "gemma": {"model": "google/gemma-4-31B-it", "n_layers": 60,
                  "jpp": ("exp12's own J++ fit for the -it model, cache/exp12/lenses/jpp_64p.pt (64 prompts; layers "
                          "0-58 even, plus 15 and 45). Not koayon's Gemma lens: it is fitted on the base model and "
                          "transferred worse in exp12 (eval_transfer_v1)"),
                  "final_norm_gain": "norm.weight", "unembedding": "embed_tokens.weight (tied)"},
    },
    "methods": {
        "swap": ("Clamped coordinate swap (exp17_swaps): for the pair V = [v(c); v(c')] at a layer, coordinates "
                 "c(h) = V^+ h; with c0 from the clean pass of the same batch, h <- h + (c0 + alpha (sigma(c0) - c0) "
                 "- c(h)) V at every J++-fitted block of a band and every window position. alpha = 1 exchanges the "
                 "two coordinates; clamping to clean targets means later layers do not swap back."),
        "window": ("Brew primary: the start colour token through the last prompt token (after the rule table; an "
                   "all-positions edit would also rewrite the table's rule lines). Secondary, test only: all "
                   "positions. Factual positive control: every position after BOS (the paper's)."),
        "families": {
            "jpp": "unit J++ vectors of the colour tokens: J_l^T (gain * W_U[' colour'])",
            "dm_full": ("unit difference of means of the final-position residual, key colour vs the rest, on correct "
                        "h=2/h=3 dev items (exp15's diffmeans direction; key = s_{h-1})"),
            "dm_jpart": ("dm_full's J++ part: non-negative gradient pursuit, k = 25 (the paper's multi-hop k), over "
                         "unit J++ vectors of every readable token (a letter or digit; special tokens excluded)"),
            "dm_rem": "dm_full - dm_jpart",
            "ar_full / ar_jpart / ar_rem": ("Qwen only, layer 42: exp15's NLA reconstructor colour vectors, centred "
                                            "over the 10 colours and unit, split the same way"),
        },
        "edit_kinds": {
            "swap": "the family's own coordinate swap (natural size)",
            "matched": ("the family's swap rescaled per position to the norm of the full direction's swap (primary for "
                        "J-part vs remainder: AGENTS.md's matched dose)"),
            "null": "a fixed random unit direction per (item, arm, layer) at the reference swap's norm",
            "clamp": "optional: the colour pair's J++ coordinates held at their clean values (the paper's control)",
        },
        "decomposition_chance_level": ("explained variance of 10 random unit vectors per layer under the same "
                                       "pursuit: the share of dm_full's variance a 25-atom J++ fit explains by "
                                       "chance"),
    },
    "items": {
        "dev": ("exp11's single_test bank (250 per h; exp15's dev bank and exp12's Gemma gate): correct h=2 items with "
                "exp15's swap targets (exp15_edits.swap_targets, same seeds); difference of means from correct "
                "h=2/h=3 dev items"),
        "test": ("a fresh bank shared by both models: seed exp17_test_20261012, single_first format, 1200 h=2 eval "
                 "items + 1 shot, unique against exp11's, exp15's and exp16's banks; written only after the lock"),
        "eligible": ("correct h=2 items (baseline argmax = gold) whose in-batch clean pass also gives gold and that "
                     "have swap targets"),
        "factual": ("probe-swap.json (anthropics/jacobian-lens @ 581d3986, Apache 2.0, sha256 a0edd27c...): 90 "
                    "two-hop prompts, raw text, no chat template; each word = its first token after a space; "
                    "eligible = clean next-token argmax is the answer's token"),
    },
    "outcomes": {
        "key arms": ("argmax over the 10 colour tokens: success T(s1') (the written colour was looked up), "
                     "start_swap T^2(s1'), echo s1' (said, not looked up), unchanged gold, other"),
        "answer arms": "success a' (said), relookup T(a') (looked up again), unchanged gold, other",
        "factual": "success = full-vocabulary next-token argmax is swap_answer's token",
        "collateral": "probability mass on the 10 colour tokens; 'other' rate; dose = mean |delta| / |h| over the edit",
        "no_llm_judge": "every metric is an argmax rate; no judged outputs",
    },
    "arms": {
        "posctl": ("per band (multi-layer) x alpha in (1, 2, 4): J++ intermediate swap, J++ answer swap (the paper's "
                   "check that the effect is not just writing the answer), random null at the intermediate swap's "
                   "size"),
        "dev": ("per band x alpha (bands (1, 2, 4, 8), single layer (2, 4, 8, 16)): jpp and dm_full key swaps, each "
                "with its own null; Qwen layer 42 also ar_full and its null; J++ answer swap at the two smallest "
                "alphas per band; one alpha-0 arm (must change nothing)"),
        "test": ("at the locked settings: jpp_raw, null_jpp, jpp_answer (at jpp's setting); dm_full, dm_jpart "
                 "(matched), dm_rem (matched), dm_jpart_own and dm_rem_own (natural size, the paper's protocol), "
                 "dm_rem_clamp (matched remainder, J++ colour coordinates clamped), null_dm, other_item (dm_full "
                 "swap of an unrelated colour pair), jpp_at_dm (raw J++ at dm's setting); Qwen: ar_full, ar_jpart, "
                 "ar_rem (matched), null_ar; all positions: jpp_raw_all, dm_full_all, null_dm_all"),
        "bands": {"qwen": {"b24_36": [24, 36], "b30_42": [30, 42], "b36_48": [36, 48], "b42_54": [42, 54],
                           "b24_48": [24, 48], "l42": [42, 42]},
                  "gemma": {"b22_34": [22, 34], "b28_40": [28, 40], "b34_46": [34, 46], "b40_52": [40, 52],
                            "b22_46": [22, 46]}},
    },
    "selection_rule": ("Per model and family (jpp, dm, ar): the dev (band, alpha) with the largest key-swap success "
                       "minus its own null's success; within 0.01 of the best, the smaller alpha, then the band with "
                       "fewer layers, then the earlier band. J-part / remainder arms use dm's setting (they are not "
                       "selected on). Computed by exp17.py select; a human approves before the lock."),
    "hypotheses": {
        "H1_positive_control": ("On the factual prompts, the J++ intermediate swap moves the answer to swap_answer "
                                "more often than the null (the paper's effect reproduces with our lenses and swap)."),
        "H2_main": ("On the brew test items, the J-part of the key direction (matched dose) swaps keys at a rate "
                    "different from the remainder's (matched dose). The paper predicts J-part > remainder; our prior "
                    "(exp14 reads, exp15 lens writes) predicts remainder >= J-part, i.e. the key is outside the "
                    "workspace."),
        "H3_clamp": ("If the remainder swaps keys, clamping the J++ colour coordinates does not remove that effect "
                     "(paper: 28% -> 6% on factual intermediates)."),
        "H4_dissociation": ("The raw J++ answer swap succeeds while the raw J++ key swap does not: the workspace "
                            "carries the output colour but not the lookup key."),
    },
    "confirmatory_tests": {
        "per_model": [
            "T1 dm_jpart - dm_rem success (paired by item): mean, 95% bootstrap CI (10k, items)",
            "T2 each of jpp_raw, dm_full, dm_jpart, dm_rem, dm_jpart_own, dm_rem_own, dm_rem_clamp (and Qwen ar_full, "
            "ar_jpart, ar_rem) vs its null: success difference, one-sided bootstrap p, Holm within model",
            "T3 dm_rem - dm_rem_clamp success (paired): CI",
            "T4 jpp_answer success - jpp_raw success (paired): CI",
        ],
        "reading": ("'Key in J++ space' needs T1's CI above 0 and dm_jpart above its null; 'key outside J++ space' "
                    "needs T1's CI below 0 and dm_rem above its null; otherwise inconclusive. Also reported: J-part "
                    "and remainder gains as fractions of dm_full's gain, with doses."),
    },
    "power": ("Expected eligible test items: Qwen ~860 (h=2 dev accuracy 0.764), Gemma ~1100 (0.928). At n=860 a "
              "rate's SE is <= 0.017; paired differences of ~0.06 are resolvable."),
    "stop_rules": {
        "K0": "exp17.py check fails (trial path, hook numbering, or dev accuracy vs exp11/exp12): stop that model.",
        "K1": ("The positive control's best J++ intermediate swap gain over its null is < 0.15 on a model: report it "
               "and ask before reading that model's brew J++ results as a test of the paper's claim."),
        "K2": ("dm_full's selected dev gain over its null is < 0.10 on a model: the decomposition has nothing to "
               "split under this protocol; stop and ask (do not switch protocol unasked)."),
        "zero_arm": "the alpha-0 arm must reproduce the clean outcome on every eligible dev item",
    },
    "deviations_from_paper": [
        "brew lookup keys, not factual intermediates (the factual set is run as the positive control)",
        "difference-of-means key direction instead of a logistic probe (exp15's best writer; one vector per colour)",
        "J++ lenses instead of the paper's J-lens; Gemma's J++ fit has only even layers (and 15, 45), so its bands "
        "edit about every other block",
        "brew window starts after the rule table (all positions run as a secondary test arm)",
        "matched-dose J-part/remainder is primary; the paper's natural-size swap is reported alongside",
    ],
    "run_plan": {
        "order": "Qwen then Gemma, never together on the GPU",
        "check_in_1": "check, prep (decomposition stats), positive control, both models",
        "check_in_2": "dev grid both models, select, proposed lock (human approval)",
        "check_in_3": "lock, test bank, test arms both models, report and figure",
        "stages": ("EXP17_MODEL=<m> exp17.py baseline:dev check prep posctl | dev select | baseline:test test; "
                   "exp17.py bank once after the lock"),
    },
}


def draft() -> None:
    if DRAFT.exists():
        raise SystemExit(f"{DRAFT} exists; written once")
    RESULTS.mkdir(parents=True, exist_ok=True)
    record = {**DRAFT_CONTENT, "status": "draft", "written": datetime.datetime.now().isoformat(timespec="seconds")}
    DRAFT.write_text(json.dumps(record, indent=1) + "\n")
    print(f"-> {DRAFT}")


ANSWER_CONTROL_MIN_GAIN = 0.9


def answer_control(run_key: str, fallback: bool = False) -> dict | None:
    """The in-task control's setting from exp17_late's dev trials: the earliest control band, then the smallest alpha,
    whose J++ answer-swap success minus its null's success is >= 0.9 on dev's eligible items. If none is and
    `fallback`: the first control band at the alpha with the largest success - null (ties: the smaller alpha),
    marked meets_rule False."""
    path = CACHE / run_key / "trials_late.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keep = {r["item_id"] for r in rows if r["arm"] == "clean" and r["outcome"] == "unchanged"}
    rate = collections.defaultdict(list)
    for r in rows:
        if r["arm"] != "clean" and r["item_id"] in keep:
            rate[r["arm"]].append(r["outcome"] == "success")
    order = CONTROL_BAND_ORDER[run_key.split("_")[0]]
    settings = []                                    # (band rank, alpha, band, success, null) per control band x alpha
    for arm, xs in rate.items():
        if arm.startswith("jpp_answer|"):
            _, band, alpha = arm.split("|")
            null = rate[f"null_answer|{band}|{alpha}"]
            if band in order:
                settings.append((order.index(band), float(alpha), band, round(sum(xs) / len(xs), 4),
                                 round(sum(null) / len(null), 4)))
    cands = [c for c in settings if c[3] - c[4] >= ANSWER_CONTROL_MIN_GAIN]
    if cands:
        _, alpha, band, s, n = min(cands)
        return {"band": band, "alpha": alpha, "dev_success": s, "dev_null": n, "meets_rule": True,
                "rule": f"earliest control band, then smallest alpha, with dev success - null >= "
                        f"{ANSWER_CONTROL_MIN_GAIN}"}
    if not fallback:
        return None
    first = [c for c in settings if c[2] == order[0]]
    _, alpha, band, s, n = max(first, key=lambda c: (round(c[3] - c[4], 4), -c[1]))
    return {"band": band, "alpha": alpha, "dev_success": s, "dev_null": n, "meets_rule": False,
            "rule": f"no control band has dev success - null >= {ANSWER_CONTROL_MIN_GAIN}; fallback: the first control "
                    f"band at the alpha with the largest success - null"}


def run_settings(run_key: str, answer_fallback: bool = False) -> dict:
    chosen = json.loads((RESULTS / run_key / "dev_selection.json").read_text())["chosen"]
    out = {fam: {"band": c["band"], "alpha": c["alpha"], "dev_gain": c["gain"]} for fam, c in chosen.items()}
    ctl = answer_control(run_key, fallback=answer_fallback)
    if ctl is not None:
        out["answer_control"] = ctl
    return out


def lock(approved_by: str, deviations: list[str], models: list[str], pending: list[str]) -> None:
    path = RESULTS / "manifest.json"
    if path.exists():
        raise SystemExit(f"{path} exists; written once")
    record = json.loads(DRAFT.read_text())
    settings = {m: run_settings(m) for m in models}
    record.update({"status": "locked", "locked": datetime.datetime.now().isoformat(timespec="seconds"),
                   "approved_by": approved_by, "locked_settings": settings,
                   "pending_models": pending, "deviations_since_draft": deviations})
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(settings, indent=1))


PATCH_PLAN = {
    "arms": ("exp17_patch: per band, the counterfactual's coordinates copied into the residual at every J++ layer and "
             "window position, in the whole residual (full) or the span of 10 vectors: dm_full, dm_jpart, dm_rem, "
             "the J++ colour vectors (jpp), a random subspace; on the control band full, jpp, random only"),
    "bands": "the run's locked dm band, jpp band and in-task control band (read from the locked settings)",
    "windows": {"start": "start colour .. last token (the swap arms' window)",
                "after": "the token after the start colour .. last token (primary: the start colour itself, which "
                         "names s0', is not patched)"},
    "items": ("the test bank's correct h=2 items whose counterfactual (start colour T^-1(s1')) is answered T(s1') "
              "clean (arm cf_clean)"),
    "success": "the answer is T(s1')",
    "tests": {"P1": "patch_jpp - patch_random at the dm band, window after",
              "P2": "patch_dm - patch_jpp at the dm band, window after",
              "P3": "patch_dm_rem - patch_dm_jpart at the dm band, window after"},
    "test_method": "paired bootstrap 95% CI (exp15_score.boot_ci), Holm over P1-P3 within run",
    "controls": ["patch_full at every band and window: the upper bound (window start: the counterfactual itself)",
                 "patch_jpp at the control band: the J++ colour span carries the answer late"],
    "secondary": ["every other arm, band and window: CIs only",
                  "specificity per arm: per item 1[success] - (1[echo] + 1[start_swap]) / 2, the success rate above "
                  "that of two other wrong colours (a patch that only scrambles the answer scores ~0)"],
}


def locked_run_settings(run_key: str) -> dict:
    """The run's settings in manifest.json and every locked addendum so far."""
    settings = dict(json.loads((RESULTS / "manifest.json").read_text())["locked_settings"].get(run_key, {}))
    for path in sorted(RESULTS.glob("manifest_addendum_*.json")):
        settings.update(json.loads(path.read_text())["locked_settings"].get(run_key, {}))
    return settings


def addendum(name: str, approved_by: str, runs: list[str], patch_runs: list[str], deviations: list[str],
             descriptive: list[str]) -> None:
    """manifest_addendum_<name>.json, written once, before the runs it locks: the settings of newly locked runs (from
    their dev selection and in-task control, as `lock`, with the control's fallback recorded when no band meets its
    rule) and, per patch run, the bands and windows of the patch arms (PATCH_PLAN). Runs in `descriptive` have their
    swap tests reported without confirmatory claims. exp17.locked_settings merges it into manifest.json's and refuses
    any setting locked twice."""
    path = RESULTS / f"manifest_addendum_{name}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; written once")
    base = json.loads((RESULTS / "manifest.json").read_text())
    assert base["status"] == "locked"
    settings = {r: run_settings(r, answer_fallback=True) for r in runs}
    for r in patch_runs:
        source = {**locked_run_settings(r), **settings.get(r, {})}
        missing = [k for k in ("dm", "jpp", "answer_control") if k not in source]
        if missing:
            raise SystemExit(f"{r}: no locked {missing}; lock the run first or in this addendum")
        bands = list(dict.fromkeys(source[k]["band"] for k in ("dm", "jpp", "answer_control")))
        settings.setdefault(r, {})["patch"] = {"bands": bands, "windows": ["start", "after"],
                                               "dm_band": source["dm"]["band"]}
    record = {"experiment": EXP_ID, "addendum": name, "status": "locked",
              "locked": datetime.datetime.now().isoformat(timespec="seconds"), "approved_by": approved_by,
              "base_manifest_locked": base["locked"], "locked_settings": settings,
              "descriptive_swap_tests": descriptive, "patch_plan": PATCH_PLAN if patch_runs else None,
              "deviations": deviations}
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(settings, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("draft")
    lk = sub.add_parser("lock")
    lk.add_argument("--approved-by", required=True)
    lk.add_argument("--deviation", action="append", default=[])
    lk.add_argument("--models", nargs="+", default=list(MODELS))
    lk.add_argument("--pending", action="append", default=[], help="a model not locked here, and why")
    ad = sub.add_parser("addendum")
    ad.add_argument("name")
    ad.add_argument("--approved-by", required=True)
    ad.add_argument("--runs", nargs="*", default=[], help="runs locked here (e.g. gemma_occupied)")
    ad.add_argument("--patch", nargs="*", default=[], help="runs whose patch arms are locked here")
    ad.add_argument("--deviation", action="append", default=[])
    ad.add_argument("--descriptive", nargs="*", default=[], help="runs whose swap tests make no confirmatory claims")
    a = p.parse_args()
    if a.cmd == "draft":
        draft()
    elif a.cmd == "lock":
        lock(a.approved_by, a.deviation, a.models, a.pending)
    else:
        addendum(a.name, a.approved_by, a.runs, a.patch, a.deviation, a.descriptive)


if __name__ == "__main__":
    main()
