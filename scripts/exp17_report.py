"""exp17 report (no model, no GPU): tables from the trial files of each model run so far, written to
results/exp17_jspace_keys/analysis/<tag>/ (a new folder per tag).

  posctl   the factual positive control per band x alpha (intermediate swap, answer swap, null), and stop rule K1
  dev      the dev grid (success, null, gain, echo, dose per family, band, alpha), the alpha-0 arm, the J++ answer
           swaps, the selection and stop rule K2
  test     the confirmatory tests T1-T4 on the locked test arms (paired bootstrap over items, Holm within model)

  python exp17_report.py --tag v1 [--models qwen gemma]
"""
from __future__ import annotations

import argparse
import collections
import json

import cc_config as cfg
import exp15_score as S

EXP_DIR = cfg.REPO_ROOT / "results" / "exp17_jspace_keys"
CACHE = cfg.REPO_ROOT / "cache" / "exp17"
K1_MIN_GAIN, K2_MIN_GAIN = 0.15, 0.10


def load(model: str, phase: str) -> list[dict] | None:
    path = CACHE / model / f"trials_{phase}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else None


def by_arm(rows: list[dict], keep: set[str]) -> dict[str, dict[str, dict]]:
    """arm -> item_id -> record, on kept items."""
    out = collections.defaultdict(dict)
    for r in rows:
        if r["arm"] != "clean" and r["item_id"] in keep:
            out[r["arm"]][r["item_id"]] = r
    return out


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# --- positive control -----------------------------------------------------------------------------------------------
def posctl(model: str) -> tuple[str, dict]:
    rows = load(model, "posctl")
    if rows is None:
        return f"{model}: no posctl trials\n", {}
    keep = {r["item_id"] for r in rows if r["arm"] == "clean" and r["kept"]}
    n_usable = sum(r["arm"] == "clean" for r in rows)
    arms = by_arm(rows, keep)
    lines = [f"{model}: factual positive control, {len(keep)} of {n_usable} usable prompts answer correctly when clean",
             f"{'band':8s} {'alpha':>5s} | {'inter':>6s} {'answer':>6s} {'null':>6s} | {'gain':>6s} | "
             f"{'kept(i)':>7s} {'dose(i)':>7s} {'dose(n)':>7s}"]
    best = None
    for name in sorted(a for a in arms if a.startswith("inter|")):
        _, band, alpha = name.split("|")
        inter, ans, null = arms[name], arms[f"answer|{band}|{alpha}"], arms[f"null|{band}|{alpha}"]
        s_i, s_a, s_n = (mean(r["success"] for r in x.values()) for x in (inter, ans, null))
        gain = s_i - s_n
        if best is None or gain > best["gain"]:
            best = {"band": band, "alpha": float(alpha), "gain": round(gain, 4), "inter": round(s_i, 4),
                    "answer": round(s_a, 4), "null": round(s_n, 4), "n": len(inter)}
        lines.append(f"{band:8s} {alpha:>5s} | {s_i:6.3f} {s_a:6.3f} {s_n:6.3f} | {gain:+6.3f} | "
                     f"{mean(r['kept'] for r in inter.values()):7.3f} {mean(r['dose'] for r in inter.values()):7.4f} "
                     f"{mean(r['dose'] for r in null.values()):7.4f}")
    if not keep:
        best = {"gain": None, "K1_stop": True, "reason": "no prompt is answered correctly when clean"}
    else:
        best["K1_stop"] = best["gain"] < K1_MIN_GAIN
    lines.append(f"best intermediate swap: {best}")
    return "\n".join(lines) + "\n", best


def posctl_orth(model: str) -> tuple[str, dict]:
    """exp17_orth: the intermediate swap with the answer pair's span removed (natural size, matched) or with the
    answer pair's coordinates clamped, on the prompts the clean model answers."""
    rows = load(model, "posctl_orth")
    if rows is None:
        return f"{model}: no posctl_orth trials\n", {}
    keep = {r["item_id"] for r in rows if r["arm"] == "clean" and r["kept"]}
    arms = by_arm(rows, keep)
    kinds = ("inter", "inter_orth", "inter_orth_m", "inter_clampans")
    lines = [f"{model}: positive-control follow-up, {len(keep)} prompts; success rate (dose)",
             f"{'band':8s} {'alpha':>5s} | " + " ".join(f"{k:>20s}" for k in kinds)]
    out = {}
    for name in sorted(a for a in arms if a.startswith("inter|")):
        _, band, alpha = name.split("|")
        cells = []
        for k in kinds:
            recs = arms[f"{k}|{band}|{alpha}"]
            s, d = mean(r["success"] for r in recs.values()), mean(r["dose"] for r in recs.values())
            out[f"{k}|{band}|{alpha}"] = {"success": round(s, 4), "dose": round(d, 4)}
            cells.append(f"{s:6.3f} ({d:.4f})".rjust(20))
        lines.append(f"{band:8s} {alpha:>5s} | " + " ".join(cells))
    return "\n".join(lines) + "\n", out


# --- dev grid -------------------------------------------------------------------------------------------------------
def outcome_rate(recs: dict, label: str) -> float:
    return mean(r["outcome"] == label for r in recs.values())


def dev(model: str) -> tuple[str, dict]:
    rows = load(model, "dev")
    if rows is None:
        return f"{model}: no dev trials\n", {}
    keep = {r["item_id"] for r in rows if r["arm"] == "clean" and r["outcome"] == "unchanged"}
    n_targets = sum(r["arm"] == "clean" for r in rows)
    arms = by_arm(rows, keep)
    zero = arms.get("zero", {})
    lines = [f"{model}: dev grid, {len(keep)} of {n_targets} targets give gold in their clean pass",
             f"alpha-0 arm: {sum(r['outcome'] == 'unchanged' for r in zero.values())} of {len(zero)} unchanged",
             f"{'family':6s} {'band':8s} {'alpha':>5s} | {'succ':>6s} {'null':>6s} {'gain':>6s} | {'echo':>6s} "
             f"{'start2':>6s} {'unch':>6s} {'other':>6s} | {'dose':>6s} {'mass':>6s}"]
    for name in sorted(arms, key=lambda a: (a.split("|")[0], a.split("|")[1:2], float(a.split("|")[-1])
                                            if "|" in a else 0)):
        fam = name.split("|")[0]
        if fam not in ("jpp", "dm", "ar"):
            continue
        _, band, alpha = name.split("|")
        recs, null = arms[name], arms[f"null_{fam}|{band}|{alpha}"]
        s, sn = outcome_rate(recs, "success"), outcome_rate(null, "success")
        lines.append(f"{fam:6s} {band:8s} {alpha:>5s} | {s:6.3f} {sn:6.3f} {s - sn:+6.3f} | "
                     f"{outcome_rate(recs, 'echo'):6.3f} {outcome_rate(recs, 'start_swap'):6.3f} "
                     f"{outcome_rate(recs, 'unchanged'):6.3f} {outcome_rate(recs, 'other'):6.3f} | "
                     f"{mean(r['dose'] for r in recs.values()):6.4f} "
                     f"{mean(r['colour_mass'] for r in recs.values()):6.3f}")
    lines.append("J++ answer swaps (gold -> a'): success = a' said, relookup = T(a')")
    for name in sorted(a for a in arms if a.startswith("jpp_answer|")):
        recs = arms[name]
        lines.append(f"  {name:22s} success {outcome_rate(recs, 'success'):.3f} relookup "
                     f"{outcome_rate(recs, 'relookup'):.3f} unchanged {outcome_rate(recs, 'unchanged'):.3f} "
                     f"dose {mean(r['dose'] for r in recs.values()):.4f}")
    sel_path = EXP_DIR / model / "dev_selection.json"
    summary = {"n_eligible": len(keep), "zero_unchanged": sum(r["outcome"] == "unchanged" for r in zero.values()),
               "n_zero": len(zero)}
    if sel_path.exists():
        chosen = json.loads(sel_path.read_text())["chosen"]
        summary["chosen"] = {f: {k: c[k] for k in ("band", "alpha", "success", "null", "gain", "echo", "dose")}
                             for f, c in chosen.items()}
        summary["K2_stop"] = chosen["dm"]["gain"] < K2_MIN_GAIN
        lines.append(f"selection: {json.dumps(summary['chosen'])}  K2 stop: {summary['K2_stop']}")
    return "\n".join(lines) + "\n", summary


# --- test -----------------------------------------------------------------------------------------------------------
def paired(arms: dict, a: str, b: str, label: str = "success") -> dict:
    """Mean and 95% bootstrap CI of 1[a's outcome = label] - 1[b's outcome = label] over items both arms ran on."""
    items = sorted(set(arms[a]) & set(arms[b]))
    diff = [(arms[a][i]["outcome"] == label) - (arms[b][i]["outcome"] == label) for i in items]
    m, lo, hi, n = S.boot_ci(diff)
    return {"mean": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n, "values": diff}


def test(model: str, phase: str = "test") -> tuple[str, dict]:
    """T1-T5 on the test arms; phase 'preview' runs the same tables on dev items at the dev selection."""
    rows = load(model, phase)
    if rows is None:
        return f"{model}: no {phase} trials\n", {}
    keep = {r["item_id"] for r in rows if r["arm"] == "clean" and r["outcome"] == "unchanged"}
    arms = by_arm(rows, keep)
    rates = {a: {lab: round(outcome_rate(recs, lab), 4) for lab in ("success", "echo", "start_swap", "unchanged",
                                                                     "relookup", "other")}
             | {"dose": round(mean(r["dose"] for r in recs.values()), 4),
                "colour_mass": round(mean(r["colour_mass"] for r in recs.values()), 4), "n": len(recs)}
             for a, recs in arms.items()}
    out = {"n_eligible": len(keep), "rates": rates, "tests": {}}
    t1 = paired(arms, "dm_jpart", "dm_rem")
    out["tests"]["T1 dm_jpart - dm_rem"] = {k: v for k, v in t1.items() if k != "values"}
    nulls = {"jpp_raw": "null_jpp", "dm_full": "null_dm", "jpp_matched": "null_dm", "dm_jpart": "null_dm",
             "dm_rem": "null_dm",
             "dm_jpart_own": "null_dm", "dm_rem_own": "null_dm", "dm_rem_clamp": "null_dm",
             "ar_full": "null_ar", "ar_jpart": "null_ar", "ar_rem": "null_ar"}
    pvals = {}
    for arm, null in nulls.items():
        if arm not in arms:
            continue
        d = paired(arms, arm, null)
        pvals[arm] = S.boot_p_greater(d["values"], 0.0)
        out["tests"][f"T2 {arm} - {null}"] = {k: v for k, v in d.items() if k != "values"}
    for arm, p in S.holm(pvals).items():
        out["tests"][f"T2 {arm} - {nulls[arm]}"].update({"p": round(pvals[arm], 5), "p_holm": round(p, 5)})
    t3 = paired(arms, "dm_rem", "dm_rem_clamp")
    out["tests"]["T3 dm_rem - dm_rem_clamp"] = {k: v for k, v in t3.items() if k != "values"}
    jpp_ans = [(r["outcome"] == "success") for r in arms["jpp_answer"].values()]
    t4 = [a - (arms["jpp_raw"][i]["outcome"] == "success") for i, a in zip(arms["jpp_answer"], jpp_ans)
          if i in arms["jpp_raw"]]
    m, lo, hi, n = S.boot_ci(t4)
    out["tests"]["T4 jpp_answer - jpp_raw"] = {"mean": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n}
    t5 = paired(arms, "dm_full", "jpp_matched")
    out["tests"]["T5 dm_full - jpp_matched"] = {k: v for k, v in t5.items() if k != "values"}
    secondary = [(f"{arm}_x{s}", f"null_dm_x{s}") for s in ("2", "4") for arm in ("dm_jpart", "jpp_matched")]
    secondary += [("jpp_answer_ctl", "null_answer_ctl")]
    for arm, null in secondary:
        if arm in arms and null in arms:
            d = paired(arms, arm, null)
            out["tests"][f"secondary {arm} - {null}"] = {k: v for k, v in d.items() if k != "values"}
    lines = [f"{model}: {phase}, {len(keep)} eligible items"]
    lines += [f"  {a:14s} " + " ".join(f"{k} {v}" for k, v in r.items()) for a, r in sorted(rates.items())]
    lines += [f"  {k}: {v}" for k, v in out["tests"].items()]
    return "\n".join(lines) + "\n", out


def test_all_positions(model: str) -> tuple[str, dict]:
    """Secondary: the locked J++ and dm swaps edited at every position (rule table included), vs dm's null."""
    rows = load(model, "test_all")
    if rows is None:
        return f"{model}: no test_all trials\n", {}
    keep = {r["item_id"] for r in rows if r["arm"] == "clean" and r["outcome"] == "unchanged"}
    arms = by_arm(rows, keep)
    out = {"n_eligible": len(keep), "rates": {}, "tests": {}}
    lines = [f"{model}: test, every position edited (secondary), {len(keep)} eligible items"]
    for arm in ("jpp_raw_all", "dm_full_all", "null_dm_all"):
        recs = arms[arm]
        out["rates"][arm] = {lab: round(outcome_rate(recs, lab), 4) for lab in ("success", "echo", "unchanged")} | {
            "dose": round(mean(r["dose"] for r in recs.values()), 4), "n": len(recs)}
        lines.append(f"  {arm:12s} " + " ".join(f"{k} {v}" for k, v in out["rates"][arm].items()))
    for arm in ("jpp_raw_all", "dm_full_all"):
        d = paired(arms, arm, "null_dm_all")
        out["tests"][f"{arm} - null_dm_all"] = {k: v for k, v in d.items() if k != "values"}
        lines.append(f"  {arm} - null_dm_all: {out['tests'][f'{arm} - null_dm_all']}")
    return "\n".join(lines) + "\n", out


# --- in-task control: J++ answer swaps at every band ------------------------------------------------------------------
def answer_swaps(model: str) -> tuple[str, dict]:
    """exp17_late: the J++ answer swap (gold -> a') and its null per band x alpha, late and early bands."""
    out, lines = {}, [f"{model}: J++ answer swaps (gold -> a') vs a random direction at their size, dev"]
    for phase in ("answer_early", "late"):
        rows = load(model, phase)
        if rows is None:
            continue
        keep = {r["item_id"] for r in rows if r["arm"] == "clean" and r["outcome"] == "unchanged"}
        arms = by_arm(rows, keep)
        for name in sorted((a for a in arms if a.startswith("jpp_answer|")),
                           key=lambda a: (a.split("|")[1], float(a.split("|")[2]))):
            _, band, alpha = name.split("|")
            s, n = outcome_rate(arms[name], "success"), outcome_rate(arms[f"null_answer|{band}|{alpha}"], "success")
            out[name] = {"success": round(s, 4), "null": round(n, 4), "n": len(arms[name]), "phase": phase,
                         "dose": round(mean(r["dose"] for r in arms[name].values()), 4)}
            lines.append(f"  {band:8s} alpha {alpha:>3s}: success {s:.3f} null {n:.3f} dose {out[name]['dose']:.4f}")
    return "\n".join(lines) + "\n", out


# --- interchange patching --------------------------------------------------------------------------------------------
PATCH_LABELS = ("success", "unchanged", "echo", "start_swap", "other")


def specificity(recs: dict) -> dict:
    """Mean and 95% bootstrap CI over items of 1[success] - (1[echo] + 1[start_swap]) / 2: the target colour's rate
    above that of two other wrong colours (manifest_addendum_patch_and_gemma, secondary). A patch that only scrambles
    the answer scores ~0."""
    values = [(r["outcome"] == "success") - ((r["outcome"] == "echo") + (r["outcome"] == "start_swap")) / 2
              for r in recs.values()]
    m, lo, hi, _ = S.boot_ci(values)
    return {"mean": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def patch(model: str, prefix: str = "patch") -> tuple[str, dict]:
    """exp17_patch (dev: prefix 'patch'; test: 'test_patch'): per window and arm the outcome rates on items whose
    clean pass gives gold and whose counterfactual gives T(s1'); at the run's dm band, window 'after', the
    pre-registered P1-P3 (paired bootstrap, Holm over the three)."""
    sel = EXP_DIR / model / "dev_selection.json"
    dm_band = json.loads(sel.read_text())["chosen"]["dm"]["band"] if sel.exists() else None
    out, lines = {}, []
    for window in ("start", "after"):
        rows = load(model, f"{prefix}_{window}")
        if rows is None:
            continue
        n_targets = sum(r["arm"] == "clean" for r in rows)
        clean_ok = {r["item_id"] for r in rows if r["arm"] == "clean" and r["outcome"] == "unchanged"}
        cf_ok = {r["item_id"] for r in rows if r["arm"] == "cf_clean" and r["outcome"] == "success"}
        keep = clean_ok & cf_ok
        arms = by_arm(rows, keep)
        arms.pop("cf_clean", None)
        rates = {a: {lab: round(outcome_rate(recs, lab), 4) for lab in PATCH_LABELS}
                 | {"dose": round(mean(r["dose"] for r in recs.values()), 4), "n": len(recs),
                    "specificity": specificity(recs)}
                 for a, recs in arms.items()}
        rec = {"n_targets": n_targets, "n_clean_gold": len(clean_ok), "n_cf_correct": len(cf_ok & clean_ok),
               "rates": rates, "tests": {}}
        lines.append(f"{model}: {prefix} window {window}: {len(keep)} of {n_targets} targets (clean gives gold and "
                     f"the counterfactual gives T(s1'))")
        for a in sorted(rates, key=lambda a: (a.split("|")[1], a)):
            r, sp = rates[a], rates[a]["specificity"]
            lines.append(f"  {a:24s} success {r['success']:.3f} unchanged {r['unchanged']:.3f} echo {r['echo']:.3f} "
                         f"other {r['other']:.3f} dose {r['dose']:.4f} | specificity {sp['mean']:+.3f} "
                         f"[{sp['lo']:+.3f}, {sp['hi']:+.3f}]")
        if window == "after" and dm_band is not None and f"patch_dm|{dm_band}" in arms:
            comparisons = {"P1": ("patch_jpp", "patch_random"), "P2": ("patch_dm", "patch_jpp"),
                           "P3": ("patch_dm_rem", "patch_dm_jpart")}
            pvals, names = {}, {}
            for key, (a, b) in comparisons.items():
                d = paired(arms, f"{a}|{dm_band}", f"{b}|{dm_band}")
                pvals[key] = S.boot_p_greater(d["values"], 0.0)
                names[key] = f"{key} {a} - {b} @ {dm_band}"
                rec["tests"][names[key]] = {k: v for k, v in d.items() if k != "values"}
            for key, p in S.holm(pvals).items():
                rec["tests"][names[key]].update({"p": round(pvals[key], 5), "p_holm": round(p, 5)})
            lines += [f"  {k}: {v}" for k, v in rec["tests"].items()]
        out[window] = rec
    if not out:
        return f"{model}: no {prefix} trials\n", {}
    return "\n".join(lines) + "\n", out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--models", nargs="+", default=["qwen", "gemma", "gemma_occupied"], help="run keys")
    a = p.parse_args()
    out_dir = EXP_DIR / "analysis" / a.tag
    out_dir.mkdir(parents=True, exist_ok=False)
    text, record = [], {}
    for model in a.models:
        record[model] = {}
        for name, fn in (("posctl", posctl), ("posctl_orth", posctl_orth), ("dev", dev),
                         ("answer_swaps", answer_swaps), ("patch_dev", patch),
                         ("preview", lambda m: test(m, "preview")), ("test", test),
                         ("test_all_positions", test_all_positions), ("patch_test", lambda m: patch(m, "test_patch"))):
            t, r = fn(model)
            text.append(t)
            record[model][name] = r
    (out_dir / "report.txt").write_text("\n".join(text))
    (out_dir / "report.json").write_text(json.dumps(record, indent=1) + "\n")
    print("\n".join(text))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
