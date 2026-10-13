"""exp16 confirmatory report: every pre-registered test of results/exp16_wrong_keys_replication/manifest.json
("confirmatory_tests"), on the exp16 test bank, one-sided, Holm-adjusted across the whole family.

The tests restate exp15's findings (several of them exploratory in exp15) as predictions:
  R*  reads    item-specific key signal on consistent off-cycle errors beyond the readers' colour preferences
               (shuffled-item permutation test); the true s_{h-1} on the same errors (prior-corrected top share)
  W*  writes   key swap above the random direction (P3); which colour comes out of an answer swap at layer 42
  P*  repair   repair above the random direction (P4); the added erase control (remove w vs remove a control colour)
  B*  behaviour  position dependence of the last lookup, and where w's rule line is printed on errors

  python exp16_report.py --tag confirmatory_v1   -> results/<exp>/<tag>/confirmatory.{json,md} (folder must be new)
"""
from __future__ import annotations

import argparse
import json

import torch
from scipy.stats import binomtest

import exp16 as E16             # applies exp16's overrides to exp15 before the exp15 modules below are imported
import exp11_brew as B
import exp15 as E
import exp15_report as R
import exp15_score as S
import exp15_shuffle_null as SN

EXP = E16.EXP
BANK = E16.TEST_BANK
N_PERM = 2000
SEED = 0
READERS = ("probe", "diffmeans", "logit", "jlens", "rlens", "jpp", "template", "nla", "olens", "ao")
WRITE_METHODS = ("diffmeans", "nla_ar", "logit", "jlens", "rlens", "jpp", "template")
KEY_WRITERS_L42 = ("diffmeans", "nla_ar")             # answer swap at layer 42: the written colour is looked up again
OUTPUT_WRITERS_L42 = ("logit", "jlens", "jpp", "template")     # ... the written colour itself is the answer
REPAIR_METHODS = ("diffmeans", "nla_ar")
ENDS = (0, 9)                                          # printed rule-line positions of the end lines
EARLY = (1, 2, 3)


def entry(values: list[float], null: float = 0.0) -> dict:
    """Mean, 95% bootstrap CI and the one-sided bootstrap p for mean > null (exp15_score.summarise)."""
    return S.summarise([{"x": v} for v in values], "x", null)


# --- reads ----------------------------------------------------------------------------------------------------------
def permutation_top_true(scores: torch.Tensor, members, index: dict, n_perm: int = N_PERM) -> dict:
    """Share of errors where the true s_{h-1} ranks top among {w, s_{h-1}, controls} on prior-corrected scores, and
    its permutation p against scores from a shuffled item (which keeps every colour preference)."""
    observed = SN.top_shares(scores, members, index)["top_true"]
    g = torch.Generator().manual_seed(SEED)
    null = torch.tensor([SN.top_shares(scores[torch.randperm(len(scores), generator=g)], members, index)["top_true"]
                         for _ in range(n_perm)])
    return {"mean": observed, "null_mean": round(float(null.mean()), 4),
            "p": float(((null >= observed).sum() + 1) / (n_perm + 1)), "n": len(members)}


def shifted(ci: dict, by: float) -> dict:
    """A bootstrap entry moved by -by (an observed value minus its null), as the forest plot's {est, lo, hi}."""
    return {"est": round(ci["mean"] - by, 4), "lo": round(ci["lo"] - by, 4), "hi": round(ci["hi"] - by, 4)}


def auc_test(scores, tgt, mask) -> dict:
    """Shuffled-item permutation test of the mean item AUC; effect = observed - shuffled null, its CI from the item
    bootstrap of the observed AUC (the null mean held fixed)."""
    null = SN.shuffled_null(scores, tgt, mask)
    observed = entry([float(v) for v in SN.item_aucs(scores, tgt, mask)], 0.5)
    return {"p": null["p_perm"], "n": null["n"], "observed_auc": null["observed"],
            "shuffled_null": null["null_mean"], "effect": shifted(observed, null["null_mean"])}


def read_tests(table, metas, colours) -> dict:
    index = {c: k for k, c in enumerate(colours)}
    groups = R.read_groups(table, metas)
    p2 = groups["P2_offcycle"]
    tests = {}
    for name, rs in R.reader_scores(BANK).items():
        if name not in READERS:
            continue
        prior = SN.colour_means(rs, [it.item_id for it in table])
        members = [it for it in p2 if rs.score(it.item_id) is not None]
        scores, tgt, mask = SN.group_tensors(rs, members, index, lambda it: it.key, lambda it: it.controls)
        corrected = SN.item_aucs(scores - prior, tgt, mask)
        tests[f"R1|{name}"] = auc_test(scores, tgt, mask) | {
            "prior_corrected_auc": entry([float(v) for v in corrected], 0.5)}
        if name in ("probe", "diffmeans"):
            eligible = [it for it in members if S.direction_eligible(it)]
            tests[f"R2|{name}"] = auc_test(*SN.group_tensors(rs, eligible, index, lambda it: it.key,
                                                             lambda it: [it.forward]))
        if name == "probe":
            corrected_scores = scores - prior
            rows = [S.read_metrics(s, it, index) for s, it in zip(corrected_scores, members)]
            r3 = permutation_top_true(corrected_scores, members, index)
            tests["R3|probe"] = r3 | {"effect": shifted(entry([r["top_true"] for r in rows]), r3["null_mean"])}
            tests["R4|probe"] = entry([r["top_true"] - r["top_key"] for r in rows]) | {
                "top_true": round(sum(r["top_true"] for r in rows) / len(rows), 4),
                "top_key": round(sum(r["top_key"] for r in rows) / len(rows), 4)}
    return tests


# --- writes and repair ----------------------------------------------------------------------------------------------
def write_tests(brews: dict) -> dict:
    lock = json.loads(E.write_lock_path().read_text())
    cells = R.by_setting(E.load_trials(BANK, "locked"))
    tests = {}
    for m in WRITE_METHODS:
        if m not in lock["primary"]:
            continue
        s = lock["primary"][m]
        tests[f"W1|{m}"] = entry(R.success_minus_null(cells, m, s["layer"], s["alpha"])) | {
            "layer": s["layer"], "alpha": s["alpha"]}
    for m in (*KEY_WRITERS_L42, *OUTPUT_WRITERS_L42):
        if m not in lock.get("layer42", {}):
            continue
        alpha = lock["layer42"][m]["alpha"]
        recs = cells.get(("answer_swap", m, 42, alpha), {})
        looked_up = [float(r["answer"] == brews[i].apply(r["c_new"], [B.SINGLE])) for i, r in recs.items()]
        written = [float(r["answer"] == r["c_new"]) for r in recs.values()]
        sign = 1.0 if m in KEY_WRITERS_L42 else -1.0
        tests[f"W2|{m}"] = entry([sign * (x - y) for x, y in zip(looked_up, written)]) | {
            "alpha": alpha, "rate_T_a": round(sum(looked_up) / max(len(looked_up), 1), 4),
            "rate_a": round(sum(written) / max(len(written), 1), 4),
            "statistic": "1[T(a')] - 1[a']" if sign > 0 else "1[a'] - 1[T(a')]"}
    return tests


def paired(cells: dict, a: tuple, b: tuple) -> list[float]:
    """Per item present in both settings: 1[a succeeded] - 1[b succeeded]."""
    x, y = cells.get(a, {}), cells.get(b, {})
    return [float(r["outcome"] == "success") - float(y[i]["outcome"] == "success") for i, r in x.items() if i in y]


def repair_tests() -> dict:
    lock = json.loads(E.write_lock_path().read_text())
    rc = R.by_setting(E.load_trials(BANK, "repair"))
    ec = R.by_setting(E.load_trials(BANK, "erase")) if E.trials_path(BANK, "erase").exists() else {}
    tests, descriptive, gold_rates = {}, {}, {}
    for m in REPAIR_METHODS:
        if m not in lock["primary"]:
            continue
        l, a = lock["primary"][m]["layer"], lock["primary"][m]["alpha"]
        tests[f"P1|{m}"] = entry(R.success_minus_null(rc, m, l, a, "repair", "repair_null"))
        if ec:
            tests[f"P2|{m}"] = entry(paired(ec, ("erase_key", m, l, a), ("erase_control", m, l, a))) | {
                "gold_after_erase_key": entry(R.rate(ec, ("erase_key", m, l, a)), None)["mean"],
                "gold_after_erase_control": entry(R.rate(ec, ("erase_control", m, l, a)), None)["mean"]}
        descriptive[f"repair_minus_redirect|{m}"] = S.summarise(
            [{"x": v} for v in paired(rc, ("repair", m, l, a), ("redirect", m, l, a))], "x", None)
        descriptive[f"redirect_rate|{m}"] = entry(R.rate(rc, ("redirect", m, l, a)), None)
        # Share of edited errors that end on the gold answer, per variant (the erase figure). 'success' is the gold
        # answer for repair, repair_null, erase_key and erase_control; redirect's success is its control colour, so
        # its gold share uses the gold from the same item's repair record.
        repaired, redirected = rc.get(("repair", m, l, a), {}), rc.get(("redirect", m, l, a), {})
        rates = {"repair": R.rate(rc, ("repair", m, l, a)), "random direction": R.rate(rc, ("repair_null", "random", l, a)),
                 "redirect": [float(r["answer"] == repaired[i]["success"]) for i, r in redirected.items()
                              if i in repaired]}
        if ec:
            rates |= {v: R.rate(ec, (v, m, l, a)) for v in ("erase_key", "erase_control")}
        gold_rates[m] = {v: entry(xs, None) for v, xs in rates.items() if xs}
    return tests, descriptive, gold_rates


# --- behaviour ------------------------------------------------------------------------------------------------------
def boot_diff(a: list[float], b: list[float], n_boot: int = 10_000) -> dict:
    """mean(a) - mean(b) for two independent groups: estimate, 95% bootstrap CI, one-sided p for > 0."""
    xa, xb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    g = torch.Generator().manual_seed(SEED)
    d = (xa[torch.randint(len(xa), (n_boot, len(xa)), generator=g)].mean(1)
         - xb[torch.randint(len(xb), (n_boot, len(xb)), generator=g)].mean(1))
    return {"mean": round(float(xa.mean() - xb.mean()), 4), "lo": round(float(d.quantile(0.025)), 4),
            "hi": round(float(d.quantile(0.975)), 4), "p": float(((d <= 0).sum() + 1) / (n_boot + 1)),
            "n": f"{len(xa)} vs {len(xb)}", "accuracy_middle": round(float(xa.mean()), 4),
            "accuracy_ends": round(float(xb.mean()), 4)}


def behaviour_tests(table, metas, brews) -> tuple[dict, dict]:
    """B1, B2, and the per-position counts behind them (the behaviour figure)."""
    h3 = [it for it in table if it.h == 3]
    pos = {it.item_id: list(brews[it.item_id].rule_order).index(it.path[2]) for it in h3}
    middle = [float(it.group == "correct") for it in h3 if pos[it.item_id] not in ENDS]
    ends = [float(it.group == "correct") for it in h3 if pos[it.item_id] in ENDS]
    p2 = R.read_groups(table, metas)["P2_offcycle"]
    key_pos = [list(brews[it.item_id].rule_order).index(it.key) for it in p2]
    early = sum(p in EARLY for p in key_pos)
    test = binomtest(early, len(p2), len(EARLY) / 10, alternative="greater")
    ci = test.proportion_ci(0.95)
    b1 = boot_diff(middle, ends)
    b2 = {"mean": round(early / len(p2), 4), "null": len(EARLY) / 10, "p": float(test.pvalue), "n": len(p2),
          "effect": {"est": round(early / len(p2) - 0.3, 4), "lo": round(ci.low - 0.3, 4),
                     "hi": round(ci.high - 0.3, 4)}}
    by_position = {"h3_accuracy": {p: {"n": sum(pos[it.item_id] == p for it in h3),
                                       "correct": sum(pos[it.item_id] == p and it.group == "correct" for it in h3)}
                                   for p in range(10)},
                   "p2_key_line": {p: key_pos.count(p) for p in range(10)}}
    return {"B1": b1, "B2": b2}, by_position


# --- report ---------------------------------------------------------------------------------------------------------
def confirmatory(tag: str) -> None:
    # The reference run (EXP16_REFERENCE=1) reads exp15's test bank and writes beside exp16's results.
    out_dir = EXP.results / (f"exp15_reference_{tag}" if E16.REFERENCE else tag)
    out_dir.mkdir(parents=True, exist_ok=False)
    items = E.load_items(BANK)
    table = E.item_table(BANK)
    metas, _ = E.load_capture(BANK, with_hidden=False)
    colours = metas[0]["colours"]
    brews = {i: B.parse(r["problem"]) for i, r in items.items()}
    tests = read_tests(table, metas, colours)
    tests |= write_tests(brews)
    repair, descriptive, gold_rates = repair_tests()
    tests |= repair
    behaviour, by_position = behaviour_tests(table, metas, brews)
    tests |= behaviour
    for t in tests.values():
        if "effect" not in t:
            t["effect"] = {"est": t["mean"], "lo": t["lo"], "hi": t["hi"]}
    holm = S.holm({k: t["p"] for k, t in tests.items() if t["p"] == t["p"]})
    for k, t in tests.items():
        t["p_holm"] = round(holm.get(k, float("nan")), 5)
        t["pass"] = bool(t["p_holm"] < 0.05)
    manifest = json.loads(EXP.manifest.read_text()) if EXP.manifest.exists() else {}
    record = {"bank": BANK, "cache": str(E.EXP.cache), "n_tests": len(tests), "alpha": 0.05,
              "family": "all tests below (Holm)", "manifest_tests": manifest.get("confirmatory_tests", {}),
              "tests": tests, "descriptive": descriptive, "gold_rates_after_edit": gold_rates,
              "by_position": by_position}
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "confirmatory.json")
    E.atomic_write_text(report_md(record), out_dir / "confirmatory.md")
    print(report_md(record))


def describe(t: dict) -> str:
    if "lo" in t:
        return f"{t['mean']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]"
    if "observed_auc" in t:
        pc = t.get("prior_corrected_auc")
        return (f"AUC {t['observed_auc']:.3f} vs shuffled {t['shuffled_null']:.3f}"
                + (f"; prior-corrected {pc['mean']:.3f} [{pc['lo']:.3f}, {pc['hi']:.3f}]" if pc else ""))
    if "null_mean" in t:
        return f"{t['mean']:.3f} vs shuffled {t['null_mean']:.3f}"
    return f"{t['mean']:.3f} vs {t['null']:.3f}"


def report_md(record: dict) -> str:
    claims = {k: v.get("claim", "") for k, v in record["manifest_tests"].items()}
    lines = [f"# exp16 confirmatory tests ({record['bank']} bank, UNVERIFIED)", "",
             f"{record['n_tests']} one-sided tests, Holm-adjusted together at 0.05. Ids as in the manifest.", "",
             "| id | claim | n | estimate | p | Holm p | pass |", "|---|---|---|---|---|---|---|"]
    for k, t in record["tests"].items():
        claim = claims.get(k, claims.get(k.split("|")[0], ""))
        lines.append(f"| {k} | {claim} | {t['n']} | {describe(t)} | {t['p']:.4g} | {t['p_holm']:.4g} | "
                     f"{'yes' if t['pass'] else 'no'} |")
    lines += ["", "Descriptive (not tested): repair - redirect (paired) and the redirect rate, per P4 method.", ""]
    for k, d in record["descriptive"].items():
        lines.append(f"- {k}: {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] n={d['n']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    confirmatory(p.parse_args().tag)


if __name__ == "__main__":
    main()
