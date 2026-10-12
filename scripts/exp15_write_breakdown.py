"""exp15 exploratory: what do the write directions write? Breakdowns of stored trials (no new model runs).

1. answer_swap at each tested setting (test, locked phase): the edit replaces the answer colour c = s2 by a'. If
   the direction writes an output colour, the model answers a'; if it writes a key that is then looked up once
   more, it answers T(a'). Both rates, the same-item random-null rates of hitting a' and T(a') at that layer and
   alpha, and the per-item difference [T(a') hit - a' hit] with a bootstrap CI.
2. key_swap success by the written colour s1' (test, locked): is the effect spread over colours or carried by a few?
3. The same answer_swap split on the dev grid (select phase), per method and layer at each layer's best alpha by
   success: does the written colour act as a key at early layers and as the output at late ones?

  python exp15_write_breakdown.py --tag test_write_breakdown_v1
      -> results/exp15_wrong_keys/<tag>/write_breakdown.{json,md}
"""
from __future__ import annotations

import argparse
import collections
import json

import exp11_brew as B
import exp15 as E
import exp15_score as S
from exp15 import DEV_BANK, EXP


def lookup(bank: str):
    """item_id -> one application of the item's table T."""
    brews = {i: B.parse(r["problem"]) for i, r in E.load_items(bank).items()}
    return lambda item_id, colour: brews[item_id].apply(colour, [B.SINGLE])


def answer_swap_split(trials: list[dict], T) -> dict:
    """Per (method, layer, alpha) of answer_swap trials: rates of answering a' (written), T(a') and the old c."""
    nulls = {(t["item_id"], t["layer"], t["alpha"]): t["answer"] for t in trials if t["variant"] == "random_null"}
    by_setting = collections.defaultdict(list)
    for t in trials:
        if t["variant"] == "answer_swap":
            by_setting[(t["method"], t["layer"], t["alpha"])].append(t)
    out = {}
    for (method, layer, alpha), ts in sorted(by_setting.items()):
        rows = []
        for t in ts:
            fwd = T(t["item_id"], t["c_new"])
            null = nulls.get((t["item_id"], layer, alpha))
            rows.append({"a": float(t["answer"] == t["c_new"]), "T_a": float(t["answer"] == fwd),
                         "unchanged": float(t["answer"] == t["c"]), "T_a_minus_a": float(t["answer"] == fwd)
                         - float(t["answer"] == t["c_new"]),
                         "null_a": None if null is None else float(null == t["c_new"]),
                         "null_T_a": None if null is None else float(null == fwd)})
        out[f"{method}|L{layer}|a{alpha}"] = {
            "n": len(rows),
            **{m: round(sum(r[m] for r in rows) / len(rows), 4) for m in ("a", "T_a", "unchanged")},
            **{m: S.summarise(rows, m, None)["mean"] for m in ("null_a", "null_T_a")},
            "T_a_minus_a": S.summarise(rows, "T_a_minus_a", 0.0)}
    return out


def success_by_colour(trials: list[dict], settings: dict) -> dict:
    """key_swap success per written colour s1' at each method's locked setting."""
    out = {}
    for method, s in settings.items():
        ts = [t for t in trials if t["variant"] == "key_swap" and t["method"] == method
              and t["layer"] == s["layer"] and t["alpha"] == s["alpha"]]
        per = collections.defaultdict(list)
        for t in ts:
            per[t["c_new"]].append(t["outcome"] == "success")
        rates = {c: round(sum(v) / len(v), 3) for c, v in sorted(per.items())}
        out[method] = {"n": len(ts), "by_colour": rates, "n_by_colour": {c: len(v) for c, v in sorted(per.items())},
                       "min": min(rates.values()), "max": max(rates.values())}
    return out


def dev_layer_split(trials: list[dict], T) -> dict:
    """answer_swap split per (method, layer) at the alpha with the highest a' rate at that layer (dev grid)."""
    split = answer_swap_split(trials, T)
    best = {}
    for key, e in split.items():
        method, layer, _ = key.split("|")
        if (method, layer) not in best or e["a"] > best[(method, layer)][1]["a"]:
            best[(method, layer)] = (key, e)
    return {key: {m: e[m] for m in ("n", "a", "T_a", "unchanged", "null_T_a")} for key, e in
            (best[k] for k in sorted(best, key=lambda k: (k[0], int(k[1][1:]))))}


def report_md(record: dict) -> str:
    lines = ["# exp15 write breakdown (exploratory, UNVERIFIED)", "",
             "## 1. answer_swap on test (locked phase): does the written colour come out, or get looked up once more?",
             "", "| method, layer, alpha | n | answer a' | answer T(a') | unchanged | null: a' / T(a') | "
                 "T(a') - a' [95% CI] |", "|---|---|---|---|---|---|---|"]
    for k, e in record["answer_swap_test"].items():
        d = e["T_a_minus_a"]
        lines.append(f"| {k} | {e['n']} | {e['a']:.3f} | {e['T_a']:.3f} | {e['unchanged']:.3f} | "
                     f"{e['null_a']:.3f} / {e['null_T_a']:.3f} | {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] |")
    lines += ["", "## 2. key_swap success by written colour s1' (test, locked setting)", "",
              "| method | n | min | max | per colour |", "|---|---|---|---|---|"]
    for m, e in record["key_swap_by_colour_test"].items():
        per = ", ".join(f"{c} {r:.2f} (n={e['n_by_colour'][c]})" for c, r in e["by_colour"].items())
        lines.append(f"| {m} | {e['n']} | {e['min']:.2f} | {e['max']:.2f} | {per} |")
    lines += ["", "## 3. answer_swap on the dev grid, per layer (alpha with the highest a' rate at that layer)", "",
              "| method, layer, alpha | n | answer a' | answer T(a') | unchanged | null T(a') |",
              "|---|---|---|---|---|---|"]
    for k, e in record["answer_swap_dev_by_layer"].items():
        lines.append(f"| {k} | {e['n']} | {e['a']:.3f} | {e['T_a']:.3f} | {e['unchanged']:.3f} | "
                     f"{e['null_T_a']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    out_dir = EXP.results / a.tag
    out_dir.mkdir(parents=True, exist_ok=False)
    lock = json.loads((EXP.results / "write_lock.json").read_text())
    test = E.load_trials("test", "locked")
    dev = E.load_trials(DEV_BANK, "select")
    record = {"exploratory": True, "source": ["trials_test_locked.jsonl", f"trials_{DEV_BANK}_select.jsonl"],
              "answer_swap_test": answer_swap_split(test, lookup("test")),
              "key_swap_by_colour_test": success_by_colour(test, lock["primary"]),
              "answer_swap_dev_by_layer": dev_layer_split(dev, lookup(DEV_BANK))}
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "write_breakdown.json")
    E.atomic_write_text(report_md(record), out_dir / "write_breakdown.md")
    print(report_md(record))


if __name__ == "__main__":
    main()
