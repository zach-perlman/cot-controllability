"""exp15 exploratory: which wrong key w does the model use on consistent off-cycle errors? (behaviour only)

T is a fresh random derangement per item, so if the model picked a wrong rule line at random, w = T^-1(a') would be
spread evenly over colours and over rule-line positions. Per colour: the share of errors with that w, of answers
a', and of true s_{h-1}; the off-cycle rate by the true s_{h-1}; by rule-line position: the share of w's line and of
s_{h-1}'s line at each printed position, and w's position relative to s_{h-1}'s; accuracy on all h=2 and h=3
items by the printed position of each lookup's rule line. Chi-square tests against uniform (colours and the 10
line positions).

  python exp15_key_distribution.py --bank test --tag test_key_distribution_v2
      -> results/exp15_wrong_keys/<tag>/key_distribution.{json,md}
"""
from __future__ import annotations

import argparse
import collections
import json

from scipy.stats import chisquare

import exp11_brew as B
import exp15 as E
import exp15_report as R
from exp15 import EXP


def shares(counter: collections.Counter, keys) -> dict:
    total = sum(counter.values())
    return {k: round(counter[k] / total, 3) for k in keys}


def key_distribution(bank: str, tag: str) -> None:
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    items = E.load_items(bank)
    table = E.item_table(bank)
    metas, _ = E.load_capture(bank, with_hidden=False)
    colours = metas[0]["colours"]
    p2 = R.read_groups(table, metas)["P2_offcycle"]
    brews = {it.item_id: B.parse(items[it.item_id]["problem"]) for it in table}

    w, answer, true_prev, pos_w, pos_true, offset = (collections.Counter() for _ in range(6))
    for it in p2:
        order = list(brews[it.item_id].rule_order)
        w[it.key] += 1
        answer[it.pred] += 1
        true_prev[it.true_prev] += 1
        pos_w[order.index(it.key)] += 1
        pos_true[order.index(it.true_prev)] += 1
        offset[order.index(it.key) - order.index(it.true_prev)] += 1

    multi = [it for it in table if it.h in (2, 3)]
    p2_ids = {it.item_id for it in p2}
    offcycle_rate = {}
    for c in colours:
        with_c = [it for it in multi if it.true_prev == c]
        offcycle_rate[c] = round(sum(it.item_id in p2_ids for it in with_c) / len(with_c), 3) if with_c else None

    n = len(p2)
    positions = range(len(colours))
    accuracy_by_line_position = {}
    for h in (2, 3):
        per_h = [it for it in table if it.h == h]
        for step in range(h):
            cells = collections.defaultdict(list)
            for it in per_h:
                order = list(brews[it.item_id].rule_order)
                cells[order.index(it.path[step])].append(it.group == "correct")
            accuracy_by_line_position[f"h{h}_lookup{step + 1}"] = {
                k: {"accuracy": round(sum(v) / len(v), 3), "n": len(v)} for k, v in sorted(cells.items())}
    record = {
        "bank": bank, "exploratory": True, "n_P2": n,
        "colour_share": {"w": shares(w, colours), "answer": shares(answer, colours),
                         "true_prev": shares(true_prev, colours)},
        "chi2_uniform_colour": {name: round(float(chisquare([c[k] for k in colours]).pvalue), 6)
                                for name, c in (("w", w), ("answer", answer), ("true_prev", true_prev))},
        "offcycle_rate_by_true_prev": offcycle_rate,
        "rule_position_share": {"w": shares(pos_w, positions), "true_prev": shares(pos_true, positions)},
        "chi2_uniform_position": {"w": round(float(chisquare([pos_w[k] for k in positions]).pvalue), 6),
                                  "true_prev": round(float(chisquare([pos_true[k] for k in positions]).pvalue), 6)},
        "w_position_minus_true_prev_position": {str(k): offset[k] for k in sorted(offset)},
        "accuracy_by_line_position": accuracy_by_line_position,
        "top_true_prev_to_w_pairs": [(f"{a}->{b}", k) for (a, b), k in collections.Counter(
            (it.true_prev, it.key) for it in p2).most_common(8)]}
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "key_distribution.json")
    E.atomic_write_text(report_md(record, colours), out_dir / "key_distribution.md")
    print(report_md(record, colours))


def report_md(r: dict, colours: list[str]) -> str:
    cs, ps = r["colour_share"], r["rule_position_share"]
    positions = range(len(colours))
    lines = [f"# exp15 wrong-key distribution: {r['bank']} (exploratory, behaviour only, UNVERIFIED)", "",
             f"Consistent off-cycle errors, n = {r['n_P2']}. Chi-square p against uniform: colours "
             f"{r['chi2_uniform_colour']}; rule-line positions {r['chi2_uniform_position']}.", "",
             "| colour | share as w | share as answer a' | share as true s_{h-1} | off-cycle rate when s_{h-1} is it |",
             "|---|---|---|---|---|"]
    lines += [f"| {c} | {cs['w'][c]:.3f} | {cs['answer'][c]:.3f} | {cs['true_prev'][c]:.3f} | "
              f"{r['offcycle_rate_by_true_prev'][c]} |" for c in colours]
    lines += ["", "| rule-line position (0 = printed first) | share as w's line | share as s_{h-1}'s line |",
              "|---|---|---|"]
    lines += [f"| {k} | {ps['w'][k]:.3f} | {ps['true_prev'][k]:.3f} |" for k in ps["w"]]
    lines += ["", f"w's line minus s_{{h-1}}'s line (positions): {r['w_position_minus_true_prev_position']}", "",
              f"Most common s_{{h-1}} -> w pairs: {r['top_true_prev_to_w_pairs']}", "",
              "Accuracy (all items of that h) by the printed position of each lookup's rule line (lookup k uses "
              "the line of s_{k-1}):", "", "| h, lookup | " + " | ".join(str(k) for k in positions) + " |",
              "|---|" + "---|" * len(positions)]
    for name, cells in r["accuracy_by_line_position"].items():
        lines.append(f"| {name} | " + " | ".join(f"{cells[k]['accuracy']:.2f}" for k in positions) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", choices=E.BANKS, required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    key_distribution(a.bank, a.tag)


if __name__ == "__main__":
    main()
