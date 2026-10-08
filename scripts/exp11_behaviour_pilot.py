"""exp11 pilot: which shortcut families explain no-CoT brew errors, on exp10's existing C0 (direct answer) test rows.

Read-only over exp10's cache; prints a table. For each model and depth h, over the wrong answers:
  observed  fraction of wrong answers that fall in the family's predicted set (gold excluded)
  uniform   expected fraction if wrong answers were uniform over the non-gold colours
  perm      expected fraction if each wrong answer were checked against ANOTHER item's family set (same h): this keeps
            the model's colour preferences and the family's set sizes, and removes only the item-specific structure.
"""
from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import exp11_brew as B

ROOT = Path(__file__).resolve().parent.parent
GRADES = ROOT / "cache/exp10/grades_v2"
ITEMS = ROOT / "cache/exp10/items_test.jsonl"
FAMILIES = ("stop_early", "skip_one_stir", "reverse_order", "last_stir_only", "wrong_column", "inverse_lookup")
N_PERM = 200


def family_sets(item: dict) -> dict[str, set[str]]:
    brew = B.parse(item["problem"])
    return {f: s - {item["answer"]} for f, s in B.hypotheses(brew).items()}


def main() -> None:
    items = {it["item_id"]: it for it in map(json.loads, open(ITEMS))}
    sets = {iid: family_sets(it) for iid, it in items.items()}
    n_colours = len(B.parse(next(iter(items.values()))["problem"]).table)
    rng = random.Random(0)
    for path in sorted(glob.glob(str(GRADES / "*__stream_abort_test_*.jsonl"))):
        if "_c1b_" in path:
            continue
        rows = [r for r in map(json.loads, open(path)) if r["condition"] == "C0"]
        if not rows:
            continue
        print(f"\n## {rows[0]['model']}  (C0, one sample per item)")
        by_h = defaultdict(list)
        for r in rows:
            by_h[items[r["item_id"]]["h"]].append(r)
        for h in sorted(by_h):
            colours = set(B.parse(items[by_h[h][0]["item_id"]]["problem"]).table)
            wrong = [r for r in by_h[h] if not r["correct"] and r["predicted"] in colours]
            invalid = sum(not r["correct"] and r["predicted"] not in colours for r in by_h[h])
            acc = sum(bool(r["correct"]) for r in by_h[h]) / len(by_h[h])
            if h == 1 or not wrong:
                print(f"h={h}: acc={acc:.2f}, wrong={len(wrong)}, invalid={invalid}")
                continue
            same_h = [iid for iid, it in items.items() if it["h"] == h]
            cells = []
            for f in FAMILIES:
                obs = sum(r["predicted"] in sets[r["item_id"]][f] for r in wrong) / len(wrong)
                uni = sum(len(sets[r["item_id"]][f]) for r in wrong) / len(wrong) / (n_colours - 1)
                perm = []
                for _ in range(N_PERM):
                    hits = []
                    for r in wrong:
                        other = rng.choice([i for i in same_h if i != r["item_id"]])
                        if r["predicted"] != items[other]["answer"]:
                            hits.append(r["predicted"] in sets[other][f])
                    perm.append(sum(hits) / max(len(hits), 1))
                cells.append(f"{f}={obs:.2f}(u{uni:.2f},p{sum(perm) / N_PERM:.2f})")
            none = sum(not any(r["predicted"] in sets[r["item_id"]][f] for f in FAMILIES) for r in wrong) / len(wrong)
            print(f"h={h}: acc={acc:.2f}, wrong={len(wrong)}, invalid={invalid}, none={none:.2f} | " + " ".join(cells))


if __name__ == "__main__":
    main()
