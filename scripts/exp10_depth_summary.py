"""A summary of exp10's depth-stage dev runs (dev_c2, depth_pilot), from the grade files only: per model, task,
set, condition and h, the row count and the rates of correct, hidden_correct, leak and closed reasoning. Rates
only, no item text, so the file can be committed. Not an analysis: no fits, no intervals.
  /venv/main/bin/python scripts/exp10_depth_summary.py
Output: results/exp10_hide_what_you_need/depth_dev_summary.json (a new file per run: _2, _3, ... if it exists).
"""

from __future__ import annotations

import collections
import glob
import json
import re
from pathlib import Path

import cc_exp10 as E

SETS = ("dev_c2", "depth_pilot")
TASKS = ("chain", "arithmetic")
NAME = re.compile(r"^(?P<model>.+?)__card__stream_abort_(?P<set>dev_c2|depth_pilot)_(?P=model)__\w+\.jsonl$")


def summarise() -> dict:
    cells = collections.defaultdict(list)
    for task in TASKS:
        for path in sorted(glob.glob(str(E.EXP.cache / task / "grades" / "*.jsonl"))):
            m = NAME.match(Path(path).name)
            if not m:
                continue
            for g in map(json.loads, open(path)):
                cells[(m["model"], task, m["set"], g["condition"], g["h"])].append(g)
    out = collections.defaultdict(dict)
    for (model, task, set_name, condition, h), rows in sorted(cells.items()):
        n = len(rows)
        rate = lambda key: round(sum(bool(r.get(key)) for r in rows) / n, 3)
        out[f"{model} | {task} | {set_name}"].setdefault(condition, {})[str(h)] = {
            "n": n, "correct": rate("correct"), "hidden_correct": rate("hidden_correct"), "leak": rate("leak"),
            "closed": round(sum(r["think_status"] == "closed" for r in rows) / n, 3)}
    return out


def main() -> None:
    path = E.EXP.results / "depth_dev_summary.json"
    k = 2
    while path.exists():
        path = E.EXP.results / f"depth_dev_summary_{k}.json"
        k += 1
    path.write_text(json.dumps(summarise(), indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
