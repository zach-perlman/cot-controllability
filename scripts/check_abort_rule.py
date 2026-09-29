"""Replay exp03's abort rule (cc_abort.confirmed_violation) on every finished exp02 trace of the abortable modes.

A trace's reasoning is fed to the rule as it would arrive while streaming: every 16th character for the first
20,000 characters, every 256th after that, and at the end. Pass criteria, per trace:
  - no false abort: whenever the rule fires, CoT-Control's grader calls the full trace non-compliant and the
    first-violation locator on the full trace gives the same character;
  - stable: the rule gives the same character on the full trace as at its first firing;
  - (reported, not required) how far past the violation the rule fires, and which violating traces it never
    catches (violation within the last ABORT_MARGIN_CHARS characters: those traces simply run to the end).
Any false abort or unstable firing fails the check (exit 1).

Run: /venv/main/bin/python scripts/check_abort_rule.py [--limit N]
Output: results/exp03_abort_survival/abort_rule_check.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import cc_abort
import cc_config as cfg
import cc_grade

DENSE_STEP, DENSE_UNTIL, SPARSE_STEP = 16, 20000, 256


def prefix_ends(n: int) -> list[int]:
    ends = list(range(DENSE_STEP, min(n, DENSE_UNTIL) + 1, DENSE_STEP))
    ends += list(range(DENSE_UNTIL + SPARSE_STEP, n + 1, SPARSE_STEP))
    return ends + [n]


def replay(task: tuple[dict, dict, str]) -> dict:
    gen, item, grading_prompt = task
    text, mode = gen["reasoning"], gen["mode"]
    compliant = cc_grade.grade_csv.grade_single_row({"mode": mode, "reasoning": text, "question": grading_prompt})
    full_char = cc_grade.first_violation_char(mode, text, item) if text else None
    out = {"request_id": gen["request_id"], "model": gen["model"], "mode": mode, "chars": len(text),
           "compliant": compliant, "fv_char": full_char, "fired_at": None, "fired_char": None, "error": None}
    for end in prefix_ends(len(text)):
        char = cc_abort.confirmed_violation(mode, text[:end], item)
        if char is None:
            continue
        out.update(fired_at=end, fired_char=char)
        if compliant:
            out["error"] = "false abort: the full trace is compliant"
        elif char != full_char:
            out["error"] = f"fired at char {char}, full-trace first violation {full_char}"
        elif cc_abort.confirmed_violation(mode, text, item) != char:
            out["error"] = "unstable: the rule gives another character on the full trace"
        break
    return out


def tasks(limit: int | None) -> list[tuple[dict, dict, str]]:
    items = {it["item_id"]: it for it in map(json.loads, cfg.ITEMS_PATH.open())}
    requests = {r["request_id"]: r for r in map(json.loads, cfg.EXP02.requests.open())}
    files = sorted(cfg.EXP02.generations.glob("*.jsonl")) + sorted(cfg.EXP02.generations.glob("*.parts/chunk_*.jsonl"))
    seen, out = set(), []
    for path in files:
        if "shard" in path.name:
            continue
        for gen in map(json.loads, path.open()):
            key = (gen["model"], gen["request_id"])
            if gen["mode"] in cfg.ABORT_MODES and key not in seen:
                seen.add(key)
                out.append((gen, items[gen["item_id"]], requests[gen["request_id"]]["grading_prompt"]))
    return out[:limit] if limit else out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="first N traces only (quick test)")
    args = parser.parse_args()
    todo = tasks(args.limit)
    with mp.Pool(min(len(todo), max(1, mp.cpu_count() - 16))) as pool:
        results = pool.map(replay, todo, chunksize=1)

    errors = [r for r in results if r["error"]]
    violating = [r for r in results if r["compliant"] == 0]
    caught = [r for r in violating if r["fired_at"] is not None]
    overshoot = sorted(r["fired_at"] - r["fv_char"] for r in caught)
    per_model = {}
    for r in results:
        m = per_model.setdefault(r["model"], {"traces": 0, "violating": 0, "caught": 0, "chars": 0,
                                              "chars_kept_if_aborted": 0})
        m["traces"] += 1
        m["chars"] += r["chars"]
        m["violating"] += r["compliant"] == 0
        m["caught"] += r["fired_at"] is not None
        m["chars_kept_if_aborted"] += r["fired_at"] if r["fired_at"] is not None else r["chars"]
    summary = {
        "traces": len(results), "violating": len(violating), "caught": len(caught),
        "false_aborts_or_unstable": len(errors), "errors": errors[:20],
        "overshoot_chars_median": overshoot[len(overshoot) // 2] if overshoot else None,
        "overshoot_chars_p95": overshoot[int(len(overshoot) * 0.95)] if overshoot else None,
        "not_caught_violating": [{k: r[k] for k in ("request_id", "model", "mode", "chars", "fv_char")}
                                 for r in violating if r["fired_at"] is None][:50],
        "per_model": per_model,
        "rule": {"margin_chars": cfg.ABORT_MARGIN_CHARS, "dense_step": DENSE_STEP, "dense_until": DENSE_UNTIL,
                 "sparse_step": SPARSE_STEP},
    }
    out = cfg.EXP03.results / "abort_rule_check.json"
    if not args.limit:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("errors", "not_caught_violating")}, indent=1))
    for e in errors[:10]:
        print("ERROR", e)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
