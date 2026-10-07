"""exp10's arithmetic banks: nocot-bench's arithmetic task (datagen/banks/arithmetic.py at cc_exp10.NOCOT_COMMIT),
regenerated with its own generator and fresh seeds. Banks live in cache/exp10/arithmetic/ (gitignored; nocot-bench
items are never republished).

One item is a fully parenthesised integer Python expression over leaves in [-99, 99] with exactly n_ops binary
nodes (+ - * // %, Python's floor semantics); the answer is its value. Two depth measures per item:
  h           n_ops, nocot-bench's difficulty (the number of operations to carry out)
  tree_depth  the longest root-to-leaf chain of operations (the critical path: operations on it must be done in
              order, the rest could in principle run in parallel)
Levels are the shipped bank's (1-8, 10, 12); every bank has one shot item per level, as the shipped bank has.

  /venv/main/bin/python scripts/exp10_arithmetic.py shipped   nocot-bench's published bank (data/ncri/arithmetic.jsonl)
  /venv/main/bin/python scripts/exp10_arithmetic.py banks     the dev and test banks (once; never repeat shipped)
"""

from __future__ import annotations

import argparse
import ast
import collections
import json

import cc_config as cfg
import cc_exp10 as E
import cc_exp10_tasks as X

CACHE = E.EXP.cache / "arithmetic"
BANK_PATHS = {bank: CACHE / f"items_{bank}.jsonl" for bank in ("dev", "test", "shipped")}
LEVELS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)
PER_LEVEL = {"dev": 10, "test": 40}
SEEDS = {"dev": "exp10_arithmetic_dev_20261007", "test": "exp10_arithmetic_test_20261007"}


def arithmetic_module():
    return X.nocot_module("datagen.banks.arithmetic")


def expression(problem: str) -> str:
    return problem.split("expression.", 1)[1].strip()


def tree_depth(expr: str) -> int:
    """Operations on the longest root-to-leaf path (a lone number: 0; unary minus on a leaf is part of the leaf)."""
    def depth(node) -> int:
        if isinstance(node, ast.BinOp):
            return 1 + max(depth(node.left), depth(node.right))
        return 0
    return depth(ast.parse(expr, mode="eval").body)


def n_ops(expr: str) -> int:
    return sum(isinstance(node, ast.BinOp) for node in ast.walk(ast.parse(expr, mode="eval")))


def leaf_values(expr: str) -> set[int]:
    """The absolute values of the expression's numbers (the number grader reads no sign)."""
    return {abs(node.value) for node in ast.walk(ast.parse(expr, mode="eval")) if isinstance(node, ast.Constant)}


def intermediate_values(expr: str) -> set[int]:
    """The absolute values of every operation but the last (the root): the states a reasoner works out on the way."""
    root = ast.parse(expr, mode="eval").body
    nodes = [node for node in ast.walk(root) if isinstance(node, ast.BinOp) and node is not root]
    return {abs(eval(compile(ast.Expression(node), "<sub>", "eval"))) for node in nodes}


def check_item(item: dict) -> dict:
    """Re-derived from the text: nocot-bench's solver gives the gold, and the expression has h operations."""
    solved = arithmetic_module().solve(type("Problem", (), {"problem": item["problem"]})())
    return {"solver_ok": solved == item["answer"], "h_ok": n_ops(expression(item["problem"])) == item["h"]}


def row(bank: str, pn: int, split: str, level: int, rung: str | None, problem: str, answer: int, chance: float,
        instruction: str) -> dict:
    return {"item_id": f"arithmetic_{bank}:{pn}", "task": "arithmetic", "bank": bank, "split": split, "h": level,
            "tree_depth": tree_depth(expression(problem)), "rung": rung, "problem": problem, "answer": answer,
            "chance": chance, "instruction": instruction, "answer_type": "int"}


def write_bank(bank: str, rows: list[dict]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if BANK_PATHS[bank].exists():
        raise SystemExit(f"{BANK_PATHS[bank]} exists; banks are generated once")
    BANK_PATHS[bank].write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"arithmetic {bank}: {sum(r['split'] == 'eval' for r in rows)} eval + "
          f"{sum(r['split'] != 'eval' for r in rows)} shot -> {BANK_PATHS[bank]}")


def load_bank(bank: str) -> list[dict]:
    return [json.loads(l) for l in BANK_PATHS[bank].open()]


def extract_shipped() -> None:
    rows = [row("shipped", it["problem_number"], it["split"], it["difficulty"], it.get("rung"), it["problem"],
                it["answer"], it["chance"], it["instruction"])
            for it in X.archive_jsonl("data/ncri/arithmetic.jsonl")]
    failed = [r["item_id"] for r in rows if not all(check_item(r).values())]
    print("shipped items failing a check:", len(failed), "of", len(rows))
    write_bank("shipped", rows)


def generate_bank(bank: str, taken: set[str]) -> list[dict]:
    """arithmetic.generate's draw (its _draw, Config and rng string form) at the shipped levels, with PER_LEVEL[bank]
    eval items per level and one shot item per level; no expression repeats one in `taken`."""
    a = arithmetic_module()
    config = a.Config(per_level={level: PER_LEVEL[bank] for level in LEVELS}, shot_levels=LEVELS)
    r = a.rng(f"arithmetic|{SEEDS[bank]}")
    seen = {expression(p) for p in taken}
    drawn = [(level, "shot", *a._draw(level, r, config, seen)) for level in config.shot_levels]
    drawn += [(level, "eval", *a._draw(level, r, config, seen))
              for level in sorted(config.per_level) for _ in range(config.per_level[level])]
    chance = a.majority_baseline([value for _, split, _, value in drawn if split == "eval"])
    rows, shot_pn, eval_pn = [], 0, 10
    for level, split, expr, value in drawn:
        pn = shot_pn if split == "shot" else eval_pn
        shot_pn, eval_pn = (shot_pn + 1, eval_pn) if split == "shot" else (shot_pn, eval_pn + 1)
        rung = None if split == "shot" else a._rung_for(config, level)
        rows.append(row(bank, pn, split, level, rung, f"{a._PREFIX}{expr}", value, chance, a.INSTRUCTION))
    return rows


def write_banks() -> None:
    record_path = E.EXP.results / "banks_arithmetic.json"
    if record_path.exists():
        raise SystemExit(f"{record_path} exists")
    if not BANK_PATHS["shipped"].exists():
        raise SystemExit("extract the shipped bank first (shipped): new items must not repeat it")
    taken = {r["problem"] for r in load_bank("shipped")}
    banks = {"test": generate_bank("test", taken)}
    banks["dev"] = generate_bank("dev", taken | {r["problem"] for r in banks["test"]})
    texts = collections.Counter([r["problem"] for rows in banks.values() for r in rows] + list(taken))
    if max(texts.values()) > 1:
        raise SystemExit("a problem text occurs twice")
    record = {"nocot_commit": E.NOCOT_COMMIT, "task": "arithmetic", "levels": LEVELS}
    for bank, rows in banks.items():
        failed = [r["item_id"] for r in rows if not all(check_item(r).values())]
        if failed:
            raise SystemExit(f"arithmetic {bank}: {len(failed)} items fail the checks: {failed[:5]}")
        evals = [r for r in rows if r["split"] == "eval"]
        depth_by_h = collections.defaultdict(collections.Counter)
        for r in evals:
            depth_by_h[r["h"]][r["tree_depth"]] += 1
        record[bank] = {"seed": SEEDS[bank], "n_eval": len(evals), "n_shot": len(rows) - len(evals),
                        "per_h": dict(collections.Counter(r["h"] for r in evals)), "chance": evals[0]["chance"],
                        "top_golds": collections.Counter(r["answer"] for r in evals).most_common(5),
                        "tree_depth_by_h": {h: dict(sorted(c.items())) for h, c in sorted(depth_by_h.items())},
                        "checks": "every item passes solver_ok and h_ok", "content_key": cfg.content_key(rows)}
        write_bank(bank, rows)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["shipped", "banks"])
    args = parser.parse_args()
    extract_shipped() if args.command == "shipped" else write_banks()


if __name__ == "__main__":
    main()
