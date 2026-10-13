"""exp16: confirmatory replication of exp15 (reading and writing the wrong intermediate) on a fresh, larger
Qwen3.6-27B brew bank, every setting fixed in advance from exp15 (results/exp16_wrong_keys_replication/manifest.json).

exp16 runs exp15's own code, unchanged, with the overrides in OVERRIDES below. What exp16 adds lives here and in
exp16_report.py:
  bank / requests   the fresh test bank (exp11's generator and format, exp16's seed and sizes), written once, after
                    the manifest is locked; no problem text shared with exp11's banks or exp15's test bank
  inherit           exp15's write directions, count directions, write lock and dev trials, copied (with sha256) so
                    every write in exp16 uses exactly exp15's locked vectors and settings
  erase             the one P4 control exp16 adds: remove the wrong key w (no key written in its place) vs remove a
                    control colour; if the model computed the true s_{h-1} and a different line won the retrieval,
                    removing w should restore the gold answer more often than removing an unrelated colour
  stage M ARGS      any exp15 module's command line (exp15, exp15_report, exp15_secondary, ...) under the overrides

EXP16_SMOKE=1 switches everything to a small smoke bank under cache/exp16_smoke and results/exp16_smoke (a dry run of
the overnight queue; no manifest needed).

Run with /venv/main/bin/python from scripts/; HF_HOME must point at the model cache.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cc_config as cfg
import exp11 as X
import exp11_brew as B
import exp15 as E15

SMOKE = os.environ.get("EXP16_SMOKE") == "1"
# EXP16_REFERENCE=1 applies no overrides, so exp16_report computes its statistics on exp15's own test bank and caches:
# the exp15 values that the manifest and the replication figure set beside exp16's.
REFERENCE = os.environ.get("EXP16_REFERENCE") == "1"
assert not (SMOKE and REFERENCE)
EXP = cfg.Exp("exp16_smoke", "exp16_smoke") if SMOKE else cfg.Exp("exp16_wrong_keys_replication", "exp16")
BANK_NAME = "exp16_smoke" if SMOKE else "exp16_test"
TEST_SPEC = ({"seed": "exp16_smoke_20261012", "format": "single_first", "per_h": {2: 60, 3: 120}} if SMOKE else
             {"seed": "exp16_test_20261012", "format": "single_first", "per_h": {1: 100, 2: 1200, 3: 4000}})
# Read cells kept in the capture: the primary cells, plus the start-colour and count tokens (the edit window starts
# at 'start'). exp15 stored 19 cells x 65 layers (12.6 MB per item); this keeps the disk use at ~3.3 MB per item.
CELLS = ("start", "count", "question_end", "nl_asst", "final")
TEXT_READ_SPEC = {"dev": E15.TEXT_READ_SPEC["dev"],
                  "test": {"seed": "exp16_text_rows", "correct_per_h": 10 if SMOKE else 300,
                           "miscount": 10 if SMOKE else 300, "offcycle": "all"}}
OVERRIDES = {
    "EXP": EXP,
    "TEST_SPEC": TEST_SPEC,
    "CELLS": CELLS,
    "TEXT_READ_SPEC": TEXT_READ_SPEC,
    "AO_EXPLORATORY_LAYERS": (),          # exp15's exploratory AO layers (20-32, 52-60 at the final cell) are dropped
}
for _name, _value in OVERRIDES.items():
    assert hasattr(E15, _name), _name
    if not REFERENCE:
        setattr(E15, _name, _value)
# Every other exp15 module does `from exp15 import ...` at import time, so they are imported only below this line.
TEST_BANK = E15.TEST_BANK
EXP15 = cfg.Exp("exp15_wrong_keys", "exp15")
INHERITED_CACHE = ("directions.pt", "count_directions.pt", "trials_single_test_count_repair.jsonl",
                   "trials_single_test_select.jsonl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- bank and requests ----------------------------------------------------------------------------------------------
def write_bank() -> None:
    """exp15's write_test_bank with exp16's seed, sizes and item ids: one h=2 shot plus per_h eval items in the
    single_first format, every gold re-derived by the text parser."""
    import cc_exp10 as E10
    if not SMOKE and (not EXP.manifest.exists() or json.loads(EXP.manifest.read_text()).get("status") != "locked"):
        raise SystemExit("the test bank is generated only after the manifest is locked")
    path = E15.bank_path(TEST_BANK)
    E15.require_absent(path)
    brew, _ = E10.nocot_import()
    fmt = TEST_SPEC["format"]
    rng = random.Random(f"{TEST_SPEC['seed']}:{fmt}")
    cells = [("shot", 2, 0)] + [("eval", h, k) for h, n in TEST_SPEC["per_h"].items() for k in range(n)]
    rows = []
    for split, h, k in cells:
        gen = B.generate(rng, h, B.FORMAT_SPEC[fmt][0], brew.COLORS, brew.INGREDIENTS)
        problem = B.render(gen, fmt, brew.COLORS)
        assert B.parse(problem).path()[-1] == gen["answer"]
        rows.append({"item_id": f"{BANK_NAME}:{fmt}:{split}:h{h}:{k}", "bank": BANK_NAME, "format": fmt,
                     "split": split, "h": h, "problem": problem, "answer": gen["answer"],
                     "instruction": brew.INSTRUCTION})
    texts = collections.Counter(r["problem"] for r in rows)
    texts.update(r["problem"] for b in X.ALL_BANKS if X.bank_path(b).exists() for r in X.load_bank(b))
    exp15_bank = EXP15.cache / "items_test.jsonl"
    texts.update(json.loads(line)["problem"] for line in exp15_bank.open())
    if max(texts.values()) > 1:
        raise SystemExit("a problem text repeats across exp11's, exp15's and exp16's banks")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = {"bank": TEST_BANK, "item_prefix": BANK_NAME, **TEST_SPEC,
              "n_eval": sum(r["split"] == "eval" for r in rows), "content_key": cfg.content_key(rows),
              "checked_unique_against": [*X.ALL_BANKS, str(exp15_bank.relative_to(cfg.REPO_ROOT))]}
    E15.atomic_write_text(json.dumps(record, indent=2) + "\n", EXP.results / "test_bank.json")
    print(f"{len(rows)} rows -> {path}")


def write_requests() -> None:
    """The C0 rows of the bank (exp10's rendering, as exp11 and exp15)."""
    import exp10_conditions as C
    path = E15.requests_path(TEST_BANK)
    E15.require_absent(path)
    items = list(E15.load_items(TEST_BANK).values())
    shot = next(r for r in items if r["split"] == "shot")
    rows = []
    for it in (r for r in items if r["split"] == "eval"):
        row = {"item_id": it["item_id"], "bank": BANK_NAME, "h": it["h"], "source": "brew", "exp10_render": True,
               "seed": 0, "format": it["format"], **C.fields("C0", it, shot)}
        row["request_id"] = cfg.content_key(row)
        rows.append(row)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {path}")


# --- inherited inputs -----------------------------------------------------------------------------------------------
def inherit() -> None:
    """Copy exp15's write inputs into exp16 (each written once), with their sha256 in results/<exp>/inherited.json:
    the write directions and count directions built on exp15's dev bank, exp15's dev trials (count-repair settings and
    the dev rows of the write breakdown), and exp15's write lock (each method's layer and alpha)."""
    record = {"source_commit": "143c06b", "files": {}}
    for name in INHERITED_CACHE:
        src, dst = EXP15.cache / name, EXP.cache / name
        E15.require_absent(dst)
        shutil.copy2(src, dst)
        record["files"][f"cache/{EXP.cache.name}/{name}"] = {"from": f"cache/exp15/{name}", "sha256": sha256(dst)}
    lock_src = EXP15.results / "write_lock.json"
    lock = json.loads(lock_src.read_text())
    lock["inherited_from"] = {"path": "results/exp15_wrong_keys/write_lock.json", "sha256": sha256(lock_src)}
    E15.require_absent(E15.write_lock_path())
    E15.atomic_write_text(json.dumps(lock, indent=1) + "\n", E15.write_lock_path())
    record["files"][f"results/{EXP.exp_id}/write_lock.json"] = lock["inherited_from"]
    E15.atomic_write_text(json.dumps(record, indent=1) + "\n", EXP.results / "inherited.json")
    print(json.dumps(record, indent=1))


# --- erase: the P4 control exp16 adds -------------------------------------------------------------------------------
ERASE_VARIANTS = ("erase_key", "erase_control")


def install_erase_vectors() -> None:
    """exp15's edit_vector, extended to the erase variants: unit(mean_k v(k) - v(c)) moves the state away from colour
    c towards the average colour without writing any particular colour in its place."""
    import exp15_edits as ED
    exp15_edit_vector = ED.edit_vector

    def edit_vector(trial, directions, colours, d):
        if trial.variant not in ERASE_VARIANTS:
            return exp15_edit_vector(trial, directions, colours, d)
        table = directions[trial.method][trial.layer].float()
        v = table.mean(0) - table[colours.index(trial.c)]
        return v / v.norm()

    ED.edit_vector = edit_vector


def erase_trials(it, colours: list[str], brew, grid: list[tuple[str, int, float]]) -> list:
    """On one consistent off-cycle error: erase w (c = the model's key) and erase a control colour r drawn from the
    same pool as exp15's redirect colour (off the true path, off the implied path, not T(a'), and T(r) off them too).
    Success = the gold answer in both; 'unchanged' = the model's original answer."""
    import exp15_edits as ED
    import exp15_score as S
    rng = ED.item_rng(it.item_id, "erase")
    excluded = set(it.path) | set(it.implied) | {it.forward}
    pool = [x for x in colours if x not in excluded and S.step(brew, x) not in excluded]
    if not pool:
        return []
    control = rng.choice(pool)
    outcomes = {"success": it.gold, "unchanged": it.pred}
    return [ED.Trial(it.item_id, variant, method, layer, alpha, c, None, it.gold, outcomes)
            for method, layer, alpha in grid for variant, c in (("erase_key", it.key), ("erase_control", control))]


def erase(bank: str) -> None:
    """The erase trials on P4's population (consistent off-cycle h=3 errors), for P4's methods at their locked
    settings -> cache/<exp>/trials_<bank>_erase.jsonl."""
    import torch
    import exp15_score as S
    install_erase_vectors()
    colours = torch.load(E15.directions_path(), weights_only=False)["colours"]
    locked = json.loads(E15.write_lock_path().read_text())
    grid = [(m, s["layer"], s["alpha"]) for m, s in locked["primary"].items() if m in locked["repair_methods"]]
    items = E15.load_items(bank)
    targets = [it for it in E15.item_table(bank) if it.h == 3 and it.group == "offcycle" and it.p_pred >= S.CONSISTENT_P]
    trials = [t for it in targets for t in erase_trials(it, colours, B.parse(items[it.item_id]["problem"]), grid)]
    E15.run_trials(bank, "erase", trials)


def zero_edit_check() -> None:
    """exp15's harness check (alpha = 0 through run_trials must reproduce the capture's answers) on exp16's test bank,
    since exp16 has no dev capture -> results/<exp>/checks_day0/zero_edit.json (read by exp15_verify)."""
    import exp15_report as R
    R.DEV_BANK = TEST_BANK
    R.check_zero_edit()


def verify_day0_records() -> dict:
    """exp15_verify.day0_records for exp16: the zero-edit check and its diagnosis are exp16's own; the text-matcher
    and ptag checks are exp15's, for the instruments exp16 inherits unchanged (exp16 does not rerun them)."""
    def load(exp: cfg.Exp, name: str) -> dict:
        return json.loads((exp.results / "checks_day0" / f"{name}.json").read_text())
    zero = load(EXP, "zero_edit")
    return {"zero_edit": {k: zero[k] for k in ("n_trials", "same_answer", "max_abs_dp", "mean_abs_dp", "pass")},
            "zero_edit_diagnosis": load(EXP, "zero_edit_diagnosis"),
            "inherited_from_exp15": {
                "matcher": {k: load(EXP15, "matcher")[k] for k in ("precision", "recall", "pass")},
                "matcher_cjk": {k: load(EXP15, "matcher_cjk")[k] for k in ("precision", "recall", "pass")},
                "ptag": {"check7_ret1_L44": load(EXP15, "ptag")["layers"]["44"]["ours_whitened"]["ret1"],
                         **{f"debug_{n}_pass": load(EXP15, f"ptag_debug_{n}")["pass"]
                            for n in ("read_site", "target_space", "pred_space")}}}}


# --- command line ---------------------------------------------------------------------------------------------------
def run_stage(module_name: str, argv: list[str]) -> None:
    """An exp15 module's own main() under the overrides (sys.argv as if it were run directly)."""
    if not module_name.startswith("exp15"):
        raise SystemExit(f"stage runs exp15 modules only, not {module_name}")
    module = importlib.import_module(module_name)
    if module_name == "exp15_verify":
        module.day0_records = verify_day0_records
    sys.argv = [f"{module_name}.py", *argv]
    module.main()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["bank", "requests", "inherit", "zero_edit", "erase", "stage"])
    p.add_argument("rest", nargs=argparse.REMAINDER, help="stage: <exp15 module> <its arguments>")
    a = p.parse_args()
    print(f"exp16 ({EXP.exp_id}; overrides: {sorted(OVERRIDES)})", flush=True)
    if a.cmd == "bank":
        write_bank()
    elif a.cmd == "requests":
        write_requests()
    elif a.cmd == "inherit":
        inherit()
    elif a.cmd == "zero_edit":
        zero_edit_check()
    elif a.cmd == "erase":
        erase(TEST_BANK)
    elif a.cmd == "stage":
        run_stage(a.rest[0], a.rest[1:])


if __name__ == "__main__":
    main()
