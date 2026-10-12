"""exp15 verification package, for the human check. Written to results/exp15_wrong_keys/verification_v1/, which is
gitignored (it shows generated brew tables and readouts); the folder must be new.

  python exp15_verify.py build --read-tag test_v1 --write-tag test_writes_v1
  python exp15_verify.py build --bank single_test --read-tag dev_v1 --out verification_dev_dryrun   (code check)

dumb_checks.json   machine checks, recomputed without the report code where possible:
  bank        eval items per h, the one-shot example (must not be an eval item); texts shared with dev (must be 0)
  groups      correct / off-cycle / miscount counts per h, recomputed by walking the table from the start colour
  keys        on every off-cycle error: T(key) = answer, and the key is off the true path
  read_auc    P1 h=2, P1 h=3 and P2 key AUC of every token reader, probe and diffmeans, recomputed with scipy's
              rankdata from the stored scores; must match read_report.json to 1e-4
  trials      every stored write-trial outcome re-derived from the brew; every edit trial has its same-item null
  day0        the zero-edit, matcher and ptag records
  lock_order  (test bank) manifest lock < test bank file created; write_lock.json < first test trial file created
p2_scores_long.csv one row per (P2 item, reader, colour): S(c) and the colour's role (key, true s_{h-1}, T(a'),
                   control, other), enough to recompute any P2 AUC in a spreadsheet
spot_check.md      a sheet for reading by hand (fixed seed): off-cycle items with their path, answer, key, each
                   reader's top colours and two NLA and two AO samples; key-swap trials with expected and observed
                   colours
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import subprocess

import numpy as np
from scipy.stats import rankdata

import exp11_brew as B
import exp15 as E
import exp15_score as S
from exp15 import DEV_BANK, EXP, PRIMARY_CELLS, PRIMARY_LAYERS, TEST_BANK

SEED = "exp15_verify"
RANK_READERS = ("logit", "jlens", "rlens", "jpp", "template", "probe", "diffmeans")
SPOT_ITEMS, SPOT_TRIALS = 8, 8


# --- independent re-derivations -------------------------------------------------------------------------------------
def walk(brew: B.Brew, colour: str, k: int) -> str:
    """T^k(colour) for k >= 0, straight from the table."""
    for _ in range(k):
        colour = brew.table[colour][B.SINGLE]
    return colour


def back(brew: B.Brew, colour: str) -> str:
    """T^-1(colour), by searching the table."""
    (found,) = [c for c, row in brew.table.items() if row[B.SINGLE] == colour]
    return found


def classify_item(brew: B.Brew, answer: str) -> str:
    h = len(brew.stirs)
    if answer == walk(brew, brew.start, h):
        return "correct"
    hops = [k for k in range(h + 4) if walk(brew, brew.start, k) == answer]
    return "offcycle" if not hops else "miscount" if len(hops) == 1 else "miscount_ambiguous"


def controls_for(brew: B.Brew, answer: str, colours: list[str]) -> list[str]:
    """Colours off the true path, off the path implied by the answer, and not T(answer)."""
    h = len(brew.stirs)
    true_path = {walk(brew, brew.start, k) for k in range(h + 1)}
    implied, c = {answer}, answer
    for _ in range(h):
        c = back(brew, c)
        implied.add(c)
    return [x for x in colours if x not in true_path | implied | {walk(brew, answer, 1)}]


def auc_vs(s: np.ndarray, target: int, others: list[int]) -> float:
    return float(np.mean([(s[target] > s[o]) + 0.5 * (s[target] == s[o]) for o in others]))


# --- checks ---------------------------------------------------------------------------------------------------------
def bank_checks(items: dict) -> dict:
    """Eval items per h (stirs parsed from the text), the one-shot example kept apart, and overlap with dev."""
    dev_problems = {r["problem"] for r in E.load_items(DEV_BANK).values()}
    evals = [r for r in items.values() if r["split"] == "eval"]
    shots = [r for r in items.values() if r["split"] == "shot"]
    by_h = {}
    for r in evals:
        h = len(B.parse(r["problem"]).stirs)
        by_h[h] = by_h.get(h, 0) + 1
    eval_problems = {r["problem"] for r in evals}
    return {"eval_items_per_h": dict(sorted(by_h.items())), "shot_items": len(shots),
            "shot_in_eval": sum(r["problem"] in eval_problems for r in shots),
            "shared_with_dev": sum(r["problem"] in dev_problems for r in items.values())}


def group_checks(items: dict, metas: list[dict], table) -> dict:
    recomputed, mismatches, key_errors = {}, 0, 0
    ours = {it.item_id: it for it in table}
    for m in metas:
        brew = B.parse(items[m["item_id"]]["problem"])
        answer = m["colours"][int(np.argmax(m["colour_logits"]))]
        group = classify_item(brew, answer)
        h = len(brew.stirs)
        recomputed.setdefault(h, {}).setdefault(group, 0)
        recomputed[h][group] += 1
        mismatches += group != ours[m["item_id"]].group
        if group == "offcycle":
            key = back(brew, answer)
            on_path = key in {walk(brew, brew.start, k) for k in range(h + 1)}
            key_errors += (walk(brew, key, 1) != answer) or on_path or key != ours[m["item_id"]].key
    return {"recomputed_groups": {h: dict(sorted(g.items())) for h, g in sorted(recomputed.items())},
            "group_mismatches_vs_report_code": mismatches, "offcycle_key_errors": key_errors}


def rank_scores(bank: str) -> dict[str, dict[str, np.ndarray]]:
    """reader -> item_id -> S [10]: the 0-based average rank of each colour within each primary (cell, layer) pair,
    averaged over pairs."""
    lr = E.load_lens_reads(bank)
    pc = [list(lr["cells"]).index(c) for c in PRIMARY_CELLS]
    raw = {k: lr["scores"][k][:, pc][:, :, list(PRIMARY_LAYERS), :10].float().numpy()
           for k in RANK_READERS if k in lr["scores"]}
    ids = {k: list(lr["item_ids"]) for k in raw}
    if E.probe_reads_path(bank).exists():
        pr = E.load_probe_reads(bank)
        for k in ("probe", "diffmeans"):
            raw[k] = pr["scores"][k].float().numpy().reshape(len(pr["item_ids"]), -1, 10)
            ids[k] = list(pr["item_ids"])
    return {k: {i: (rankdata(x[n].reshape(-1, 10), axis=-1) - 1).mean(0) for n, i in enumerate(ids[k])}
            for k, x in raw.items()}


def read_auc_checks(bank: str, items: dict, table, read_tag: str) -> dict:
    report = json.loads((EXP.results / read_tag / "read_report.json").read_text())
    colours = list(E.load_lens_reads(bank)["labels"][:10])
    scores = rank_scores(bank)
    groups = {"P1_correct_h2": [it for it in table if it.h == 2 and it.group == "correct"],
              "P1_correct_h3": [it for it in table if it.h == 3 and it.group == "correct"],
              "P2_offcycle": [it for it in table if it.h in (2, 3) and it.group == "offcycle"
                              and it.p_pred >= S.CONSISTENT_P]}
    out = {}
    for reader, per_item in scores.items():
        for g, members in groups.items():
            aucs = []
            for it in members:
                brew = B.parse(items[it.item_id]["problem"])
                key = back(brew, it.pred)
                ctl = [colours.index(c) for c in controls_for(brew, it.pred, colours)]
                aucs.append(auc_vs(per_item[it.item_id], colours.index(key), ctl))
            ours, theirs = float(np.mean(aucs)), report["readers"][reader][g]["key_auc"]["mean"]
            out[f"{reader}|{g}"] = {"recomputed": round(ours, 4), "report": theirs, "n": len(aucs),
                                    "match": abs(ours - theirs) < 1e-4}
    return {"all_match": all(v["match"] for v in out.values()), "rows": out}


def expected_outcomes(t: dict, brew: B.Brew, it) -> dict:
    """The outcome colours of a stored trial, re-derived from the brew. key_swap: from its c' = s1'. other_item
    (an unrelated colour pair, scored against the item's key-swap targets): from the stored echo colour s1'."""
    if t["variant"] in ("key_swap", "other_item"):
        s1_new = t["c_new"] if t["variant"] == "key_swap" else t["outcomes"]["echo"]
        return {"success": walk(brew, s1_new, 1), "start_swap": walk(brew, s1_new, 2), "echo": s1_new,
                "unchanged": it.gold}
    if t["variant"] == "answer_swap":
        return {"success": t["c_new"], "unchanged": it.gold}
    if t["variant"] == "repair":
        return {"success": it.gold, "unchanged": it.pred}
    if t["variant"] == "redirect":
        return {"success": walk(brew, t["c_new"], 1), "unchanged": it.pred}
    return None                          # nulls carry their partner's targets; checked through pairing below


def first_label(outcomes: dict, answer: str) -> str:
    return next((label for label, colour in outcomes.items() if colour == answer), "other")


def trial_phases(bank: str) -> tuple[str, str]:
    """(P3 phase, P4 phase): dev's P3 trials are the selection grid."""
    return ("select" if bank == DEV_BANK else "locked"), "repair"


def trial_checks(bank: str, items: dict, table) -> dict:
    ours = {it.item_id: it for it in table}
    out = {}
    for phase in trial_phases(bank):
        if not E.trials_path(bank, phase).exists():
            continue
        records = E.load_trials(bank, phase)
        wrong_outcome, wrong_c, checked = 0, 0, 0
        for t in records:
            it, brew = ours[t["item_id"]], B.parse(items[t["item_id"]]["problem"])
            if t["variant"] == "key_swap":
                wrong_c += t["c"] != it.path[1] or t["c_new"] in it.path
            if t["variant"] in ("repair", "redirect"):
                wrong_c += t["c"] != back(brew, it.pred)
            if t["variant"] == "repair":
                wrong_c += t["c_new"] != it.path[-2]
            exp = expected_outcomes(t, brew, it)
            if exp is not None:
                checked += 1
                wrong_outcome += first_label(exp, t["answer"]) != t["outcome"]
            elif t["variant"] in ("random_null", "repair_null"):
                checked += 1
                wrong_outcome += first_label(t["outcomes"], t["answer"]) != t["outcome"]
        null_of = {"key_swap": "random_null", "answer_swap": "random_null", "other_item": "random_null",
                   "repair": "repair_null", "redirect": "repair_null"}
        nulls = {(t["item_id"], t["layer"], t["alpha"], t["variant"]) for t in records if t["method"] == "random"}
        unpaired = sum((t["item_id"], t["layer"], t["alpha"], null_of[t["variant"]]) not in nulls
                       for t in records if t["variant"] in null_of)
        counts = {}
        for t in records:
            counts[t["variant"]] = counts.get(t["variant"], 0) + 1
        out[phase] = {"n_records": len(records), "per_variant": counts, "outcomes_checked": checked,
                      "outcome_mismatches": wrong_outcome, "c_or_c_new_mismatches": wrong_c,
                      "edit_trials_without_null": unpaired}
    return out


def birth_time(path) -> datetime.datetime:
    """The file's creation time from the filesystem (statx via coreutils; the files are written once)."""
    seconds = int(subprocess.run(["stat", "-c", "%W", str(path)], capture_output=True, text=True, check=True).stdout)
    if seconds == 0:
        raise RuntimeError(f"no birth time recorded for {path}")
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)


def lock_order_checks(bank: str) -> dict:
    """The test bank was generated after the manifest lock, and the first test write trial after the write lock."""
    def written(name: str) -> datetime.datetime:
        stamp = json.loads((EXP.results / name).read_text())["written"]
        return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    manifest, write_lock = written("manifest.json"), written("write_lock.json")
    bank_born = birth_time(E.bank_path(bank))
    trial_files = [E.trials_path(bank, phase) for phase in ("locked", "repair")]
    trials_born = min(birth_time(p) for p in trial_files)
    return {"manifest_locked": manifest.isoformat(), "test_bank_created": bank_born.isoformat(),
            "write_lock_written": write_lock.isoformat(), "first_test_trial_file_created": trials_born.isoformat(),
            "pass": manifest < bank_born and write_lock < trials_born}


def day0_records() -> dict:
    def load(name):
        return json.loads((E.check_dir() / f"{name}.json").read_text())
    zero = load("zero_edit")
    return {"zero_edit": {k: zero[k] for k in ("n_trials", "same_answer", "max_abs_dp", "mean_abs_dp", "pass")},
            "matcher": {k: load("matcher")[k] for k in ("precision", "recall", "pass")},
            "matcher_cjk": {k: load("matcher_cjk")[k] for k in ("precision", "recall", "pass")},
            "ptag": {"check7_ret1_L44": load("ptag")["layers"]["44"]["ours_whitened"]["ret1"],
                     **{f"debug_{n}_pass": load(f"ptag_debug_{n}")["pass"]
                        for n in ("read_site", "target_space", "pred_space")}}}


# --- human material -------------------------------------------------------------------------------------------------
def text_scores(bank: str, colours: list[str]) -> tuple[dict, dict]:
    """reader -> item_id -> S [10] for the text readers at their primary pairs, and the raw samples."""
    import exp15_readers as RD
    import exp15_report as R
    scores, samples = {}, {}
    for name in R.TEXT_READERS:
        if not E.text_reads_path(bank, name).exists():
            continue
        pairs = [f"{c}|{l}" for c, l in E.text_jobs(name) if l in R.TEXT_PRIMARY_LAYERS[name]]
        reads = E.load_text_reads(bank, name)
        samples[name] = {i: {p: [RD.explanation_text(name, s) for s in r[p]] for p in pairs} for i, r in reads.items()}
        scores[name] = {i: np.mean([[np.mean([c in S.colours_named(x) for x in texts]) for c in colours]
                                    for texts in by_pair.values()], axis=0)
                        for i, by_pair in samples[name].items()}
    return scores, samples


def role(colour: str, it, controls: list[str]) -> str:
    if colour == it.key:
        return "key"
    if colour == it.true_prev:
        return "true_prev"
    if colour == it.forward:
        return "forward"
    return "control" if colour in controls else "other"


def write_p2_csv(path, p2, readers: dict, colours: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "h", "answer", "p_answer", "reader", "colour", "role", "S"])
        for it in p2:
            for name, per_item in readers.items():
                if it.item_id not in per_item:
                    continue
                for k, c in enumerate(colours):
                    w.writerow([it.item_id, it.h, it.pred, round(it.p_pred, 4), name, c, role(c, it, it.controls),
                                round(float(per_item[it.item_id][k]), 4)])


def spot_sheet(p2, items, readers, samples, colours, trials, best_method) -> str:
    rng = random.Random(SEED)
    lines = ["# exp15 spot-check sheet (fixed seed 'exp15_verify'; UNVERIFIED until a human signs VERIFIED.md)", "",
             "For each item: check from the table that key = the colour the table sends to the answer, and that the "
             "answer is not on the start colour's cycle. For each sample, check the colours you see named against the "
             "matcher's list.", ""]
    for it in rng.sample(p2, min(SPOT_ITEMS, len(p2))):
        brew = B.parse(items[it.item_id]["problem"])
        lines += [f"## {it.item_id} (h={it.h})", "",
                  "table: " + ", ".join(f"{c} -> {row[B.SINGLE]}" for c, row in brew.table.items()), "",
                  f"start {brew.start}; true path {' -> '.join(it.path)}; model answer {it.pred} (p={it.p_pred:.2f}); "
                  f"key w = {it.key}; T(answer) = {it.forward}; controls {', '.join(it.controls)}", "",
                  "| reader | top 3 colours by S |", "|---|---|"]
        for name, per_item in readers.items():
            if it.item_id in per_item:
                s = per_item[it.item_id]
                top = sorted(range(10), key=lambda k: -s[k])[:3]
                lines.append(f"| {name} | " + ", ".join(f"{colours[k]} {s[k]:.2f}" for k in top) + " |")
        for name in ("nla", "ao"):
            by_pair = samples.get(name, {}).get(it.item_id, {})
            pair = next((p for p in by_pair if p.startswith("final|")), next(iter(by_pair), None))
            for x in (by_pair[pair][:2] if pair else []):
                lines += ["", f"{name} {pair}: matcher -> {sorted(S.colours_named(x))}", "",
                          "> " + x[:600].replace("\n", " ")]
        lines.append("")
    key_trials = [t for t in trials if t["variant"] == "key_swap" and t["method"] == best_method]
    lines += [f"## Key-swap trials ({best_method}, the method with the largest test key swap - null)", "",
              "Check: success colour = the table applied to s1'; outcome = the first label whose colour is the answer.",
              "", "| item | s1 | s1' | T(s1') (success) | gold (unchanged) | answer | stored outcome |",
              "|---|---|---|---|---|---|---|"]
    for t in rng.sample(key_trials, min(SPOT_TRIALS, len(key_trials))):
        lines.append(f"| {t['item_id']} | {t['c']} | {t['c_new']} | {t['outcomes']['success']} | "
                     f"{t['outcomes']['unchanged']} | {t['answer']} | {t['outcome']} |")
    return "\n".join(lines) + "\n"


def build(bank: str, read_tag: str, write_tag: str | None, out_name: str) -> None:
    out_dir = EXP.results / out_name
    out_dir.mkdir(parents=True, exist_ok=False)
    items = E.load_items(bank)
    metas, _ = E.load_capture(bank, with_hidden=False)
    table = E.item_table(bank)
    colours = metas[0]["colours"]
    checks = {"bank": bank_checks(items), "groups": group_checks(items, metas, table),
              "read_auc": read_auc_checks(bank, items, table, read_tag), "trials": trial_checks(bank, items, table),
              "day0": day0_records()}
    if bank == TEST_BANK:
        checks["lock_order"] = lock_order_checks(bank)
    checks["pass"] = (checks["bank"]["shared_with_dev"] == 0 and checks["bank"]["shot_in_eval"] == 0
                      and checks.get("lock_order", {"pass": True})["pass"]
                      and checks["groups"]["group_mismatches_vs_report_code"] == 0
                      and checks["groups"]["offcycle_key_errors"] == 0 and checks["read_auc"]["all_match"]
                      and all(v["outcome_mismatches"] == 0 and v["c_or_c_new_mismatches"] == 0
                              and v["edit_trials_without_null"] == 0 for v in checks["trials"].values()))
    (out_dir / "dumb_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    readers = rank_scores(bank)
    t_scores, samples = text_scores(bank, colours)
    readers |= t_scores
    p2 = [it for it in table if it.h in (2, 3) and it.group == "offcycle" and it.p_pred >= S.CONSISTENT_P]
    write_p2_csv(out_dir / "p2_scores_long.csv", p2, readers, colours)
    if write_tag:
        write = json.loads((EXP.results / write_tag / "write_report.json").read_text())
        best = max(write["P3"], key=lambda m: write["P3"][m]["key_minus_null"]["mean"])
    else:
        best = "diffmeans"
    sheet = spot_sheet(p2, items, readers, samples, colours, E.load_trials(bank, trial_phases(bank)[0]), best)
    (out_dir / "spot_check.md").write_text(sheet)
    print(json.dumps({k: v for k, v in checks.items() if k != "read_auc"} | {"read_auc_all_match":
                                                                             checks["read_auc"]["all_match"]},
                     indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    p.add_argument("--bank", choices=E.BANKS, default=TEST_BANK)
    p.add_argument("--read-tag", required=True)
    p.add_argument("--write-tag", help="write report folder (the spot sheet's key-swap method is its best)")
    p.add_argument("--out", default="verification_v1", help="folder under results/exp15_wrong_keys (must be new)")
    a = p.parse_args()
    build(a.bank, a.read_tag, a.write_tag, a.out)


if __name__ == "__main__":
    main()
