"""Analysis of exp09's extension 3 (results/exp09_final_test/manifest_extension3.json): examples only, openings with
thinking off, and every main thinking on/off and prompt contrast split by CoT necessity.

Cells and inference as cc_exp09_ext2_analysis (mean over model x rule cells of the arm difference; exp09's paired
question bootstrap; two-sided bootstrap p). Holm over E1 and OF1.
  examples only   S(1000), thinking on (exp09's P1/P2 metric)
  openings off    S(200) of the graded continuation (extension 2's thinking-off metric); OFX compares the opening's
                  effect thinking off and on, both in S(200). Thinking-off responses capped at 1200 tokens (part
                  "offopenc", deviations.json); pooled over the stated 5 models, and over all 7 as a secondary pool
  necessity       per model and question, from 5 direct answers: necessary (<= 1 right), mixed, unnecessary (>= 4
                  right); C1, C2, C3 (exp09's A - baseline thinking on, S(1000)) and E1 within each label

Run: /venv/main/bin/python scripts/cc_exp09_ext3_analysis.py --run NAME [--skip-missing]
Writes results/exp09_final_test/analysis_ext3/<run>/ (never overwritten).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import cc_config as cfg
import cc_exp07 as e7
import cc_exp09 as e9
import cc_exp09_ext2 as x2
import cc_exp09_ext2_analysis as an2
import cc_exp09_ext3 as x3
import exp09_rules as R
from cc_exp08_analysis import contrast, fmt
from cc_survival import holm

OUT_ROOT = e9.EXP.results / "analysis_ext3"
OPEN_OFF_RULES = x2.OPENING_RULES
# name: (metric, cell a, cell b, rules, primary)
CONTRASTS = {
    "E1 A - examples only (thinking on)": ("S_1000", "A|none|on", "examples_only|none|on", R.ALL_RULES, True),
    "OF1 compliant opening - none (A, thinking off)": ("S_200", "A|compliant|off", "A|none|off", OPEN_OFF_RULES, True),
    "E2 examples only - CoT-Control prompt (thinking on)": ("S_1000", "examples_only|none|on", "baseline|none|on",
                                                            R.ALL_RULES, False),
    "OF2 rule + opening - opening only (thinking off)": ("S_200", "A|compliant|off", "no_rule|compliant|off",
                                                         OPEN_OFF_RULES, False),
    "OF3 commitment - compliant opening (A, thinking off)": ("S_200", "A|commitment|off", "A|compliant|off",
                                                             OPEN_OFF_RULES, False),
    "OF4 non-compliant opening - none (A, thinking off)": ("S_200", "A|noncompliant|off", "A|none|off",
                                                           [r for r in OPEN_OFF_RULES if r != "word_suppression"],
                                                           False),
}
BY_NECESSITY = {
    "C1 thinking off - on (CoT-Control prompt)": ("S_200", "baseline|none|off", "baseline|none|on", R.ALL_RULES),
    "C2 thinking off - on (A)": ("S_200", "A|none|off", "A|none|on", R.ALL_RULES),
    "C3 A - CoT-Control prompt, thinking on": ("S_1000", "A|none|on", "baseline|none|on", R.ALL_RULES),
    "E1 A - examples only (thinking on)": CONTRASTS["E1 A - examples only (thinking on)"][:4],
}
LABELS = ["necessary", "mixed", "unnecessary"]


# --- Data -----------------------------------------------------------------------------------------------------------
def load(skip_missing: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Extension 2's and 3's graded rows with exp09's short rows (survival columns), and the necessity labels."""
    frames, labels, present = [], [], []
    base, base_models = an2.load(skip_missing)
    for model in base_models:
        parts = ["exonly", "nec", x3.OFFOPEN_CAPPED]
        paths = {}
        for p in parts:
            try:
                paths[p] = x2.grades_path(model, p)
            except SystemExit:
                continue
        if model in x3.OFFOPEN_SKIPPED and not paths.get(x3.OFFOPEN_CAPPED, cfg.REPO_ROOT / "-").exists():
            parts.remove(x3.OFFOPEN_CAPPED)  # queued last, optional (not in the stated 5-model pool)
        if not all(p in paths and paths[p].exists() for p in parts):
            if skip_missing:
                continue
            raise SystemExit(f"{model}: extension-3 grades missing (pass --skip-missing)")
        for p in parts[:1] + parts[2:]:
            frames.append(pd.DataFrame([json.loads(line) for line in paths[p].open()]).assign(model=model))
        nec = pd.DataFrame([json.loads(line) for line in paths["nec"].open()])
        right = nec.groupby("item_id")["correct"].sum().rename("direct_correct").reset_index().assign(model=model)
        labels.append(right)
        present.append(model)
    if not present:
        raise SystemExit("no graded model")
    df = pd.concat([base[base["model"].isin(present)], e7.survival(pd.concat(frames, ignore_index=True))],
                   ignore_index=True)
    lab = pd.concat(labels, ignore_index=True)
    lab["label"] = np.select([lab["direct_correct"] >= cfg.EXP04_NECESSITY_UNNECESSARY_MIN,
                              lab["direct_correct"] <= cfg.EXP04_NECESSITY_NECESSARY_MAX],
                             ["unnecessary", "necessary"], "mixed")
    return df, lab, present


def interaction(values: dict, cells: list[tuple[str, str]]) -> dict | None:
    """OFX: (A|compliant|off - A|none|off) - (A|compliant|on - A|none|on), S(200), mean over cells."""
    keys = ["A|compliant|off", "A|none|off", "A|compliant|on", "A|none|on"]
    cells = [(m, r) for m, r in cells if all(("S_200", k, m, r) in values for k in keys)]
    if not cells:
        return None

    def diff(m: str, r: str, i: int) -> np.ndarray | float:
        v = [values[("S_200", k, m, r)][i] for k in keys]
        return (v[0] - v[1]) - (v[2] - v[3])

    point = float(np.mean([diff(m, r, 0) for m, r in cells]))
    d = np.mean([diff(m, r, 1) for m, r in cells], axis=0)
    return {"value": point, "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))), "n_cells": len(cells)}


def openings_off_models(models: list[str], all_models: bool) -> list[str]:
    """The stated pool for the openings-off contrasts: the 5 models not dropped for cost (all_models: every model
    with openings-off grades, a secondary pool)."""
    return [m for m in models if all_models or m not in x3.OFFOPEN_SKIPPED]


def contrasts(values: dict, models: list[str], all_openings_off_models: bool = False) -> dict:
    out = {}
    for name, (metric, a, b, rules, primary) in CONTRASTS.items():
        pool = openings_off_models(models, all_openings_off_models) if name.startswith("OF") else models
        cells = [(m, r) for m in pool for r in rules]
        res = contrast(values, metric, a, b, cells)
        if res is not None:
            res.update(metric=metric, a=a, b=b, primary=primary,
                       per_model={m: contrast(values, metric, a, b, [c for c in cells if c[0] == m]) for m in models},
                       per_rule={r: contrast(values, metric, a, b, [c for c in cells if c[1] == r]) for r in rules},
                       per_rule_set={s: contrast(values, metric, a, b, [c for c in cells if c[1] in rs])
                                     for s, rs in an2.RULE_SETS.items()})
            out[name] = res
    for n, p in holm({n: v["p_two_sided"] for n, v in out.items() if v["primary"]}).items():
        out[n]["p_holm"] = p
    cells = [(m, r) for m in openings_off_models(models, all_openings_off_models) for r in OPEN_OFF_RULES]
    ofx = interaction(values, cells)
    if ofx is not None:
        ofx.update(metric="S_200", primary=False,
                   per_model={m: interaction(values, [c for c in cells if c[0] == m]) for m in models},
                   per_rule={r: interaction(values, [c for c in cells if c[1] == r]) for r in OPEN_OFF_RULES},
                   per_rule_set={})
        out["OFX opening effect, thinking off - on (A, S(200))"] = ofx
    return out


def by_necessity(df: pd.DataFrame, labels: pd.DataFrame, models: list[str]) -> dict:
    """Each BY_NECESSITY contrast within the (model, question) pairs of each label."""
    out = {}
    for label in LABELS:
        keep = labels[labels["label"] == label][["model", "item_id"]]
        sub = df.merge(keep, on=["model", "item_id"])
        if sub.empty:
            continue
        values = an2.cell_values(sub)
        out[label] = {name: contrast(values, metric, a, b, [(m, r) for m in models for r in rules])
                      for name, (metric, a, b, rules) in BY_NECESSITY.items()}
    return out


# --- Report ---------------------------------------------------------------------------------------------------------
def report(summary: dict) -> str:
    models = summary["models"]
    lines = [f"# exp09 extension 3 (examples only, openings thinking off, CoT necessity): {summary['run']} "
             "(UNVERIFIED until a human adds it to VERIFIED.md)", "",
             f"Models: {', '.join(models)}." + (f" Missing: {', '.join(summary['missing'])}." if summary["missing"]
                                                 else ""),
             "Not part of exp09's pre-registered test; Holm over E1 and OF1 only. Rule-grader and accuracy metrics "
             "(no LLM judge).", "", "## Contrasts", ""]
    lines += an2.contrast_table(summary["contrasts"], models, with_holm=True)
    lines += ["", "## Openings off on every model with openings-off grades (secondary pool)", ""]
    lines += an2.contrast_table(summary["openings_off_all_models"], models, with_holm=False)
    lines += ["", "## Per rule set", ""] + an2.breakdown_table(
        {n: v for n, v in summary["contrasts"].items() if v["per_rule_set"]}, "per_rule_set", list(an2.RULE_SETS))
    lines += ["", "## By CoT necessity (5 direct answers per model and question)", "",
              "Questions per label: " + json.dumps(summary["necessity_counts"]), "",
              "| contrast | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for name in BY_NECESSITY:
        lines.append(f"| {name} | " + " | ".join(fmt(summary["by_necessity"].get(lab, {}).get(name))
                                                 for lab in LABELS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    out_dir = OUT_ROOT / args.run
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; analysis runs are never overwritten")
    df, labels, models = load(args.skip_missing)
    values = an2.cell_values(df)
    out_dir.mkdir(parents=True)
    labels.to_csv(out_dir / "necessity_labels.csv", index=False)
    all_off = {n: v for n, v in contrasts(values, models, all_openings_off_models=True).items() if n.startswith("OF")}
    summary = {"run": args.run, "models": models, "missing": [m for m in x2.MODEL_ORDER if m not in models],
               "contrasts": contrasts(values, models), "openings_off_all_models": all_off,
               "by_necessity": by_necessity(df, labels, models),
               "necessity_counts": labels.groupby(["model", "label"]).size().unstack(fill_value=0).to_dict("index")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "REPORT_auto.md").write_text(report(summary))
    print(report(summary))


if __name__ == "__main__":
    main()
