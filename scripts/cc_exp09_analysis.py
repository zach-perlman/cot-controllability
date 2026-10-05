"""exp09_final_test analysis, as pre-registered in results/exp09_final_test/manifest.json (this file was written and
committed before any exp09 generation was read).

Per cell (model x rule x arm): S(200) and S(1000) of the short rows (Kaplan-Meier, cc_exp07.survival's columns), and
on the full rows whole-trace compliance and accuracy. A pool (models x rules) is the mean over its cells, each cell
weighted equally. Uncertainty: a bootstrap that resamples the questions (stratified by source; one draw shared by
every model and arm, so contrasts are paired) and, independently, the pool's rules; the question-only interval is
reported too. Primary contrasts P1-P3 are Holm-corrected.

  /venv/main/bin/python scripts/cc_exp09_analysis.py --run v1 [--skip-missing]
Output: results/exp09_final_test/analysis/<run>/ (never overwritten): primary.json, secondary.json, cells.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import cc_config as cfg
import cc_exp07 as e7
import cc_exp09 as e9
import exp09_rules as R
from cc_survival import holm, kaplan_meier

BOOT_SEED = 20261005
N_BOOT = 2000
T_VALUES = np.array([200, 1000])
PRIMARY = {  # name -> (metric, arm, minus arm, models, rules)
    "P1": ("S_1000", "A", "baseline", e9.FRESH_MODELS, R.ALL_RULES),
    "P2": ("S_1000", "A", "baseline", e9.MODELS, R.NEW_RULES),
    "P3": ("whole_trace_compliance", "A", "baseline", e9.FULL_MODELS, e9.FULL_RULES),
}


# --- Data -----------------------------------------------------------------------------------------------------------
def load(skip_missing: bool) -> tuple[pd.DataFrame, list[str]]:
    frames, present = [], []
    for model in e9.MODELS:
        try:
            path = e9.grades_path(model)
        except SystemExit:
            path = None
        if path is None or not path.exists():
            if skip_missing:
                continue
            raise SystemExit(f"{model}: no grades (grade it, or pass --skip-missing)")
        frames.append(pd.DataFrame([json.loads(line) for line in path.open()]).assign(model=model))
        present.append(model)
    if not frames:
        raise SystemExit("no graded model")
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = df["prompt"]
    short = e7.survival(df[df["channel"] == "short"])
    full = df[df["channel"] == "full"].copy()
    full["whole_trace_compliance"] = full["compliant"].fillna(False).astype(float)  # empty / no rule: not compliant
    full["accuracy"] = full["correct"].astype(float)
    return pd.concat([short, full], ignore_index=True), present


def question_draws() -> tuple[dict, np.ndarray]:
    """item_id -> column, and B x questions bootstrap counts (questions resampled within each source)."""
    items = e9.items()
    index = {it["item_id"]: k for k, it in enumerate(items)}
    rng = np.random.default_rng(BOOT_SEED)
    weights = np.zeros((N_BOOT, len(items)))
    for source in sorted({it["source"] for it in items}):
        cols = [k for k, it in enumerate(items) if it["source"] == source]
        weights[:, cols] = rng.multinomial(len(cols), np.ones(len(cols)) / len(cols), size=N_BOOT)
    return index, weights


def rule_draws(rules: list[str]) -> np.ndarray:
    """B x rules bootstrap counts; the generator is seeded by the pool's rules, so a pool's draw is reproducible."""
    rng = np.random.default_rng([BOOT_SEED, *[R.ALL_RULES.index(r) for r in rules]])
    return rng.multinomial(len(rules), np.ones(len(rules)) / len(rules), size=N_BOOT).astype(float)


# --- Cells ----------------------------------------------------------------------------------------------------------
def cell_values(df: pd.DataFrame) -> dict:
    """(metric, model, rule, arm) -> (point, B draws). Accuracy of the no-rule rows is keyed by rule no_constraint."""
    index, weights = question_draws()
    out = {}
    for (channel, model, mode, arm), c in df.groupby(["channel", "model", "mode", "arm"]):
        w = weights[:, c["item_id"].map(index).to_numpy()]
        if channel == "short":
            times, events = c["time"].to_numpy(), c["event"].to_numpy().astype(bool)
            point = kaplan_meier(times, events, np.ones((1, len(c))), T_VALUES)[0]
            draws = kaplan_meier(times, events, w, T_VALUES)
            out[("S_200", model, mode, arm)] = (100 * point[0], 100 * draws[:, 0])
            out[("S_1000", model, mode, arm)] = (100 * point[1], 100 * draws[:, 1])
        else:
            for metric in ("whole_trace_compliance", "accuracy"):
                y = c[metric].to_numpy()
                out[(metric, model, mode, arm)] = (100 * y.mean(), 100 * (w @ y) / w.sum(axis=1))
    return out


def pooled(values: dict, metric: str, arm: str, models: list[str], rules: list[str],
           resample_rules: bool) -> tuple[float, np.ndarray] | None:
    cells = [[values.get((metric, m, r, arm)) for r in rules] for m in models]
    if any(c is None for row in cells for c in row):
        return None
    point = float(np.mean([c[0] for row in cells for c in row]))
    rule_w = rule_draws(rules) if resample_rules else np.ones((N_BOOT, len(rules)))
    per_rule = np.stack([np.mean([cells[i][j][1] for i in range(len(models))], axis=0)
                         for j in range(len(rules))], axis=1)  # B x rules: mean over models
    return point, (per_rule * rule_w).sum(axis=1) / rule_w.sum(axis=1)


def summary(point: float, draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": round(point, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)]}


def contrast(values: dict, metric: str, a: str, b: str, models: list[str], rules: list[str]) -> dict | None:
    out = {"metric": metric, "arm": a, "minus": b, "models": models, "rules": rules}
    for resample_rules, key in ((True, "questions_and_rules"), (False, "questions_only")):
        pa = pooled(values, metric, a, models, rules, resample_rules)
        pb = pooled(values, metric, b, models, rules, resample_rules)
        if pa is None or pb is None:
            return None
        d = pa[1] - pb[1]
        out[key] = summary(pa[0] - pb[0], d) | {"p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
        out[f"{a}_{key}"], out[f"{b}_{key}"] = summary(*pa), summary(*pb)
    return out


# --- Analyses -------------------------------------------------------------------------------------------------------
def primary(values: dict, present: list[str]) -> dict:
    out = {}
    for name, (metric, a, b, models, rules) in PRIMARY.items():
        missing = [m for m in models if m not in present]
        out[name] = contrast(values, metric, a, b, models, rules) if not missing else {"missing_models": missing}
    p_values = {k: v["questions_and_rules"]["p"] for k, v in out.items() if "questions_and_rules" in v}
    for k, p in holm(p_values).items():
        out[k]["holm_p"] = round(p, 4)
        out[k]["supported"] = p < 0.05 and out[k]["questions_and_rules"]["ci95"][0] > 0
    return out


def secondary(df: pd.DataFrame, values: dict, present: list[str]) -> dict:
    design = [m for m in e9.DESIGN_MODELS if m in present]
    fresh = [m for m in e9.FRESH_MODELS if m in present]
    out = {"A_minus_baseline_by_kind": {}, "A_minus_baseline_by_kind_S200": {}, "per_model": {}}
    for kind in ("formatting", "insertion", "content"):
        rules = [r for r in R.ALL_RULES if R.RULE_KIND[r] == kind]
        out["A_minus_baseline_by_kind"][kind] = contrast(values, "S_1000", "A", "baseline", present, rules)
        # Added after the v1 analysis (descriptive, not pre-registered).
        out["A_minus_baseline_by_kind_S200"][kind] = contrast(values, "S_200", "A", "baseline", present, rules)
    for model in present:
        out["per_model"][model] = contrast(values, "S_1000", "A", "baseline", [model], R.ALL_RULES)
    if design and fresh:  # selection check: is A's gain larger on the models it was chosen on?
        gain = {}
        for group, models in (("design", design), ("fresh", fresh)):
            pa = pooled(values, "S_1000", "A", models, R.ALL_RULES, True)
            pb = pooled(values, "S_1000", "baseline", models, R.ALL_RULES, True)
            gain[group] = (pa[0] - pb[0], pa[1] - pb[1])
        out["design_gain_minus_fresh_gain"] = summary(gain["design"][0] - gain["fresh"][0],
                                                      gain["design"][1] - gain["fresh"][1])
    full_models = [m for m in e9.FULL_MODELS if m in present]
    if full_models:
        out["accuracy"] = {arm: summary(*pooled(values, "accuracy", arm, full_models, e9.FULL_RULES, False))
                           for arm in e9.FULL_ARMS}
        out["accuracy"]["no_rule"] = summary(*pooled(values, "accuracy", "no_rule", full_models,
                                                     [cfg.NO_CONSTRAINT], False))
        out["accuracy_A_minus_baseline"] = contrast(values, "accuracy", "A", "baseline", full_models, e9.FULL_RULES)
        out["whole_trace_by_rule"] = {r: contrast(values, "whole_trace_compliance", "A", "baseline", full_models, [r])
                                      for r in e9.FULL_RULES}
    short = df[df["channel"] == "short"]
    out["ways_to_look_better"] = {  # per arm, % of short rows
        arm: {"empty": round(100 * d["empty"].mean(), 2), "degenerate": round(100 * d["degenerate"].mean(), 2),
              "meta_regex": round(100 * d["meta_regex"].astype(float).mean(), 2),
              "ended_clean_before_200": round(100 * d["ended_clean_early"].mean(), 2)}
        for arm, d in short.groupby("arm")}
    full = df[df["channel"] == "full"]
    out["full_reasoning_tokens_median"] = {f"{arm}|{mode}": float(d["reasoning_tokens"].median())
                                           for (arm, mode), d in full.groupby(["arm", "mode"])}
    return out


def cells_table(values: dict) -> pd.DataFrame:
    rows = []
    for (metric, model, rule, arm), (point, draws) in values.items():
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"metric": metric, "model": model, "rule": rule, "arm": arm, "kind": R.RULE_KIND.get(rule, ""),
                     "model_group": "design" if model in e9.DESIGN_MODELS else "fresh",
                     "value": round(point, 2), "ci95_questions_lo": round(float(lo), 2),
                     "ci95_questions_hi": round(float(hi), 2)})
    return pd.DataFrame(rows).sort_values(["metric", "model", "rule", "arm"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--skip-missing", action="store_true", help="analyse the models graded so far")
    args = parser.parse_args()
    out_dir = e9.EXP.results / "analysis" / args.run
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; analysis runs are never overwritten")
    df, present = load(args.skip_missing)
    values = cell_values(df)
    result = {"models_present": present, "complete": sorted(present) == sorted(e9.MODELS),
              "primary": primary(values, present)}
    out_dir.mkdir(parents=True)
    (out_dir / "primary.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "secondary.json").write_text(json.dumps(secondary(df, values, present), indent=2) + "\n")
    cells_table(values).to_csv(out_dir / "cells.csv", index=False)
    for name, r in result["primary"].items():
        q = r.get("questions_and_rules")
        print(name, PRIMARY[name][:3], q and f"{q['value']:+.1f} [{q['ci95'][0]:+.1f}, {q['ci95'][1]:+.1f}] "
              f"holm p={r.get('holm_p')}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
