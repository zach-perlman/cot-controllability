"""exp07b analysis: each candidate minus upgraded, S(1000) points, pre-registered win rule
(results/exp07b_prompt_compare/manifest.json):

  a candidate's score    mean over the model x rule cells where it ran of S_candidate - S_upgraded
  test                   two-sided paired bootstrap over questions (exp07's: 2000 draws, one set of question weights
                         shared by every cell), Holm-adjusted over the 5 candidates
  win                    difference > 0 and Holm p < 0.05; if several win, the largest difference; else upgraded

Per model and per rule differences, and the guard flags, are descriptive. A winner whose meta-regex or
ended-clean-early share is more than GUARD_FLAG points above upgraded's (same cells) is flagged for review.

Run: /venv/main/bin/python scripts/cc_exp07b_analysis.py --run NAME [--skip-missing]
Writes results/exp07b_prompt_compare/analysis/<run>/ and figures/exp07b_prompt_compare/<run>/ (never overwritten).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cc_exp04_analysis as a4
import cc_exp07 as e7
import cc_exp07b as e
import exp07_rules as rules
import exp07b_candidates as b
from cc_exp07_analysis import BOOT_SEED, GUARDS, N_BOOT, cell_draws, stat
from cc_survival import holm

ROOT = Path(__file__).resolve().parent.parent
T_STAR = e7.T_STAR
CANDIDATES = list(b.LETTER)
ALPHA = 0.05
GUARD_FLAG = 5.0  # points
COLORS = {"upgraded": "#B5532A", "many_examples": "#0072B2", "own_compliant": "#009E73", "named_once": "#CC79A7",
          "own_guide": "#E69F00", "monitored": "#56B4E9"}


def load(skip_missing: bool) -> tuple[pd.DataFrame, list[str]]:
    frames, missing = [], []
    for model in e.MODELS:
        try:
            path = e.grades_path(model)
        except SystemExit:
            path = None
        if path is None or not path.exists():
            if not skip_missing:
                raise SystemExit(f"{model}: no grades yet")
            missing.append(model)
            continue
        frames.append(pd.DataFrame([json.loads(line) for line in path.open()]).assign(model=model))
    return e7.survival(pd.concat(frames, ignore_index=True)), missing


def difference(draws: dict, cand: str, cells: list[tuple[str, str]]) -> dict:
    """Mean over cells of S_cand - S_upgraded, its 95% CI and two-sided bootstrap p."""
    point = float(np.mean([draws[(cand, m, r)][0] - draws[("upgraded", m, r)][0] for m, r in cells]))
    d = np.mean([draws[(cand, m, r)][1] - draws[("upgraded", m, r)][1] for m, r in cells], axis=0)
    return {**stat(point, d), "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))),
            "n_cells": len(cells)}


def cells_of(draws: dict, cand: str, models: list[str]) -> list[tuple[str, str]]:
    return [(m, r) for m in models for r in rules.ALL_RULES if (cand, m, r) in draws and ("upgraded", m, r) in draws]


def compare(draws: dict, models: list[str]) -> dict:
    out = {}
    for cand in CANDIDATES:
        cells = cells_of(draws, cand, models)
        if not cells:
            continue
        out[cand] = {"letter": b.LETTER[cand], "cells": [f"{m}|{r}" for m, r in cells],
                     "primary": difference(draws, cand, cells),
                     "per_model": {m: difference(draws, cand, [c for c in cells if c[0] == m])
                                   for m in models if any(c[0] == m for c in cells)},
                     "per_rule": {r: difference(draws, cand, [c for c in cells if c[1] == r])
                                  for r in rules.ALL_RULES if any(c[1] == r for c in cells)}}
    adjusted = holm({c: v["primary"]["p_two_sided"] for c, v in out.items()})
    for c, v in out.items():
        v["primary"]["p_holm"] = adjusted[c]
    return out


def winner_of(results: dict) -> str:
    wins = {c: v["primary"]["value"] for c, v in results.items()
            if v["primary"]["value"] > 0 and v["primary"]["p_holm"] < ALPHA}
    return max(wins, key=wins.get) if wins else "upgraded"


def guards(df: pd.DataFrame, results: dict) -> list[dict]:
    """Per candidate: guard shares (%) for it and for upgraded on the cells where it ran."""
    d = df.assign(cell=df["model"] + "|" + df["mode"])
    d[GUARDS] = d[GUARDS].astype(float)
    out = []
    for cand, v in results.items():
        on = d[d["cell"].isin(v["cells"])]
        for arm in (cand, "upgraded"):
            shares = 100 * on.loc[on["prompt"] == arm, GUARDS].mean()
            out.append({"candidate": cand, "arm": arm, **shares.round(1).to_dict()})
    return out


def guard_flags(guard_rows: list[dict], cand: str) -> list[str]:
    if cand == "upgraded":
        return []
    rows = {g["arm"]: g for g in guard_rows if g["candidate"] == cand}
    return [k for k in ("meta_regex", "ended_clean_early") if rows[cand][k] - rows["upgraded"][k] > GUARD_FLAG]


# --- Figures --------------------------------------------------------------------------------------------------------
def fig_differences(results: dict, models: list[str], fig_dir: Path) -> str:
    """B1. Each candidate - upgraded: the primary mean (large marker, 95% CI) and per model (small markers)."""
    fig = go.Figure()
    labels = [f"{b.LETTER[c]}: {c}" for c in results]
    prim = [v["primary"] for v in results.values()]
    fig.add_trace(go.Scatter(
        x=[p["value"] for p in prim], y=labels, mode="markers", name="mean over its cells",
        marker={"color": [COLORS[c] for c in results], "size": 13},
        error_x={"type": "data", "symmetric": False, "thickness": 1.5, "color": "#444444",
                 "array": [p["ci"][1] - p["value"] for p in prim], "arrayminus": [p["value"] - p["ci"][0] for p in prim]},
        text=[f"Holm p {p['p_holm']:.3f}" for p in prim], hovertemplate="%{y}: %{x:+.1f} (%{text})<extra></extra>"))
    for k, (model, symbol) in enumerate(zip(models, ("circle-open", "square-open", "diamond-open"))):
        pts = [(lab, v["per_model"][model]["value"]) for lab, v in zip(labels, results.values())
               if model in v["per_model"]]
        fig.add_trace(go.Scatter(x=[x for _, x in pts], y=[lab for lab, _ in pts], mode="markers", name=model,
                                 marker={"symbol": symbol, "size": 9, "color": "#666666"},
                                 hovertemplate=f"{model}: %{{x:+.1f}}<extra></extra>"))
    fig.add_vline(x=0, line={"color": "#999999", "width": 1})
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text=f"S({T_STAR}) candidate minus upgraded, points")
    fig.update_layout(title=f"B1. exp07b: one change to the upgraded prompt, on {len(models)} models and 90 new "
                            f"questions<br><sup>S({T_STAR}): share of reasoning texts with no violation through "
                            f"{T_STAR} tokens (Kaplan-Meier). Large marker: mean over the model x rule cells where the "
                            "candidate ran, 95% CI from a bootstrap over questions. Open markers: per model.</sup>",
                      legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center"},
                      margin={"l": 200, "t": 100, "b": 110})
    return a4.save(fig, fig_dir, "B1_differences", 1300, 520)


def fig_per_rule(draws: dict, models: list[str], fig_dir: Path) -> str:
    """B2. S(1000) per rule (x) and model (panel), one marker per arm."""
    arms = ["upgraded"] + CANDIDATES
    offsets = np.linspace(-0.33, 0.33, len(arms))
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.02, subplot_titles=models)
    for c, model in enumerate(models, 1):
        for offset, arm in zip(offsets, arms):
            pts = [(i, draws[(arm, model, r)]) for i, r in enumerate(rules.ALL_RULES) if (arm, model, r) in draws]
            fig.add_trace(go.Scatter(
                x=[i + offset for i, _ in pts], y=[p for _, (p, _) in pts], mode="markers",
                name=arm if arm == "upgraded" else f"{b.LETTER[arm]}: {arm}", legendgroup=arm, showlegend=c == 1,
                marker={"color": COLORS[arm], "size": 8},
                error_y={"type": "data", "symmetric": False, "color": COLORS[arm], "thickness": 1,
                         "array": [np.percentile(d, 97.5) - p for _, (p, d) in pts],
                         "arrayminus": [p - np.percentile(d, 2.5) for _, (p, d) in pts]}), row=1, col=c)
        fig.update_xaxes(tickangle=60, tickvals=list(range(len(rules.ALL_RULES))), ticktext=rules.ALL_RULES,
                         range=[-0.6, len(rules.ALL_RULES) - 0.4], row=1, col=c)
    fig.update_yaxes(range=[-3, 103], title_text=f"S({T_STAR}), %", row=1, col=1)
    fig.update_layout(title=f"B2. S({T_STAR}) per rule and model, every arm",
                      legend={"orientation": "h", "y": -0.45, "x": 0.5, "xanchor": "center"},
                      margin={"t": 90, "b": 190})
    return a4.save(fig, fig_dir, "B2_per_rule", 1900, 620)


# --- Report ---------------------------------------------------------------------------------------------------------
def fmt(s: dict) -> str:
    return f"{s['value']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]"


def report(summary: dict) -> str:
    winner = summary["winner"]
    lines = [f"# exp07b analysis: {summary['run']} (UNVERIFIED until a human adds it to VERIFIED.md)", "",
             f"Models: {', '.join(summary['models'])}."
             + (f" Missing: {', '.join(summary['missing'])}." if summary["missing"] else ""), "",
             f"**Winner (pre-registered rule): `{winner}`**"
             + (f" (guard flags for review: {', '.join(summary['winner_guard_flags'])})"
                if summary["winner_guard_flags"] else ""), "",
             f"## Candidate - upgraded (S({T_STAR}) points, 95% CI over questions)", "",
             "| candidate | cells | difference | p | Holm p | " + " | ".join(summary["models"]) + " |",
             "|---|---|---|---|---|" + "---|" * len(summary["models"])]
    for c, v in summary["results"].items():
        p = v["primary"]
        per_model = [fmt(v["per_model"][m]) if m in v["per_model"] else "not run" for m in summary["models"]]
        lines.append(f"| {v['letter']}: {c} | {p['n_cells']} | {fmt(p)} | {p['p_two_sided']:.3f} | "
                     f"{p['p_holm']:.3f} | " + " | ".join(per_model) + " |")
    lines += ["", "## Per rule (mean over models where it ran)", "",
              "| candidate | " + " | ".join(rules.ALL_RULES) + " |", "|---|" + "---|" * len(rules.ALL_RULES)]
    for c, v in summary["results"].items():
        lines.append(f"| {v['letter']} | " + " | ".join(f"{v['per_rule'][r]['value']:+.1f}" if r in v["per_rule"]
                                                         else "-" for r in rules.ALL_RULES) + " |")
    lines += ["", "## Guards (% of texts, on the candidate's cells)", "",
              "| candidate | arm | " + " | ".join(GUARDS) + " |", "|---|---|" + "---|" * len(GUARDS)]
    for g in summary["guards"]:
        lines.append(f"| {g['candidate']} | {g['arm']} | " + " | ".join(f"{g[k]:.1f}" for k in GUARDS) + " |")
    lines += ["", "## Figures", ""] + [f"- [{f}](../../../../figures/exp07b_prompt_compare/{summary['run']}/{f}.html)"
                                       for f in summary["figures"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="analysis run name")
    parser.add_argument("--skip-missing", action="store_true", help="analyse the models graded so far")
    args = parser.parse_args()
    out_dir = e.EXP.results / "analysis" / args.run
    fig_dir = e.EXP.figure_dir(args.run)
    for d in (out_dir, fig_dir):
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")
    df, missing = load(args.skip_missing)
    models = [m for m in e.MODELS if m not in missing]
    draws = cell_draws(df)
    results = compare(draws, models)
    winner = winner_of(results)
    guard_rows = guards(df, results)
    out_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)
    pd.DataFrame([{"prompt": a, "model": m, "mode": r, "S_1000": p, "ci_low": np.percentile(d, 2.5),
                   "ci_high": np.percentile(d, 97.5)} for (a, m, r), (p, d) in draws.items()]
                 ).round(2).to_csv(out_dir / "per_cell.csv", index=False)
    summary = {"exp_id": e.EXP.exp_id, "run": args.run, "models": models, "missing": missing, "t_star": T_STAR,
               "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "alpha": ALPHA,
               "n_questions": int(df["item_id"].nunique()), "results": results, "winner": winner,
               "winner_guard_flags": guard_flags(guard_rows, winner), "guards": guard_rows,
               "figures": [fig_differences(results, models, fig_dir), fig_per_rule(draws, models, fig_dir)]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "REPORT_auto.md").write_text(report(summary))
    print(f"wrote {out_dir}")
    print(report(summary))


if __name__ == "__main__":
    main()
