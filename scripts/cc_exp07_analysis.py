"""exp07 test analysis (results/exp07_prompt_search/manifest.json, "test").

Arms: baseline (CoT-Control's prompt), upgraded (exp06's), the validation winner. Metric: S(1000) per (arm, model,
rule) cell, as in the search (cc_exp07.survival: an empty or degenerate text is a violation at token 0); a group's
score is the mean over its model x rule cells. 95% intervals: one bootstrap over test questions (2000 resamples),
the same resampled questions for every arm, rule and model.

Primary: winner - upgraded on the held-out rules except multiple_word_suppression (alternating_case, no_digits,
no_first_person) x the 4 held-out models. Secondary: the same on the other rule x model groups; upgraded - baseline
(does exp06's upgrade generalise?); multiple_word_suppression alone; each contrast per rule and per model.

Outputs (never overwritten): results/exp07_prompt_search/analysis/<run>/{summary.json, per_cell.csv, REPORT_auto.md},
figures/exp07_prompt_search/<run>/{T1_contrasts, T2_per_rule}.{html,png}.
Run: /venv/main/bin/python scripts/cc_exp07_analysis.py --run NAME [--skip-missing]
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
import cc_exp07 as e
import exp07_rules as rules
from cc_survival import kaplan_meier

ROOT = Path(__file__).resolve().parent.parent
N_BOOT, BOOT_SEED = 2000, 20261002
T_STAR = e.T_STAR
HELDOUT_PRIMARY_RULES = ["alternating_case", "no_digits", "no_first_person"]
NEAR_DUPLICATE_RULE = "multiple_word_suppression"
GROUPS = {  # name: (rules, models)
    "held-out rules x held-out models (primary)": (HELDOUT_PRIMARY_RULES, e.HELDOUT_MODELS),
    "search rules x held-out models": (rules.SEARCH_RULES, e.HELDOUT_MODELS),
    "held-out rules x search models": (HELDOUT_PRIMARY_RULES, e.SEARCH_MODELS),
    "search rules x search models": (rules.SEARCH_RULES, e.SEARCH_MODELS),
    "multiple_word_suppression x held-out models": ([NEAR_DUPLICATE_RULE], e.HELDOUT_MODELS),
    "multiple_word_suppression x search models": ([NEAR_DUPLICATE_RULE], e.SEARCH_MODELS),
}
GUARDS = ["clean_200", "ended_clean_early", "empty", "degenerate", "meta_regex"]
ARM_COLOR = {"baseline": "#999999", "upgraded": "#B5532A"}
WINNER_COLOR = "#0072B2"


def load(skip_missing: bool) -> tuple[pd.DataFrame, list[str]]:
    frames, missing = [], []
    for model in e.MODELS:
        try:
            path = e.grades_path(e.generation_path("test", model))
        except SystemExit:
            path = None
        if path is None or not path.exists():
            missing.append(model)
            continue
        frames.append(pd.DataFrame([json.loads(line) for line in path.open()]).assign(model=model))
    if missing and not skip_missing:
        raise SystemExit(f"no test grades for {missing} (use --skip-missing); nothing written")
    if not frames:
        raise SystemExit("no test grades yet")
    return e.survival(pd.concat(frames, ignore_index=True)), missing


def cell_draws(df: pd.DataFrame) -> dict:
    """(arm, model, rule) -> (point S(1000), bootstrap draws), every cell with the same question weights."""
    items = sorted(df["item_id"].unique())
    index = {it: i for i, it in enumerate(items)}
    weights = np.random.default_rng(BOOT_SEED).multinomial(len(items), np.ones(len(items)) / len(items),
                                                           size=N_BOOT).astype(float)
    out = {}
    for (arm, model, mode), c in df.groupby(["prompt", "model", "mode"]):
        times, events = c["time"].to_numpy(), c["event"].to_numpy().astype(bool)
        w = weights[:, c["item_id"].map(index).to_numpy()]
        point = kaplan_meier(times, events, np.ones((1, len(c))), np.array([T_STAR]))[0, 0]
        out[(arm, model, mode)] = (100 * point, 100 * kaplan_meier(times, events, w, np.array([T_STAR]))[:, 0])
    return out


def stat(point: float, draws: np.ndarray) -> dict:
    return {"value": float(point), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def group_score(draws: dict, arm: str, group_rules: list[str], models: list[str]) -> tuple[float, np.ndarray] | None:
    cells = [draws[(arm, m, r)] for m in models for r in group_rules if (arm, m, r) in draws]
    if len(cells) != len(models) * len(group_rules):
        return None
    return float(np.mean([p for p, _ in cells])), np.mean([d for _, d in cells], axis=0)


def contrast(draws: dict, a: str, b: str, group_rules: list[str], models: list[str]) -> dict | None:
    sa, sb = group_score(draws, a, group_rules, models), group_score(draws, b, group_rules, models)
    if sa is None or sb is None:
        return None
    return {"a": stat(*sa), "b": stat(*sb), "difference": stat(sa[0] - sb[0], sa[1] - sb[1]),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(sa[1] - sb[1] <= 0), np.mean(sa[1] - sb[1] >= 0))))}


def contrasts(draws: dict, winner: str, models: list[str]) -> list[dict]:
    pairs = [(winner, "upgraded"), ("upgraded", "baseline")]
    out = []
    for name, (group_rules, group_models) in GROUPS.items():
        present = [m for m in group_models if m in models]
        for a, b in pairs:
            c = contrast(draws, a, b, group_rules, present) if present else None
            if c:
                out.append({"group": name, "contrast": f"{a} - {b}", "models": present, **c})
    for a, b in pairs:  # per rule (all present models) and per model (all rules)
        for r in rules.ALL_RULES:
            c = contrast(draws, a, b, [r], models)
            if c:
                out.append({"group": f"rule: {r}", "contrast": f"{a} - {b}", "models": models, **c})
        for m in models:
            c = contrast(draws, a, b, rules.ALL_RULES, [m])
            if c:
                out.append({"group": f"model: {m}", "contrast": f"{a} - {b}", "models": [m], **c})
    return out


def guard_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm and rule set: shares (%) of the flags an arm could win by."""
    d = df.assign(rule_set=np.where(df["mode"].isin(rules.SEARCH_RULES), "search rules", "held-out rules"))
    d[GUARDS] = d[GUARDS].astype(float)
    return (100 * d.groupby(["prompt", "rule_set"])[GUARDS].mean()).round(1).reset_index()


# --- Figures --------------------------------------------------------------------------------------------------------
def fig_contrasts(rows: list[dict], winner: str, fig_dir: Path) -> str:
    """T1. The group contrasts with 95% CIs: winner - upgraded (left) and upgraded - baseline (right)."""
    groups = [g for g in GROUPS]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[f"{winner} - upgraded (did the search help?)",
                                        "upgraded - baseline (does exp06's upgrade generalise?)"])
    for col, (name, color) in enumerate(((f"{winner} - upgraded", WINNER_COLOR),
                                         ("upgraded - baseline", ARM_COLOR["upgraded"])), 1):
        pts = [next((r for r in rows if r["group"] == g and r["contrast"] == name), None) for g in groups]
        y = [g for g, p in zip(groups, pts) if p]
        s = [p["difference"] for p in pts if p]
        fig.add_trace(go.Scatter(x=[v["value"] for v in s], y=y, mode="markers", showlegend=False,
                                 marker={"color": color, "size": 11},
                                 error_x={"type": "data", "symmetric": False, "color": color, "thickness": 1.5,
                                          "array": [v["ci"][1] - v["value"] for v in s],
                                          "arrayminus": [v["value"] - v["ci"][0] for v in s]},
                                 hovertemplate="%{y}: %{x:+.1f}<extra></extra>"), row=1, col=col)
        fig.add_vline(x=0, line={"color": "#999999", "width": 1}, row=1, col=col)
        fig.update_xaxes(title_text="difference in S(1000), points", row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="T1. exp07 test: the search winner vs the upgraded prompt, and the upgraded prompt vs "
                            "CoT-Control's, on 160 unseen questions"
                            f"<br><sup>S({T_STAR}): share of reasoning texts with no violation through {T_STAR} "
                            "tokens (Kaplan-Meier; an empty or degenerate text is a violation at token 0), mean over "
                            "the group's model x rule cells; 95% CIs from one bootstrap over questions. Top row: the "
                            "pre-registered primary contrast.</sup>", margin={"l": 330, "t": 110})
    return a4.save(fig, fig_dir, "T1_contrasts", 1500, 520)


def fig_per_rule(draws: dict, winner: str, models: list[str], fig_dir: Path) -> str:
    """T2. S(1000) per rule (x) and model (panel), one marker per arm."""
    arms = [("baseline", ARM_COLOR["baseline"]), ("upgraded", ARM_COLOR["upgraded"]), (winner, WINNER_COLOR)]
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.015,
                        subplot_titles=[m + (" (held out)" if m in e.HELDOUT_MODELS else "") for m in models])
    for c, model in enumerate(models, 1):
        for offset, (arm, color) in zip((-0.22, 0, 0.22), arms):
            pts = [(i, draws[(arm, model, r)]) for i, r in enumerate(rules.ALL_RULES) if (arm, model, r) in draws]
            fig.add_trace(go.Scatter(
                x=[i + offset for i, _ in pts], y=[p for _, (p, _) in pts], mode="markers", name=arm,
                legendgroup=arm,
                showlegend=c == 1, marker={"color": color, "size": 9},
                error_y={"type": "data", "symmetric": False, "color": color, "thickness": 1,
                         "array": [np.percentile(d, 97.5) - p for _, (p, d) in pts],
                         "arrayminus": [p - np.percentile(d, 2.5) for _, (p, d) in pts]}), row=1, col=c)
        fig.add_vrect(x0=len(rules.SEARCH_RULES) - 0.5, x1=len(rules.ALL_RULES) - 0.5, fillcolor="#eeeeee",
                      line_width=0, layer="below", row=1, col=c)
        fig.update_xaxes(tickangle=60, tickvals=list(range(len(rules.ALL_RULES))), ticktext=rules.ALL_RULES,
                         range=[-0.6, len(rules.ALL_RULES) - 0.4], row=1, col=c)
    fig.update_yaxes(range=[-3, 103], title_text=f"S({T_STAR}), %", row=1, col=1)
    fig.update_layout(title=f"T2. S({T_STAR}) per rule and model (shaded: held-out rules)",
                      legend={"orientation": "h", "y": -0.45, "x": 0.5, "xanchor": "center"},
                      margin={"t": 90, "b": 190})
    return a4.save(fig, fig_dir, "T2_per_rule", 1900, 620)


# --- Report ---------------------------------------------------------------------------------------------------------
def fmt(s: dict) -> str:
    return f"{s['value']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]"


def report(summary: dict) -> str:
    lines = [f"# exp07 test analysis: {summary['run']} (UNVERIFIED until a human adds it to VERIFIED.md)", "",
             f"Winner (validation): `{summary['winner']}`. Models: {', '.join(summary['models'])}."
             + (f" Missing: {', '.join(summary['missing'])}." if summary["missing"] else ""), "",
             f"## Contrasts (S({T_STAR}) points, 95% CI over questions)", "",
             "| group | contrast | a | b | difference | p |", "|---|---|---|---|---|---|"]
    for r in summary["contrasts"]:
        lines.append(f"| {r['group']} | {r['contrast']} | {r['a']['value']:.1f} | {r['b']['value']:.1f} | "
                     f"{fmt(r['difference'])} | {r['p_two_sided']:.3f} |")
    lines += ["", "## Guards (% of texts)", "", "| arm | rules | " + " | ".join(GUARDS) + " |",
              "|---|---|" + "---|" * len(GUARDS)]
    for g in summary["guards"]:
        lines.append(f"| {g['prompt']} | {g['rule_set']} | " + " | ".join(f"{g[k]:.1f}" for k in GUARDS) + " |")
    lines += ["", "## Figures", ""] + [f"- [{f}](../../../../figures/exp07_prompt_search/{summary['run']}/{f}.html)"
                                       for f in summary["figures"]]
    return "\n".join(lines) + "\n"


def winner_name() -> str:
    arms = {json.loads(line)["prompt"] for line in e.requests_path("test", e.HELDOUT_MODELS[0]).open()}
    others = sorted(arms - {"baseline", "upgraded"})
    return others[0] if others else "upgraded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="analysis run name")
    parser.add_argument("--skip-missing", action="store_true", help="analyse the models graded so far")
    args = parser.parse_args()
    out_dir = ROOT / "results/exp07_prompt_search/analysis" / args.run
    fig_dir = ROOT / "figures/exp07_prompt_search" / args.run
    for d in (out_dir, fig_dir):
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")
    df, missing = load(args.skip_missing)
    winner = winner_name()
    models = [m for m in e.MODELS if m not in missing]
    draws = cell_draws(df)
    out_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)
    rows = contrasts(draws, winner, models)
    per_cell = pd.DataFrame([{"prompt": a, "model": m, "mode": r, "S_1000": p, "ci_low": np.percentile(d, 2.5),
                              "ci_high": np.percentile(d, 97.5)} for (a, m, r), (p, d) in draws.items()])
    per_cell.round(2).to_csv(out_dir / "per_cell.csv", index=False)
    summary = {"exp_id": "exp07_prompt_search", "run": args.run, "winner": winner, "models": models,
               "missing": missing, "t_star": T_STAR, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
               "n_questions": int(df["item_id"].nunique()), "contrasts": rows,
               "guards": guard_table(df).to_dict("records"),
               "figures": [fig_contrasts(rows, winner, fig_dir), fig_per_rule(draws, winner, models, fig_dir)]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "REPORT_auto.md").write_text(report(summary))
    print(f"wrote {out_dir}")
    print(report(summary))


if __name__ == "__main__":
    main()
