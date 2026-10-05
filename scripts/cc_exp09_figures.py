"""Figures for exp09 (results/exp09_final_test/manifest.json, "secondary_descriptive"), from the data that exists:
exp09's main run (all 7 models) and extension 3's thinking-on parts (examples only, CoT necessity).

  F1_primary              P1-P3: A and CoT-Control's prompt, and A - baseline, with 95% CIs (questions and rules)
  F2_heatmap_S1000        per model x rule S(1000), both arms and A - baseline
  F3_heatmap_S200         the same at S(200)
  F4_design_vs_fresh      A - baseline per model, design models vs fresh ones (the selection check)
  F5_by_rule_kind         A - baseline by rule kind (formatting, insertion, content)
  F6_accuracy_and_length  accuracy per arm (full rows) and reasoning length per model and arm
  F7_ways_to_look_better  empty, degenerate and meta-regex rates per model and arm (short rows)
  F8_length_matched       added check (not pre-registered): whole-trace compliance and accuracy by reasoning length
  F9_examples_only        extension 3: A - examples only and examples only - CoT-Control's prompt per model (thinking on)
  F10_necessity           extension 3: A - baseline and A - examples only within CoT-necessity labels

Estimates (levels, differences, CIs) are read from the analysis outputs (analysis/<run>/primary.json, secondary.json,
cells.csv; analysis_ext3/<ext3 run>/summary.json). Rates, length distributions and length bins (F6 right, F7, F8)
are counted from the graded rows. No question text appears.

Outputs (never overwritten): figures/exp09_final_test/<run>/F*.{png,html}.
Run: /venv/main/bin/python scripts/cc_exp09_figures.py --run NAME [--analysis-run v1] [--ext3-run thinking_on_v1]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cc_config as cfg
import cc_exp07 as e7
import cc_exp09 as e9
import exp09_rules as R

FIG_ROOT = cfg.REPO_ROOT / "figures" / "exp09_final_test"
MODELS = e9.DESIGN_MODELS + e9.FRESH_MODELS
SHORT = {"Qwen3.8-27B-FP8": "Qwen3.8-27B", "Gemma-4-31B-FP8": "Gemma-4-31B", "Qwen3-32B": "Qwen3-32B",
         "Qwen3.6-27B-FP8": "Qwen3.6-27B", "Qwen3.6-35B-A3B-FP8": "Qwen3.6-35B-A3B",
         "GLM-4.7-Flash-FP8": "GLM-4.7-Flash", "Gemma-4-12B-FP8": "Gemma-4-12B"}
RULE_LABEL = {"lowercase_thinking": "lowercase", "uppercase_thinking": "UPPERCASE", "meow_between_words": "meow",
              "end_of_sentence": "end of sentence", "word_suppression": "word suppression",
              "alternating_case": "aLtErNaTiNg", "multiple_word_suppression": "multi-word suppr.",
              "no_digits": "no digits", "no_first_person": "no first person", "no_commas": "no commas*",
              "now_sentence_start": "'Now' starts*", "no_question_words": "no question words*"}
ARM = {"A": ("Prompt A", "#B15928"), "baseline": ("CoT-Control's prompt", "#9E9E9E"),
       "no_rule": ("No rule", "#000000"), "examples_only": ("A's examples, no rule", "#1F78B4")}
GROUP_COLOR = {"design": "#6A3D9A", "fresh": "#33A02C"}
FOOT = ("exp09: 120 never-used questions, 7 models (design: chose A; fresh: never used to choose), "
        "12 rules (*new in exp09). UNVERIFIED.")


# --- Helpers --------------------------------------------------------------------------------------------------------
def save(fig: go.Figure, fig_dir, name: str, width: int, height: int, top: int = 120, bottom: int = 120) -> str:
    """Title in the top margin; legend and footer in the bottom margin, below the x-axis titles."""
    plot_height = height - top - bottom
    below_plot = lambda px: -px / plot_height  # paper y for a point px pixels below the plot area
    fig.update_layout(width=width, height=height, template="plotly_white", font=dict(size=13),
                      margin=dict(t=top, b=bottom, l=70, r=60),
                      title=dict(x=0.01, xanchor="left", y=1 - 30 / height, yanchor="top", yref="container"),
                      legend=dict(orientation="h", x=0, xanchor="left", y=below_plot(bottom - 40), yanchor="bottom"))
    fig.add_annotation(text=FOOT, xref="paper", yref="paper", x=0, y=below_plot(bottom - 10), showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=10, color="#666"))
    fig.write_html(fig_dir / f"{name}.html", include_plotlyjs="cdn")
    fig.write_image(fig_dir / f"{name}.png", scale=2)
    return name


def title(main: str, sub: str) -> str:
    return f"<b>{main}</b><br><span style='font-size:12px;color:#555'>{sub}</span>"


def err(ci: list[float], value: float) -> dict:
    return dict(type="data", symmetric=False, array=[ci[1] - value], arrayminus=[value - ci[0]], thickness=1.5)


def fmt(v: dict, key: str = "ci95") -> str:
    return f"{v['value']:+.1f} [{v[key][0]:+.1f}, {v[key][1]:+.1f}]"


def load_grades() -> pd.DataFrame:
    frames = [pd.DataFrame([json.loads(line) for line in e9.grades_path(m).open()]).assign(model=m) for m in MODELS]
    return e7.survival(pd.concat(frames, ignore_index=True))


# --- Figures from the analysis outputs ------------------------------------------------------------------------------
def fig_primary(primary: dict, fig_dir) -> str:
    names = {"P1": "P1: fresh models x 12 rules<br>S(1000)", "P2": "P2: 7 models x 3 new rules<br>S(1000)",
             "P3": "P3: 7 models x 2 rules<br>whole trace complies"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=[names[p] for p in names], horizontal_spacing=0.08)
    for col, p in enumerate(names, start=1):
        c = primary["primary"][p]
        for arm in ("baseline", "A"):
            v = c[f"{arm}_questions_and_rules"]
            fig.add_trace(go.Bar(x=[ARM[arm][0]], y=[v["value"]], marker_color=ARM[arm][1], error_y=err(v["ci95"],
                          v["value"]), showlegend=False, text=[f"{v['value']:.1f}"], textposition="outside"),
                          row=1, col=col)
        d = c["questions_and_rules"]
        axis = "" if col == 1 else col
        fig.add_annotation(text=f"A - baseline {fmt(d)}<br>Holm p {c['holm_p']:.3f}", xref=f"x{axis} domain",
                           yref=f"y{axis} domain", x=0.5, y=0.97, showarrow=False, font=dict(size=12))
        fig.update_yaxes(range=[0, 60], title_text="% of traces" if col == 1 else None, row=1, col=col)
    fig.update_layout(title=title("exp09's pre-registered test: prompt A beats CoT-Control's prompt on all three",
                                  "Pooled over model x rule cells; 95% CIs resample questions (by source) and rules"))
    return save(fig, fig_dir, "F1_primary", 1100, 560, top=150, bottom=80)


def fig_heatmap(cells: pd.DataFrame, metric: str, fig_dir, name: str) -> str:
    d = cells[cells["metric"] == metric]
    rules = R.ALL_RULES
    panels = {}
    for arm in ("baseline", "A"):
        panels[arm] = d[d["arm"] == arm].pivot(index="model", columns="rule", values="value").loc[MODELS, rules]
    panels["diff"] = panels["A"] - panels["baseline"]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06, shared_yaxes=True,
                        subplot_titles=[ARM["baseline"][0], ARM["A"][0], "A - baseline"])
    ylabels = [f"{SHORT[m]} ({'design' if m in e9.DESIGN_MODELS else 'fresh'})" for m in MODELS]
    xlabels = [RULE_LABEL[r] for r in rules]
    for col, key in enumerate(("baseline", "A", "diff"), start=1):
        z = panels[key].to_numpy()
        diff = key == "diff"
        fig.add_trace(go.Heatmap(
            z=z, x=xlabels, y=ylabels, zmin=-60 if diff else 0, zmax=60 if diff else 100,
            colorscale="RdBu" if diff else "Viridis", zmid=0 if diff else None, text=np.round(z).astype(int),
            texttemplate="%{text}", textfont=dict(size=10), showscale=col in (2, 3),
            colorbar=dict(x=1.005 if diff else 0.652, xanchor="left", len=0.9, thickness=10,
                          title="pp" if diff else "%")),
            row=1, col=col)
        fig.update_xaxes(tickangle=-50, row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    t = metric.replace("S_", "S(") + ")"
    fig.update_layout(title=title(f"{t} per model and rule: every cell shown, not only the pooled number",
                                  f"{t} = share of reasoning traces still following the rule at token "
                                  f"{t[2:-1]} (Kaplan-Meier; empty or degenerate = violation at 0)"))
    return save(fig, fig_dir, name, 1500, 640, bottom=170)


def fig_design_vs_fresh(secondary: dict, fig_dir) -> str:
    fig = go.Figure()
    for m in MODELS:
        v = secondary["per_model"][m]["questions_and_rules"]
        group = "design" if m in e9.DESIGN_MODELS else "fresh"
        fig.add_trace(go.Scatter(x=[v["value"]], y=[SHORT[m]], mode="markers", marker=dict(size=12,
                      color=GROUP_COLOR[group]), error_x=err(v["ci95"], v["value"]), name=group,
                      legendgroup=group, showlegend=m in (e9.DESIGN_MODELS[0], e9.FRESH_MODELS[0])))
    fig.add_vline(x=0, line_color="#999")
    g = secondary["design_gain_minus_fresh_gain"]
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="A - baseline, S(1000), mean over 12 rules (pp)")
    fig.update_layout(title=title("A gains most on the models it was chosen on",
                                  f"Design-model gain minus fresh-model gain: {fmt(g)} pp. 95% CIs resample "
                                  "questions and rules"))
    return save(fig, fig_dir, "F4_design_vs_fresh", 900, 560, top=100)


BY_KIND_KEY = {"S_1000": "A_minus_baseline_by_kind", "S_200": "A_minus_baseline_by_kind_S200"}


def fig_by_kind(secondary: dict, fig_dir) -> str:
    """One row per metric: S(1000) (pre-registered) on top, S(200) (added after v1) below."""
    kinds = ["formatting", "insertion", "content"]
    metrics = {"S_1000": "S(1000)", "S_200": "S(200), added after v1"}
    fig = make_subplots(rows=2, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.12, vertical_spacing=0.14,
                        subplot_titles=[s for t in metrics.values() for s in (f"{t}: per arm, 7 models",
                                                                               f"{t}: A - baseline (pp)")])
    ranked = {}
    for row, metric in enumerate(metrics, start=1):
        by_kind = secondary[BY_KIND_KEY[metric]]
        for arm in ("baseline", "A"):
            vals = [by_kind[k][f"{arm}_questions_and_rules"] for k in kinds]
            fig.add_trace(go.Bar(x=kinds, y=[v["value"] for v in vals], name=ARM[arm][0], marker_color=ARM[arm][1],
                                 legendgroup=arm, showlegend=row == 1,
                                 error_y=dict(type="data", symmetric=False,
                                              array=[v["ci95"][1] - v["value"] for v in vals],
                                              arrayminus=[v["value"] - v["ci95"][0] for v in vals])), row=row, col=1)
        diffs = [by_kind[k]["questions_and_rules"] for k in kinds]
        fig.add_trace(go.Scatter(x=kinds, y=[v["value"] for v in diffs], mode="markers", marker=dict(size=12,
                      color="#333"), showlegend=False, error_y=dict(type="data", symmetric=False,
                      array=[v["ci95"][1] - v["value"] for v in diffs],
                      arrayminus=[v["value"] - v["ci95"][0] for v in diffs])), row=row, col=2)
        fig.update_yaxes(rangemode="tozero", row=row, col=2)
        ranked[metric] = sorted(kinds, key=lambda k: by_kind[k]["questions_and_rules"]["value"], reverse=True)
    if ranked["S_1000"][::2] == ranked["S_200"][::2]:  # same largest and smallest at both thresholds
        headline = (f"By rule kind: A's gain is largest for {ranked['S_1000'][0]} rules and smallest for "
                    f"{ranked['S_1000'][-1]} rules, at both S(1000) and S(200)")
    else:
        headline = "By rule kind: the ordering of A's gain differs between S(1000) and S(200)"
    rules = {k: ", ".join(RULE_LABEL[r] for r in secondary[BY_KIND_KEY["S_1000"]][k]["rules"]) for k in kinds}
    fig.update_layout(barmode="group", title=title(headline, "<br>".join(f"{k}: {v}" for k, v in rules.items())))
    return save(fig, fig_dir, "F5_by_rule_kind", 1150, 950, top=150)


def fig_accuracy_and_length(secondary: dict, cells: pd.DataFrame, grades: pd.DataFrame, fig_dir) -> str:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, column_widths=[0.42, 0.58],
                        subplot_titles=["Accuracy, full rows (7 models; A and baseline: 2 rules)",
                                        "Reasoning length, full rows (tokens, log scale)"])
    for arm in ("no_rule", "baseline", "A"):
        v = secondary["accuracy"][arm]
        fig.add_trace(go.Bar(x=[ARM[arm][0]], y=[v["value"]], marker_color=ARM[arm][1], showlegend=False,
                             error_y=err(v["ci95"], v["value"]), text=[f"{v['value']:.1f}"],
                             textposition="outside"), row=1, col=1)
    d = cells[cells["metric"] == "accuracy"].groupby(["model", "arm"])["value"].mean().reset_index()
    for arm in ("no_rule", "baseline", "A"):
        a = d[d["arm"] == arm]
        fig.add_trace(go.Scatter(x=[ARM[arm][0]] * len(a), y=a["value"], mode="markers", showlegend=False,
                                 marker=dict(color="white", line=dict(color="#333", width=1), size=7),
                                 hovertext=a["model"]), row=1, col=1)
    full = grades[grades["channel"] == "full"]
    for arm in ("no_rule", "baseline", "A"):
        f = full[full["prompt"] == arm]
        fig.add_trace(go.Box(x=f["model"].map(SHORT), y=f["reasoning_tokens"].clip(lower=1), name=ARM[arm][0],
                             marker_color=ARM[arm][1], boxpoints=False), row=1, col=2)
    fig.update_yaxes(range=[0, 75], title_text="% correct", row=1, col=1)
    fig.update_yaxes(type="log", dtick=1, title_text="reasoning tokens (empty shown at 1)", row=1, col=2)
    fig.update_layout(boxmode="group",
                      title=title("Accuracy is not significantly lower under A, but A's reasoning is much shorter",
                                  f"A - baseline accuracy {fmt(secondary['accuracy_A_minus_baseline']['questions_and_rules'])} "
                                  "pp (bars: pooled with 95% CI; dots: models)"))
    return save(fig, fig_dir, "F6_accuracy_and_length", 1400, 600, bottom=150)


# --- Figures counted from the graded rows ---------------------------------------------------------------------------
def fig_ways_to_look_better(grades: pd.DataFrame, fig_dir) -> str:
    short = grades[grades["channel"] == "short"].copy()
    empty = short["empty"].astype(bool)
    short["degenerate_not_empty"] = short["degenerate"].astype(bool) & ~empty
    short["empty"] = empty
    keys = ["empty", "degenerate_not_empty", "meta_regex"]
    short[keys] = short[keys].astype(float)
    rates = short.groupby(["model", "prompt"])[keys].mean().mul(100).reset_index()
    names = {"empty": "Empty reasoning", "degenerate_not_empty": "Degenerate but not empty (few distinct words)",
             "meta_regex": "Talks about the rule (meta regex)"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(names.values()), shared_yaxes=True,
                        horizontal_spacing=0.04)
    for col, key in enumerate(names, start=1):
        for arm in ("baseline", "A"):
            r = rates[rates["prompt"] == arm].set_index("model").loc[MODELS]
            fig.add_trace(go.Bar(y=[SHORT[m] for m in MODELS], x=r[key], orientation="h", name=ARM[arm][0],
                                 marker_color=ARM[arm][1], showlegend=col == 1), row=1, col=col)
        fig.update_xaxes(title_text="% of short rows", row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group",
                      title=title("Ways a prompt could look better without following the rule",
                                  "Short rows, 12 rules x 120 questions per model and arm. Empty and degenerate "
                                  "traces count as violations at token 0 in S(t)"))
    return save(fig, fig_dir, "F7_ways_to_look_better", 1300, 560)


LENGTH_BINS = [0, 500, 1000, 2000, 4000, 8000, 10 ** 6]
BIN_LABELS = ["0-500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k+"]
MIN_ROWS_PER_BIN = 10


def fig_length_matched(grades: pd.DataFrame, fig_dir) -> str:
    full = grades[(grades["channel"] == "full") & grades["prompt"].isin(["A", "baseline"])].copy()
    full["bin"] = pd.cut(full["reasoning_tokens"], LENGTH_BINS, labels=BIN_LABELS, right=False)
    full[["compliant", "correct"]] = full[["compliant", "correct"]].astype(float)
    g = full.groupby(["prompt", "bin"], observed=False).agg(n=("compliant", "size"), compliant=("compliant", "mean"),
                                                            correct=("correct", "mean")).reset_index()
    # Pooled bins mix models (A moves different models into different bins), so also compare within model.
    per_model = full.groupby(["model", "bin", "prompt"], observed=False).agg(
        n=("correct", "size"), correct=("correct", "mean")).unstack("prompt")
    enough = (per_model[("n", "A")] >= MIN_ROWS_PER_BIN) & (per_model[("n", "baseline")] >= MIN_ROWS_PER_BIN)
    within = (per_model[("correct", "A")] - per_model[("correct", "baseline")])[enough]
    within_by_bin = within.groupby(level="bin", observed=False).mean().dropna()
    n_lower = int((within_by_bin < 0).sum())
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=["Whole trace complies (%)", "Accuracy (%)", "Rows per length bin"])
    for arm in ("baseline", "A"):
        a = g[g["prompt"] == arm]
        for col, key, scale in ((1, "compliant", 100), (2, "correct", 100), (3, "n", 1)):
            fig.add_trace(go.Scatter(x=a["bin"].astype(str), y=a[key] * scale, mode="lines+markers",
                                     name=ARM[arm][0], line=dict(color=ARM[arm][1]), showlegend=col == 1),
                          row=1, col=col)
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="reasoning tokens", row=1, col=col)
    fig.update_layout(                      title=title("Added check: at matched reasoning length, A still complies far more; its "
                                  f"accuracy is lower in {n_lower} of {len(within_by_bin)} bins within model",
                                  "Full rows (2 rules x 120 questions x 7 models per arm), pooled. Within-model "
                                  f"accuracy A - baseline per bin (models with >= {MIN_ROWS_PER_BIN} rows per arm): "
                                  + ", ".join(f"{b} {100 * v:+.1f}" for b, v in within_by_bin.items())
                                  + "<br>Length is an outcome of the prompt, so bins are not a randomised "
                                  "comparison; not pre-registered"))
    return save(fig, fig_dir, "F8_length_matched", 1400, 600, top=140)


# --- Extension 3 (thinking-on parts) --------------------------------------------------------------------------------
def fig_examples_only(ext3: dict, fig_dir) -> str:
    names = {"E1 A - examples only (thinking on)": "A - A's examples without the rule",
             "E2 examples only - CoT-Control prompt (thinking on)": "A's examples without the rule - CoT-Control's prompt"}
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[f"{v}<br>pooled {fmt(ext3['contrasts'][k], 'ci')}" for k, v in names.items()])
    for col, key in enumerate(names, start=1):
        c = ext3["contrasts"][key]
        for m in MODELS:
            v = c["per_model"][m]
            group = "design" if m in e9.DESIGN_MODELS else "fresh"
            fig.add_trace(go.Scatter(x=[v["value"]], y=[SHORT[m]], mode="markers", marker=dict(
                size=11, color=GROUP_COLOR[group]), error_x=err(v["ci"], v["value"]), name=group, legendgroup=group,
                showlegend=col == 1 and m in (e9.DESIGN_MODELS[0], e9.FRESH_MODELS[0])), row=1, col=col)
        fig.add_vline(x=0, line_color="#999", row=1, col=col)
        fig.update_xaxes(title_text="difference in S(1000), pp", row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title("Extension 3: the stated rule, not just the examples, carries A's gain (thinking on)",
                                  "12 rules x 120 questions per model; 95% CIs resample questions only (narrower "
                                  "than F1's, which also resample rules). Extension, not pre-registered in exp09"))
    return save(fig, fig_dir, "F9_examples_only", 1300, 600, top=150)


def fig_necessity(ext3: dict, fig_dir) -> str:
    labels = ["necessary", "mixed", "unnecessary"]
    names = {"C3 A - CoT-Control prompt, thinking on": ("A - CoT-Control's prompt", ARM["A"][1]),
             "E1 A - examples only (thinking on)": ("A - A's examples without the rule", ARM["examples_only"][1])}
    counts = pd.DataFrame(ext3["necessity_counts"]).T.reindex(MODELS)[labels]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.12,
                        subplot_titles=["Gain in S(1000) within each label (pp)", "Questions per label and model"])
    for offset, (key, (label, color)) in zip((-0.08, 0.08), names.items()):
        vals = [ext3["by_necessity"][lab][key] for lab in labels]
        fig.add_trace(go.Scatter(x=np.arange(len(labels)) + offset, y=[v["value"] for v in vals], mode="markers",
                                 name=label,
                                 marker=dict(size=12, color=color),
                                 error_y=dict(type="data", symmetric=False, array=[v["ci"][1] - v["value"] for v in vals],
                                              arrayminus=[v["value"] - v["ci"][0] for v in vals])), row=1, col=1)
    for lab, color in zip(labels, ("#E31A1C", "#FDBF6F", "#A6CEE3")):
        fig.add_trace(go.Bar(y=[SHORT[m] for m in MODELS], x=counts[lab], orientation="h", name=lab,
                             marker_color=color), row=1, col=2)
    fig.update_xaxes(tickvals=list(range(len(labels))), ticktext=labels, range=[-0.5, len(labels) - 0.5], row=1, col=1)
    fig.update_yaxes(range=[0, 30], row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(barmode="stack",
                      title=title("Extension 3: A's gain is about the same whether or not the question needs reasoning",
                                  "Labels per model and question from 5 direct one-letter answers: necessary <= 1 "
                                  "right, unnecessary >= 4. Thinking-off splits wait for extension 2"))
    return save(fig, fig_dir, "F10_necessity", 1300, 580)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--analysis-run", default="v1")
    parser.add_argument("--ext3-run", default="thinking_on_v1")
    args = parser.parse_args()
    fig_dir = FIG_ROOT / args.run
    if fig_dir.exists():
        raise SystemExit(f"{fig_dir} exists; figure runs are never overwritten")
    analysis = e9.EXP.results / "analysis" / args.analysis_run
    primary = json.loads((analysis / "primary.json").read_text())
    secondary = json.loads((analysis / "secondary.json").read_text())
    cells = pd.read_csv(analysis / "cells.csv")
    ext3 = json.loads((e9.EXP.results / "analysis_ext3" / args.ext3_run / "summary.json").read_text())
    grades = load_grades()
    fig_dir.mkdir(parents=True)
    made = [fig_primary(primary, fig_dir), fig_heatmap(cells, "S_1000", fig_dir, "F2_heatmap_S1000"),
            fig_heatmap(cells, "S_200", fig_dir, "F3_heatmap_S200"), fig_design_vs_fresh(secondary, fig_dir),
            fig_by_kind(secondary, fig_dir), fig_accuracy_and_length(secondary, cells, grades, fig_dir),
            fig_ways_to_look_better(grades, fig_dir), fig_length_matched(grades, fig_dir),
            fig_examples_only(ext3, fig_dir), fig_necessity(ext3, fig_dir)]
    (fig_dir / "inputs.json").write_text(json.dumps({"analysis_run": args.analysis_run, "ext3_run": args.ext3_run,
                                                     "figures": made}, indent=1) + "\n")
    print(fig_dir, made)


if __name__ == "__main__":
    main()
