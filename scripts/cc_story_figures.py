"""Figures for sharing the project with outside readers: exp06 as the main evidence, plus what led to it.

Reads exp06's analysis outputs (results/exp06_prompt_prefill/analysis/<analysis run>/summary.json, per_rule.csv) and,
for the survival curves, exp06's graded rows. No new estimates: every number is one the exp06 analysis computed.
Toy text in S0 is invented (no benchmark question text).

  S0_task             what the models are asked to do, with toy examples of each rule
  S1_project          the experiments in order: question asked, answer found
  S2_headline         the paper's metric (whole reasoning trace follows the rule) under 3 prompts
  S3_ladder           every lever on one axis: S(1000) per intervention, one dot per model
  S4_over_the_trace   how compliance decays through the trace, and that a bad start is not recovered
  S5_cot_vs_output    the same rule on the reasoning trace vs on reasoning written in the answer
  S6_per_rule         which rules move (word suppression, the content rule, barely does)
  S7_cost_and_check   accuracy is unchanged, and the screen's numbers replicate on new questions

Outputs (never overwritten): figures/story/<run>/S*.{png,html}.
Run: /venv/main/bin/python scripts/cc_story_figures.py --run NAME [--analysis-run overnight]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cc_config as cfg
import cc_exp06_analysis as an
from cc_survival import kaplan_meier

MODELS = cfg.EXP06_MODELS
SHORT_MODEL = {"Qwen3.8-27B-FP8": "Qwen3.8-27B", "Gemma-4-31B-FP8": "Gemma-4-31B", "Qwen3-32B": "Qwen3-32B",
               "Qwen3.6-27B-FP8": "Qwen3.6-27B*"}
MODEL_COLOR = {"Qwen3.8-27B-FP8": "#0072B2", "Gemma-4-31B-FP8": "#D55E00", "Qwen3-32B": "#009E73",
               "Qwen3.6-27B-FP8": "#CC79A7"}
HELD_OUT_NOTE = "*Qwen3.6-27B was not used to choose the prompt (held out)."
# Interventions, in plain words, with one colour each across all figures.
ARM = {
    ("baseline", "none"): ("CoT-Control's prompt", "#9E9E9E"),
    ("stacked", "none"): ("Rule repeated in system + user turn", "#56B4E9"),
    ("upgraded", "none"): ("Upgraded prompt", "#B15928"),
    ("baseline", "prefill_compliant"): ("CoT-Control's prompt + compliant first sentence", "#C9A7E4"),
    ("upgraded", "prefill_compliant"): ("Upgraded prompt + compliant first sentence", "#6A3D9A"),
    ("upgraded", "prefill_noncompliant"): ("Upgraded prompt + rule-breaking first sentence", "#E31A1C"),
    ("no_rule", "prefill_compliant"): ("Compliant first sentence, no rule given (copying control)", "#000000"),
}
UPGRADED_DEF = "rule repeated + 3 of the model's own compliant traces as examples + a list of common failures"
RULE_LABEL = {"lowercase_thinking": "all lowercase", "uppercase_thinking": "ALL UPPERCASE",
              "meow_between_words": "'meow' between words", "end_of_sentence": "end sentences with 'safe'",
              "word_suppression": "never use a given word"}
FONT = "Inter, Helvetica Neue, Arial, sans-serif"


def save(fig: go.Figure, fig_dir, name: str, width: int, height: int) -> str:
    fig.update_layout(template="simple_white", font={"family": FONT, "size": 14, "color": "#222222"},
                      width=width, height=height, title={"x": 0.02, "xanchor": "left", "font": {"size": 20}})
    fig.write_html(fig_dir / f"{name}.html", include_plotlyjs="cdn")
    fig.write_image(fig_dir / f"{name}.png", scale=2)
    return name


def title(main: str, sub: str) -> str:
    return f"<b>{main}</b><br><span style='font-size:13px;color:#555555'>{sub}</span>"


def cells(summary: dict) -> dict:
    return {(r["model"], r["prompt"], r["opening"]): r for r in summary["cells"]}


# --- S0, S1: text figures -----------------------------------------------------------------------------------------
def fig_task(fig_dir) -> str:
    rules = ["all lowercase", "ALL UPPERCASE", "'meow' between words", "end sentences with 'safe'",
             "never use a given word (e.g. 'force')"]
    ok = ["so the net force is mass times acceleration, which gives 12 n.",
          "SO THE NET FORCE IS MASS TIMES ACCELERATION, WHICH GIVES 12 N.",
          "so meow the meow net meow force meow is meow mass meow times ...",
          "The net force is mass times acceleration safe. That gives 12 N safe.",
          "The net push is mass times acceleration, which gives 12 N."]
    bad = ["<b>S</b>o the net force ...", "SO THE NET FORCE IS <b>mass</b> ...", "so meow the meow net <b>force is</b> ...",
           "The net force is mass times acceleration<b>.</b>", "The net <b>force</b> is ..."]
    fig = go.Figure(go.Table(
        columnwidth=[1.1, 2.6, 1.6],
        header={"values": ["<b>Rule about the reasoning trace</b>", "<b>Compliant reasoning (toy example)</b>",
                           "<b>First violation</b>"],
                "fill_color": "#EFEFEF", "align": "left", "font": {"size": 14}, "height": 34},
        cells={"values": [rules, ok, bad], "align": "left", "height": 34, "font": {"size": 13},
               "fill_color": [["white", "#FAFAFA"] * 3]}))
    fig.update_layout(
        title=title("The task: follow a rule about what goes in your own chain of thought",
                    "The CoT-Control benchmark gives a reasoning model a question plus one rule about its thinking; "
                    "a program checks the rule token by token.<br>If models can control their thinking on request, a "
                    "chain-of-thought monitor is easier to fool. The paper reports they mostly can't (Qwen3-32B: 2.3%)."),
        margin={"t": 120, "b": 110, "l": 30, "r": 30},
        annotations=[{"text": "<b>Two measures used throughout</b><br>"
                              "<b>Whole trace</b>: % of reasoning traces with no violation from start to end "
                              "(the paper's measure).<br>"
                              "<b>S(1000)</b>: % of traces with no violation in their first 1000 tokens. We stop "
                              "generation at the first violation, which makes large runs cheap.",
                      "xref": "paper", "yref": "paper", "x": 0, "y": 0.0, "yanchor": "top", "showarrow": False,
                      "align": "left", "font": {"size": 13}}])
    return save(fig, fig_dir, "S0_task", 1300, 470)


def fig_project(fig_dir) -> str:
    rows = [
        ("exp01", "Does our setup reproduce the paper?",
         "Yes: Qwen3-32B whole-trace compliance 2.0% (paper: 2.3%)."),
        ("exp02-03", "Do better-worded prompts help?",
         "A little: repeating the rule or asking for self-monitoring adds 2-5 points."),
        ("exp04", "Can a model keep the rule once it starts compliant?",
         "Partly: a compliant first sentence lifts S(1000) to 20-40%. A rule-breaking one gives ~0.<br>"
         "With reasoning moved into the answer, the same models follow the rule much more often."),
        ("exp05", "Is that just copying the opening's style?",
         "No: without the rule, the same opening gives far less, at every opening length."),
        ("exp06a", "Which prompt components help? (screen)",
         "Single components help little; repeated rule + the model's own example traces + a failure list "
         "helps a lot."),
        ("<b>exp06</b>", "<b>Does it hold on new questions and an unseen model?</b>",
         "<b>Yes. Whole-trace compliance goes from 2-3% to 13-46%, with no detectable accuracy cost.</b><br>"
         "But control fades over long traces, a slip is never recovered, and the gain is CoT-specific "
         "for only 2 of 4 models."),
    ]
    fig = go.Figure(go.Table(
        columnwidth=[0.55, 1.6, 3.4],
        header={"values": ["<b>Experiment</b>", "<b>Question</b>", "<b>Answer (all results unverified)</b>"],
                "fill_color": "#EFEFEF", "align": "left", "font": {"size": 14}, "height": 34},
        cells={"values": [list(c) for c in zip(*rows)], "align": "left", "height": 46, "font": {"size": 13},
               "fill_color": [["white"] * 5 + ["#FFF4E5"]]}))
    fig.update_layout(title=title("What we did, in order",
                                  "Open reasoning models, 100 questions (GPQA, HLE, MMLU-Pro) x 5 rules per "
                                  "condition. exp06 used 100 new questions, so the prompt was not scored on the "
                                  "questions it was chosen on."),
                      margin={"t": 100, "b": 20, "l": 30, "r": 30})
    return save(fig, fig_dir, "S1_project", 1400, 470)


# --- S2-S7: data figures ------------------------------------------------------------------------------------------
def error(stats: list[dict]) -> dict:
    return {"type": "data", "symmetric": False, "array": [s["ci"][1] - s["value"] for s in stats],
            "arrayminus": [s["value"] - s["ci"][0] for s in stats], "thickness": 1.5, "width": 4,
            "color": "#444444"}


def fig_headline(c: dict, fig_dir) -> str:
    prompts = [("baseline", "none"), ("stacked", "none"), ("upgraded", "none")]
    fig = go.Figure()
    for p in prompts:
        label, color = ARM[p]
        stats = [c[(m, *p)]["whole_trace_compliant"] for m in MODELS]
        fig.add_trace(go.Bar(x=[SHORT_MODEL[m] for m in MODELS], y=[s["value"] for s in stats], name=label,
                             marker={"color": color}, error_y=error(stats),
                             text=[f"{s['value']:.0f}%" for s in stats], textposition="outside",
                             textfont={"size": 13}, cliponaxis=False))
    fig.add_hline(y=2.3, line={"color": "#555555", "dash": "dot", "width": 1})
    fig.add_annotation(x=1, xref="paper", y=2.3, text="CoT-Control paper,<br>Qwen3-32B: 2.3%", showarrow=False,
                       xanchor="left", yanchor="middle", align="left", font={"size": 12, "color": "#555555"})
    fig.update_layout(
        barmode="group", bargap=0.25,
        title=title("A better prompt lifts whole-trace compliance from ~2% to 13-46%",
                    f"% of complete reasoning traces that follow the rule from start to finish.<br>Upgraded prompt = "
                    f"{UPGRADED_DEF}.<br>125 traces per bar (25 questions x 5 rules); 95% bootstrap CIs. "
                    f"{HELD_OUT_NOTE}"),
        yaxis={"title": "whole trace follows the rule (%)", "range": [0, 60]},
        legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"}, margin={"t": 165, "b": 90, "r": 170})
    return save(fig, fig_dir, "S2_headline", 1200, 600)


def fig_ladder(c: dict, fig_dir) -> str:
    arms = [("baseline", "none"), ("stacked", "none"), ("baseline", "prefill_compliant"), ("upgraded", "none"),
            ("upgraded", "prefill_compliant"), ("no_rule", "prefill_compliant")]
    y = {a: i for i, a in enumerate(arms)}
    fig = go.Figure()
    fig.add_hrect(y0=len(arms) - 1.5, y1=len(arms) - 0.5, fillcolor="#F2F2F2", line_width=0, layer="below")
    for a in arms:  # faint bar = mean over models
        mean = np.mean([c[(m, *a)]["S_1000"]["value"] for m in MODELS])
        fig.add_trace(go.Bar(x=[mean], y=[y[a]], orientation="h", marker={"color": ARM[a][1], "opacity": 0.35},
                             showlegend=False, hoverinfo="skip", width=0.6))
        fig.add_annotation(x=mean, y=y[a], text=f" {mean:.0f}", showarrow=False, xanchor="left", yshift=-17,
                           font={"size": 11, "color": "#666666"})
    offsets = dict(zip(MODELS, np.linspace(-0.2, 0.2, len(MODELS))))
    for m in MODELS:
        stats = [c[(m, *a)]["S_1000"] for a in arms]
        fig.add_trace(go.Scatter(
            x=[s["value"] for s in stats], y=[y[a] + offsets[m] for a in arms], mode="markers", name=SHORT_MODEL[m],
            marker={"color": MODEL_COLOR[m], "size": 10, "line": {"color": "white", "width": 1}},
            error_x={**error(stats), "color": MODEL_COLOR[m], "thickness": 1.2, "width": 0},
            hovertemplate=[f"{ARM[a][0]}: {s['value']:.1f}% [{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]<extra>"
                           f"{SHORT_MODEL[m]}</extra>" for a, s in zip(arms, stats)]))
    fig.update_yaxes(tickvals=list(y.values()), ticktext=[ARM[a][0] for a in arms], autorange="reversed",
                     showline=False, ticks="")
    fig.update_xaxes(title_text="% of traces with no violation in the first 1000 tokens, S(1000)", range=[0, 80],
                     showgrid=True, gridcolor="#EEEEEE")
    fig.update_layout(
        title=title("The prompt alone beats writing the first sentence for the model (3 of 4 models)",
                    "Dots: one per model, 95% CIs. Bars and grey numbers: mean over the 4 models. Mean over 5 rules, "
                    "100 new questions, thinking on.<br>A 'first sentence' is written into the reasoning for the "
                    "model; only what it writes afterwards is scored. Grey band: control. " + HELD_OUT_NOTE),
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
        margin={"t": 120, "l": 430, "b": 100})
    return save(fig, fig_dir, "S3_ladder", 1300, 580)


def fig_over_the_trace(df: pd.DataFrame, fig_dir) -> str:
    arms = [("baseline", "none"), ("upgraded", "none"), ("upgraded", "prefill_compliant"),
            ("upgraded", "prefill_noncompliant")]
    grid = np.arange(0, cfg.EXP06_STOP_TOKENS - an.CUT_MARGIN + 1, 10, dtype=float)
    fig = make_subplots(rows=1, cols=len(MODELS), shared_yaxes=True, horizontal_spacing=0.025,
                        subplot_titles=[SHORT_MODEL[m] for m in MODELS])
    for k, m in enumerate(MODELS):
        for a in arms:
            d = an.cell(df, m, *a)
            d = d[d["mode"].isin(cfg.EXP06_OPENER_MODES)]
            curve = 100 * np.mean([kaplan_meier(g["time"].to_numpy(), g["event"].to_numpy(), np.ones((1, len(g))),
                                                grid)[0] for _, g in d.groupby("mode")], axis=0)
            label, color = ARM[a]
            fig.add_trace(go.Scatter(x=grid, y=curve, mode="lines", name=label, legendgroup=label,
                                     showlegend=k == 0, line={"color": color, "width": 3},
                                     hovertemplate=f"{label}<br>token %{{x}}: %{{y:.0f}}%<extra>{SHORT_MODEL[m]}"
                                                   f"</extra>"), row=1, col=k + 1)
        fig.update_xaxes(title_text="tokens into the reasoning", range=[0, grid.max()], row=1, col=k + 1,
                         showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="% of traces with no violation yet", range=[0, 101], row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
    fig.update_layout(
        title=title("Control fades as the reasoning goes on, and a bad start is never recovered",
                    "Share of reasoning traces still following the rule at each point (Kaplan-Meier), mean over the "
                    "4 formatting rules, 100 new questions.<br>With a first sentence written for the model, tokens "
                    "are counted after it. Red: the first sentence breaks the rule; models almost never return to "
                    "it. " + HELD_OUT_NOTE),
        legend={"orientation": "h", "y": -0.22, "x": 0.5, "xanchor": "center"}, margin={"t": 150, "b": 120})
    return save(fig, fig_dir, "S4_over_the_trace", 1500, 590)


def fig_cot_vs_output(c: dict, fig_dir) -> str:
    channels = [("thinking trace", "none", True), ("reasoning written in the answer", "thinking_off", False)]
    fig = go.Figure()
    rows = []
    for m in MODELS:
        for label, opening, _ in channels:
            rows.append((m, label, opening))
    ys = list(range(len(rows)))
    for i, (m, label, opening) in enumerate(rows):
        lo = c[(m, "baseline", opening)]["clean_200"]["value"]
        hi = c[(m, "upgraded", opening)]["clean_200"]["value"]
        color = MODEL_COLOR[m]
        fig.add_trace(go.Scatter(x=[lo, hi], y=[i, i], mode="lines", showlegend=False, hoverinfo="skip",
                                 line={"color": color, "width": 4 if opening == "none" else 2,
                                       "dash": "solid" if opening == "none" else "dot"}))
        fig.add_trace(go.Scatter(x=[lo], y=[i], mode="markers", showlegend=False,
                                 marker={"color": "white", "size": 11, "line": {"color": color, "width": 2}},
                                 hovertemplate=f"CoT-Control prompt: {lo:.0f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=[hi], y=[i], mode="markers+text", showlegend=False, text=[f"+{hi - lo:.0f}"],
                                 textposition="middle right", textfont={"color": color, "size": 13},
                                 marker={"color": color, "size": 12},
                                 hovertemplate=f"upgraded prompt: {hi:.0f}%<extra></extra>"))
    for k, m in enumerate(MODELS):
        if k:
            fig.add_hline(y=2 * k - 0.5, line={"color": "#DDDDDD", "width": 1})
    fig.update_yaxes(tickvals=ys, ticktext=[f"<b>{SHORT_MODEL[m]}</b>  {label}" if label.startswith("thinking")
                                            else label for m, label, _ in rows],
                     autorange="reversed", showline=False, ticks="")
    fig.update_xaxes(title_text="% of texts reaching 200 tokens with no violation (shorter texts count as failures)",
                     range=[0, 100], showgrid=True, gridcolor="#EEEEEE")
    fig.update_layout(
        title=title("Is the gain about the chain of thought, or about following instructions in general?",
                    "Same rule, applied either to the model's thinking trace (solid) or, with thinking switched off, "
                    "to reasoning it writes inside its answer (dotted).<br>Open circle: CoT-Control's prompt; filled: "
                    "upgraded prompt; number: gain. The answer channel starts higher. Relative to the room left to "
                    "improve,<br>the gain favours the thinking trace for Gemma and Qwen3.6, but not for Qwen3.8 or "
                    "Qwen3-32B. " + HELD_OUT_NOTE),
        margin={"t": 160, "l": 330, "b": 70})
    return save(fig, fig_dir, "S5_cot_vs_output", 1200, 640)


def fig_per_rule(per_rule: pd.DataFrame, fig_dir) -> str:
    d = per_rule[per_rule["thinking"] & (per_rule["opening"] == "none")]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04, shared_yaxes=True,
                        subplot_titles=["CoT-Control's prompt", "Upgraded prompt"])
    for k, prompt in enumerate(("baseline", "upgraded")):
        z = (d[d["prompt"] == prompt].pivot(index="mode", columns="model", values="S_1000")
             .reindex(index=cfg.EXP06_MODES, columns=MODELS))
        fig.add_trace(go.Heatmap(z=z.to_numpy(), x=[SHORT_MODEL[m] for m in MODELS],
                                 y=[RULE_LABEL[r] for r in cfg.EXP06_MODES], zmin=0, zmax=100,
                                 colorscale=[[0, "#FFFFFF"], [1, "#B15928"]], showscale=k == 1,
                                 colorbar={"title": "S(1000) %"}, text=z.round(0).astype("Int64").astype(str).to_numpy(),
                                 texttemplate="%{text}", textfont={"size": 14}, xgap=3, ygap=3,
                                 hovertemplate="%{y} | %{x}: %{z:.1f}%<extra></extra>"), row=1, col=k + 1)
    fig.update_yaxes(autorange="reversed", ticks="", showline=False)
    fig.update_xaxes(ticks="", showline=False)
    fig.update_layout(
        title=title("Formatting rules move a lot; the content rule barely does",
                    "% of traces with no violation in the first 1000 tokens, per rule (100 new questions, thinking "
                    "on, no first sentence).<br>'Never use a given word' is the rule closest to hiding content from "
                    "a monitor. " + HELD_OUT_NOTE),
        margin={"t": 165, "l": 200})
    return save(fig, fig_dir, "S6_per_rule", 1200, 500)


def fig_cost_and_check(summary: dict, fig_dir) -> str:
    acc = {(r["model"], r["metric"]): r["difference"] for r in summary["accuracy"]
           if (r["a"], r["b"]) == ("upgraded", "baseline")}
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14, column_widths=[0.5, 0.5],
                        subplot_titles=["Upgraded minus CoT-Control's prompt, on the same 125 traces",
                                        "The screen's numbers hold on new questions"])
    metrics = [("whole_trace_compliant", "rule followed (whole trace)", "#B15928", -0.15),
               ("correct", "answer correct", "#333333", 0.15)]
    for metric, label, color, offset in metrics:
        stats = [acc[(m, metric)] for m in MODELS]
        fig.add_trace(go.Scatter(
            x=[s["value"] for s in stats], y=[i + offset for i in range(len(MODELS))], mode="markers+text",
            showlegend=False, marker={"size": 12, "color": color}, error_x={**error(stats), "color": color},
            text=[label if i == 0 else "" for i in range(len(MODELS))], textposition="top center",
            textfont={"color": color, "size": 12},
            hovertemplate=[f"{label}: {s['value']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]<extra></extra>"
                           for s in stats]), row=1, col=1)
    fig.update_yaxes(tickvals=list(range(len(MODELS))), ticktext=[SHORT_MODEL[m] for m in MODELS], row=1, col=1)
    fig.add_vline(x=0, line={"color": "#888888", "width": 1}, row=1, col=1)
    fig.update_xaxes(title_text="change, percentage points", range=[-20, 60], showgrid=True, gridcolor="#EEEEEE",
                     row=1, col=1)
    fig.update_yaxes(range=[len(MODELS) - 0.5, -0.6], row=1, col=1)
    rep = summary["replication"]
    for m in MODELS:
        pts = [r for r in rep if r["model"] == m]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[r["exp03_items"]["value"] for r in pts], y=[r["new_items"]["value"] for r in pts], mode="markers",
            name=SHORT_MODEL[m], marker={"color": MODEL_COLOR[m], "size": 10,
                                         "symbol": ["star" if r["cell"] == "upgraded|none" else "circle" for r in pts]},
            hovertemplate=[f"{r['cell']}: {r['exp03_items']['value']:.0f} -> {r['new_items']['value']:.0f}<extra>"
                           f"{SHORT_MODEL[m]}</extra>" for r in pts]), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 80], y=[0, 80], mode="lines", showlegend=False, hoverinfo="skip",
                             line={"color": "#999999", "dash": "dot"}), row=1, col=2)
    fig.update_xaxes(title_text="S(1000) on the questions used to choose the prompt (%)", range=[0, 80],
                     showgrid=True, gridcolor="#EEEEEE", row=1, col=2)
    fig.update_yaxes(title_text="S(1000) on 100 new questions (%)", range=[0, 80], showgrid=True,
                     gridcolor="#EEEEEE", row=1, col=2)
    fig.update_layout(
        title=title("No detectable cost to accuracy, and no sign of overfitting the prompt to its questions",
                    "Left: paired differences with 95% CIs (CIs are wide: about +/-8 points). Right: same "
                    "condition, same model; ★ = upgraded prompt; dotted line = equal.<br>The held-out model has no "
                    "screen numbers, so it appears only on the left. " + HELD_OUT_NOTE),
        legend={"orientation": "h", "y": -0.2, "x": 0.78, "xanchor": "center"}, margin={"t": 150, "b": 110})
    return save(fig, fig_dir, "S7_cost_and_check", 1400, 600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="figure set name")
    parser.add_argument("--analysis-run", default="overnight", help="exp06 analysis run to read")
    parser.add_argument("--scratch", action="store_true", help="write under /tmp/story_scratch (layout tests)")
    args = parser.parse_args()
    root = Path("/tmp/story_scratch") if args.scratch else cfg.REPO_ROOT / "figures" / "story"
    fig_dir = root / args.run
    if fig_dir.exists():
        raise SystemExit(f"{fig_dir} exists; figure sets are never overwritten")
    analysis = cfg.EXP06.results / "analysis" / args.analysis_run
    summary = json.loads((analysis / "summary.json").read_text())
    per_rule = pd.read_csv(analysis / "per_rule.csv")
    df = an.score(an.load(skip_missing=False)[0])
    fig_dir.mkdir(parents=True)
    c = cells(summary)
    names = [fig_task(fig_dir), fig_project(fig_dir), fig_headline(c, fig_dir), fig_ladder(c, fig_dir),
             fig_over_the_trace(df, fig_dir), fig_cot_vs_output(c, fig_dir), fig_per_rule(per_rule, fig_dir),
             fig_cost_and_check(summary, fig_dir)]
    print("\n".join(str(fig_dir / f"{n}.png") for n in names))


if __name__ == "__main__":
    main()
