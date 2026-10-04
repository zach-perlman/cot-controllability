"""Figures for sharing the project with outside readers: exp06 as the main evidence, plus what led to it.

Reads exp06's analysis outputs (results/exp06_prompt_prefill/analysis/<analysis run>/summary.json, per_rule.csv) and,
for the survival curves, exp06's graded rows. No new estimates: every number is one the exp06 analysis computed.
Toy text in S0 is invented (no benchmark question text).

  S0_task             what the models are asked to do, with toy examples of each rule
  S1_project          the experiments in order: question asked, answer found
  S2_headline         the paper's metric (whole reasoning trace follows the rule) under 3 prompts
  S3_ladder           every lever on one axis: S(1000) per intervention, one dot per model
  S4_over_the_trace   how compliance decays through the trace, and that a bad start is not recovered
  S5_cot_vs_output    the same rule on the reasoning trace vs on reasoning written in the answer (raw, and as a
                      share of the room left to improve)
  S5b_opening_on_vs_off  the same for a compliant first sentence (thinking off: prefilled after the opening tag)
  S6_per_rule         which rules move (word suppression, the content rule, barely does)
  S7_cost_and_check   accuracy is unchanged, and the screen's numbers replicate on new questions
After exp06 (read from exp07's test analysis, exp07b's and exp08's; S1 gains their rows):
  S8_heldout          the upgraded prompt on rules and models never used to choose it (exp07 test)
  S9_what_helps       one change at a time to the upgraded prompt: only more examples clearly helps (exp07b)
  S10_openings        under prompt A, first sentences on the held-out rules vs exp06's rules (exp08 + extension)
  S11_channels        prompt A and CoT-Control's prompt, thinking trace vs reasoning written in the answer (exp08)
  S12_cost_and_length prompt A's accuracy cost and its shorter reasoning (exp08)

Outputs (never overwritten): figures/exp6_story/<run>/S*.{png,html}.
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
               "Qwen3.6-27B-FP8": "Qwen3.6-27B*", "Qwen3.5-9B": "Qwen3.5-9B*"}
MODEL_COLOR = {"Qwen3.8-27B-FP8": "#0072B2", "Gemma-4-31B-FP8": "#D55E00", "Qwen3-32B": "#009E73",
               "Qwen3.6-27B-FP8": "#CC79A7", "Qwen3.5-9B": "#E69F00"}
HELD_OUT_NOTE = "*Held out: not used to choose the prompt."
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


def cells(summary: dict, thinking: bool = True) -> dict:
    """(model, prompt, opening) -> the analysis's estimates, for one channel (thinking on, or off)."""
    return {(r["model"], r["prompt"], r["opening"]): r for r in summary["cells"] if r["thinking"] == thinking}


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


FLOOR = 5  # % whole-trace compliance under the upgraded prompt below which a model counts as not lifted


def whole_trace_lift(c: dict) -> tuple[list[str], list[str], float, float]:
    """(models the upgraded prompt lifts above FLOOR, models it does not, lowest and highest lifted value)."""
    up = {m: c[(m, "upgraded", "none")]["whole_trace_compliant"]["value"] for m in MODELS}
    lifted = [m for m in MODELS if up[m] >= FLOOR]
    return lifted, [m for m in MODELS if up[m] < FLOOR], min(up[m] for m in lifted), max(up[m] for m in lifted)


def cot_specific_models(summary: dict) -> list[str]:
    """Models whose upgraded-prompt gain, as a share of the room left, is larger on the trace (95% CI above 0)."""
    return [r["model"] for r in summary["cot_specificity"]
            if r["prompt"] == "upgraded" and r["on_minus_off_headroom"]["ci"][0] > 0]


def short_names(models: list[str]) -> str:
    return ", ".join(SHORT_MODEL[m].rstrip("*") for m in models)


def fig_project(c: dict, summary: dict, fig_dir, later_rows: list[tuple[str, str, str]] | None = None,
                name: str = "S1_project") -> str:
    lifted, floor, lo, hi = whole_trace_lift(c)
    held_out = len(cfg.EXP06_HELD_OUT_MODELS)
    exp06_answer = (f"<b>Yes for {len(lifted)} of {len(MODELS)} models. Whole-trace compliance goes from 2-3% to "
                    f"{lo:.0f}-{hi:.0f}%, with no detectable accuracy cost.</b><br>"
                    + (f"{short_names(floor)} stays near 0. " if floor else "")
                    + "Control fades over long traces and a slip is never recovered.<br>The gain is CoT-specific "
                    f"for only {len(cot_specific_models(summary))} of {len(MODELS)} models.")
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
        ("<b>exp06</b>", f"<b>Does it hold on new questions and {held_out} unseen models?</b>", exp06_answer),
    ]
    sub = ("Open reasoning models, 100 questions (GPQA, HLE, MMLU-Pro) x 5 rules per condition. exp06 used 100 new "
           "questions, so the prompt was not scored on the questions it was chosen on.")
    if later_rows:
        rows = [(e, q.replace("<b>", "").replace("</b>", ""), a.replace("<b>", "").replace("</b>", ""))
                for e, q, a in rows] + later_rows
        sub += ("<br>exp07-exp08: 160 further new questions and 9 rules, 4 of which no prompt was ever chosen on.")
    fill = ["white"] * (len(rows) - 1) + ["#FFF4E5"]
    fig = go.Figure(go.Table(
        columnwidth=[0.55, 1.6, 3.4],
        header={"values": ["<b>Experiment</b>", "<b>Question</b>", "<b>Answer (all results unverified)</b>"],
                "fill_color": "#EFEFEF", "align": "left", "font": {"size": 14}, "height": 34},
        cells={"values": [list(col) for col in zip(*rows)], "align": "left", "height": 46, "font": {"size": 13},
               "fill_color": [fill]}))
    fig.update_layout(title=title("What we did, in order", sub),
                      margin={"t": 120 if later_rows else 100, "b": 20, "l": 30, "r": 30})
    return save(fig, fig_dir, name, 1400, 470 + 80 * len(later_rows or []))


# --- S2-S7: data figures ------------------------------------------------------------------------------------------
def error(stats: list[dict]) -> dict:
    return {"type": "data", "symmetric": False, "array": [s["ci"][1] - s["value"] for s in stats],
            "arrayminus": [s["value"] - s["ci"][0] for s in stats], "thickness": 1.5, "width": 4,
            "color": "#444444"}


def fig_headline(c: dict, fig_dir) -> str:
    prompts = [("baseline", "none"), ("stacked", "none"), ("upgraded", "none")]
    lifted, floor, lo, hi = whole_trace_lift(c)
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
        title=title(f"A better prompt lifts whole-trace compliance from ~2% to {lo:.0f}-{hi:.0f}%"
                    + (f" in {len(lifted)} of {len(MODELS)} models" if floor else ""),
                    "% of complete reasoning traces that follow the rule from start to finish"
                    + (f"; {short_names(floor)} stays near 0" if floor else "") + ".<br>Upgraded prompt = "
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
    beats = sum(c[(m, "upgraded", "none")]["S_1000"]["value"] > c[(m, "baseline", "prefill_compliant")]["S_1000"]["value"]
                for m in MODELS)
    fig.update_layout(
        title=title(f"The prompt alone beats writing the first sentence for the model ({beats} of {len(MODELS)} "
                    "models)",
                    f"Dots: one per model, 95% CIs. Bars and grey numbers: mean over the {len(MODELS)} models. Mean "
                    "over 5 rules, "
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


def headroom_verdict(spec: dict) -> str:
    """Which models' gain, as a share of the room left, is larger on the trace than in the answer (95% CI of the
    difference above 0), larger in the answer (below 0), or neither."""
    groups = {"trace": [], "answer": [], "neither": []}
    for m in MODELS:
        lo, hi = spec[m]["on_minus_off_headroom"]["ci"]
        groups["trace" if lo > 0 else "answer" if hi < 0 else "neither"].append(SHORT_MODEL[m].rstrip("*"))
    parts = [f"larger on the thinking trace for {', '.join(groups['trace']) or 'no model'}",
             f"larger in the answer for {', '.join(groups['answer']) or 'no model'}"]
    if groups["neither"]:
        parts.append(f"no clear difference for {', '.join(groups['neither'])}")
    return "; ".join(parts)


CHANNELS = [("thinking trace", True, "on"), ("reasoning written in the answer", False, "off")]
LENIENT_SHOWN = 5  # % of a model's thinking-off rows rescored (closed by </think>) before its lenient values are drawn


def lenient_marker(x: float, y: int, color: str, hover: str) -> go.Scatter:
    return go.Scatter(x=[x], y=[y], mode="markers", showlegend=False, hovertemplate=hover + "<extra></extra>",
                      marker={"color": color, "size": 12, "symbol": "diamond-open", "line": {"width": 2}})


def fig_channels(c_on: dict, c_off: dict, spec: dict, before: dict, after: dict, labels: tuple[str, str],
                 copying: tuple | None, name: str, main: str, sub: str, fig_dir, lenient_spec: dict | None = None) -> str:
    """One row per (model, channel): left, the clean-at-200 rate of the `before` cell (open circle) and the `after`
    cell (filled), with the gain in points (and, if given, the `copying` cell as a black x); right, the gain as a
    share of the room left (spec: per model, cc_exp06_analysis's on/off headroom stats). With lenient_spec (the same
    stats with </think> accepted as the closing tag), thinking-off values it moves by over LENIENT_SHOWN points get a
    hollow diamond."""
    rows = [(m, label, thinking, key) for m in MODELS for label, thinking, key in CHANNELS]
    lenient_models = []
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05, column_widths=[0.55, 0.45],
                        subplot_titles=["Raw: % compliant, before and after", "Gain as a share of the room left to improve"])
    for i, (m, label, thinking, key) in enumerate(rows):
        c = c_on if thinking else c_off
        color = MODEL_COLOR[m]
        lo, hi = (c[(m, *cell[key])]["clean_200"]["value"] for cell in (before, after))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[i, i], mode="lines", showlegend=False, hoverinfo="skip",
                                 line={"color": color, "width": 4 if thinking else 2,
                                       "dash": "solid" if thinking else "dot"}), row=1, col=1)
        fig.add_trace(go.Scatter(x=[lo], y=[i], mode="markers", showlegend=False,
                                 marker={"color": "white", "size": 11, "line": {"color": color, "width": 2}},
                                 hovertemplate=f"{labels[0]}: {lo:.0f}%<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[hi], y=[i], mode="markers+text", showlegend=False, text=[f"{hi - lo:+.0f}"],
                                 textposition="middle right", textfont={"color": color, "size": 13},
                                 marker={"color": color, "size": 12},
                                 hovertemplate=f"{labels[1]}: {hi:.0f}%<extra></extra>"), row=1, col=1)
        if copying:
            x = c[(m, *copying)]["clean_200"]["value"]
            fig.add_trace(go.Scatter(x=[x], y=[i], mode="markers", showlegend=False,
                                     marker={"color": "black", "size": 10, "symbol": "x-thin",
                                             "line": {"color": "black", "width": 2}},
                                     hovertemplate=f"{ARM[copying][0]}: {x:.0f}%<extra></extra>"), row=1, col=1)
        share = spec[m][f"{key}_headroom"]
        fig.add_trace(go.Scatter(
            x=[share["value"]], y=[i], mode="markers+text", showlegend=False, text=[f"{share['value']:.0f}%"],
            textposition="top center", textfont={"color": color, "size": 12},
            marker={"color": color if thinking else "white", "size": 12, "line": {"color": color, "width": 2}},
            error_x={**error([share]), "color": color},
            hovertemplate=f"gain = {share['value']:.0f}% of the room left [{share['ci'][0]:.0f}, "
                          f"{share['ci'][1]:.0f}]<extra></extra>"), row=1, col=2)
        off_cells = [c[(m, *cell[key])] for cell in (before, after)]
        if lenient_spec and not thinking and max(r["lenient_rescored_share"] for r in off_cells) > LENIENT_SHOWN:
            lenient_models.append(SHORT_MODEL[m].rstrip("*"))
            for r, what in zip(off_cells, labels):
                x = r["clean_200_lenient"]["value"]
                fig.add_trace(lenient_marker(x, i, color, f"{what}, &lt;/think&gt; counted: {x:.0f}%"), row=1, col=1)
            x = lenient_spec[m]["off_headroom"]["value"]
            fig.add_trace(lenient_marker(x, i, color, f"&lt;/think&gt; counted: {x:.0f}% of the room left"),
                          row=1, col=2)
    for k in range(1, len(MODELS)):
        for col in (1, 2):
            fig.add_hline(y=2 * k - 0.5, line={"color": "#DDDDDD", "width": 1}, row=1, col=col)
    fig.update_yaxes(tickvals=list(range(len(rows))),
                     ticktext=[f"<b>{SHORT_MODEL[m]}</b>  {label}" if key == "on" else label
                               for m, label, _, key in rows],
                     range=[len(rows) - 0.5, -0.7], showline=False, ticks="")
    fig.update_xaxes(title_text="% of texts reaching 200 tokens with no violation", range=[0, 100], showgrid=True,
                     gridcolor="#EEEEEE", row=1, col=1)
    lowest = min(spec[m][f"{key}_headroom"]["ci"][0] for m in MODELS for key in ("on", "off"))
    fig.update_xaxes(title_text="gain / (100 - the open circle's %), with 95% CI", range=[min(0, lowest - 5), 100],
                     showgrid=True, gridcolor="#EEEEEE", row=1, col=2)
    sub += (". On that scale (95% CI of the difference),<br>the gain is " + headroom_verdict(spec) + ". "
            + HELD_OUT_NOTE)
    if lenient_models:
        sub += (f"<br>◇ = {', '.join(lenient_models)} with thinking off, if reasoning closed by &lt;/think&gt; instead "
                f"of the tag counted (over {LENIENT_SHOWN}% of its rows; graded as no reasoning, so a failure)")
    top = 130 + 20 * (sub.count("<br>") + 1)
    fig.update_layout(title=title(main, sub), margin={"t": top, "l": 330, "b": 70}, title_y=0.955,
                      title_yanchor="top")
    return save(fig, fig_dir, name, 1400, top + 70 * len(rows) + 70)


def upgraded_rows(summary: dict, key: str) -> dict | None:
    """Per model, the upgraded-prompt row of a specificity table (None for analysis runs without it)."""
    return {r["model"]: r for r in summary[key] if r["prompt"] == "upgraded"} if key in summary else None


def fig_cot_vs_output(c_on: dict, c_off: dict, summary: dict, fig_dir) -> str:
    """The upgraded prompt's gain, thinking on vs off, both without an opening sentence."""
    return fig_channels(
        c_on, c_off, {r["model"]: r for r in summary["cot_specificity"] if r["prompt"] == "upgraded"},
        before={"on": ("baseline", "none"), "off": ("baseline", "thinking_off")},
        after={"on": ("upgraded", "none"), "off": ("upgraded", "thinking_off")},
        labels=("CoT-Control's prompt", "upgraded prompt"), copying=None, name="S5_cot_vs_output",
        main="Is the gain about the chain of thought, or about following instructions in general?",
        sub="Same rule on the thinking trace (solid) or, with thinking off, on reasoning written in the answer "
            "(dotted); no opening sentence in either.<br>Left: open circle = CoT-Control's prompt, filled = upgraded, "
            "number = gain in points (texts under 200 tokens count as failures).<br>Right: each gain divided by the "
            "room it had, because the answer channel starts higher", fig_dir=fig_dir,
        lenient_spec=upgraded_rows(summary, "cot_specificity_lenient"))


def fig_opening_on_vs_off(c_on: dict, c_off: dict, summary: dict, fig_dir) -> str:
    """The compliant opening sentence's gain under the upgraded prompt, thinking on vs off."""
    return fig_channels(
        c_on, c_off, {r["model"]: r for r in summary["opening_specificity"] if r["prompt"] == "upgraded"},
        before={"on": ("upgraded", "none"), "off": ("upgraded", "thinking_off")},
        after={"on": ("upgraded", "prefill_compliant"), "off": ("upgraded", "prefill_compliant")},
        labels=("upgraded prompt", "upgraded prompt + compliant first sentence"), copying=("no_rule", "prefill_compliant"),
        name="S5b_opening_on_vs_off",
        main="Does writing a compliant first sentence help the same way in the answer?",
        sub="Upgraded prompt; the first sentence written for the model into its thinking trace (solid) or, with "
            "thinking off, right after &lt;output_reasoning&gt; (dotted).<br>Left: open circle = no first sentence, "
            "filled = compliant first sentence, number = gain in points, × = the sentence with no rule given "
            "(copying).<br>Graded on what the model writes after the sentence. Thinking off, every first sentence "
            "also opens the tag, which some models otherwise skip.<br>Right: each gain divided by the room it had",
        fig_dir=fig_dir, lenient_spec=upgraded_rows(summary, "opening_specificity_lenient"))


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
                    "condition, same model; ★ = upgraded prompt; dotted line = equal.<br>The held-out models have no "
                    "screen numbers, so they appear only on the left. " + HELD_OUT_NOTE),
        legend={"orientation": "h", "y": -0.2, "x": 0.78, "xanchor": "center"}, margin={"t": 170, "b": 110})
    return save(fig, fig_dir, "S7_cost_and_check", 1400, 600)


# --- S8-S12: after exp06 ------------------------------------------------------------------------------------------
LATER_RUNS = {"exp07": ("exp07_prompt_search", "test_v1"), "exp07b": ("exp07b_prompt_compare", "v1"),
              "exp08": ("exp08_openings_channels", "v2")}
LATER_SHORT = {**{m: s.rstrip("*") for m, s in SHORT_MODEL.items()}, "GLM-4.7-Flash": "GLM-4.7-Flash",
               "Qwen3.6-35B-A3B-FP8": "Qwen3.6-35B-A3B", "Gemma-4-12B": "Gemma-4-12B"}
LATER_COLOR = {**MODEL_COLOR, "GLM-4.7-Flash": "#56B4E9", "Qwen3.6-35B-A3B-FP8": "#F0E442", "Gemma-4-12B": "#999999"}
LATER_RULE_LABEL = {**RULE_LABEL, "alternating_case": "AlTeRnAtInG cAsE",
                    "multiple_word_suppression": "never use several given words",
                    "no_digits": "no digits (numbers in words)", "no_first_person": "no first person (I, we, ...)"}
EXP08_A_RULES = ["alternating_case", "no_digits", "no_first_person"]  # the rules prompt A runs on in every exp08 part
CANDIDATE_LABEL = {
    "many_examples": "A: up to 12 example traces instead of 3",
    "own_compliant": "B: the model's own unedited compliant openings as examples",
    "named_once": "C: banned items named once, then described positively",
    "own_guide": "L: the list of common failures written by the model itself",
    "monitored": "G: a note that the reasoning is monitored",
}
OPENING_CONTRASTS = [  # (exp08 contrast, its extension on exp06's rules, label)
    ("O1 compliant opening - none (A)", "O1x compliant opening - none (A), exp06 rules",
     "compliant first sentence vs none"),
    ("O2 rule + opening - opening only", "O2x rule + opening - opening only, exp06 rules",
     "rule + compliant sentence vs the sentence alone (copying)"),
    ("O3 commitment - compliant opening (A)", "O3x commitment - compliant opening (A), exp06 rules",
     "'I will follow the rule' sentence vs compliant sentence"),
    ("O4 non-compliant opening - none (A)", "O4x non-compliant opening - none (A), exp06 rules",
     "rule-breaking first sentence vs none"),
]


def load_later() -> dict:
    out = {}
    for key, (exp_dir, run) in LATER_RUNS.items():
        out[key] = json.loads((cfg.REPO_ROOT / "results" / exp_dir / "analysis" / run / "summary.json").read_text())
    out["exp08_cells"] = pd.read_csv(cfg.REPO_ROOT / "results" / LATER_RUNS["exp08"][0] / "analysis"
                                     / LATER_RUNS["exp08"][1] / "per_cell.csv")
    return out


def exp07_contrast(s: dict, group: str, contrast: str = "upgraded - baseline") -> dict:
    [c] = [c for c in s["contrasts"] if c["group"] == group and c["contrast"] == contrast]
    return c


EXP07_GROUPS = [("search rules x search models", "rules and models it was chosen on"),
                ("search rules x held-out models", "same rules, 4 new models"),
                ("held-out rules x search models", "4 new rules, same models"),
                ("held-out rules x held-out models (primary)", "4 new rules and 4 new models")]


def later_project_rows(later: dict) -> list[tuple[str, str, str]]:
    s7, s7b, s8 = later["exp07"], later["exp07b"], later["exp08"]["contrasts"]
    same = exp07_contrast(s7, EXP07_GROUPS[0][0])["difference"]["value"]
    new = exp07_contrast(s7, EXP07_GROUPS[3][0])["difference"]["value"]
    best_search = exp07_contrast(s7, EXP07_GROUPS[3][0], f"{s7['winner']} - upgraded")["difference"]["value"]
    r = s7b["results"]
    a, own_guide, monitored = (r[k]["primary"] for k in ("many_examples", "own_guide", "monitored"))
    o1, o1x, o4x, c2, acc1 = (s8[k] for k in ("O1 compliant opening - none (A)", OPENING_CONTRASTS[0][1],
                                               OPENING_CONTRASTS[3][1], "C2 thinking off - on (A)",
                                               "ACC1 A - no rule (accuracy)"))
    return [
        ("exp07", "Does the upgraded prompt hold on rules and models never used to choose it? Can a search beat it?",
         f"Partly: +{new:.0f} points S(1000) on 4 new rules and 4 new models, vs +{same:.0f} on the rules and "
         f"models it was chosen on.<br>The best of the searched variants matches it ({best_search:+.1f})."),
        ("exp07b", "Which single change to the upgraded prompt helps?",
         f"More examples (up to 12 instead of 3; this is prompt A): {fmt_ci(a)}. The model's own failure list hurts "
         f"({own_guide['value']:+.1f}); a 'you are monitored' note does nothing ({monitored['value']:+.1f})."),
        ("<b>exp08</b>", "<b>With that prompt (A): do first sentences, the channel and accuracy matter?</b>",
         f"<b>A compliant first sentence helps on formatting rules ({o1x['value']:+.0f}), not on the held-out "
         f"rules ({o1['value']:+.0f});<br>a rule-breaking one costs {o4x['value']:.0f}. Reasoning in the answer is "
         f"easier to control ({c2['value']:+.0f}).<br>Accuracy {acc1['value']:+.0f} vs no rule (Holm p = "
         f"{acc1['p_holm']:.2f}), with much shorter reasoning.</b>"),
    ]


def fmt_ci(s: dict, digits: int = 1) -> str:
    return f"{s['value']:+.{digits}f} [{s['ci'][0]:+.{digits}f}, {s['ci'][1]:+.{digits}f}]"


def dumbbell(fig, row: int, col: int, y: int, before: float, after: float, color: str, labels: tuple[str, str]):
    """Open circle (before) -> filled circle (after), with the change in points written beside it."""
    fig.add_trace(go.Scatter(x=[before, after], y=[y, y], mode="lines", showlegend=False, hoverinfo="skip",
                             line={"color": color, "width": 3}), row=row, col=col)
    fig.add_trace(go.Scatter(x=[before], y=[y], mode="markers", showlegend=False,
                             marker={"color": "white", "size": 11, "line": {"color": color, "width": 2}},
                             hovertemplate=f"{labels[0]}: {before:.1f}%<extra></extra>"), row=row, col=col)
    fig.add_trace(go.Scatter(x=[after], y=[y], mode="markers+text", showlegend=False,
                             text=[f"{after - before:+.0f}"], textposition="middle right",
                             textfont={"color": color, "size": 13}, marker={"color": color, "size": 12},
                             hovertemplate=f"{labels[1]}: {after:.1f}%<extra></extra>"), row=row, col=col)


def fig_heldout(s7: dict, fig_dir) -> str:
    """S8. exp07's test: CoT-Control's prompt (open) -> upgraded (filled), by how new the rules and models are, and
    per rule."""
    held_rules = ["alternating_case", "multiple_word_suppression", "no_digits", "no_first_person"]
    rule_order = list(cfg.EXP06_MODES) + held_rules
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.28, column_widths=[0.45, 0.55],
                        subplot_titles=["By how new the rules and models are", "Per rule (all 6 models)"])
    labels = ("CoT-Control's prompt", "upgraded prompt")
    for i, (group, label) in enumerate(EXP07_GROUPS):
        c = exp07_contrast(s7, group)
        dumbbell(fig, 1, 1, i, c["b"]["value"], c["a"]["value"], "#B15928", labels)
    fig.update_yaxes(tickvals=list(range(len(EXP07_GROUPS))), ticktext=[g[1] for g in EXP07_GROUPS],
                     range=[len(EXP07_GROUPS) - 0.5, -0.5], row=1, col=1, showline=False, ticks="")
    for i, rule in enumerate(rule_order):
        c = exp07_contrast(s7, f"rule: {rule}")
        dumbbell(fig, 1, 2, i, c["b"]["value"], c["a"]["value"], "#6A3D9A" if rule in held_rules else "#B15928",
                 labels)
    fig.add_hline(y=len(cfg.EXP06_MODES) - 0.5, line={"color": "#BBBBBB", "dash": "dot"}, row=1, col=2)
    fig.update_yaxes(tickvals=list(range(len(rule_order))),
                     ticktext=[LATER_RULE_LABEL[r] + (" (new)" if r in held_rules else "") for r in rule_order],
                     range=[len(rule_order) - 0.5, -0.5], row=1, col=2, showline=False, ticks="")
    for col in (1, 2):
        fig.update_xaxes(title_text="S(1000): % with no violation in the first 1000 tokens", range=[0, 70],
                         showgrid=True, gridcolor="#EEEEEE", row=1, col=col)
    same = exp07_contrast(s7, EXP07_GROUPS[0][0])["difference"]
    new = exp07_contrast(s7, EXP07_GROUPS[3][0])["difference"]
    held_out_models = ", ".join(LATER_SHORT[m] for m in exp07_contrast(s7, EXP07_GROUPS[3][0])["models"])
    fig.update_layout(
        title=title(f"The upgraded prompt carries over to new rules and models, but its gain shrinks from "
                    f"+{same['value']:.0f} to +{new['value']:.0f} points",
                    f"exp07's test: {s7['n_questions']} new questions, thinking on. Open circle = CoT-Control's "
                    "prompt, filled = upgraded, number = gain in points.<br>New rules (purple) and new models "
                    f"({held_out_models}) were never used to choose the prompt. Gain on 4 new rules x 4 new models: "
                    f"{fmt_ci(new)} (95% CI)."),
        margin={"t": 170, "l": 270, "r": 40, "b": 70})
    return save(fig, fig_dir, "S8_heldout", 1500, 600)


def fig_what_helps(s7b: dict, fig_dir, name: str = "S9_what_helps") -> str:
    """S9. exp07b: each single change to the upgraded prompt, its difference in S(1000) (95% CI, Holm over 5)."""
    names = list(CANDIDATE_LABEL)
    res = s7b["results"]
    fig = go.Figure()
    stats = [res[n]["primary"] for n in names]
    fig.add_trace(go.Scatter(
        x=[s["value"] for s in stats], y=list(range(len(names))), mode="markers+text", name="mean over cells",
        marker={"size": 13, "color": ["#B15928" if s["p_holm"] < 0.05 else "#888888" for s in stats]},
        error_x=error(stats), text=[f"{s['value']:+.1f}   (Holm p {'< 0.001' if s['p_holm'] < 0.001 else '= %.2f' % s['p_holm']})"
                                    for s in stats],
        textposition="top center", textfont={"size": 12}))
    for m, symbol in zip(s7b["models"], ("circle-open", "square-open", "diamond-open")):
        fig.add_trace(go.Scatter(x=[res[n]["per_model"][m]["value"] for n in names], y=list(range(len(names))),
                                 mode="markers", name=LATER_SHORT[m],
                                 marker={"symbol": symbol, "size": 10, "color": LATER_COLOR[m], "line": {"width": 2}}))
    fig.add_vline(x=0, line={"color": "#888888", "width": 1})
    fig.update_yaxes(tickvals=list(range(len(names))), ticktext=[CANDIDATE_LABEL[n] for n in names],
                     range=[len(names) - 0.5, -0.7], showline=False, ticks="")
    fig.update_xaxes(title_text="change in S(1000) vs the upgraded prompt (points)", showgrid=True, gridcolor="#EEEEEE")
    a = res["many_examples"]["primary"]
    fig.update_layout(
        title=title(f"One change at a time: only more examples clearly helps ({a['value']:+.1f} points)",
                    f"exp07b: the upgraded prompt with exactly one change, {s7b['n_questions']} new questions, "
                    f"{len(s7b['models'])} models x up to 9 rules, thinking on. A fits 12 examples on every rule but "
                    "'never use several given words' (7-8: that prompt is long).<br>Large markers: mean over model x "
                    "rule cells with 95% CI (orange: Holm p < 0.05 over the 5 changes); open markers: per model."),
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
        margin={"t": 130, "l": 430, "b": 100})
    return save(fig, fig_dir, name, 1400, 520)


def fig_openings(s8: dict, fig_dir, name: str = "S10_openings") -> str:
    """S10. exp08 under prompt A: the first-sentence contrasts on exp08's held-out rules and on exp06's rules."""
    contrasts = s8["contrasts"]
    panels = [(0, "held-out rules: AlTeRnAtInG cAsE, no first person"),
              (1, "exp06's 5 formatting and word rules")]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04, subplot_titles=[p[1] for p in panels])
    for col, (which, _) in enumerate(panels, start=1):
        stats = [contrasts[c[which]] for c in OPENING_CONTRASTS]
        fig.add_trace(go.Scatter(x=[s["value"] for s in stats], y=list(range(len(stats))), mode="markers+text",
                                 showlegend=False, marker={"size": 13, "color": "#6A3D9A"}, error_x=error(stats),
                                 text=[f"{s['value']:+.1f}" for s in stats], textposition="top center",
                                 textfont={"size": 12}), row=1, col=col)
        for m, symbol in zip(s8["models"], ("circle-open", "square-open", "diamond-open", "triangle-up-open")):
            fig.add_trace(go.Scatter(x=[s["per_model"][m]["value"] for s in stats], y=list(range(len(stats))),
                                     mode="markers", name=LATER_SHORT[m], showlegend=col == 1,
                                     marker={"symbol": symbol, "size": 10, "color": LATER_COLOR[m],
                                             "line": {"width": 2}}), row=1, col=col)
        fig.add_vline(x=0, line={"color": "#888888", "width": 1}, row=1, col=col)
        fig.update_xaxes(title_text="change in S(1000) of the text after the sentence (points)", range=[-70, 80],
                         showgrid=True, gridcolor="#EEEEEE", row=1, col=col)
    fig.update_yaxes(tickvals=list(range(len(OPENING_CONTRASTS))), ticktext=[c[2] for c in OPENING_CONTRASTS],
                     range=[len(OPENING_CONTRASTS) - 0.5, -0.7], showline=False, ticks="", row=1, col=1)
    o1, o1x, o4x = contrasts[OPENING_CONTRASTS[0][0]], contrasts[OPENING_CONTRASTS[0][1]], contrasts[OPENING_CONTRASTS[3][1]]
    fig.update_layout(
        title=title(f"A compliant first sentence still helps on formatting rules ({o1x['value']:+.0f}), not on the "
                    f"held-out rules ({o1['value']:+.0f}); a rule-breaking one costs {o4x['value']:.0f}",
                    "exp08, prompt A (the upgraded prompt with 12 example traces instead of 3), 160 new questions x 4 "
                    "models, thinking on. The first "
                    "sentence is written for the model; only what follows is scored. Purple: mean, 95% CI;<br>open: "
                    f"per model. Left: pre-registered (Holm p = {o1['p_holm']:.2f} for the top row). Right: added "
                    "after the first model's left result; its bottom row omits 'never use a word' (no rule-breaking "
                    "sentence)."),
        legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"},
        margin={"t": 200, "l": 380, "b": 100})
    return save(fig, fig_dir, name, 1500, 600)


def exp08_level(cells: pd.DataFrame, metric: str, cell: str, model: str, rules_: list[str]) -> float:
    d = cells[(cells["metric"] == metric) & (cells["cell"] == cell) & (cells["model"] == model)
              & cells["mode"].isin(rules_)]
    if len(d) != len(rules_):
        raise ValueError(f"{metric} {cell} {model}: {len(d)} of {len(rules_)} rules")
    return float(d["value"].mean())


def fig_channels_a(s8: dict, cells: pd.DataFrame, fig_dir) -> str:
    """S11. exp08: S(200) on the thinking trace (solid) and on reasoning written in the answer (dotted), CoT-Control's
    prompt (open) -> prompt A (filled), mean over A's 3 rules."""
    models = s8["models"]
    rows = [(m, label, ch) for m in models for label, ch in (("thinking trace", "on"),
                                                             ("reasoning written in the answer", "off"))]
    fig = make_subplots(rows=1, cols=1)
    for i, (m, label, ch) in enumerate(rows):
        before = exp08_level(cells, "S_200", f"baseline|none|{ch}", m, EXP08_A_RULES)
        after = exp08_level(cells, "S_200", f"A|none|{ch}", m, EXP08_A_RULES)
        dumbbell(fig, 1, 1, i, before, after, LATER_COLOR[m], ("CoT-Control's prompt", "prompt A"))
        if ch == "off":
            fig.data[-3].line.dash = "dot"
    for k in range(1, len(models)):
        fig.add_hline(y=2 * k - 0.5, line={"color": "#DDDDDD", "width": 1})
    fig.update_yaxes(tickvals=list(range(len(rows))),
                     ticktext=[f"<b>{LATER_SHORT[m]}</b>  {label}" if ch == "on" else label for m, label, ch in rows],
                     range=[len(rows) - 0.5, -0.7], showline=False, ticks="")
    fig.update_xaxes(title_text="S(200): % of texts with no violation in the first 200 tokens", range=[0, 100],
                     showgrid=True, gridcolor="#EEEEEE")
    answer_easier = sum(exp08_level(cells, "S_200", "A|none|off", m, EXP08_A_RULES)
                        > exp08_level(cells, "S_200", "A|none|on", m, EXP08_A_RULES) for m in models)
    con, tagged = s8["contrasts"], s8["tagged_only"]
    c2, c3, c4 = (con[k] for k in ("C2 thinking off - on (A)", "C3 A - CoT-Control prompt, thinking on",
                                   "C4 A - CoT-Control prompt, thinking off"))
    c2t, c4t = (tagged[k] for k in ("C2 thinking off - on (A)", "C4 A - CoT-Control prompt, thinking off"))
    fig.update_layout(
        title=title(f"Prompt A helps on the trace ({c3['value']:+.0f}) and in the answer ({c4['value']:+.0f}); the "
                    f"answer stays easier to control for {answer_easier} of {len(models)} models",
                    "exp08, 160 new questions, mean over 3 held-out rules (AlTeRnAtInG cAsE, no digits, no first "
                    "person). Open circle = CoT-Control's prompt, filled = prompt A.<br>Thinking off, the reasoning "
                    "goes inside &lt;output_reasoning&gt; tags in the answer; an answer with no closed tag counts as "
                    f"a failure. Answer minus trace under A: {fmt_ci(c2)} (Holm p {'< 0.001' if c2['p_holm'] < 0.001 else '= %.2f' % c2['p_holm']}).<br>"
                    f"Counting only answers that closed the tag: answer minus trace under A {fmt_ci(c2t)} (Qwen3.6 "
                    f"{c2t['per_model']['Qwen3.6-27B-FP8']['value']:+.0f}); A minus "
                    f"CoT-Control's prompt in the answer {fmt_ci(c4t)}."),
        margin={"t": 170, "l": 330, "b": 70})
    return save(fig, fig_dir, "S11_channels", 1400, 170 + 62 * len(rows) + 70)


def fig_cost_and_length(s8: dict, fig_dir, name: str = "S12_cost_and_length") -> str:
    """S12. exp08's accuracy part (40 questions, full traces): accuracy differences (95% CI) and median reasoning
    length per arm, A's 3 rules."""
    import cc_exp08_analysis as an8
    df, _ = an8.load(skip_missing=False)
    full = df[(df["part"] == "accuracy") & (df["mode"].isin(EXP08_A_RULES + [cfg.NO_CONSTRAINT]))]
    models = s8["models"]
    arms = [("no_rule|none|full", "no rule", "#000000"), ("baseline|none|full", "CoT-Control's prompt", "#9E9E9E"),
            ("A|none|full", "prompt A", "#6A3D9A")]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                        subplot_titles=["Accuracy: paired differences (95% CI)", "Median reasoning length (tokens)"])
    con = s8["contrasts"]
    diffs = [("ACC1 A - no rule (accuracy)", "prompt A - no rule", "#6A3D9A", -0.15),
             ("ACC3 A - CoT-Control prompt (accuracy)", "prompt A - CoT-Control's prompt", "#B15928", 0.15)]
    for contrast_name, label, color, offset in diffs:
        stats = [con[contrast_name]["per_model"][m] for m in models] + [con[contrast_name]]
        fig.add_trace(go.Scatter(x=[s["value"] for s in stats], y=[i + offset for i in range(len(stats))],
                                 mode="markers", name=label, marker={"size": 11, "color": color},
                                 error_x={**error(stats), "color": color}), row=1, col=1)
    fig.add_vline(x=0, line={"color": "#888888", "width": 1}, row=1, col=1)
    ticks = [LATER_SHORT[m] for m in models] + ["<b>all 4 models</b>"]
    fig.update_yaxes(tickvals=list(range(len(ticks))), ticktext=ticks, range=[len(ticks) - 0.5, -0.6], row=1, col=1)
    fig.update_xaxes(title_text="change in % correct (points)", range=[-30, 20], showgrid=True, gridcolor="#EEEEEE",
                     row=1, col=1)
    for k, (cell, label, color) in enumerate(arms):
        med = [float(full[(full["cell"] == cell) & (full["model"] == m)]["reasoning_tokens"].median()) for m in models]
        fig.add_trace(go.Bar(x=med, y=ticks[:-1], orientation="h", name=label, marker={"color": color},
                             text=[f"{v:.0f}" for v in med], textposition="outside", cliponaxis=False,
                             showlegend=True), row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="median reasoning tokens (cap 16000)", showgrid=True, gridcolor="#EEEEEE", row=1, col=2)
    acc1 = con["ACC1 A - no rule (accuracy)"]
    pooled = {cell: float(full[full["cell"] == cell]["reasoning_tokens"].median()) for cell, _, _ in arms}
    whole = {cell: 100 * float(full[(full["cell"] == cell)]["compliant"].astype(float).mean())
             for cell in ("baseline|none|full", "A|none|full")}
    trunc = {cell: 100 * float(full[full["cell"] == cell]["truncated"].astype(float).mean()) for cell, _, _ in arms}
    fig.update_layout(
        barmode="group",
        title=title(f"Prompt A may cost some accuracy ({acc1['value']:+.0f} points vs no rule, Holm p = "
                    f"{acc1['p_holm']:.2f}) and its reasoning is far shorter",
                    f"exp08, 40 new questions x 3 held-out rules x 4 models, full reasoning traces then an answer. "
                    f"Median reasoning: no rule {pooled['no_rule|none|full']:.0f}, CoT-Control's prompt "
                    f"{pooled['baseline|none|full']:.0f}, prompt A {pooled['A|none|full']:.0f} tokens.<br>Whole trace "
                    f"follows the rule: {whole['baseline|none|full']:.0f}% with CoT-Control's prompt vs "
                    f"{whole['A|none|full']:.0f}% with A, partly because shorter traces have fewer chances to slip. "
                    f"Traces cut by the cap: no rule {trunc['no_rule|none|full']:.0f}%,<br>CoT-Control's prompt "
                    f"{trunc['baseline|none|full']:.0f}%, A {trunc['A|none|full']:.0f}% (cut traces lower the no-rule "
                    "accuracy most, so A's true cost may be larger). 40 questions: CIs are wide."),
        legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"}, margin={"t": 205, "b": 110})
    return save(fig, fig_dir, name, 1500, 640)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="figure set name")
    parser.add_argument("--analysis-run", default="overnight", help="exp06 analysis run to read")
    parser.add_argument("--scratch", action="store_true", help="write under /tmp/story_scratch (layout tests)")
    parser.add_argument("--exp06-only", action="store_true", help="S0-S7 only, as in sets v1-v2.2")
    args = parser.parse_args()
    root = Path("/tmp/story_scratch") if args.scratch else cfg.REPO_ROOT / "figures" / "exp6_story"
    fig_dir = root / args.run
    if fig_dir.exists():
        raise SystemExit(f"{fig_dir} exists; figure sets are never overwritten")
    analysis = cfg.EXP06.results / "analysis" / args.analysis_run
    summary = json.loads((analysis / "summary.json").read_text())
    per_rule = pd.read_csv(analysis / "per_rule.csv")
    raw, missing = an.load(skip_missing=True)  # S4 reads only the thinking-on rows
    if [p for p in missing if p.endswith("_main")]:
        raise SystemExit(f"no graded thinking-on rows for {missing}")
    df = an.score(raw)
    later = None if args.exp06_only else load_later()
    fig_dir.mkdir(parents=True)
    c, c_off = cells(summary), cells(summary, thinking=False)
    rows = later_project_rows(later) if later else None
    names = [fig_task(fig_dir), fig_project(c, summary, fig_dir, rows), fig_headline(c, fig_dir), fig_ladder(c, fig_dir),
             fig_over_the_trace(df, fig_dir), fig_cot_vs_output(c, c_off, summary, fig_dir)]
    if set(MODELS) <= set((upgraded_rows(summary, "opening_specificity") or {})):  # thinking-off prefill data in
        names.append(fig_opening_on_vs_off(c, c_off, summary, fig_dir))
    names += [fig_per_rule(per_rule, fig_dir), fig_cost_and_check(summary, fig_dir)]
    if later:
        names += [fig_heldout(later["exp07"], fig_dir), fig_what_helps(later["exp07b"], fig_dir),
                  fig_openings(later["exp08"], fig_dir),
                  fig_channels_a(later["exp08"], later["exp08_cells"], fig_dir),
                  fig_cost_and_length(later["exp08"], fig_dir)]
    print("\n".join(str(fig_dir / f"{n}.png") for n in names))


if __name__ == "__main__":
    main()
