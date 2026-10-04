"""Story figures with prompt A as the treatment: the exp06 story (cc_story_figures.py) redone on exp08's data.

Prompt A is exp07b's winner (the upgraded prompt with up to 12 example traces instead of 3). It is compared with
CoT-Control's prompt (the control) and the upgraded prompt (the previous best) on exp07's 160 test questions and 9
rules: exp06's 5 and 4 held-out rules no prompt was chosen on.

Sources: exp08's graded rows plus exp07's test rows for the CoT-Control and upgraded prompts (thinking on; same
questions, seeds, stop and engine; exp07 tested Qwen3.8, Gemma and Qwen3.6, not Qwen3-32B), and the analysis
summaries of exp06, exp07b and exp08. Estimates not in an analysis report (ladder, survival curves, per-rule cells,
whole-trace rates, A vs upgraded, gain as a share of the room left) use exp08's estimator: Kaplan-Meier per (model,
rule), the mean over cells, and exp08's 2000 bootstrap draws of question weights. Comparisons that use exp07's rows
are cross-run and exploratory.

  E0_task              the 9 rules, with toy examples
  E1_project           the experiments in order (as story v3.1's S1)
  E2_prompts           what each arm is
  E3_how_A_was_chosen  one change at a time to the upgraded prompt (exp07b; as S9)
  E4_headline          the paper's metric: whole trace follows the rule, and survival over the whole trace
  E5_ladder            every arm on one axis: S(1000), one dot per model
  E6_over_the_trace    survival over the first 1150 tokens, per model
  E7_per_rule          S(1000) per rule and model, for the three prompts
  E8_A_vs_upgraded     A's gain over the upgraded prompt, on the questions it was chosen on and on new ones
  E9_openings          first-sentence contrasts under A (as S10)
  E10_cot_vs_output    A on the thinking trace vs on reasoning written in the answer, raw and as a share of the room
  E11_cost_and_length  accuracy and reasoning length (as S12)

Outputs (never overwritten): figures/exp8_story/<run>/E*.{png,html}.
Run: /venv/main/bin/python scripts/cc_exp08_story.py --run NAME
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
import cc_exp06_analysis as an6
import cc_exp07 as e7
import cc_exp08 as e8
import cc_exp08_analysis as an8
import cc_exp08_ext as x
import cc_story_figures as cs
from cc_story_figures import error, save, title
from cc_survival import kaplan_meier

EXP06_ANALYSIS_RUN = "with_qwen35_lenient"  # the run story v2.2-v3.1 used
MODELS = e8.MODELS
WITH_UPGRADED = [m for m in MODELS if m in e7.MODELS]  # exp07 tested them: upgraded rows on the same questions
WITHOUT_UPGRADED = [m for m in MODELS if m not in WITH_UPGRADED]
SHORT = {m: cs.LATER_SHORT[m] for m in MODELS}
COLOR = {m: cs.LATER_COLOR[m] for m in MODELS}
RULE_LABEL = cs.LATER_RULE_LABEL
EXP06_RULES, HELD_OUT_RULES = list(x.RULES), list(e8.RULES)
FORMATTING_RULES = list(cfg.EXP06_OPENER_MODES)  # exp06's 4 formatting rules (S4 used these)
A_RULES = an8.A_RULES  # the held-out rules A ran on thinking off and in the accuracy part
STOP = e7.STOP_TOKENS
T_LAST = STOP - an6.CUT_MARGIN  # survival curves end here (as in S4)

ARM = {  # cell -> (label, colour), the same in every figure
    "baseline|none|on": ("CoT-Control's prompt", "#9E9E9E"),
    "upgraded|none|on": ("Upgraded prompt (3 example traces)", "#B15928"),
    "A|none|on": ("Prompt A (12 example traces)", "#5B2C83"),
    "A|compliant|on": ("A + compliant first sentence", "#A27DD0"),
    "A|commitment|on": ("A + 'I will follow the rule' sentence", "#1F78B4"),
    "A|noncompliant|on": ("A + rule-breaking first sentence", "#E31A1C"),
    "no_rule|compliant|on": ("Compliant first sentence, no rule given (copying control)", "#000000"),
}
FULL_ARM = {"baseline|none|full": ARM["baseline|none|on"], "A|none|full": ARM["A|none|on"]}
NO_UPGRADED_NOTE = (f"{', '.join(SHORT[m] for m in WITHOUT_UPGRADED)} was not in exp07, so it has no "
                    "upgraded-prompt rows and no CoT-Control rows on exp06's rules.")


# --- Data -----------------------------------------------------------------------------------------------------------
def load_rows() -> pd.DataFrame:
    """exp08's graded rows (with exp07's CoT-Control rows on the held-out rules, as exp08's analysis loads them) plus
    exp07's CoT-Control rows on exp06's rules and its upgraded rows on all 9 rules."""
    rows, _ = an8.load(skip_missing=False)
    old = e7.load_grades("test", WITH_UPGRADED)
    old = old[((old["prompt"] == "baseline") & old["mode"].isin(EXP06_RULES)) | (old["prompt"] == "upgraded")]
    old = e7.survival(old.assign(cell=old["prompt"] + "|none|on", channel="on", part="prompts", condition="none"))
    return pd.concat([rows, old], ignore_index=True)


def pooled(values: dict, metric: str, cell: str, models: list[str], rules_: list[str]) -> dict | None:
    """Mean over the (model, rule) cells, with its bootstrap draws; None unless every cell exists."""
    keys = [(metric, cell, m, r) for m in models for r in rules_]
    if not keys or any(k not in values for k in keys):
        return None
    draws = np.mean([values[k][1] for k in keys], axis=0)
    return {"value": float(np.mean([values[k][0] for k in keys])), "draws": draws,
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def summarise(draws: np.ndarray, value: float) -> dict:
    return {"value": float(value), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "draws": draws}


def difference(a: dict, b: dict) -> dict:
    return summarise(a["draws"] - b["draws"], a["value"] - b["value"])


def full_trace_rates(rows: pd.DataFrame, column: str) -> dict:
    """(cell, model, rule) -> % of full-trace rows (accuracy part) with `column` true, with bootstrap draws."""
    index, weights = an8.question_weights(rows)
    out = {}
    full = rows[(rows["part"] == "accuracy") & rows["mode"].isin(A_RULES)]
    for (cell, model, mode), c in full.groupby(["cell", "model", "mode"]):
        y = c[column].astype(float).to_numpy()
        w = weights[:, c["item_id"].map(index).to_numpy()]
        out[(column, cell, model, mode)] = (100 * y.mean(), 100 * (w @ y) / w.sum(axis=1))
    return out


def survival_curve(rows: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Mean over rules of each rule's Kaplan-Meier curve, in %."""
    return 100 * np.mean([kaplan_meier(g["time"].to_numpy(), g["event"].to_numpy().astype(bool),
                                       np.ones((1, len(g))), grid)[0] for _, g in rows.groupby("mode")], axis=0)


def fmt(s: dict, digits: int = 1) -> str:
    return f"{s['value']:+.{digits}f} [{s['ci'][0]:+.{digits}f}, {s['ci'][1]:+.{digits}f}]"


# --- E0, E2: text figures -------------------------------------------------------------------------------------------
def fig_task(fig_dir) -> str:
    rules_ = [("all lowercase", "so the net force is mass times acceleration, which gives 12 n.",
               "<b>S</b>o the net force ..."),
              ("ALL UPPERCASE", "SO THE NET FORCE IS MASS TIMES ACCELERATION, WHICH GIVES 12 N.",
               "SO THE NET FORCE IS <b>mass</b> ..."),
              ("'meow' between words", "so meow the meow net meow force meow is meow mass ...",
               "so meow the meow net <b>force is</b> ..."),
              ("end sentences with 'safe'", "The net force is mass times acceleration safe. That gives 12 N safe.",
               "The net force is mass times acceleration<b>.</b>"),
              ("never use a given word (e.g. 'force')", "The net push is mass times acceleration, which gives 12 N.",
               "The net <b>force</b> is ..."),
              ("AlTeRnAtInG cAsE (new)", "So ThE NeT FoRcE Is MaSs TiMeS AcCeLeRaTiOn, WhIcH GiVeS 12 N.",
               "So ThE <b>net</b> ..."),
              ("never use several given words (new; e.g. 'force', 'mass')",
               "The net push is the object's inertia times its acceleration.", "The net push is <b>mass</b> ..."),
              ("no digits (new)", "so the net force is mass times acceleration, which gives twelve newtons.",
               "... which gives <b>12</b> N."),
              ("no first person (new)", "The net force is mass times acceleration, which gives 12 N.",
               "<b>I</b> think the net force ...")]
    fig = go.Figure(go.Table(
        columnwidth=[1.4, 2.6, 1.5],
        header={"values": ["<b>Rule about the reasoning trace</b>", "<b>Compliant reasoning (toy example)</b>",
                           "<b>First violation</b>"],
                "fill_color": "#EFEFEF", "align": "left", "font": {"size": 14}, "height": 34},
        cells={"values": [list(col) for col in zip(*rules_)], "align": "left", "height": 32, "font": {"size": 13},
               "fill_color": [["white"] * 5 + ["#F3EEF9"] * 4]}))
    fig.update_layout(
        title=title("The task: follow a rule about what goes in your own chain of thought",
                    "The CoT-Control benchmark gives a reasoning model a question plus one rule about its thinking; a "
                    "program checks the rule token by token.<br>If models can control their thinking on request, a "
                    "chain-of-thought monitor is easier to fool. Shaded: 4 rules no prompt was ever chosen on."),
        margin={"t": 110, "b": 130, "l": 30, "r": 30},
        annotations=[{"text": "<b>Measures</b><br><b>Whole trace</b>: % of reasoning traces with no violation from "
                              "start to end (the paper's measure).<br><b>S(1000)</b>, <b>S(200)</b>: % of traces "
                              "with no violation in their first 1000 (200) tokens. Generation stops at the first "
                              "violation, which makes large runs cheap.",
                      "xref": "paper", "yref": "paper", "x": 0, "y": 0.0, "yanchor": "top", "showarrow": False,
                      "align": "left", "font": {"size": 13}}])
    return save(fig, fig_dir, "E0_task", 1300, 590)


def fig_prompts(fig_dir) -> str:
    rows_ = [
        ("<b>Control</b>", ARM["baseline|none|on"][0], "CoT-Control's prompt: the question and the rule, stated once.",
         "CoT-Control paper"),
        ("Previous best", ARM["upgraded|none|on"][0], cs.UPGRADED_DEF.capitalize() + ".", "exp06a screen, exp06"),
        ("<b>Treatment</b>", "<b>" + ARM["A|none|on"][0] + "</b>",
         "<b>The upgraded prompt with 12 example traces instead of 3</b> (7-8 for 'never use several given words', "
         "whose prompt is long).", "exp07b (best of 5 single changes)"),
        ("Opening", ARM["A|compliant|on"][0], "A, with the first sentence of the reasoning written for the model, "
         "following the rule. Only what the model writes after it is scored.", "exp04-exp06, exp08"),
        ("Opening", ARM["A|commitment|on"][0], "A, with a first sentence that restates the rule and commits to it, "
         "written so it follows the rule.", "exp08"),
        ("Opening", ARM["A|noncompliant|on"][0], "A, with a neutral first sentence that breaks the rule.",
         "exp04-exp06, exp08"),
        ("Control", ARM["no_rule|compliant|on"][0], "No rule anywhere; the same compliant first sentence. Measures "
         "how much the model just copies the opening's style.", "exp05, exp08"),
        ("Reference", "No rule", "The question alone: the accuracy reference.", "exp08"),
    ]
    fig = go.Figure(go.Table(
        columnwidth=[0.7, 1.7, 3.6, 1.3],
        header={"values": ["<b>Role</b>", "<b>Arm</b>", "<b>What the model gets</b>", "<b>From</b>"],
                "fill_color": "#EFEFEF", "align": "left", "font": {"size": 14}, "height": 34},
        cells={"values": [list(col) for col in zip(*rows_)], "align": "left", "height": 40, "font": {"size": 13},
               "fill_color": [["white", "white", "#F3EEF9"] + ["white"] * 5]}))
    fig.update_layout(
        title=title("What is being compared",
                    "Example traces are the model's own reasoning on other questions, rewritten to follow the rule. "
                    "Thinking on unless stated; with thinking off, the reasoning goes inside &lt;output_reasoning&gt; "
                    "tags in the answer."),
        margin={"t": 100, "b": 20, "l": 30, "r": 30})
    return save(fig, fig_dir, "E2_prompts", 1400, 500)


# --- E4: headline (whole trace) ---------------------------------------------------------------------------------------
def fig_headline(rows: pd.DataFrame, fig_dir) -> str:
    rates = full_trace_rates(rows, "compliant")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, column_widths=[0.5, 0.5],
                        subplot_titles=["Whole trace follows the rule (%)",
                                        "% of traces with no violation yet, over the whole trace"])
    pooled_rate = {}
    for cell, (label, color) in FULL_ARM.items():
        stats = [pooled(rates, "compliant", cell, [m], A_RULES) for m in MODELS]
        pooled_rate[cell] = pooled(rates, "compliant", cell, MODELS, A_RULES)
        fig.add_trace(go.Bar(x=[SHORT[m] for m in MODELS], y=[s["value"] for s in stats], name=label,
                             marker={"color": color}, error_y=error(stats), text=[f"{s['value']:.0f}%" for s in stats],
                             textposition="outside", cliponaxis=False, legendgroup=label), row=1, col=1)
    fig.add_hline(y=2.3, line={"color": "#555555", "dash": "dot", "width": 1}, row=1, col=1)
    fig.update_yaxes(range=[0, 75], title_text="%", row=1, col=1)
    grid = np.unique(np.round(np.logspace(1, np.log10(e8.ACCURACY_REASONING_CAP), 150)))
    full = rows[(rows["part"] == "accuracy") & rows["mode"].isin(A_RULES)]
    medians = {}
    for cell, (label, color) in FULL_ARM.items():
        d = full[full["cell"] == cell]
        curve = np.mean([survival_curve(d[d["model"] == m], grid) for m in MODELS], axis=0)
        medians[cell] = float(d["reasoning_tokens"].median())
        fig.add_trace(go.Scatter(x=grid, y=curve, mode="lines", name=label, legendgroup=label, showlegend=False,
                                 line={"color": color, "width": 3},
                                 hovertemplate=f"{label}<br>token %{{x}}: %{{y:.0f}}%<extra></extra>"), row=1, col=2)
        fig.add_vline(x=medians[cell], line={"color": color, "dash": "dot", "width": 1.5}, row=1, col=2)
    fig.update_xaxes(type="log", title_text="tokens into the reasoning (log scale); dotted: median trace length",
                     showgrid=True, gridcolor="#EEEEEE", row=1, col=2)
    fig.update_yaxes(range=[0, 101], showgrid=True, gridcolor="#EEEEEE", row=1, col=2)
    b, a = pooled_rate["baseline|none|full"], pooled_rate["A|none|full"]
    fig.update_layout(
        barmode="group",
        title=title(f"Prompt A lifts whole-trace compliance from {b['value']:.0f}% to {a['value']:.0f}% "
                    f"({fmt(difference(a, b))} points), and is ahead at every point in the trace",
                    "The paper's measure. exp08's accuracy part: 40 new questions x 3 held-out rules (AlTeRnAtInG "
                    f"cAsE, no digits, no first person), full reasoning up to {e8.ACCURACY_REASONING_CAP} tokens;<br>"
                    "120 traces per bar, 95% CIs; dotted: the paper's Qwen3-32B, 2.3%.<br>Right: Kaplan-Meier, mean "
                    "over models and rules. The whole-trace rate also rewards A's shorter traces (median "
                    f"{medians['A|none|full']:.0f} vs {medians['baseline|none|full']:.0f} tokens, dotted); the curves "
                    "compare the prompts at the same token."),
        legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"}, margin={"t": 220, "b": 100})
    return save(fig, fig_dir, "E4_headline", 1500, 665)


# --- E5: ladder ---------------------------------------------------------------------------------------------------
LADDER = list(ARM)
LADDER_PANELS = [("exp06's 5 rules", EXP06_RULES), ("held-out rules with openings (AlTeRnAtInG, no first person)",
                                                     list(e8.OPENING_RULES))]


def row_rules(cell: str, rules_: list[str]) -> list[str]:
    """'never use a given word' has no rule-breaking first sentence."""
    return [r for r in rules_ if r in an8.EXTENSION_NONCOMPLIANT_RULES or r in e8.RULES] \
        if cell == "A|noncompliant|on" else rules_


def fig_ladder(values: dict, fig_dir) -> str:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[p[0] for p in LADDER_PANELS])
    y = {cell: i for i, cell in enumerate(LADDER)}
    offsets = dict(zip(MODELS, np.linspace(-0.24, 0.24, len(MODELS))))
    means = {}
    for col, (_, rules_) in enumerate(LADDER_PANELS, start=1):
        fig.add_hrect(y0=len(LADDER) - 1.5, y1=len(LADDER) - 0.5, fillcolor="#F2F2F2", line_width=0, layer="below",
                      row=1, col=col)
        for cell in LADDER:
            mean = pooled(values, "S_1000", cell, WITH_UPGRADED, row_rules(cell, rules_))
            means[(col, cell)] = mean
            fig.add_trace(go.Bar(x=[mean["value"]], y=[y[cell]], orientation="h", width=0.66, showlegend=False,
                                 marker={"color": ARM[cell][1], "opacity": 0.3}, hoverinfo="skip"), row=1, col=col)
            fig.add_annotation(x=mean["value"], y=y[cell], text=f" {mean['value']:.0f}", showarrow=False,
                               xanchor="left", yshift=-19, font={"size": 11, "color": "#666666"}, row=1, col=col)
        for m in MODELS:
            stats = [(cell, pooled(values, "S_1000", cell, [m], row_rules(cell, rules_))) for cell in LADDER]
            stats = [(cell, s) for cell, s in stats if s is not None]
            hollow = m in WITHOUT_UPGRADED
            fig.add_trace(go.Scatter(
                x=[s["value"] for _, s in stats], y=[y[cell] + offsets[m] for cell, _ in stats], mode="markers",
                name=SHORT[m], legendgroup=m, showlegend=col == 1,
                marker={"color": "white" if hollow else COLOR[m], "size": 10,
                        "line": {"color": COLOR[m], "width": 2 if hollow else 1}},
                error_x={**error([s for _, s in stats]), "color": COLOR[m], "thickness": 1.2, "width": 0},
                hovertemplate=[f"{ARM[cell][0]}: {s['value']:.1f}% [{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]<extra>"
                               f"{SHORT[m]}</extra>" for cell, s in stats]), row=1, col=col)
        fig.update_xaxes(title_text="S(1000): % with no violation in the first 1000 tokens", range=[0, 100],
                         showgrid=True, gridcolor="#EEEEEE", row=1, col=col)
    fig.update_yaxes(tickvals=list(y.values()), ticktext=[ARM[c][0] for c in LADDER], autorange="reversed",
                     showline=False, ticks="", row=1, col=1)
    base6, a6 = means[(1, "baseline|none|on")], means[(1, "A|none|on")]
    up6 = means[(1, "upgraded|none|on")]
    baseh, ah = means[(2, "baseline|none|on")], means[(2, "A|none|on")]
    fig.update_layout(
        title=title(f"Prompt A: {base6['value']:.0f}% → {a6['value']:.0f}% on exp06's rules (upgraded: "
                    f"{up6['value']:.0f}%), but only {baseh['value']:.0f}% → {ah['value']:.0f}% on the two held-out "
                    "rules here",
                    "exp08 + exp07's test rows, 160 new questions, thinking on. Bars and grey numbers: mean over "
                    f"{', '.join(SHORT[m] for m in WITH_UPGRADED)}; dots: per model, 95% CIs. Grey band: control."
                    "<br>A first sentence is written into the reasoning for the model; only what it writes after it "
                    "is scored. The rule-breaking row leaves out 'never use a given word'.<br>Hollow: "
                    + NO_UPGRADED_NOTE),
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
        margin={"t": 210, "l": 420, "b": 100})
    return save(fig, fig_dir, "E5_ladder", 1500, 700)


# --- E6: over the trace -----------------------------------------------------------------------------------------------
TRACE_ARMS = ["baseline|none|on", "upgraded|none|on", "A|none|on", "A|compliant|on", "A|noncompliant|on"]
TRACE_PANELS = [("exp06's 4 formatting rules", FORMATTING_RULES),
                ("held-out rules with openings", list(e8.OPENING_RULES))]


def fig_over_the_trace(rows: pd.DataFrame, fig_dir) -> str:
    grid = np.arange(0, T_LAST + 1, 10, dtype=float)
    on = rows[rows["channel"] == "on"]
    fig = make_subplots(rows=2, cols=len(MODELS), shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.025,
                        vertical_spacing=0.1, subplot_titles=[SHORT[m] for m in MODELS],
                        row_titles=[p[0] for p in TRACE_PANELS])
    for r, (_, rules_) in enumerate(TRACE_PANELS, start=1):
        for k, m in enumerate(MODELS, start=1):
            for cell in TRACE_ARMS:
                d = on[(on["cell"] == cell) & (on["model"] == m) & on["mode"].isin(rules_)]
                if d["mode"].nunique() != len(rules_):
                    continue
                label, color = ARM[cell]
                fig.add_trace(go.Scatter(x=grid, y=survival_curve(d, grid), mode="lines", name=label,
                                         legendgroup=label, showlegend=r == 1 and k == 1,
                                         line={"color": color, "width": 3 if cell == "A|none|on" else 2},
                                         hovertemplate=f"{label}<br>token %{{x}}: %{{y:.0f}}%<extra>{SHORT[m]}"
                                                       "</extra>"), row=r, col=k)
            fig.update_xaxes(range=[0, T_LAST], showgrid=True, gridcolor="#EEEEEE", row=r, col=k)
        fig.update_yaxes(title_text="% with no violation yet", range=[0, 101], row=r, col=1)
    for k in range(1, len(MODELS) + 1):
        fig.update_xaxes(title_text="tokens into the reasoning", row=2, col=k)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
    fig.update_layout(
        title=title("Control fades as the reasoning goes on; both example-based prompts slow the fade, and a bad "
                    "start is not recovered",
                    "Share of reasoning traces still following the rule at each point (Kaplan-Meier, mean over the "
                    "panel's rules), 160 new questions, thinking on. After a first sentence written for the<br>model, "
                    f"tokens count from after it. {NO_UPGRADED_NOTE}"),
        legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"}, margin={"t": 160, "b": 110})
    return save(fig, fig_dir, "E6_over_the_trace", 1550, 860)


# --- E7: per rule -------------------------------------------------------------------------------------------------
def fig_per_rule(rows: pd.DataFrame, values: dict, fig_dir) -> str:
    prompts = ["baseline|none|on", "upgraded|none|on", "A|none|on"]
    rules_ = EXP06_RULES + HELD_OUT_RULES
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.03, shared_yaxes=True,
                        subplot_titles=[ARM[p][0] for p in prompts])
    for k, cell in enumerate(prompts, start=1):
        z = np.array([[values[("S_1000", cell, m, r)][0] if ("S_1000", cell, m, r) in values else np.nan
                       for m in MODELS] for r in rules_])
        text = np.where(np.isnan(z), "not run", np.char.mod("%.0f", np.nan_to_num(z)))
        fig.add_trace(go.Heatmap(z=z, x=[SHORT[m] for m in MODELS], y=[RULE_LABEL[r] for r in rules_], zmin=0,
                                 zmax=100, colorscale=[[0, "#FFFFFF"], [1, "#5B2C83"]], showscale=k == 3,
                                 colorbar={"title": "S(1000) %"}, text=text, texttemplate="%{text}",
                                 textfont={"size": 13}, xgap=3, ygap=3,
                                 hovertemplate="%{y} | %{x}: %{z:.1f}%<extra></extra>"), row=1, col=k)
    fig.add_hline(y=len(EXP06_RULES) - 0.5, line={"color": "#333333", "width": 2})
    fig.update_yaxes(autorange="reversed", ticks="", showline=False)
    fig.update_xaxes(ticks="", showline=False, tickangle=-30)
    gain = {r: pooled(values, "S_1000", "A|none|on", WITH_UPGRADED, [r])["value"]
            - pooled(values, "S_1000", "baseline|none|on", WITH_UPGRADED, [r])["value"] for r in rules_}
    best = max(gain, key=gain.get)
    small = [RULE_LABEL[r] for r in rules_ if gain[r] < 5]
    a_rows = rows[(rows["cell"] == "A|none|on")]
    first_word = a_rows.assign(first_word=a_rows["event"].astype(bool) & (a_rows["fv_char"] <= 4)) \
        .groupby(["model", "mode"])["first_word"].mean()
    habit = [f"{SHORT[m]} on {RULE_LABEL[r]} ({100 * v:.0f}% of traces)" for (m, r), v in first_word.items()
             if v >= 0.9]
    fig.update_layout(
        title=title(f"A helps most on {RULE_LABEL[best]} (+{gain[best]:.0f}); under 5 points on {len(small)} "
                    f"rule{'s' if len(small) != 1 else ''}",
                    "S(1000) per rule (160 new questions, thinking on, no first sentence). Below the line: the 4 "
                    "held-out rules. Gains: A minus CoT-Control's prompt, mean over<br>"
                    f"{', '.join(SHORT[m] for m in WITH_UPGRADED)}. Under 5 points: {', '.join(small)}."
                    + (f"<br>Broken in the first word under A (e.g. a habitual 'Okay,'): {'; '.join(habit)}."
                       if habit else "")),
        margin={"t": 215, "l": 240, "b": 90})
    return save(fig, fig_dir, "E7_per_rule", 1500, 680)


# --- E8: A vs upgraded, chosen-on vs new questions -------------------------------------------------------------------
def fig_a_vs_upgraded(values: dict, s7b: dict, fig_dir) -> str:
    """exp07b's A - upgraded (90 questions A was chosen on) next to the same difference on exp08's 160 new questions."""
    res = s7b["results"]["many_examples"]
    rules_ = EXP06_RULES + HELD_OUT_RULES
    models = list(dict.fromkeys(s7b["models"] + WITH_UPGRADED))
    rows_ = []  # (label, chosen-on stat or None, new-questions stat or None)
    for m in models:
        new = None
        if m in WITH_UPGRADED:
            new = difference(pooled(values, "S_1000", "A|none|on", [m], rules_),
                             pooled(values, "S_1000", "upgraded|none|on", [m], rules_))
        rows_.append((SHORT.get(m, cs.LATER_SHORT[m]), res["per_model"].get(m), new))
    new_all = difference(pooled(values, "S_1000", "A|none|on", WITH_UPGRADED, rules_),
                         pooled(values, "S_1000", "upgraded|none|on", WITH_UPGRADED, rules_))
    by_set = {name: difference(pooled(values, "S_1000", "A|none|on", WITH_UPGRADED, rs),
                               pooled(values, "S_1000", "upgraded|none|on", WITH_UPGRADED, rs))
              for name, rs in (("exp06's rules", EXP06_RULES), ("held-out rules", HELD_OUT_RULES))}
    rows_.append(("<b>mean over models</b>", res["primary"], new_all))
    on_exp06, on_held_out = by_set["exp06's rules"], by_set["held-out rules"]
    in_both = [m for m in s7b["models"] if m in WITH_UPGRADED]
    both_text = ", ".join(f"{SHORT[m]} {res['per_model'][m]['value']:+.1f} → {r[2]['value']:+.1f}"
                          for m in in_both for r in rows_ if r[0] == SHORT[m])
    fig = go.Figure()
    for which, (label, color, symbol, offset) in enumerate(
            [("90 questions A was chosen on (exp07b)", "#999999", "circle-open", -0.13),
             ("160 new questions (exp08 vs exp07's rows)", "#5B2C83", "circle", 0.13)]):
        pts = [(i, r[1 + which]) for i, r in enumerate(rows_) if r[1 + which] is not None]
        fig.add_trace(go.Scatter(x=[s["value"] for _, s in pts], y=[i + offset for i, _ in pts], mode="markers+text",
                                 name=label, marker={"size": 12, "color": color, "symbol": symbol,
                                                     "line": {"width": 2, "color": color}},
                                 error_x={**error([s for _, s in pts]), "color": color},
                                 text=[f"{s['value']:+.1f}" for _, s in pts], textposition="top center",
                                 textfont={"size": 11, "color": color}))
    fig.add_vline(x=0, line={"color": "#888888", "width": 1})
    fig.update_yaxes(tickvals=list(range(len(rows_))), ticktext=[r[0] for r in rows_],
                     range=[len(rows_) - 0.5, -0.6], showline=False, ticks="")
    fig.update_xaxes(title_text="A minus the upgraded prompt, S(1000) points (95% CI)", showgrid=True,
                     gridcolor="#EEEEEE")
    fig.update_layout(
        title=title(f"A's edge over the upgraded prompt shrinks on new questions: {res['primary']['value']:+.1f} → "
                    f"{new_all['value']:+.1f} points",
                    "Same 9 rules, thinking on. exp07b compared the two prompts within one run; on the new questions "
                    "A's rows (exp08) are compared with exp07's<br>upgraded rows (same questions, seeds, stop and "
                    "engine; cross-run, exploratory). On new questions: exp06's rules "
                    f"{fmt(on_exp06)}, held-out rules {fmt(on_held_out)}.<br>The model sets differ; for the models "
                    f"in both: {both_text}."),
        legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"}, margin={"t": 160, "l": 200, "b": 100})
    return save(fig, fig_dir, "E8_A_vs_upgraded", 1300, 580)


# --- E10: CoT vs output -------------------------------------------------------------------------------------------
CHANNELS = [("thinking trace", "on"), ("reasoning written in the answer", "off")]


def headroom(before: dict, after: dict) -> dict:
    """The gain as a % of the room the control left: (after - before) / (100 - before)."""
    return summarise(100 * (after["draws"] - before["draws"]) / (100 - before["draws"]),
                     100 * (after["value"] - before["value"]) / (100 - before["value"]))


def fig_cot_vs_output(rows: pd.DataFrame, values: dict, fig_dir) -> str:
    """Per model: S(200) under CoT-Control's prompt (open) and A (filled), thinking trace (solid) and answer (dotted),
    mean over A's 3 held-out rules; right, each gain as a share of the room left. Diamonds: thinking off, counting
    only answers that closed the reasoning tag."""
    tagged = an8.cell_values(rows[(rows["channel"] == "off") & (rows["external_blocks"].fillna(0) > 0)])
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05, column_widths=[0.55, 0.45],
                        subplot_titles=["Raw S(200): CoT-Control's prompt → prompt A",
                                        "Gain as a share of the room left to improve"])
    spec, tag_driven, i = {}, [], 0
    for m in MODELS:
        color = COLOR[m]
        for label, ch in CHANNELS:
            before = pooled(values, "S_200", f"baseline|none|{ch}", [m], A_RULES)
            after = pooled(values, "S_200", f"A|none|{ch}", [m], A_RULES)
            share = headroom(before, after)
            spec[(m, ch)] = share
            dash = "solid" if ch == "on" else "dot"
            fig.add_trace(go.Scatter(x=[before["value"], after["value"]], y=[i, i], mode="lines", showlegend=False,
                                     hoverinfo="skip", line={"color": color, "width": 4 if ch == "on" else 2,
                                                             "dash": dash}), row=1, col=1)
            fig.add_trace(go.Scatter(x=[before["value"]], y=[i], mode="markers", showlegend=False,
                                     marker={"color": "white", "size": 11, "line": {"color": color, "width": 2}},
                                     hovertemplate=f"CoT-Control's prompt: {before['value']:.0f}%<extra></extra>"),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=[after["value"]], y=[i], mode="markers+text", showlegend=False,
                                     text=[f"{after['value'] - before['value']:+.0f}"], textposition="middle right",
                                     textfont={"color": color, "size": 13}, marker={"color": color, "size": 12},
                                     hovertemplate=f"prompt A: {after['value']:.0f}%<extra></extra>"), row=1, col=1)
            if ch == "off":
                t_before, t_after = (pooled(tagged, "S_200", cell, [m], A_RULES)
                                     for cell in ("baseline|none|off", "A|none|off"))
                for what, t in (("CoT-Control's prompt", t_before), ("prompt A", t_after)):
                    fig.add_trace(cs.lenient_marker(t["value"], i, color, f"{what}, tagged answers only: "
                                                    f"{t['value']:.0f}%"), row=1, col=1)
                if after["value"] < before["value"] and t_after["value"] > t_before["value"]:
                    tag_driven.append(SHORT[m])
            fig.add_trace(go.Scatter(
                x=[share["value"]], y=[i], mode="markers+text", showlegend=False, text=[f"{share['value']:.0f}%"],
                textposition="top center", textfont={"color": color, "size": 12},
                marker={"color": color if ch == "on" else "white", "size": 12, "line": {"color": color, "width": 2}},
                error_x={**error([share]), "color": color}), row=1, col=2)
            i += 1
    for k in range(1, len(MODELS)):
        for col in (1, 2):
            fig.add_hline(y=2 * k - 0.5, line={"color": "#DDDDDD", "width": 1}, row=1, col=col)
    fig.update_yaxes(tickvals=list(range(i)),
                     ticktext=[f"<b>{SHORT[m]}</b>  {label}" if ch == "on" else label
                               for m in MODELS for label, ch in CHANNELS],
                     range=[i - 0.5, -0.7], showline=False, ticks="")
    fig.update_xaxes(title_text="% of texts with no violation in the first 200 tokens", range=[0, 100],
                     showgrid=True, gridcolor="#EEEEEE", row=1, col=1)
    lowest = min(s["ci"][0] for s in spec.values())
    fig.update_xaxes(title_text="gain / (100 - CoT-Control's %), with 95% CI", range=[min(0, lowest - 5), 100],
                     showgrid=True, gridcolor="#EEEEEE", row=1, col=2)
    verdict = {"trace": [], "answer": [], "neither": []}
    for m in MODELS:
        d = spec[(m, "on")]["draws"] - spec[(m, "off")]["draws"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict["trace" if lo > 0 else "answer" if hi < 0 else "neither"].append(SHORT[m])
    verdict_text = "; ".join(f"{k}: {', '.join(v)}" for k, v in
                             (("larger on the trace", verdict["trace"]), ("larger in the answer", verdict["answer"]),
                              ("no clear difference", verdict["neither"])) if v)
    fig.update_layout(
        title=title(f"Is A's gain about the chain of thought? As a share of the room left, it is larger on the "
                    f"trace for {len(verdict['trace'])} of {len(MODELS)} models",
                    "exp08, 160 new questions, mean over AlTeRnAtInG cAsE, no digits, no first person; no first "
                    "sentence. Solid: rule on the thinking trace; dotted: thinking off, rule on the reasoning<br>"
                    "written in &lt;output_reasoning&gt; tags (an answer with no closed tag counts as a failure; ◇ = "
                    "counting only answers that closed it). Right: gain / room left, because the answer channel "
                    f"starts higher.<br>On that scale (95% CI of the difference): {verdict_text}."
                    + (f"<br>{', '.join(tag_driven)}: A's loss in the answer comes from answers that never close the "
                       "tag (◇ goes up)." if tag_driven else "")),
        margin={"t": 245, "l": 330, "b": 70})
    return save(fig, fig_dir, "E10_cot_vs_output", 1450, 245 + 70 * i + 70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="figure set name")
    parser.add_argument("--scratch", action="store_true", help="write under /tmp/exp8_story_scratch (layout tests)")
    args = parser.parse_args()
    root = Path("/tmp/exp8_story_scratch") if args.scratch else cfg.REPO_ROOT / "figures" / "exp8_story"
    fig_dir = root / args.run
    if fig_dir.exists():
        raise SystemExit(f"{fig_dir} exists; figure sets are never overwritten")
    s6 = json.loads((cfg.EXP06.results / "analysis" / EXP06_ANALYSIS_RUN / "summary.json").read_text())
    later = cs.load_later()
    rows = load_rows()
    values = an8.cell_values(rows)
    fig_dir.mkdir(parents=True)
    names = [fig_task(fig_dir),
             cs.fig_project(cs.cells(s6), s6, fig_dir, cs.later_project_rows(later), name="E1_project"),
             fig_prompts(fig_dir),
             cs.fig_what_helps(later["exp07b"], fig_dir, name="E3_how_A_was_chosen"),
             fig_headline(rows, fig_dir), fig_ladder(values, fig_dir), fig_over_the_trace(rows, fig_dir),
             fig_per_rule(rows, values, fig_dir), fig_a_vs_upgraded(values, later["exp07b"], fig_dir),
             cs.fig_openings(later["exp08"], fig_dir, name="E9_openings"),
             fig_cot_vs_output(rows, values, fig_dir),
             cs.fig_cost_and_length(later["exp08"], fig_dir, name="E11_cost_and_length")]
    print("\n".join(str(fig_dir / f"{n}.png") for n in names))


if __name__ == "__main__":
    main()
