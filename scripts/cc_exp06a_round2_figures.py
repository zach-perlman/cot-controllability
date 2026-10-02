"""exp06a round 2 figures (deviations_round2.json, deviations_round2_decision.json), organised by question rather
than one strip of every arm. Exploratory, like all of round 2. Same rows and scoring as cc_exp06a_analysis.

  R2_1_ladder        per model: S(1000), thinking on, as pieces are added to the prompt, next to exp04's prefill arms
  R2_2_piece_effects the paired gain from adding one piece (3-shot, failure guide, start-with, stacked), by what the
                     prompt already contains: equal bars = additive, growing bars = the pieces need each other
  R2_3_per_rule      S(1000) per rule for the ladder arms (thinking on)
  R2_4_cot_vs_output gain over baseline with thinking on vs off, length-robust (share of texts reaching 200 tokens
                     with no violation), on the 51 thinking-off questions: below the diagonal = specific to the CoT
                     (only arms whose thinking-off rows are graded)

Outputs (never overwritten): figures/exp06a_prompt_screen/<run>/R2_*.{html,png} and
results/exp06a_prompt_screen/analysis/<run>/round2_figure_data.json.
Run: /venv/main/bin/python scripts/cc_exp06a_round2_figures.py --run NAME
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cc_analysis as ca
import cc_config as cfg
import cc_exp04_analysis as a4
import cc_exp06a
import cc_exp06a_analysis as an

EXP = cfg.EXP06A
MODEL_STYLE = {"Qwen3.8-27B-FP8": ("#0072B2", "circle"), "Gemma-4-31B-FP8": ("#D55E00", "diamond"),
               "Qwen3-32B": ("#009E73", "square")}
# (group, arm, label), top to bottom. In-environment reruns for baseline and stacked.
LADDER = [
    ("reference", "baseline_rerun", "CoT-Control prompt (baseline)"),
    ("reference", "stacked_rerun", "stacked prompt (Jozdien)"),
    ("baseline + pieces", "start_with", "baseline + start-with"),
    ("baseline + pieces", "failure_guide", "baseline + failure guide"),
    ("baseline + pieces", "fewshot", "baseline + 3-shot"),
    ("baseline + pieces", "fewshot_failure_guide", "baseline + 3-shot + guide"),
    ("stacked + pieces", "stacked_start_with", "stacked + start-with"),
    ("stacked + pieces", "stacked_failure_guide", "stacked + failure guide"),
    ("stacked + pieces", "stacked_fewshot", "stacked + 3-shot"),
    ("stacked + pieces", "stacked_fewshot_failure_guide", "stacked + 3-shot + guide"),
    ("stacked + pieces", "stacked_all", "stacked + 3-shot + guide + start-with"),
    ("prefill (exp04)", "exp04_prefill", "compliant prefill (scored after it)"),
    ("prefill (exp04)", "exp04_prefill_stacked", "compliant prefill + stacked"),
]
GROUP_SHADE = {"reference": "#f2f2f2", "baseline + pieces": "#fff7e6", "stacked + pieces": "#e8f1fa",
               "prefill (exp04)": "#f2f2f2"}
LABEL = {arm: label for _, arm, label in LADDER}
# piece -> [(what the prompt already contains, arm without the piece, arm with it)]
PIECE_EFFECTS = {
    "+ 3-shot": [("baseline", "baseline_rerun", "fewshot"),
                 ("baseline + guide", "failure_guide", "fewshot_failure_guide"),
                 ("stacked", "stacked_rerun", "stacked_fewshot"),
                 ("stacked + guide", "stacked_failure_guide", "stacked_fewshot_failure_guide")],
    "+ failure guide": [("baseline", "baseline_rerun", "failure_guide"),
                        ("baseline + 3-shot", "fewshot", "fewshot_failure_guide"),
                        ("stacked", "stacked_rerun", "stacked_failure_guide"),
                        ("stacked + 3-shot", "stacked_fewshot", "stacked_fewshot_failure_guide")],
    "+ start-with": [("baseline", "baseline_rerun", "start_with"),
                     ("stacked", "stacked_rerun", "stacked_start_with"),
                     ("stacked + 3-shot + guide", "stacked_fewshot_failure_guide", "stacked_all")],
    "+ stacked": [("baseline", "baseline_rerun", "stacked_rerun"),
                  ("start-with", "start_with", "stacked_start_with"),
                  ("guide", "failure_guide", "stacked_failure_guide"),
                  ("3-shot", "fewshot", "stacked_fewshot"),
                  ("3-shot + guide", "fewshot_failure_guide", "stacked_fewshot_failure_guide")],
}
COT_VS_OUTPUT_ARMS = ["stacked", "start_with", "warmup", "fewshot", "style_guide", "failure_guide",
                      "monitor_stealth"] + an.ROUND2
SHORT = {"stacked": "stacked", "start_with": "start-with", "warmup": "warm-up", "fewshot": "3-shot",
         "style_guide": "style guide", "failure_guide": "guide", "monitor_stealth": "monitor",
         "stacked_failure_guide": "st+guide", "stacked_fewshot": "st+3shot", "stacked_start_with": "st+start",
         "stacked_all": "st+all", "fewshot_failure_guide": "3shot+guide", "stacked_fewshot_failure_guide": "st+3shot+guide"}


def load() -> tuple[pd.DataFrame, list[str], list[str]]:
    new, missing = an.exp06a_rows(an.MODELS, skip_missing=True)
    models = [m for m in an.MODELS if f"{m}{cc_exp06a.request_parts(m)[0]}" not in missing]
    raw = pd.concat([new, an.reference_rows(models)], ignore_index=True)
    return an.score(raw[[c for c in an.COLUMNS if c in raw.columns]], True), models, missing


def value(est) -> dict | None:
    return ca.stat(*est) if est else None


def fig_ladder(draws: dict, models: list[str], fig_dir) -> tuple[str, list[dict]]:
    arms = [a for _, a, _ in LADDER if any(draws.get((m, a, True, "S_t_star")) for m in models)]
    y = {a: i for i, a in enumerate(arms)}
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=models)
    data = []
    for k, model in enumerate(models):
        color, symbol = MODEL_STYLE[model]
        pts = [(a, value(draws.get((model, a, True, "S_t_star")))) for a in arms]
        pts = [(a, s) for a, s in pts if s]
        data += [{"model": model, "arm": a, "S_1000": s} for a, s in pts]
        fig.add_trace(go.Scatter(
            x=[s["value"] for _, s in pts], y=[y[a] for a, _ in pts], mode="markers+text", showlegend=False,
            marker={"color": color, "symbol": symbol, "size": 11, "line": {"color": "black", "width": 0.6}},
            error_x={"type": "data", "symmetric": False, "array": [s["ci"][1] - s["value"] for _, s in pts],
                     "arrayminus": [s["value"] - s["ci"][0] for _, s in pts], "color": color, "thickness": 1.5},
            text=[f"{s['value']:.0f}" for _, s in pts], textposition="top center", textfont={"size": 10, "color": color},
            hovertemplate=[f"{LABEL[a]}: {s['value']:.1f} [{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]<extra>{model}</extra>"
                           for a, s in pts]), row=1, col=k + 1)
        for arm, dash in (("stacked_rerun", "dot"), ("exp04_prefill_stacked", "dash")):
            s = value(draws.get((model, arm, True, "S_t_star")))
            if s:
                fig.add_vline(x=s["value"], line={"color": "#555555", "dash": dash, "width": 1}, row=1, col=k + 1)
        for group, shade in GROUP_SHADE.items():
            rows = [y[a] for g, a, _ in LADDER if g == group and a in y]
            if rows:
                fig.add_hrect(y0=min(rows) - 0.5, y1=max(rows) + 0.5, fillcolor=shade, line_width=0, layer="below",
                              row=1, col=k + 1)
        fig.update_xaxes(range=[0, 100], title_text="% with no violation in the first 1000 tokens", row=1, col=k + 1)
    fig.update_yaxes(tickvals=list(range(len(arms))), ticktext=[LABEL[a] for a in arms], autorange="reversed",
                     row=1, col=1)
    fig.update_layout(title="R2-1. Adding pieces to the prompt, thinking on (100 questions, mean over the 5 rules)"
                            "<br><sup>Dots: S(1000) with 95% question-level bootstrap CIs. Dotted line: stacked; "
                            "dashed: exp04's compliant prefill + stacked. Shaded bands: references; one or two "
                            "pieces added to the baseline prompt;<br>pieces added to the stacked prompt; exp04 "
                            "prefill arms (scored after the prefill). Harness arms are scored on the whole trace. "
                            "Exploratory: arms chosen on these questions</sup>",
                      margin={"l": 270, "t": 120, "b": 70})
    return a4.save(fig, fig_dir, "R2_1_ladder", 1600, 650), data


def fig_piece_effects(draws: dict, models: list[str], fig_dir) -> tuple[str, list[dict]]:
    pieces = list(PIECE_EFFECTS)
    fig = make_subplots(rows=1, cols=len(pieces), horizontal_spacing=0.09, subplot_titles=pieces)
    offsets = dict(zip(models, np.linspace(-0.22, 0.22, len(models))))
    data = []
    for c, piece in enumerate(pieces):
        contexts = PIECE_EFFECTS[piece]
        for model in models:
            color, symbol = MODEL_STYLE[model]
            xs, ys, lo, hi, hover = [], [], [], [], []
            for i, (context, without, with_) in enumerate(contexts):
                ea, eb = draws.get((model, with_, True, "S_t_star")), draws.get((model, without, True, "S_t_star"))
                if not ea or not eb:
                    continue
                d = ca.stat(ea[0] - eb[0], ea[1] - eb[1])
                data.append({"model": model, "piece": piece, "already_in_prompt": context, "without": without,
                             "with": with_, "gain": d})
                xs.append(d["value"]); ys.append(i + offsets[model])
                lo.append(d["value"] - d["ci"][0]); hi.append(d["ci"][1] - d["value"])
                hover.append(f"{piece} to {context}: {d['value']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]")
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=model, legendgroup=model, showlegend=c == 0,
                marker={"color": color, "symbol": symbol, "size": 10, "line": {"color": "black", "width": 0.6}},
                error_x={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo, "color": color,
                         "thickness": 1.5},
                hovertemplate=[h + f"<extra>{model}</extra>" for h in hover]), row=1, col=c + 1)
        fig.add_vline(x=0, line={"color": "#555555", "width": 1}, row=1, col=c + 1)
        fig.update_yaxes(tickvals=list(range(len(contexts))), ticktext=[f"to {ctx}" for ctx, _, _ in contexts],
                         autorange="reversed", row=1, col=c + 1)
        fig.update_xaxes(title_text="gain in S(1000), points", range=[-20, 60], row=1, col=c + 1)
    fig.update_layout(title="R2-2. What each piece adds, by what the prompt already contains (thinking on, 100 "
                            "questions)<br><sup>Paired difference in S(1000) (with the piece minus without), 95% "
                            "bootstrap CIs. If pieces simply added up, the rows in a panel would match; rows that "
                            "grow downwards mean the pieces work together</sup>",
                      legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"},
                      margin={"l": 60, "t": 110, "b": 110})
    return a4.save(fig, fig_dir, "R2_2_piece_effects", 1700, 520), data


def fig_per_rule(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap, fig_dir) -> tuple[str, list[dict]]:
    arms = [a for _, a, _ in LADDER]
    d = an.per_rule_table(df[df["thinking"] & df["arm"].isin(arms)], models, boot)
    d = d[d["t"] == an.T_STAR]
    arms = [a for a in arms if a in set(d["arm"])]
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.02, subplot_titles=models)
    for k, model in enumerate(models):
        m = (d[d["model"] == model].pivot(index="arm", columns="mode", values="S")
             .reindex(index=arms, columns=an.RULES))
        fig.add_trace(go.Heatmap(z=m.to_numpy(), x=[a4.MODE_LABEL[r].split(" (")[0] for r in an.RULES],
                                 y=[LABEL[a] for a in arms], zmin=0, zmax=100, colorscale="Viridis",
                                 showscale=k == len(models) - 1,
                                 text=m.map(lambda v: "" if pd.isna(v) else f"{v:.0f}").to_numpy(),
                                 texttemplate="%{text}", hovertemplate="%{y} | %{x}: %{z:.1f}%<extra></extra>"),
                      row=1, col=k + 1)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="R2-3. Where the gains come from: S(1000) per rule, thinking on, % of 100 questions",
                      margin={"l": 270, "t": 90})
    return a4.save(fig, fig_dir, "R2_3_per_rule", 1500, 620), d.to_dict("records")


def fig_cot_vs_output(draws_half: dict, models: list[str], n_half: int, fig_dir) -> tuple[str, list[dict]]:
    fig = make_subplots(rows=1, cols=len(models), horizontal_spacing=0.05, subplot_titles=models)
    data = []
    for k, model in enumerate(models):
        color, _ = MODEL_STYLE[model]
        base_on = draws_half.get((model, "baseline_rerun", True, "clean_short"))
        base_off = draws_half.get((model, "baseline_rerun", False, "clean_short"))
        pts = []
        for arm in COT_VS_OUTPUT_ARMS:
            on_arm = "stacked_rerun" if arm == "stacked" else arm
            on, off = draws_half.get((model, on_arm, True, "clean_short")), draws_half.get((model, arm, False,
                                                                                               "clean_short"))
            if not (on and off and base_on and base_off):
                continue
            g_on, g_off = ca.stat(on[0] - base_on[0], on[1] - base_on[1]), ca.stat(off[0] - base_off[0],
                                                                                   off[1] - base_off[1])
            pts.append((arm, g_on, g_off))
            data.append({"model": model, "arm": arm, "gain_thinking_on": g_on, "gain_thinking_off": g_off})
        round2 = [arm in an.ROUND2 for arm, _, _ in pts]
        fig.add_trace(go.Scatter(
            x=[p[1]["value"] for p in pts], y=[p[2]["value"] for p in pts], mode="markers+text", showlegend=False,
            marker={"color": [color if r2 else "white" for r2 in round2], "size": 10,
                    "line": {"color": color, "width": 1.5}},
            text=[SHORT[p[0]] for p in pts], textposition="top center", textfont={"size": 10},
            hovertemplate=[f"{SHORT[a]}: on {on['value']:+.1f}, off {off['value']:+.1f}<extra>{model}</extra>"
                           for a, on, off in pts]), row=1, col=k + 1)
        fig.add_trace(go.Scatter(x=[-50, 100], y=[-50, 100], mode="lines", showlegend=False, hoverinfo="skip",
                                 line={"color": "#999999", "dash": "dot", "width": 1}), row=1, col=k + 1)
        fig.update_xaxes(range=[-15, 85], title_text="gain with thinking on (points)", row=1, col=k + 1)
        fig.update_yaxes(range=[-50, 85], row=1, col=k + 1)
    fig.update_yaxes(title_text="gain with thinking off (points)", row=1, col=1)
    fig.update_layout(title=f"R2-4. Is the gain specific to the CoT? Gain over baseline with the rule on the thinking "
                            f"trace vs on the &lt;output_reasoning&gt; tag content ({n_half} questions)"
                            f"<br><sup>Metric: share of ALL texts reaching {an.T_SHORT} tokens with no violation (a "
                            f"short or missing text counts as a failure). Dotted: equal gain. Below the line = more "
                            f"specific to the CoT; above = general instruction following.<br>Open: round-1 arms; "
                            f"filled: round-2 arms (shown once their thinking-off rows are graded). Qwen3.8's "
                            f"thinking-off gains include using the tags at all (notes/exp06_changes_v3.md, item 33)"
                            f"</sup>",
                      margin={"t": 130})
    return a4.save(fig, fig_dir, "R2_4_cot_vs_output", 1600, 600), data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="figure run name")
    parser.add_argument("--scratch", action="store_true", help="write under /tmp/exp06a_scratch (layout tests)")
    args = parser.parse_args()
    if args.scratch:
        out_dir = fig_dir = Path("/tmp/exp06a_scratch") / args.run
    else:
        out_dir, fig_dir = EXP.results / "analysis" / args.run, EXP.figure_dir(args.run)
    for d in {out_dir, fig_dir}:
        if d.exists():
            raise SystemExit(f"{d} exists; runs are never overwritten")
    df, models, missing = load()
    for d in {out_dir, fig_dir}:
        d.mkdir(parents=True)
    test_items, half_items = cc_exp06a.load_items()[0], cc_exp06a.thinking_off_items()
    boot = ca.Bootstrap(test_items, cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    boot_half = ca.Bootstrap(half_items, cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    _, draws = an.arm_table(df[df["thinking"]], models, boot, [(a, True) for _, a, _ in LADDER])
    df_half = df[df["item_id"].isin({it["item_id"] for it in half_items})]
    half_arms = ["baseline_rerun", "stacked_rerun"] + COT_VS_OUTPUT_ARMS
    _, draws_half = an.arm_table(df_half, models, boot_half, [(a, t) for t in (True, False) for a in half_arms])
    figures, data = {}, {"models": models, "missing_parts": missing}
    for name, (path, rows) in {
        "ladder": fig_ladder(draws, models, fig_dir),
        "piece_effects": fig_piece_effects(draws, models, fig_dir),
        "per_rule": fig_per_rule(df, models, boot, fig_dir),
        "cot_vs_output": fig_cot_vs_output(draws_half, models, len(half_items), fig_dir),
    }.items():
        figures[name], data[name] = path, rows
    (out_dir / "round2_figure_data.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps(figures, indent=2))


if __name__ == "__main__":
    main()
