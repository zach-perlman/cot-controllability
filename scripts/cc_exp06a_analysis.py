"""exp06a analysis (results/exp06a_prompt_screen/manifest.json): which prompt harness raises CoT controllability, and
is its gain specific to the CoT?

Rows (all on exp03's 100 items, same seeds per (item, rule)):
  exp06a       every harness, thinking on (reasoning trace) and off (<output_reasoning> tag content)
  references   baseline / stacked, thinking on: exp03's rows (Qwen3-32B) or exp04's requests_none rows (others);
               baseline, thinking off: exp04's external_ceiling rows
  earlier arms (combined figure only) exp04's compliant-prefill arms and exp05's own-opening d3 arm; prefill rows are
               scored on the continuation after the prefill, as in those experiments
Scoring: KM S(t) with event = first violation; an empty graded text (empty thinking trace, no tag content) is a
violation at token 0 (manifest; --censor-empty-traces gives the sensitivity variant for thinking-on rows). P1 for
thinking on is "no violation before the stop" (cfg.EXP06A_REASONING_STOP_TOKENS), the same for reused rows that ran
longer; for thinking off it is the whole tag content.

Primary: S(t*) thinking on, mean over the 5 rules; each non-diagnostic harness minus stacked per model, Holm over
harnesses x models. Secondary: harness minus baseline; the no-rule twins; the CoT-vs-output comparison at S(200).

Outputs (never overwritten): results/exp06a_prompt_screen/analysis/<run>/{summary.json, per_cell.csv,
REPORT_auto.md} and figures/exp06a_prompt_screen/<run>/.
Run: /venv/main/bin/python scripts/cc_exp06a_analysis.py --run NAME [--skip-missing] [--censor-empty-traces]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cc_analysis as ca
import cc_config as cfg
import cc_exp04
import cc_exp04_analysis as a4
import cc_exp06a
from cc_survival import bootstrap_p, holm

EXP = cfg.EXP06A
T_STAR, T_SHORT = a4.T_STAR, a4.T_SHORT
RULES = cfg.EXP06A_MODES
OPENERS = cfg.EXP04_OPENER_MODES
MODELS = cfg.EXP06A_MODELS
STOP = cfg.EXP06A_REASONING_STOP_TOKENS
VARIANTS = [h for h in cfg.EXP06A_HARNESSES if h not in cfg.EXP06A_DIAGNOSTIC]
ON_ARMS = cfg.EXP06A_REFERENCES + list(cfg.EXP06A_RERUNS) + cfg.EXP06A_HARNESSES + list(cfg.EXP06A_NO_RULE_TWINS)
OFF_ARMS = ["baseline", "baseline_rerun", "stacked"] + cfg.EXP06A_HARNESSES
# Earlier experiments' arms on the same items (thinking on), for the combined figure.
EARLIER = {
    "exp04_prefill": {"label": "exp04 compliant prefill (scored after it)", "color": "#666666"},
    "exp04_prefill_stacked": {"label": "exp04 compliant prefill + stacked", "color": "#2F6B4F"},
    "exp04_prefill_no_rule": {"label": "exp04 compliant prefill, no rule", "color": "#9DC3E6"},
    "exp05_own_d3": {"label": "exp05 own opening ≥150 tokens (scored after it)", "color": "#7F5FA0"},
}
EARLIER_OF = {("prefill_compliant", "baseline"): "exp04_prefill", ("prefill_compliant", "stacked"): "exp04_prefill_stacked",
              ("prefill_no_rule", cfg.NO_CONSTRAINT): "exp04_prefill_no_rule"}
COLUMNS = ["model", "item_id", "source", "mode", "arm", "thinking", "origin", "reasoning", "reasoning_tokens",
           "fv_token", "compliant", "think_status", "meta_regex", "response_reasoning_tokens_cut"]


def style(arm: str) -> dict:
    return cfg.HARNESS_STYLE.get(arm) or EARLIER[arm]


# --- Loading ----------------------------------------------------------------------------------------------------------
def exp06a_rows(models: list[str], skip_missing: bool) -> tuple[pd.DataFrame, list[str]]:
    requests = {r["request_id"]: r for p in sorted(EXP.cache.glob("requests_*.jsonl")) for r in cc_exp04.load_requests(p)}
    frames, missing = [], []
    for model in models:
        paths = [Path(p) for p in glob.glob(str(EXP.generations / f"{model}__card__stream_abort_{model}__*.jsonl"))]
        if len(paths) != 1 or not a4.grades_path(paths[0]).exists():
            missing.append(model)
            continue
        df = a4.read_run(paths[0], "exp06a")
        df["arm"] = df["prompt"]
        df["thinking"] = df["request_id"].map(lambda i: requests[i]["thinking"])
        df["origin"] = "exp06a"
        frames.append(df)
    if missing and not skip_missing:
        raise SystemExit(f"no graded exp06a rows for {missing} (use --skip-missing); nothing written")
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)), missing


def reference_rows(models: list[str]) -> pd.DataFrame:
    """exp03/exp04's baseline and stacked rows, exp04's thinking-off baseline, exp04's prefill arms."""
    old = a4.analysis_rows(a4.load(empty_trace_is_violation=True))
    old = old[old["model"].isin(models) & old["mode"].isin(RULES)].copy()
    frames = []
    refs = old[(old["condition"] == "none") & old["prompt"].isin(cfg.EXP06A_REFERENCES)]
    frames.append(refs.assign(arm=refs["prompt"], thinking=True, origin=refs["source_run"]))
    ceiling = old[old["condition"] == "external_ceiling"]
    frames.append(ceiling.assign(arm="baseline", thinking=False, origin="exp04"))
    for (condition, prompt), arm in EARLIER_OF.items():
        d = old[(old["condition"] == condition) & (old["prompt"] == prompt)]
        frames.append(d.assign(arm=arm, thinking=True, origin="exp04"))
    return pd.concat(frames, ignore_index=True)


def exp05_rows(models: list[str]) -> pd.DataFrame:
    import cc_exp05_analysis as a5
    d = a5.load(True, models, skip_ungraded=True)
    d = d[(d["source_run"] == "exp05") & (d["dose"] == "d3") & (d["condition"] == "prefill_compliant")
          & d["model"].isin(models) & d["mode"].isin(OPENERS)]
    return d.assign(arm="exp05_own_d3", thinking=True, origin="exp05")


def score(df: pd.DataFrame, empty_is_violation: bool) -> pd.DataFrame:
    """Survival columns for every row. Thinking off: no tag content is always a violation at token 0 (exp04)."""
    df = df.copy()
    df["empty"] = df["reasoning_tokens"] == 0
    df["event_at_0"] = df["empty"] & (empty_is_violation | ~df["thinking"].astype(bool))
    df["event"] = df["fv_token"].notna() | df["event_at_0"]
    df["time"] = np.where(df["fv_token"].notna(), df["fv_token"].fillna(0),
                          np.where(df["event_at_0"], 0, df["reasoning_tokens"])).astype(float)
    within_stop = ~(df["event"] & (df["time"] < STOP))
    off_compliant = np.where(df["event_at_0"], 0.0, pd.to_numeric(df["compliant"], errors="coerce"))
    df["P1"] = np.where(df["thinking"].astype(bool), within_stop.astype(float), off_compliant)
    df["graded_tokens_capped"] = np.minimum(df["reasoning_tokens"], STOP)
    starts = {m: cc_exp06a.start_sentence(m) for m in RULES}
    df["starts_with_sentence"] = [str(r).lstrip().startswith(starts[m]) for r, m in zip(df["reasoning"], df["mode"])]
    df["meta_regex"] = df["meta_regex"].astype(float)
    return df


# --- Tables -------------------------------------------------------------------------------------------------------------
def cell(df: pd.DataFrame, model: str, arm: str, thinking: bool) -> pd.DataFrame:
    return df[(df["model"] == model) & (df["arm"] == arm) & (df["thinking"] == thinking)]


def arm_table(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap) -> tuple[list[dict], dict]:
    rows, draws = [], {}
    arms = [(a, True) for a in ON_ARMS + list(EARLIER)] + [(a, False) for a in OFF_ARMS]
    for model in models:
        for arm, thinking in arms:
            d = cell(df, model, arm, thinking)
            if d.empty:
                continue
            rec = {"model": model, "arm": arm, "thinking": thinking, "n": len(d),
                   "origin": "/".join(sorted(set(d["origin"])))}
            for name, modes, t in (("S_t_star", RULES, T_STAR), ("S_t_star_openers", OPENERS, T_STAR),
                                   ("S_short", RULES, T_SHORT), ("S_short_openers", OPENERS, T_SHORT)):
                est = a4.macro_km(d, modes, t, boot)
                draws[(model, arm, thinking, name)] = est
                rec[name] = ca.stat(*est) if est else None
            rec["P1"] = ca.stat(*ca.pooled(d, "P1", boot)) if set(RULES) <= set(d["mode"]) else None
            rec.update(empty_share=float(d["empty"].mean() * 100),
                       median_graded_tokens_capped=float(d["graded_tokens_capped"].median()),
                       meta_regex_share=float(d["meta_regex"].mean() * 100),
                       starts_with_sentence_share=float(d["starts_with_sentence"].mean() * 100),
                       reasoning_cut_share=float((d["response_reasoning_tokens_cut"].fillna(0) > 0).mean() * 100))
            rows.append(rec)
    return rows, draws


def contrast_family(draws: dict, models: list[str], pairs: list[tuple[str, str]], thinking: bool,
                    metric: str = "S_t_star") -> list[dict]:
    out = []
    for model in models:
        for a, b in pairs:
            ea, eb = draws.get((model, a, thinking, metric)), draws.get((model, b, thinking, metric))
            if ea is None or eb is None:
                continue
            out.append({"model": model, "a": a, "b": b, "thinking": thinking, "metric": metric,
                        "difference": ca.stat(ea[0] - eb[0], ea[1] - eb[1]), "p": bootstrap_p(ea[1] - eb[1])})
    adjusted = holm({(c["model"], c["a"], c["b"]): c["p"] for c in out})
    for c in out:
        c["p_holm"] = adjusted[(c["model"], c["a"], c["b"])]
    return out


def cot_specificity(draws: dict, models: list[str], metric: str = "S_short") -> list[dict]:
    """Per harness: its gain over baseline with thinking on, with thinking off, and on - off (paired draws)."""
    out = []
    for model in models:
        for arm in ["stacked"] + cfg.EXP06A_HARNESSES:
            keys = [(model, arm, True, metric), (model, "baseline", True, metric),
                    (model, arm, False, metric), (model, "baseline", False, metric)]
            if any(draws.get(k) is None for k in keys):
                continue
            (pa, da), (pb, db), (pc, dc), (pd_, dd) = (draws[k] for k in keys)
            out.append({"model": model, "arm": arm, "metric": metric,
                        "gain_thinking_on": ca.stat(pa - pb, da - db), "gain_thinking_off": ca.stat(pc - pd_, dc - dd),
                        "on_minus_off": ca.stat((pa - pb) - (pc - pd_), (da - db) - (dc - dd))})
    return out


def per_rule_table(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap) -> pd.DataFrame:
    recs = []
    for model in models:
        for (arm, thinking), d in df[df["model"] == model].groupby(["arm", "thinking"]):
            for mode in RULES:
                c = d[d["mode"] == mode]
                if c.empty:
                    continue
                for t in (T_STAR, T_SHORT):
                    point, dr = a4.km_at(c, t, boot)
                    lo, hi = ca.ci(dr)
                    recs.append({"model": model, "arm": arm, "thinking": thinking, "mode": mode, "t": t, "n": len(c),
                                 "S": point * 100, "ci_lo": lo * 100, "ci_hi": hi * 100,
                                 "P1": float(c["P1"].mean() * 100), "empty_share": float(c["empty"].mean() * 100)})
    return pd.DataFrame(recs)


def by_source_table(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap) -> list[dict]:
    """fewshot's examples are MMLU-Pro items: S(t*) per question source for fewshot and the references."""
    out = []
    for model in models:
        for arm in ("baseline", "stacked", "fewshot", "fewshot_no_rule"):
            d = cell(df, model, arm, True)
            for source, ds in d.groupby("source"):
                est = a4.macro_km(ds, RULES, T_STAR, boot)
                if est:
                    out.append({"model": model, "arm": arm, "source": source, "n_items": ds["item_id"].nunique(),
                                "S_t_star": ca.stat(*est)})
    return out


def checks(df: pd.DataFrame, models: list[str]) -> dict:
    new = df[df["origin"] == "exp06a"]
    on, off = new[new["thinking"]], new[~new["thinking"]]
    records = {m: json.loads(cc_exp06a.record_path(m).read_text()) for m in models if cc_exp06a.record_path(m).exists()}
    return {
        "thinking_on_status_share": {k: float(v) * 100 for k, v in on["think_status"].value_counts(normalize=True).items()},
        "thinking_on_reached_stop_share": float((on["reasoning_tokens"] >= STOP).mean() * 100) if len(on) else None,
        "thinking_off_no_tag_share": float(off["empty"].mean() * 100) if len(off) else None,
        "thinking_off_reasoning_channel_cut_share": float((off["response_reasoning_tokens_cut"].fillna(0) > 0).mean() * 100)
                                                    if len(off) else None,
        "fewshot_items": {m: r["fewshot_items"] for m, r in records.items()},
        "style_guide_turn1_grader_pass_rate": {m: r["style_guide_turn1_grader_pass_rate"] for m, r in records.items()},
    }


# --- Figures ------------------------------------------------------------------------------------------------------------
def fig_combined(rows: list[dict], models: list[str], fig_dir) -> str:
    """H1: every arm, thinking on, S(t*) per model. Top: the 5 rules; bottom: the 4 opener rules (where exp05 has data)."""
    order = (cfg.EXP06A_REFERENCES + list(EARLIER) + list(cfg.EXP06A_RERUNS) + VARIANTS + cfg.EXP06A_DIAGNOSTIC
             + list(cfg.EXP06A_NO_RULE_TWINS))
    labels = [style(a)["label"] for a in order]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[f"S({T_STAR}), mean over the 5 rules", f"S({T_STAR}), mean over the 4 opener "
                                        "rules (lowercase, uppercase, meow, end 'safe')"])
    markers = ["circle", "diamond", "square"]
    offsets = np.linspace(-0.25, 0.25, len(models))
    for k, model in enumerate(models):
        for r, metric in ((1, "S_t_star"), (2, "S_t_star_openers")):
            ys, lo, hi, colors = [], [], [], []
            for arm in order:
                rec = next((x for x in rows if (x["model"], x["arm"], x["thinking"]) == (model, arm, True)), None)
                s = rec[metric] if rec else None
                ys.append(s["value"] if s else None)
                lo.append(s["value"] - s["ci"][0] if s else None)
                hi.append(s["ci"][1] - s["value"] if s else None)
                colors.append(style(arm)["color"])
            fig.add_trace(go.Scatter(
                x=np.arange(len(order)) + offsets[k], y=ys, mode="markers", name=model, legendgroup=model,
                showlegend=r == 1, marker={"symbol": markers[k], "size": 10, "color": colors,
                                           "line": {"color": "black", "width": 0.8}},
                error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo, "thickness": 1,
                         "color": "#555555"},
                customdata=labels, hovertemplate=f"{model}<br>%{{customdata}}: %{{y:.1f}}%<extra></extra>"),
                row=r, col=1)
    stacked = [x for x in rows if x["arm"] == "stacked" and x["thinking"]]
    for r, metric in ((1, "S_t_star"), (2, "S_t_star_openers")):
        mean_stacked = np.mean([x[metric]["value"] for x in stacked if x["model"] in models and x[metric]])
        fig.add_hline(y=mean_stacked, line={"color": cfg.HARNESS_STYLE["stacked"]["color"], "dash": "dot", "width": 1},
                      row=r, col=1)
        n_ref = len(cfg.EXP06A_REFERENCES) + len(EARLIER)
        fig.add_vrect(x0=-0.5, x1=n_ref - 0.5, fillcolor="#f2f2f2", line_width=0, layer="below", row=r, col=1)
        fig.update_yaxes(title_text="% with no violation<br>in the first %d tokens" % T_STAR, range=[-2, 102], row=r, col=1)
    fig.update_xaxes(tickvals=list(range(len(order))), ticktext=[l.replace(" (", "<br>(") for l in labels],
                     tickangle=-40, row=2, col=1)
    fig.update_layout(title=f"H1. Every prompt harness next to earlier experiments' arms (thinking on, same 100 "
                            f"questions)<br><sup>Shaded: references and exp03-exp05 arms. Marker = model, colour = arm; "
                            f"95% question-level bootstrap CIs. Dotted: stacked, mean over models. Prefill arms are "
                            f"scored after the prefill; harness arms on the whole trace</sup>",
                      legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"}, margin={"t": 140, "b": 220})
    return a4.save(fig, fig_dir, "H1_combined_S_tstar", 1500, 950)


def fig_cot_vs_output(rows: list[dict], models: list[str], fig_dir) -> str:
    """H2: per harness, S(200) and P1 with thinking on (reasoning trace) and off (reasoning in output tags)."""
    order = OFF_ARMS
    fig = make_subplots(rows=2, cols=len(models), shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.02,
                        vertical_spacing=0.08, subplot_titles=models + [""] * len(models))
    for k, model in enumerate(models):
        for r, metric in ((1, "S_short"), (2, "P1")):
            for thinking, off, symbol, name in ((True, -0.15, "circle", "thinking on: rule on the reasoning trace"),
                                                (False, 0.15, "circle-open", "thinking off: rule on <output_reasoning> "
                                                                             "tag content")):
                ys, lo, hi = [], [], []
                for arm in order:
                    rec = next((x for x in rows if (x["model"], x["arm"], x["thinking"]) == (model, arm, thinking)), None)
                    s = rec[metric] if rec else None
                    ys.append(s["value"] if s else None)
                    lo.append(s["value"] - s["ci"][0] if s else None)
                    hi.append(s["ci"][1] - s["value"] if s else None)
                fig.add_trace(go.Scatter(
                    x=np.arange(len(order)) + off, y=ys, mode="markers", name=name, legendgroup=str(thinking),
                    showlegend=k == 0 and r == 1,
                    marker={"symbol": symbol, "size": 9, "color": [style(a)["color"] for a in order],
                            "line": {"color": "black", "width": 1}},
                    error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo, "thickness": 1,
                             "color": "#555555"},
                    hovertemplate=f"{model} | {name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>"), row=r, col=k + 1)
            fig.update_xaxes(tickvals=list(range(len(order))), ticktext=[style(a)["label"].split(" (")[0] for a in order],
                             tickangle=-50, row=2, col=k + 1)
    fig.update_yaxes(title_text=f"S({T_SHORT}): no violation<br>in the first {T_SHORT} tokens", range=[-2, 102],
                     row=1, col=1)
    fig.update_yaxes(title_text=f"P1: no violation in the graded<br>text (thinking on: first {STOP} tokens)",
                     range=[-2, 102], row=2, col=1)
    fig.update_layout(title="H2. Is a harness's gain specific to the CoT? The same harness with the reasoning in the "
                            "thinking trace vs in the output<br><sup>Mean over the 5 rules; 95% CIs. A harness that "
                            "lifts both by the same amount works through general instruction following</sup>",
                      legend={"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right"}, margin={"t": 130, "b": 200})
    return a4.save(fig, fig_dir, "H2_cot_vs_output", 1500, 850)


def fig_per_rule(per_rule: pd.DataFrame, models: list[str], fig_dir) -> str:
    """H3: S(t*) per (arm, rule), thinking on, one heatmap per model."""
    order = cfg.EXP06A_REFERENCES + cfg.EXP06A_HARNESSES + list(cfg.EXP06A_NO_RULE_TWINS)
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, subplot_titles=models, horizontal_spacing=0.02)
    d = per_rule[(per_rule["t"] == T_STAR) & per_rule["thinking"]]
    for k, model in enumerate(models):
        m = d[d["model"] == model].pivot(index="arm", columns="mode", values="S").reindex(index=order, columns=RULES)
        fig.add_trace(go.Heatmap(z=m.to_numpy(), x=[a4.MODE_LABEL[r].split(" (")[0] for r in RULES],
                                 y=[style(a)["label"] for a in order], zmin=0, zmax=100, colorscale="Viridis",
                                 showscale=k == len(models) - 1, text=np.round(m.to_numpy()).astype("float"),
                                 texttemplate="%{text:.0f}", hovertemplate="%{y} | %{x}: %{z:.1f}%<extra></extra>"),
                      row=1, col=k + 1)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"H3. S({T_STAR}) per rule and harness (thinking on), % of 100 questions",
                      margin={"l": 260, "t": 90})
    return a4.save(fig, fig_dir, "H3_per_rule", 1500, 620)


# --- Report ---------------------------------------------------------------------------------------------------------------
def report(s: dict, figures: list[str], fig_rel: str) -> str:
    def f(x):
        return ca.fmt(x) if x else "n/a"
    lines = [f"# exp06a prompt screen: automatic report (run `{s['run']}`)", "",
             "**UNVERIFIED** until a human adds it to VERIFIED.md. Every number is grader-scored (no LLM judge).", "",
             f"Models: {', '.join(s['models'])}" + (f"; missing: {', '.join(s['missing_models'])}" if s["missing_models"]
                                                     else ""), "",
             f"## S({T_STAR}), thinking on (mean over the 5 rules; opener-rule mean in brackets)", "",
             "| arm | " + " | ".join(s["models"]) + " |", "|---|" + "---|" * len(s["models"])]
    for arm in ON_ARMS + list(EARLIER):
        cells = []
        for model in s["models"]:
            rec = next((x for x in s["arms"] if (x["model"], x["arm"], x["thinking"]) == (model, arm, True)), None)
            cells.append(f"{f(rec['S_t_star'])} [{f(rec['S_t_star_openers'])}]" if rec else "n/a")
        lines.append(f"| {style(arm)['label']} | " + " | ".join(cells) + " |")
    lines += ["", f"## Primary: harness - stacked, S({T_STAR}) thinking on (Holm over {len(VARIANTS)} x models)", "",
              "| model | harness | difference (pts) | p | p Holm |", "|---|---|---|---|---|"]
    for c in s["primary"]:
        lines.append(f"| {c['model']} | {c['a']} | {ca.fmt(c['difference'])} | {c['p']:.3f} | {c['p_holm']:.3f} |")
    lines += ["", "## Reruns in this environment minus the reused rows (5-rule mean)", "",
              "| model | arm | thinking | metric | difference (pts) | p |", "|---|---|---|---|---|---|"]
    for c in s["rerun_minus_reused"]:
        lines.append(f"| {c['model']} | {c['b']} | {'on' if c['thinking'] else 'off'} | {c['metric']} | "
                     f"{ca.fmt(c['difference'])} | {c['p']:.3f} |")
    lines += ["", "## Primary contrasts against the in-environment stacked rerun", "",
              "| model | harness | difference (pts) | p Holm |", "|---|---|---|---|"]
    for c in s["primary_vs_rerun"]:
        lines.append(f"| {c['model']} | {c['a']} | {ca.fmt(c['difference'])} | {c['p_holm']:.3f} |")
    lines += ["", "## Copying controls (thinking on): harness - its no-rule twin", "",
              "| model | harness | difference (pts) | p Holm |", "|---|---|---|---|"]
    for c in s["twins"]:
        lines.append(f"| {c['model']} | {c['a']} | {ca.fmt(c['difference'])} | {c['p_holm']:.3f} |")
    lines += ["", f"## CoT specificity: gain over baseline at S({T_SHORT}), thinking on vs off", "",
              "| model | arm | gain, thinking on | gain, thinking off | on - off |", "|---|---|---|---|---|"]
    for c in s["cot_specificity"]:
        lines.append(f"| {c['model']} | {c['arm']} | {ca.fmt(c['gain_thinking_on'])} | {ca.fmt(c['gain_thinking_off'])} "
                     f"| {ca.fmt(c['on_minus_off'])} |")
    lines += ["", "## Secondary per-arm metrics", "",
              "| model | arm | thinking | P1 | empty % | median tokens (capped) | meta regex % | starts with sentence % |",
              "|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        lines.append(f"| {x['model']} | {x['arm']} | {'on' if x['thinking'] else 'off'} | {f(x['P1'])} | "
                     f"{x['empty_share']:.1f} | {x['median_graded_tokens_capped']:.0f} | {x['meta_regex_share']:.1f} | "
                     f"{x['starts_with_sentence_share']:.1f} |")
    lines += ["", "## Checks", "", "```", json.dumps(s["checks"], indent=1), "```", "", "## Figures", ""]
    lines += [f"- [{name}]({fig_rel}/{name}.html)" for name in figures]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="analysis run name")
    parser.add_argument("--skip-missing", action="store_true", help="leave out models with no graded exp06a rows")
    parser.add_argument("--censor-empty-traces", action="store_true",
                        help="sensitivity: censor empty thinking traces at token 0 instead of scoring a violation")
    parser.add_argument("--scratch", action="store_true", help="write under /tmp/exp06a_scratch (tests)")
    args = parser.parse_args()
    if args.scratch:
        out_dir = fig_dir = Path("/tmp/exp06a_scratch") / args.run
    else:
        out_dir, fig_dir = EXP.results / "analysis" / args.run, EXP.figure_dir(args.run)
    for d in {out_dir, fig_dir}:
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")

    new, missing = exp06a_rows(MODELS, args.skip_missing)
    models = [m for m in MODELS if m not in missing]
    raw = pd.concat([new, reference_rows(models), exp05_rows(models)], ignore_index=True)
    df = score(raw[[c for c in COLUMNS if c in raw.columns]], not args.censor_empty_traces)
    for d in {out_dir, fig_dir}:
        d.mkdir(parents=True)
    boot = ca.Bootstrap(cc_exp06a.load_items()[0], cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    arms, draws = arm_table(df, models, boot)
    per_rule = per_rule_table(df, models, boot)
    summary = {
        "exp_id": EXP.exp_id, "run": args.run, "models": models, "missing_models": missing,
        "censor_empty": args.censor_empty_traces, "t_star": T_STAR, "t_short": T_SHORT, "stop": STOP,
        "primary": contrast_family(draws, models, [(v, "stacked") for v in VARIANTS], True),
        "primary_vs_rerun": contrast_family(draws, models, [(v, "stacked_rerun") for v in VARIANTS], True),
        "rerun_minus_reused": (contrast_family(draws, models, [(r, ref) for r, ref in cfg.EXP06A_RERUNS.items()], True)
                               + contrast_family(draws, models, [("baseline_rerun", "baseline")], False, "S_short")),
        "vs_baseline": contrast_family(draws, models, [(h, "baseline") for h in cfg.EXP06A_HARNESSES], True),
        "twins": contrast_family(draws, models, [(h, twin) for twin, h in cfg.EXP06A_NO_RULE_TWINS.items()], True),
        "cot_specificity": cot_specificity(draws, models),
        "by_source": by_source_table(df, models, boot),
        "arms": arms, "checks": checks(df, models),
    }
    figures = [fig_combined(arms, models, fig_dir), fig_cot_vs_output(arms, models, fig_dir),
               fig_per_rule(per_rule, models, fig_dir)]
    per_rule.to_csv(out_dir / "per_cell.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    text = report(summary, figures, os.path.relpath(fig_dir, out_dir))
    (out_dir / "REPORT_auto.md").write_text(text)
    print(f"wrote {out_dir}")
    print(text)


if __name__ == "__main__":
    main()
