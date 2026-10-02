"""exp06 analysis (results/exp06_prompt_prefill/manifest.json): exp04's prefill design with exp06a's upgraded prompt,
on 100 new items.

Cells (thinking on): prompt (baseline / stacked / upgraded / no_rule) x opening (none / compliant / non-compliant
prefill). Thinking off: baseline / stacked / upgraded on the <output_reasoning> tag content (opening label
"thinking_off"), and, added 2026-10-02 (manifest_extension_off_prefill.json), cc_exp06.CELLS_OFF_PREFILL: the same
sentences prefilled after the opening tag. Prefill rows are scored on the continuation after the prefill.
Thinking-off comparisons use the length-robust rate at 200 tokens (thinking-off texts are short).
Scoring (as exp06a): KM S(t), event = first violation; an empty graded text (no trace, no continuation, no tag
content) is a violation at token 0. Length-robust companion clean_<t>: the share of ALL texts that reach t tokens with
no violation (a text that ends earlier counts as not reaching t).

Primary (manifest): S(1000), 5-rule mean, thinking on; C1 upgraded|none - stacked|none, C2 upgraded|compliant -
upgraded|none, C3 upgraded|compliant - no_rule|compliant; Holm over the 3 contrasts x the 3 screened models, and
separately over each held-out model's 3.

Figures (figures/exp06_prompt_prefill/<run>/):
  E1_prefill_comparison   S(1000) per opening, one line per prompt (top: the 4 opener rules; bottom: word
                          suppression, which has no non-compliant opening)
  E2_what_each_lever_adds paired differences: compliant opening, rule beyond copying, compliant vs non-compliant
                          opening, upgraded vs stacked
  E3_survival_and_ends    S(t) curves per (model, opening), and under each panel how the traces ended
  E4_cot_vs_output        gain over baseline with thinking on vs off (points, and % of headroom)
  E5_per_rule             S(1000) per rule and cell
  E6_accuracy             full-trace cells: accuracy and whole-trace compliance per prompt
  E7_replication          exp06a/exp04 estimates on exp03's items vs exp06's on new items, same cells
  E8_openings_both_channels  rate at 200 tokens per opening, thinking on vs off (4 opener rules)
Outputs (never overwritten): results/exp06_prompt_prefill/analysis/<run>/{summary.json, per_rule.csv, REPORT_auto.md}.
Run: /venv/main/bin/python scripts/cc_exp06_analysis.py --run NAME [--skip-missing] [--scratch]
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
import cc_exp06
from cc_survival import bootstrap_p, holm, kaplan_meier

EXP = cfg.EXP06
T_STAR, T_SHORT, STOP = cfg.SURVIVAL_T_STAR, 200, cfg.EXP06_STOP_TOKENS
RULES, OPENERS = cfg.EXP06_MODES, cfg.EXP06_OPENER_MODES
MODELS, HELD_OUT = cfg.EXP06_MODELS, cfg.EXP06_HELD_OUT_MODELS
PROMPTS = cc_exp06.RULE_PROMPTS  # baseline, stacked, upgraded
OPENINGS = cc_exp06.OPENINGS  # none, prefill_compliant, prefill_noncompliant
CELLS_ON = cc_exp06.CELLS
OFF = "thinking_off"  # opening label of the no-prefill thinking-off cells
CELLS_OFF = [(p, OFF) for p in PROMPTS] + cc_exp06.CELLS_OFF_PREFILL
PRIMARY = {"C1 upgraded vs stacked (no opening)": (("upgraded", "none"), ("stacked", "none")),
           "C2 compliant opening (upgraded)": (("upgraded", "prefill_compliant"), ("upgraded", "none")),
           "C3 rule beyond copying (compliant opening)": (("upgraded", "prefill_compliant"),
                                                          ("no_rule", "prefill_compliant"))}
MODEL_STYLE = {"Qwen3.8-27B-FP8": ("#0072B2", "circle"), "Gemma-4-31B-FP8": ("#D55E00", "diamond"),
               "Qwen3-32B": ("#009E73", "square"), "Qwen3.6-27B-FP8": ("#CC79A7", "star"),
               "Qwen3.5-9B": ("#E69F00", "triangle-up")}
PROMPT_LABEL = {p: s["label"] for p, s in cfg.EXP06_STYLE.items()}
CUT_MARGIN = 50
SHORT_PROMPT = {"baseline": "CoT-Control", "stacked": "stacked", "upgraded": "upgraded", "no_rule": "no rule"}
OPENING_LABEL = {o: s["label"] for o, s in cfg.EXP06_OPENING_STYLE.items()}
COLUMNS = ["model", "item_id", "source", "mode", "prompt", "opening", "thinking", "reasoning", "reasoning_tokens",
           "fv_token", "compliant", "correct", "think_status", "meta_regex", "full_trace_cell", "external_blocks",
           "think_close_no_tag"]


def model_name(model: str) -> str:
    return model + (" (held out)" if model in HELD_OUT else "")


# --- Loading and scoring --------------------------------------------------------------------------------------------
def load(skip_missing: bool) -> tuple[pd.DataFrame, list[str]]:
    frames, missing = [], []
    for model in MODELS:
        for part in cc_exp06.PARTS + [cc_exp06.OFF_PREFILL_PART]:
            paths = [Path(p) for p in glob.glob(str(EXP.generations / f"{model}__card__stream_abort_{model}{part}__*.jsonl"))]
            if len(paths) != 1 or not a4.grades_path(paths[0]).exists():
                missing.append(f"{model}{part}")
                continue
            requests = {r["request_id"]: r for r in cc_exp04.load_requests(cc_exp06.requests_path(model, part))}
            df = a4.read_run(paths[0], "exp06")
            df["opening"] = df["request_id"].map(lambda i: requests[i]["condition"])
            df["thinking"] = df["request_id"].map(lambda i: requests[i]["thinking"])
            # thinking off: no <output_reasoning> block, but the response closes with </think> (scored as no tags)
            blocks = df["external_blocks"] if "external_blocks" in df else pd.Series(0, index=df.index)
            df["think_close_no_tag"] = (~df["thinking"].astype(bool) & (blocks.fillna(0) == 0)
                                        & df["answer"].fillna("").str.contains("</think>", regex=False))
            frames.append(df)
    if missing and not skip_missing:
        raise SystemExit(f"no graded rows for {missing} (use --skip-missing); nothing written")
    if not frames:
        raise SystemExit("no graded exp06 rows yet")
    df = pd.concat(frames, ignore_index=True)
    return df[[c for c in COLUMNS if c in df.columns]], missing


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["thinking"] = df["thinking"].astype(bool)
    df["empty"] = df["reasoning_tokens"] == 0
    df["event"] = df["fv_token"].notna() | df["empty"]
    df["time"] = np.where(df["fv_token"].notna(), df["fv_token"].fillna(0),
                          np.where(df["empty"], 0, df["reasoning_tokens"])).astype(float)
    for t in (T_SHORT, T_STAR):
        df[f"clean_{t}"] = ((df["reasoning_tokens"] >= t) & ~(df["event"] & (df["time"] < t))).astype(float)
    # How each trace ended, at t*: violated before t*; no violation but the text ended before t*; clean through t*.
    df["end"] = np.where(df["event"] & (df["time"] < T_STAR), "violation",
                         np.where(df["reasoning_tokens"] < T_STAR, "ended_clean_early", "clean_through"))
    df["no_tags"] = ~df["thinking"] & df["empty"]
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").astype(float)
    df["whole_trace_compliant"] = pd.to_numeric(df["compliant"], errors="coerce").astype(float)
    df["meta_regex"] = df["meta_regex"].astype(float)
    return df


def cell(df: pd.DataFrame, model: str, prompt: str, opening: str, thinking: bool = True) -> pd.DataFrame:
    return df[(df["model"] == model) & (df["prompt"] == prompt) & (df["opening"] == opening)
              & (df["thinking"] == thinking)]


# --- Estimates --------------------------------------------------------------------------------------------------------
def estimates(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap) -> tuple[list[dict], dict]:
    """Per (model, prompt, opening, thinking): S(t) over the 5 rules and over the 4 opener rules, the length-robust
    rates, and descriptive shares. draws[(model, prompt, opening, thinking, metric)] = (point, bootstrap draws)."""
    rows, draws = [], {}
    keys = [(p, o, True) for p, o in CELLS_ON] + [(p, o, False) for p, o in CELLS_OFF]
    for model in models:
        for prompt, opening, thinking in keys:
            d = cell(df, model, prompt, opening, thinking)
            if d.empty:
                continue
            rec = {"model": model, "prompt": prompt, "opening": opening, "thinking": thinking, "n": len(d)}
            for name, modes, t in (("S_1000", RULES, T_STAR), ("S_1000_openers", OPENERS, T_STAR),
                                   ("S_200", RULES, T_SHORT), ("S_1000_word_suppression", ["word_suppression"], T_STAR)):
                est = a4.macro_km(d, modes, t, boot) if set(modes) <= set(d["mode"]) else None
                draws[(model, prompt, opening, thinking, name)] = est
                rec[name] = ca.stat(*est) if est else None
            for name in ("clean_200", "clean_1000"):
                for suffix, sub in (("", d), ("_openers", d[d["mode"].isin(OPENERS)])):
                    est = ca.pooled(sub, name, boot) if len(sub) else None
                    draws[(model, prompt, opening, thinking, name + suffix)] = est
                    rec[name + suffix] = ca.stat(*est) if est else None
            if not thinking:
                tagged = d[~d["no_tags"]]
                est = ca.pooled(tagged, "clean_200", boot) if len(tagged) else None
                draws[(model, prompt, opening, thinking, "clean_200_tagged")] = est
                rec["clean_200_tagged"] = ca.stat(*est) if est else None
                rec["think_close_no_tag_share"] = float(d["think_close_no_tag"].mean() * 100)
            full = d[d["full_trace_cell"].fillna(False).astype(bool)]
            for name in ("correct", "whole_trace_compliant"):
                est = ca.pooled(full, name, boot) if len(full) else None
                draws[(model, prompt, opening, thinking, name)] = est
                rec[name] = ca.stat(*est) if est else None
            ends = d[d["mode"].isin(OPENERS)]["end"].value_counts(normalize=True) * 100
            rec.update(end_shares_openers={k: float(ends.get(k, 0.0)) for k in
                                           ("violation", "ended_clean_early", "clean_through")},
                       empty_share=float(d["empty"].mean() * 100),
                       median_graded_tokens=float(d["reasoning_tokens"].median()),
                       meta_regex_share=float(d["meta_regex"].mean() * 100),
                       n_full_trace=int(len(full)))
            rows.append(rec)
    return rows, draws


def difference(draws: dict, model: str, a: tuple, b: tuple, metric: str = "S_1000") -> dict | None:
    """Paired difference a - b of two cells (prompt, opening[, thinking]) for one metric."""
    ea = draws.get((model, *a, *(() if len(a) == 3 else (True,)), metric))
    eb = draws.get((model, *b, *(() if len(b) == 3 else (True,)), metric))
    if not ea or not eb:
        return None
    return {"difference": ca.stat(ea[0] - eb[0], ea[1] - eb[1]), "p": bootstrap_p(ea[1] - eb[1])}


def primary_contrasts(draws: dict, models: list[str]) -> list[dict]:
    out = []
    families = {"screened": [m for m in models if m not in HELD_OUT]}
    families.update({f"held_out:{m}": [m] for m in models if m in HELD_OUT})
    for family, fam_models in families.items():
        rows = []
        for model in fam_models:
            for name, (a, b) in PRIMARY.items():
                d = difference(draws, model, a, b)
                if d:
                    rows.append({"model": model, "contrast": name, "a": "|".join(a), "b": "|".join(b),
                                 "family": family, **d})
        adjusted = holm({(r["model"], r["contrast"]): r["p"] for r in rows})
        for r in rows:
            r["p_holm"] = adjusted[(r["model"], r["contrast"])]
        out += rows
    return out


# Secondary contrasts: name -> list of (row label, cell a, cell b, metric)
SECONDARY = {
    "compliant opening vs none": [(PROMPT_LABEL[p], (p, "prefill_compliant"), (p, "none"), "S_1000") for p in PROMPTS],
    "rule beyond copying (compliant opening)": [(PROMPT_LABEL[p], (p, "prefill_compliant"),
                                                 ("no_rule", "prefill_compliant"), "S_1000") for p in PROMPTS],
    "compliant vs non-compliant opening (4 opener rules)": [(PROMPT_LABEL[p], (p, "prefill_compliant"),
                                                             (p, "prefill_noncompliant"), "S_1000_openers")
                                                            for p in PROMPTS],
    "upgraded vs stacked prompt": [(OPENING_LABEL[o], ("upgraded", o), ("stacked", o),
                                    "S_1000_openers" if o == "prefill_noncompliant" else "S_1000") for o in OPENINGS],
}
# Thinking off (cells (prompt, opening, False)), on the rate at 200 tokens. Every prefill forces the opening tag, which
# the no-prefill cell may never write (a violation at token 0), so "vs none" is also given on tagged rows only.
OFF_PREFILL_PROMPTS = ("baseline", "upgraded")
SECONDARY_OFF = {
    "thinking off: compliant opening vs none": [
        (PROMPT_LABEL[p], (p, "prefill_compliant", False), (p, OFF, False), "clean_200") for p in OFF_PREFILL_PROMPTS],
    "thinking off: compliant opening vs none, tagged rows only": [
        (PROMPT_LABEL[p], (p, "prefill_compliant", False), (p, OFF, False), "clean_200_tagged")
        for p in OFF_PREFILL_PROMPTS],
    "thinking off: rule beyond copying (compliant opening)": [
        (PROMPT_LABEL[p], (p, "prefill_compliant", False), ("no_rule", "prefill_compliant", False), "clean_200")
        for p in OFF_PREFILL_PROMPTS],
    "thinking off: compliant vs non-compliant opening (4 opener rules)": [
        (PROMPT_LABEL["upgraded"], ("upgraded", "prefill_compliant", False), ("upgraded", "prefill_noncompliant", False),
         "clean_200_openers")],
}


def secondary_contrasts(draws: dict, models: list[str]) -> list[dict]:
    out = []
    for name, rows in {**SECONDARY, **SECONDARY_OFF}.items():
        for model in models:
            for label, a, b, metric in rows:
                d = difference(draws, model, a, b, metric)
                if d:
                    out.append({"family": name, "row": label, "model": model, "a": "|".join(a), "b": "|".join(b),
                                "metric": metric, **d})
    # Does a compliant opening add less under the upgraded prompt? (upgraded gain) - (stacked gain), paired.
    for model in models:
        keys = [(model, p, o, True, "S_1000") for p in ("upgraded", "stacked") for o in ("prefill_compliant", "none")]
        if all(draws.get(k) for k in keys):
            (p1, d1), (p2, d2), (p3, d3), (p4, d4) = (draws[k] for k in keys)
            out.append({"family": "interaction", "row": "opening gain under upgraded - under stacked", "model": model,
                        "metric": "S_1000", "difference": ca.stat((p1 - p2) - (p3 - p4), (d1 - d2) - (d3 - d4)),
                        "p": bootstrap_p((d1 - d2) - (d3 - d4))})
    return out


def gain(arm, base, normalize: bool) -> tuple[float, np.ndarray]:
    """Gain over baseline in points, or as a share of the baseline's headroom (1 - baseline)."""
    point, draws = arm[0] - base[0], arm[1] - base[1]
    if normalize:  # undefined (nan) when the baseline is already at 100%
        with np.errstate(invalid="ignore", divide="ignore"):
            point = point / (1 - base[0]) if base[0] < 1 else float("nan")
            draws = draws / (1 - base[1])
    return point, draws


def cot_specificity(draws: dict, models: list[str]) -> list[dict]:
    """Per prompt: gain over baseline in clean_200 with thinking on (no opening) and off, raw and as % of headroom;
    and the tags-only version of the thinking-off gain."""
    out = []
    for model in models:
        base_on = draws.get((model, "baseline", "none", True, "clean_200"))
        base_off = draws.get((model, "baseline", "thinking_off", False, "clean_200"))
        base_tag = draws.get((model, "baseline", "thinking_off", False, "clean_200_tagged"))
        for prompt in ("stacked", "upgraded"):
            on = draws.get((model, prompt, "none", True, "clean_200"))
            off = draws.get((model, prompt, "thinking_off", False, "clean_200"))
            tag = draws.get((model, prompt, "thinking_off", False, "clean_200_tagged"))
            if not all((base_on, base_off, on, off)):
                continue
            rec = {"model": model, "prompt": prompt}
            for normalize in (False, True):
                g_on, g_off = gain(on, base_on, normalize), gain(off, base_off, normalize)
                k = "headroom" if normalize else "points"
                rec[f"on_{k}"], rec[f"off_{k}"] = ca.stat(*g_on), ca.stat(*g_off)
                rec[f"on_minus_off_{k}"] = ca.stat(g_on[0] - g_off[0], g_on[1] - g_off[1])
            if tag and base_tag:
                g_tag = gain(tag, base_tag, False)
                rec["off_points_tagged_only"] = ca.stat(*g_tag)
                g_on = gain(on, base_on, False)
                rec["on_minus_off_points_tagged_only"] = ca.stat(g_on[0] - g_tag[0], g_on[1] - g_tag[1])
            out.append(rec)
    return out


def opening_specificity(draws: dict, models: list[str]) -> list[dict]:
    """Per prompt (baseline, upgraded): the compliant opening's gain over no opening in clean_200, with thinking on
    vs off, raw and as % of headroom (as cot_specificity, with the opening in place of the prompt)."""
    out = []
    for model in models:
        for prompt in OFF_PREFILL_PROMPTS:
            on, base_on = (draws.get((model, prompt, o, True, "clean_200")) for o in ("prefill_compliant", "none"))
            off, base_off = (draws.get((model, prompt, o, False, "clean_200")) for o in ("prefill_compliant", OFF))
            if not all((on, base_on, off, base_off)):
                continue
            rec = {"model": model, "prompt": prompt}
            for normalize in (False, True):
                g_on, g_off = gain(on, base_on, normalize), gain(off, base_off, normalize)
                k = "headroom" if normalize else "points"
                rec[f"on_{k}"], rec[f"off_{k}"] = ca.stat(*g_on), ca.stat(*g_off)
                rec[f"on_minus_off_{k}"] = ca.stat(g_on[0] - g_off[0], g_on[1] - g_off[1])
            out.append(rec)
    return out


def accuracy_contrasts(draws: dict, models: list[str]) -> list[dict]:
    out = []
    for model in models:
        for a, b in (("upgraded", "stacked"), ("stacked", "baseline"), ("upgraded", "baseline")):
            for metric in ("correct", "whole_trace_compliant"):
                d = difference(draws, model, (a, "none"), (b, "none"), metric)
                if d:
                    out.append({"model": model, "a": a, "b": b, "metric": metric, **d})
    return out


def per_rule_table(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap) -> pd.DataFrame:
    recs = []
    for model in models:
        for (prompt, opening, thinking), d in df[df["model"] == model].groupby(["prompt", "opening", "thinking"]):
            for mode in RULES:
                c = d[d["mode"] == mode]
                if c.empty:
                    continue
                point, dr = a4.km_at(c, T_STAR, boot)
                lo, hi = ca.ci(dr)
                recs.append({"model": model, "prompt": prompt, "opening": opening, "thinking": thinking,
                             "mode": mode, "n": len(c), "S_1000": point * 100, "ci_lo": lo * 100, "ci_hi": hi * 100,
                             "clean_200": float(c["clean_200"].mean() * 100),
                             "empty_share": float(c["empty"].mean() * 100)})
    return pd.DataFrame(recs)


def screen_estimates() -> dict:
    """exp06a's (and exp04's, through exp06a's analysis) estimates on exp03's items for the cells exp06 repeats."""
    path = cfg.EXP06A.results / "analysis" / "round2_complete_report" / "summary.json"
    arms = json.loads(path.read_text())["arms"]
    same_cell = {"baseline_rerun": ("baseline", "none"), "stacked_rerun": ("stacked", "none"),
                 cfg.EXP06_PROMPTS["upgraded"]: ("upgraded", "none"), "exp04_prefill": ("baseline", "prefill_compliant"),
                 "exp04_prefill_stacked": ("stacked", "prefill_compliant"),
                 "exp04_prefill_no_rule": ("no_rule", "prefill_compliant")}
    return {(a["model"], *same_cell[a["arm"]]): a["S_t_star"] for a in arms
            if a["thinking"] and a["arm"] in same_cell and a.get("S_t_star")}


def checks(df: pd.DataFrame, models: list[str]) -> dict:
    on, off = df[df["thinking"]], df[~df["thinking"]]
    records = {m: json.loads(cc_exp06.record_path(m).read_text()) for m in models}
    off_records = {m: json.loads(p.read_text()) for m in models
                   if (p := cc_exp06.record_path(m, cc_exp06.OFF_PREFILL_PART)).exists()}
    return {
        "thinking_on_status_share": {k: float(v) * 100 for k, v in on["think_status"].value_counts(normalize=True).items()},
        "thinking_on_empty_share_by_cell": {f"{m}|{p}|{o}": float(g["empty"].mean() * 100)
                                            for (m, p, o), g in on.groupby(["model", "prompt", "opening"])},
        # no tag content: no closed tag block, or (prefill rows) nothing after the prefill inside it
        "thinking_off_no_tag_share": {f"{m}|{p}|{o}": float(g["no_tags"].mean() * 100)
                                      for (m, p, o), g in off.groupby(["model", "prompt", "opening"])},
        "fewshot_items": {m: r["fewshot_items"] for m, r in records.items()},
        "rendering_checks": {m: r["rendering_checks"] for m, r in records.items()},
        "off_prefill_rendering_checks": {m: r["rendering_checks"] for m, r in off_records.items()},
    }


# --- Figures ----------------------------------------------------------------------------------------------------------
def ci_bars(stats: list[dict | None]) -> dict:
    return {"type": "data", "symmetric": False, "array": [s["ci"][1] - s["value"] if s else None for s in stats],
            "arrayminus": [s["value"] - s["ci"][0] if s else None for s in stats], "thickness": 1.5}


def fig_prefill_comparison(rows: list[dict], models: list[str], fig_dir) -> str:
    """E1. x: opening; one line per prompt; the no-rule copying control as a black marker at the compliant opening."""
    est = {(r["model"], r["prompt"], r["opening"]): r for r in rows if r["thinking"]}
    fig = make_subplots(rows=2, cols=len(models), shared_yaxes=True, vertical_spacing=0.16, horizontal_spacing=0.03,
                        row_heights=[0.62, 0.38],
                        subplot_titles=[model_name(m) for m in models] + [""] * len(models))
    for row, (metric, openings, what) in enumerate(((("S_1000_openers", OPENINGS, "4 opener rules")),
                                                    ("S_1000_word_suppression", OPENINGS[:2], "word suppression")), 1):
        x = {o: i for i, o in enumerate(OPENINGS)}
        for k, model in enumerate(models):
            for prompt in PROMPTS + ["no_rule"]:
                pts = [(o, est.get((model, prompt, o), {}).get(metric)) for o in openings]
                pts = [(o, s) for o, s in pts if s]
                if not pts:
                    continue
                color = cfg.EXP06_STYLE[prompt]["color"]
                fig.add_trace(go.Scatter(
                    x=[x[o] + (0.08 if prompt == "no_rule" else 0) for o, _ in pts], y=[s["value"] for _, s in pts],
                    mode="lines+markers+text" if prompt != "no_rule" else "markers+text",
                    name=PROMPT_LABEL[prompt], legendgroup=prompt, showlegend=row == 1 and k == 0,
                    line={"color": color, "width": 2.5},
                    marker={"color": color, "size": 10, "symbol": "x" if prompt == "no_rule" else "circle",
                            "line": {"color": "black", "width": 0.6}},
                    error_y={**ci_bars([s for _, s in pts]), "color": color},
                    text=[f"{s['value']:.0f}" for _, s in pts], textposition="middle right",
                    textfont={"size": 10, "color": color},
                    hovertemplate=[f"{PROMPT_LABEL[prompt]} | {OPENING_LABEL[o]}: {s['value']:.1f} "
                                   f"[{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]<extra>{model}</extra>" for o, s in pts]),
                    row=row, col=k + 1)
            fig.update_xaxes(tickvals=list(x.values()), ticktext=[OPENING_LABEL[o] for o in OPENINGS],
                             range=[-0.3, len(OPENINGS) - 0.6], row=row, col=k + 1)
        fig.update_yaxes(title_text=f"S(1000), %<br>{what}", range=[0, 100], row=row, col=1)
    fig.update_layout(title="E1. Does a compliant start still help once the prompt is upgraded? Thinking on, 100 new "
                            "questions<br><sup>% of traces with no violation in the first 1000 tokens (KM), mean over "
                            "rules, 95% question-level bootstrap CIs. Openings are scored on what the model writes "
                            "after them. ×: compliant opening with no rule in the prompt<br>(how much is copying the "
                            "opening's style). Word suppression (bottom) has no non-compliant opening; its opening "
                            "is the neutral sentence</sup>",
                      legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"},
                      margin={"l": 80, "t": 125, "b": 90})
    return a4.save(fig, fig_dir, "E1_prefill_comparison", 250 + 400 * len(models), 860)


def fig_levers(secondary: list[dict], primary: list[dict], models: list[str], fig_dir) -> str:
    """E2. Paired differences in S(1000); primary contrasts marked with their Holm p."""
    families = list(SECONDARY)
    fig = make_subplots(rows=1, cols=len(families), horizontal_spacing=0.1, subplot_titles=families)
    offsets = dict(zip(models, np.linspace(-0.24, 0.24, len(models))))
    p_holm = {(p["model"], p["a"], p["b"]): p["p_holm"] for p in primary}
    for c, family in enumerate(families):
        labels = [label for label, *_ in SECONDARY[family]]
        for model in models:
            color, symbol = MODEL_STYLE[model]
            rows = [r for r in secondary if r["family"] == family and r["model"] == model]
            ys = [labels.index(r["row"]) + offsets[model] for r in rows]
            stats = [r["difference"] for r in rows]
            hover = [f"{r['a']} - {r['b']}: {s['value']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]"
                     + (f", Holm p {p_holm[(model, r['a'], r['b'])]:.3f} (primary)"
                        if (model, r["a"], r["b"]) in p_holm else "") for r, s in zip(rows, stats)]
            primary_row = [(model, r["a"], r["b"]) in p_holm for r in rows]
            fig.add_trace(go.Scatter(
                x=[s["value"] for s in stats], y=ys, mode="markers", name=model_name(model), legendgroup=model,
                showlegend=c == 0,
                marker={"color": [color if p else "white" for p in primary_row], "symbol": symbol, "size": 10,
                        "line": {"color": color, "width": 1.8}},
                error_x={"type": "data", "symmetric": False, "array": [s["ci"][1] - s["value"] for s in stats],
                         "arrayminus": [s["value"] - s["ci"][0] for s in stats], "color": color, "thickness": 1.5},
                hovertemplate=[h + f"<extra>{model}</extra>" for h in hover]), row=1, col=c + 1)
        fig.add_vline(x=0, line={"color": "#555555", "width": 1}, row=1, col=c + 1)
        fig.update_yaxes(tickvals=list(range(len(labels))), ticktext=[label.split(" (")[0] for label in labels],
                         range=[len(labels) - 0.5, -0.5], row=1, col=c + 1)
        fig.update_xaxes(title_text="difference in S(1000), points", range=[-40, 90], row=1, col=c + 1)
    fig.update_layout(title="E2. What each lever adds (thinking on, 100 new questions)<br><sup>Paired differences in "
                            "S(1000) with 95% bootstrap CIs. Filled markers: pre-registered primary contrasts (hover "
                            "for Holm p); open: secondary. Third panel uses the 4 opener rules (word suppression has "
                            "no non-compliant opening)</sup>",
                      legend={"orientation": "h", "y": -0.22, "x": 0.5, "xanchor": "center"},
                      margin={"l": 60, "t": 110, "b": 120})
    return a4.save(fig, fig_dir, "E2_what_each_lever_adds", 1900, 520)


# Neutral colours, so the bars cannot be read as prompts.
END_STYLE = {"violation": ("#4D4D4D", f"bars: violation before token {T_STAR}"),
             "ended_clean_early": ("#C8C8C8", f"bars: no violation, but the text ended before token {T_STAR}"),
             "clean_through": ("#F0E442", f"bars: no violation through token {T_STAR}")}


def fig_survival_and_ends(df: pd.DataFrame, rows: list[dict], models: list[str], fig_dir) -> str:
    """E3. Per (model, opening): KM S(t) per prompt, mean over the 4 opener rules; under it, one bar per prompt with
    how the traces ended (so a high S(t) from short traces is visible)."""
    # The cut at the stop can split a word or sentence, which the grader scores as a violation (tokens ~1198), so
    # the curves end CUT_MARGIN tokens before it. Estimates at t* are unaffected.
    grid = np.arange(0, STOP - CUT_MARGIN + 1, 10, dtype=float)
    est = {(r["model"], r["prompt"], r["opening"]): r for r in rows if r["thinking"]}
    specs_rows = []
    for _ in models:
        specs_rows += [0.75, 0.25]
    titles = []
    for m in models:
        titles += [f"{model_name(m)} | {OPENING_LABEL[o]}" for o in OPENINGS] + [""] * len(OPENINGS)
    fig = make_subplots(rows=2 * len(models), cols=len(OPENINGS), row_heights=specs_rows, vertical_spacing=0.05,
                        horizontal_spacing=0.06, subplot_titles=titles)
    for i, model in enumerate(models):
        r_curve, r_bar = 2 * i + 1, 2 * i + 2
        for j, opening in enumerate(OPENINGS):
            prompts = PROMPTS + (["no_rule"] if opening == "prefill_compliant" else [])
            for prompt in prompts:
                d = cell(df, model, prompt, opening)
                d = d[d["mode"].isin(OPENERS)]
                if d.empty:
                    continue
                curves = [kaplan_meier(g["time"].to_numpy(), g["event"].to_numpy(), np.ones((1, len(g))), grid)[0]
                          for _, g in d.groupby("mode")]
                color = cfg.EXP06_STYLE[prompt]["color"]
                fig.add_trace(go.Scatter(x=grid, y=100 * np.mean(curves, axis=0), mode="lines",
                                         line={"color": color, "width": 2.2,
                                               "dash": "dot" if prompt == "no_rule" else "solid"},
                                         name=PROMPT_LABEL[prompt], legendgroup=prompt,
                                         showlegend=i == 0 and j == 1,
                                         hovertemplate=f"{PROMPT_LABEL[prompt]}<br>t=%{{x}}: %{{y:.0f}}%"
                                                       f"<extra>{model}</extra>"),
                              row=r_curve, col=j + 1)
            fig.add_vline(x=T_STAR, line={"color": "black", "width": 0.6, "dash": "dot"}, row=r_curve, col=j + 1)
            fig.update_yaxes(range=[0, 101], row=r_curve, col=j + 1, showticklabels=j == 0)
            fig.update_xaxes(range=[0, STOP - CUT_MARGIN], row=r_curve, col=j + 1, tickfont={"size": 9})
            for end, (color, label) in END_STYLE.items():
                ys = [p for p in prompts if (model, p, opening) in est]
                fig.add_trace(go.Bar(
                    y=[SHORT_PROMPT[p] for p in ys],
                    x=[est[(model, p, opening)]["end_shares_openers"][end] for p in ys], orientation="h",
                    marker={"color": color, "line": {"color": "#888888", "width": 0.5}}, name=label,
                    legendgroup=end, showlegend=i == 0 and j == 0,
                    hovertemplate="%{y}: %{x:.0f}%<extra>" + label + "</extra>"), row=r_bar, col=j + 1)
            fig.update_xaxes(range=[0, 100], row=r_bar, col=j + 1, showticklabels=i == len(models) - 1,
                             title_text="% of traces" if i == len(models) - 1 else None)
            fig.update_yaxes(tickfont={"size": 9}, row=r_bar, col=j + 1)
        fig.update_yaxes(title_text="% no violation yet", row=r_curve, col=1)
    height, top, bottom = 330 * len(models) + 310, 170, 140
    fig.update_layout(barmode="stack",
                      title="E3. When violations happen, and how traces end (thinking on, 4 opener rules, 100 new "
                            "questions)<br><sup>Curves: KM share of traces with no violation yet (y) by tokens into "
                            "the graded text, after the opening (x; dotted line: 1000; rows stop at 1200,<br>curves "
                            "end at 1150 because the cut at the stop can split a word). Bars: % of traces that "
                            "violated before token 1000 / ended clean but shorter than 1000 tokens / were clean "
                            "through 1000.<br>A long light-grey segment means part of S(1000) rests on short "
                            "texts</sup>",
                      legend={"orientation": "h", "y": -70 / (height - top - bottom), "yanchor": "top", "x": 0.5,
                              "xanchor": "center"},
                      margin={"l": 90, "t": top, "b": bottom})
    return a4.save(fig, fig_dir, "E3_survival_and_ends", 1600, height)


def fig_cot_vs_output(spec: list[dict], opening_spec: list[dict], models: list[str], fig_dir) -> str:
    """E4. Gain with thinking on vs off, of each prompt over the CoT-Control prompt (circles) and of the compliant
    opening over none under a prompt (triangles); points (left), % of headroom (right)."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["gain in points", "gain as % of the room left (100 - the cell it is over)"])
    lever = {("prompt", "stacked"): ("circle-open", "stacked prompt over CoT-Control's"),
             ("prompt", "upgraded"): ("circle", "upgraded prompt over CoT-Control's"),
             ("opening", "baseline"): ("triangle-up-open", "compliant opening over none, CoT-Control's prompt"),
             ("opening", "upgraded"): ("triangle-up", "compliant opening over none, upgraded prompt")}
    gains = [("prompt", r) for r in spec] + [("opening", r) for r in opening_spec]
    for c, unit in enumerate(("points", "headroom"), 1):
        for model in models:
            color, _ = MODEL_STYLE[model]
            pts = [(kind, r) for kind, r in gains if r["model"] == model]
            fig.add_trace(go.Scatter(
                x=[r[f"on_{unit}"]["value"] for _, r in pts], y=[r[f"off_{unit}"]["value"] for _, r in pts],
                mode="markers", name=model_name(model), legendgroup=model, showlegend=c == 1,
                marker={"color": color, "size": 12, "symbol": [lever[(kind, r["prompt"])][0] for kind, r in pts],
                        "line": {"color": color, "width": 2}},
                error_x={**ci_bars([r[f"on_{unit}"] for _, r in pts]), "color": color},
                error_y={**ci_bars([r[f"off_{unit}"] for _, r in pts]), "color": color},
                hovertemplate=[f"{lever[(kind, r['prompt'])][1]}: on {r[f'on_{unit}']['value']:+.1f}, off "
                               f"{r[f'off_{unit}']['value']:+.1f}; on - off {r[f'on_minus_off_{unit}']['value']:+.1f} "
                               f"[{r[f'on_minus_off_{unit}']['ci'][0]:+.1f}, {r[f'on_minus_off_{unit}']['ci'][1]:+.1f}]"
                               f"<extra>{model}</extra>" for kind, r in pts]), row=1, col=c)
        fig.add_trace(go.Scatter(x=[-100, 100], y=[-100, 100], mode="lines", showlegend=False, hoverinfo="skip",
                                 line={"color": "#999999", "dash": "dot", "width": 1}), row=1, col=c)
        lows = [s["ci"][0] for _, r in gains for s in (r[f"on_{unit}"], r[f"off_{unit}"]) if np.isfinite(s["ci"][0])]
        lo = min([-10] + [v - 5 for v in lows])
        fig.update_xaxes(range=[lo, 100], title_text=f"gain with thinking on ({unit})", row=1, col=c)
        fig.update_yaxes(range=[lo, 100], title_text=f"gain with thinking off ({unit})", row=1, col=c)
    for symbol, label in lever.values():  # symbol legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker={"symbol": symbol, "color": "#555555", "size": 11}), row=1, col=1)
    fig.update_layout(title="E4. Is the gain specific to the CoT? Rule on the thinking trace vs on the "
                            "&lt;output_reasoning&gt; tag content (100 new questions)"
                            f"<br><sup>Metric: share of ALL texts reaching {T_SHORT} tokens with no violation (a short "
                            "or missing text counts as a failure). Dotted: equal gain; below it = more specific to the "
                            "CoT. Circles: a prompt's gain (no opening); triangles:<br>a compliant opening's gain "
                            "(thinking off: prefilled after the opening tag, which the no-opening cell may never "
                            "write). Right: gain / room left, since thinking-off cells start higher. Hover for on - off "
                            "with its CI</sup>",
                      legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
                      margin={"t": 125, "b": 150})
    return a4.save(fig, fig_dir, "E4_cot_vs_output", 1400, 700)


def fig_per_rule(per_rule: pd.DataFrame, models: list[str], fig_dir) -> str:
    """E5. S(1000) per rule and thinking-on cell."""
    d = per_rule[per_rule["thinking"]]
    order = [f"{PROMPT_LABEL[p].split(' (')[0]} | {OPENING_LABEL[o]}" for p, o in CELLS_ON]
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=[model_name(m) for m in models])
    for k, model in enumerate(models):
        dm = d[d["model"] == model].assign(
            cell=lambda x: [f"{PROMPT_LABEL[p].split(' (')[0]} | {OPENING_LABEL[o]}" for p, o in zip(x["prompt"], x["opening"])])
        m = dm.pivot(index="cell", columns="mode", values="S_1000").reindex(index=order, columns=RULES)
        fig.add_trace(go.Heatmap(z=m.to_numpy(), x=[a4.MODE_LABEL[r].split(" (")[0] for r in RULES], y=order,
                                 zmin=0, zmax=100, colorscale="Viridis", showscale=k == len(models) - 1,
                                 text=m.map(lambda v: "" if pd.isna(v) else f"{v:.0f}").to_numpy(),
                                 texttemplate="%{text}", hovertemplate="%{y} | %{x}: %{z:.1f}%<extra></extra>"),
                      row=1, col=k + 1)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="E5. Where the effects come from: S(1000) per rule, thinking on, % of 100 new questions"
                            "<br><sup>Blank: no non-compliant opening for word suppression</sup>",
                      margin={"l": 300, "t": 100})
    return a4.save(fig, fig_dir, "E5_per_rule", 400 + 350 * len(models), 560)


def fig_accuracy(rows: list[dict], models: list[str], fig_dir) -> str:
    """E6. Full-trace cells (25% of items per rule, no opening): accuracy and whole-trace compliance per prompt."""
    est = {(r["model"], r["prompt"]): r for r in rows if r["thinking"] and r["opening"] == "none"}
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=["accuracy (answer correct)", "whole trace follows the rule (grader)"])
    offsets = dict(zip(PROMPTS, np.linspace(-0.25, 0.25, len(PROMPTS))))
    for c, metric in enumerate(("correct", "whole_trace_compliant"), 1):
        for prompt in PROMPTS:
            stats = [est.get((m, prompt), {}).get(metric) for m in models]
            color = cfg.EXP06_STYLE[prompt]["color"]
            fig.add_trace(go.Scatter(
                x=[i + offsets[prompt] for i in range(len(models))], y=[s["value"] if s else None for s in stats],
                mode="markers", name=PROMPT_LABEL[prompt], legendgroup=prompt, showlegend=c == 1,
                marker={"color": color, "size": 11, "line": {"color": "black", "width": 0.6}},
                error_y={**ci_bars(stats), "color": color},
                hovertemplate="%{y:.1f}%<extra>" + PROMPT_LABEL[prompt] + "</extra>"), row=1, col=c)
        fig.update_xaxes(tickvals=list(range(len(models))), ticktext=[model_name(m) for m in models], row=1, col=c)
        fig.update_yaxes(range=[0, 100], title_text="% of full-trace rows", row=1, col=c)
    n = est.get((models[0], "baseline"), {}).get("n_full_trace", 0)
    fig.update_layout(title="E6. Does control cost accuracy? Full traces with an answer, no opening "
                            f"({n} rows per prompt and model: 25% of the questions per rule)<br><sup>95% bootstrap "
                            "CIs; the same questions, rules and seeds under every prompt, so differences are paired "
                            "(REPORT_auto.md)</sup>",
                      legend={"orientation": "h", "y": -0.15, "x": 0.5, "xanchor": "center"},
                      margin={"t": 100, "b": 100})
    return a4.save(fig, fig_dir, "E6_accuracy", 1400, 560)


def fig_replication(rows: list[dict], screen: dict, models: list[str], fig_dir) -> tuple[str, list[dict]]:
    """E7. Same cell, exp03's items (exp06a/exp04) vs exp06's new items: below the diagonal = lower on new items."""
    est = {(r["model"], r["prompt"], r["opening"]): r["S_1000"] for r in rows if r["thinking"] and r.get("S_1000")}
    cells = [("baseline", "none"), ("stacked", "none"), ("upgraded", "none"), ("baseline", "prefill_compliant"),
             ("stacked", "prefill_compliant"), ("no_rule", "prefill_compliant")]
    symbol = {("baseline", "none"): "circle-open", ("stacked", "none"): "square-open", ("upgraded", "none"): "star",
              ("baseline", "prefill_compliant"): "triangle-up-open", ("stacked", "prefill_compliant"): "triangle-up",
              ("no_rule", "prefill_compliant"): "x"}
    fig = go.Figure()
    data = []
    for model in models:
        color, _ = MODEL_STYLE[model]
        pts = [(c, screen.get((model, *c)), est.get((model, *c))) for c in cells]
        pts = [(c, a, b) for c, a, b in pts if a and b]
        data += [{"model": model, "cell": "|".join(c), "exp03_items": a, "new_items": b} for c, a, b in pts]
        fig.add_trace(go.Scatter(
            x=[a["value"] for _, a, _ in pts], y=[b["value"] for _, _, b in pts], mode="markers", name=model,
            marker={"color": color, "size": 12, "symbol": [symbol[c] for c, _, _ in pts],
                    "line": {"color": color, "width": 2}},
            error_x={**ci_bars([a for _, a, _ in pts]), "color": color},
            error_y={**ci_bars([b for _, _, b in pts]), "color": color},
            hovertemplate=[f"{'|'.join(c)}: exp03 items {a['value']:.1f}, new items {b['value']:.1f}<extra>{model}"
                           f"</extra>" for c, a, b in pts]))
    for c in cells:  # symbol legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="|".join(c),
                                 marker={"symbol": symbol[c], "color": "#555555", "size": 10}))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", showlegend=False, hoverinfo="skip",
                             line={"color": "#999999", "dash": "dot", "width": 1}))
    fig.update_xaxes(range=[0, 100], title_text="S(1000) on exp03's 100 questions (exp06a / exp04), %")
    fig.update_yaxes(range=[0, 100], title_text="S(1000) on exp06's 100 new questions, %")
    fig.update_layout(title="E7. Do the screen's numbers hold on new questions? Same cell, same model<br><sup>The "
                            "upgraded prompt (★) was chosen on exp03's questions, so it should sit a little below the "
                            "diagonal if the screen overfit them.<br>Held-out model not shown (no screen numbers). "
                            "exp04's prefill rows ran in an earlier environment. 95% CIs on both axes</sup>",
                      margin={"t": 120})
    return a4.save(fig, fig_dir, "E7_replication", 1000, 760), data


def fig_openings_both_channels(rows: list[dict], models: list[str], fig_dir) -> str:
    """E8. Rate at 200 tokens (4 opener rules) per opening, thinking on (solid) vs off (dotted), for the prompts
    that have thinking-off openings; ×: the no-rule copying control."""
    est = {(r["model"], r["prompt"], OPENINGS[0] if r["opening"] == OFF else r["opening"], r["thinking"]): r
           for r in rows}
    x = {o: i for i, o in enumerate(OPENINGS)}
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[model_name(m) for m in models])
    for k, model in enumerate(models):
        for thinking, dash, channel in ((True, "solid", "thinking on"), (False, "dot", "thinking off")):
            for prompt in OFF_PREFILL_PROMPTS + ("no_rule",):
                pts = [(o, est.get((model, prompt, o, thinking), {}).get("clean_200_openers")) for o in OPENINGS]
                pts = [(o, s) for o, s in pts if s]
                color, copying = cfg.EXP06_STYLE[prompt]["color"], prompt == "no_rule"
                fig.add_trace(go.Scatter(
                    x=[x[o] + (0.08 if copying else 0) + (0 if thinking else 0.04) for o, _ in pts],
                    y=[s["value"] for _, s in pts], mode="markers" if copying else "lines+markers",
                    name=f"{PROMPT_LABEL[prompt]}, {channel}", legendgroup=f"{prompt}{thinking}",
                    showlegend=k == 0, line={"color": color, "width": 2.5, "dash": dash},
                    marker={"color": color if thinking else "white", "size": 10,
                            "symbol": "x" if copying else "circle", "line": {"color": color, "width": 1.5}},
                    error_y={**ci_bars([s for _, s in pts]), "color": color},
                    hovertemplate=[f"{PROMPT_LABEL[prompt]} | {OPENING_LABEL[o]} | {channel}: {s['value']:.1f} "
                                   f"[{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]<extra>{model}</extra>" for o, s in pts]),
                    row=1, col=k + 1)
        fig.update_xaxes(tickvals=list(x.values()), ticktext=[OPENING_LABEL[o] for o in OPENINGS],
                         range=[-0.3, len(OPENINGS) - 0.6], row=1, col=k + 1)
    fig.update_yaxes(title_text=f"% reaching {T_SHORT} tokens clean<br>4 opener rules", range=[0, 100], row=1, col=1)
    fig.update_layout(title="E8. Does an opening sentence work the same in the thinking trace and in reasoning "
                            "written in the answer? (100 new questions)<br><sup>Share of ALL texts reaching "
                            f"{T_SHORT} tokens with no violation after the opening, 4 opener rules, 95% CIs. Solid / "
                            "filled: thinking on; dotted / open: thinking off, the sentence prefilled after "
                            "&lt;output_reasoning&gt;.<br>Thinking off with no opening must write the tag itself "
                            "(no tag = violation at token 0); every prefill writes it. ×: the sentence with no rule "
                            "in the prompt (copying control)</sup>",
                      legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
                      margin={"l": 80, "t": 130, "b": 120})
    return a4.save(fig, fig_dir, "E8_openings_both_channels", 1700, 560)


# --- Report -------------------------------------------------------------------------------------------------------------
def report(s: dict, figures: list[str], fig_rel: str) -> str:
    fmt = ca.fmt
    lines = [f"# exp06 analysis: {s['run']} (UNVERIFIED until a human adds it to VERIFIED.md)", "",
             f"Models: {', '.join(model_name(m) for m in s['models'])}. Missing parts: {s['missing_parts'] or 'none'}.",
             "100 new questions x 5 rules per cell; thinking on unless stated. S(1000) = KM % of traces with no "
             "violation in the first 1000 graded tokens (prefill rows: after the prefill).", "",
             "## Primary contrasts: S(1000), 5-rule mean (Holm within the screened family and within each held-out model)", "",
             "| model | contrast | difference (pts) | p | p Holm |", "|---|---|---|---|---|"]
    for r in s["primary"]:
        lines.append(f"| {model_name(r['model'])} | {r['contrast']} | {fmt(r['difference'])} | {r['p']:.3f} | "
                     f"{r['p_holm']:.3f} |")
    lines += ["", "## Cells", "", "| model | prompt | opening | S(1000) 5 rules | S(1000) 4 opener rules | "
              "clean through 1000 % | S(1000) word suppression | empty % | median graded tokens |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in s["cells"]:
        if r["thinking"]:
            lines.append(f"| {model_name(r['model'])} | {r['prompt']} | {r['opening']} | {fmt(r['S_1000'])} | "
                         f"{fmt(r['S_1000_openers'])} | {fmt(r['clean_1000'])} | {fmt(r['S_1000_word_suppression'])} | "
                         f"{r['empty_share']:.1f} | {r['median_graded_tokens']:.0f} |")
    lines += ["", "## Thinking-off cells (rule on the <output_reasoning> tag content; openings prefilled after the "
              "opening tag)", "", "| model | prompt | opening | clean at 200 % | clean at 200 %, 4 opener rules | "
              "clean at 200 %, tagged rows only | no tag content % | of which closed with </think> % of rows | "
              "median graded tokens |", "|---|---|---|---|---|---|---|---|---|"]
    for r in s["cells"]:
        if not r["thinking"]:
            lines.append(f"| {model_name(r['model'])} | {r['prompt']} | {r['opening']} | {fmt(r['clean_200'])} | "
                         f"{fmt(r['clean_200_openers'])} | {fmt(r.get('clean_200_tagged'))} | {r['empty_share']:.1f} | "
                         f"{r['think_close_no_tag_share']:.1f} | {r['median_graded_tokens']:.0f} |")
    lines += ["", "## Secondary contrasts (paired; metric per row)", "",
              "| family | row | model | metric | difference (pts) | p |", "|---|---|---|---|---|---|"]
    for r in s["secondary"]:
        lines.append(f"| {r['family']} | {r['row']} | {model_name(r['model'])} | {r['metric']} | "
                     f"{fmt(r['difference'])} | {r['p']:.3f} |")
    lines += ["", "## Opening specificity: compliant opening's gain over none, % of texts reaching 200 tokens clean, "
              "thinking on vs off", "", "| model | prompt | on (pts) | off (pts) | on - off (pts) | on (% headroom) | "
              "off (% headroom) | on - off (% headroom) |", "|---|---|---|---|---|---|---|---|"]
    for r in s["opening_specificity"]:
        lines.append(f"| {model_name(r['model'])} | {r['prompt']} | {fmt(r['on_points'])} | {fmt(r['off_points'])} | "
                     f"{fmt(r['on_minus_off_points'])} | {fmt(r['on_headroom'])} | {fmt(r['off_headroom'])} | "
                     f"{fmt(r['on_minus_off_headroom'])} |")
    lines += ["", "## CoT specificity: gain over baseline in % of texts reaching 200 tokens clean, thinking on vs off",
              "", "| model | prompt | on (pts) | off (pts) | on - off (pts) | on - off, tagged rows only | on (% headroom) "
              "| off (% headroom) | on - off (% headroom) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in s["cot_specificity"]:
        lines.append(f"| {model_name(r['model'])} | {r['prompt']} | {fmt(r['on_points'])} | {fmt(r['off_points'])} | "
                     f"{fmt(r['on_minus_off_points'])} | {fmt(r.get('on_minus_off_points_tagged_only'))} | "
                     f"{fmt(r['on_headroom'])} | {fmt(r['off_headroom'])} | {fmt(r['on_minus_off_headroom'])} |")
    lines += ["", "## Accuracy and whole-trace compliance (full-trace cells, no opening): paired differences", "",
              "| model | a - b | metric | difference (pts) | p |", "|---|---|---|---|---|"]
    for r in s["accuracy"]:
        lines.append(f"| {model_name(r['model'])} | {r['a']} - {r['b']} | {r['metric']} | {fmt(r['difference'])} | "
                     f"{r['p']:.3f} |")
    lines += ["", "## Checks", "", "```", json.dumps(s["checks"], indent=1), "```", "", "## Figures", ""]
    lines += [f"- [{f}]({fig_rel}/{f}.html)" for f in figures]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="analysis run name")
    parser.add_argument("--skip-missing", action="store_true", help="analyse the parts graded so far")
    parser.add_argument("--scratch", action="store_true", help="write under /tmp/exp06_scratch (layout tests)")
    args = parser.parse_args()
    if args.scratch:
        out_dir = fig_dir = Path("/tmp/exp06_scratch") / args.run
    else:
        out_dir, fig_dir = EXP.results / "analysis" / args.run, EXP.figure_dir(args.run)
    for d in {out_dir, fig_dir}:
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")
    raw, missing = load(args.skip_missing)
    df = score(raw)
    models = [m for m in MODELS if m in set(df["model"])]
    for d in {out_dir, fig_dir}:
        d.mkdir(parents=True)
    boot = ca.Bootstrap(cc_exp06.load_items()[0], cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    rows, draws = estimates(df, models, boot)
    per_rule = per_rule_table(df, models, boot)
    summary = {"exp_id": EXP.exp_id, "run": args.run, "models": models, "missing_parts": missing,
               "t_star": T_STAR, "t_short": T_SHORT, "stop": STOP,
               "primary": primary_contrasts(draws, models), "secondary": secondary_contrasts(draws, models),
               "cot_specificity": cot_specificity(draws, models),
               "opening_specificity": opening_specificity(draws, models), "accuracy": accuracy_contrasts(draws, models),
               "cells": rows, "checks": checks(df, models)}
    screened = [m for m in models if m not in HELD_OUT]
    figures = [fig_prefill_comparison(rows, models, fig_dir),
               fig_levers(summary["secondary"], summary["primary"], models, fig_dir),
               fig_survival_and_ends(df, rows, models, fig_dir),
               fig_cot_vs_output(summary["cot_specificity"], summary["opening_specificity"], models, fig_dir),
               fig_per_rule(per_rule, models, fig_dir),
               fig_accuracy(rows, models, fig_dir)]
    name, summary["replication"] = fig_replication(rows, screen_estimates(), screened, fig_dir)
    figures += [name, fig_openings_both_channels(rows, models, fig_dir)]
    per_rule.to_csv(out_dir / "per_rule.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    text = report(summary, figures, os.path.relpath(fig_dir, out_dir))
    (out_dir / "REPORT_auto.md").write_text(text)
    print(f"wrote {out_dir}")
    print(text)


if __name__ == "__main__":
    main()
