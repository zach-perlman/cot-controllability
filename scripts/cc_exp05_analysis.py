"""exp05 analysis (results/exp05_dose/manifest.json): rule-following or imitation, as the compliant opening grows?

Survival per trace as exp04 (event = first violation in the continuation; an empty thinking trace is a violation at
token 0, the manifest's scoring; --censor-empty-traces gives the censored variant). Estimators, bootstrap, the lenient
case-rule grader and plotting helpers come from cc_exp04_analysis.

Cells: per model (cfg.EXP05_ALL_MODELS), opening dose (d1 / d2 / d3: the model's own opening, rewritten to comply)
and condition (prefill_compliant: rule in the prompt; prefill_no_rule: no rule). Reference points (not in the tests):
d0 = no opening (baseline prompt), "generic" = exp04's one generic compliant sentence. Both come from exp04, except
for cfg.EXP05_NEW_MODELS, whose d0 is exp05's requests_base rows and which have no generic point.

Primary: R_d1 and R_d3 (rule arm - no-rule arm, S(t*) over the 4 opener rules) and R_trend = R_d3 - R_d1; Holm over
the 3 contrasts x every analyzed model (deviations_one_analysis.json; manifest.json's 9-contrast Holm is kept in
summary.json as p_holm_manifest). Secondary: rule-arm d3 - d1; lenient grader; answer correctness vs rule-keeping on
full-trace cells (G5); the effort check (Qwen3.8 xhigh / medium / low on exp04's requests of two arms; Holm over the
4 effort contrasts).

Outputs (never overwritten): results/exp05_dose/analysis/<run>/ and figures/exp05_dose/<run>/ (--scratch: under
/tmp/exp05_scratch/<run>/, for tests on partial data).
Run: /venv/main/bin/python scripts/cc_exp05_analysis.py --run NAME [--skip-models M ...] [--skip-missing]
     [--censor-empty-traces] [--scratch]
"""

from __future__ import annotations

import argparse
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
from cc_survival import bootstrap_p, holm

EXP = cfg.EXP05
T_STAR, T_SHORT = a4.T_STAR, a4.T_SHORT
OPENERS = cfg.EXP05_MODES
DOSES = list(cfg.EXP05_DOSE_MIN_TOKENS)
X_ORDER = ["d0", "generic"] + DOSES  # reference points first
X_LABEL = {"d0": "no opening", "generic": "exp04 generic<br>sentence", "d1": "own opening:<br>1 sentence",
           "d2": "own opening:<br>≥ 60 tokens", "d3": "own opening:<br>≥ 150 tokens"}
ARM_STYLE = {"prefill_compliant": {"color": cfg.CONDITION_STYLE["prefill_compliant"]["color"], "label": "rule in prompt"},
             "prefill_no_rule": {"color": cfg.CONDITION_STYLE["prefill_no_rule"]["color"],
                                 "label": "no rule (imitation only)"},
             "none": {"color": cfg.CONDITION_STYLE["none"]["color"], "label": "no opening, rule in prompt"}}
EFFORTS = {"Qwen3.8-27B-FP8-xhigh": "xhigh", "Qwen3.8-27B-FP8": "medium", "Qwen3.8-27B-FP8-low": "low"}
PRIMARY = [("R_d1", "d1"), ("R_d3", "d3")]
MODEL_COLORS = dict(zip(cfg.EXP05_ALL_MODELS,  # Okabe-Ito
                        ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#F0E442"]))
GRID_COLS = 4


# --- Loading ---------------------------------------------------------------------------------------------------------
def score(df: pd.DataFrame, empty_is_violation: bool) -> pd.DataFrame:
    """exp04's survival columns for exp05 rows (no thinking-off rows here)."""
    df = df.copy()
    df["empty_trace"] = df["reasoning_tokens"] == 0
    df["event_at_0"] = df["empty_trace"] & empty_is_violation
    df["event"] = df["fv_token"].notna() | df["event_at_0"]
    df["time"] = np.where(df["fv_token"].notna(), df["fv_token"].fillna(0),
                          np.where(df["event_at_0"], 0, df["reasoning_tokens"])).astype(float)
    df["compliant_final"] = np.where(df["event_at_0"], 0.0, pd.to_numeric(df["compliant"], errors="coerce"))
    df["aborted"] = df["think_status"] == "aborted"
    df["correct"] = df["correct"].astype(float)
    df.loc[df["aborted"], "correct"] = np.nan
    return df


def load(empty_is_violation: bool, models: list[str], skip_ungraded: bool) -> pd.DataFrame:
    """exp05's rows (dose cells, base rows of the new models, effort rows) and exp04's reference rows of the same
    models and rules. skip_ungraded: leave out generation files with no grades yet (partial-data tests)."""
    requests = {r["request_id"]: r for p in sorted(EXP.cache.glob("requests*.jsonl")) for r in cc_exp04.load_requests(p)}
    frames = []
    for p in sorted(EXP.generations.glob("*.jsonl")):
        if p.name.split("__")[0] not in models + list(EFFORTS):
            continue
        if skip_ungraded and not a4.grades_path(p).exists():
            print(f"skipping {p.name}: not graded yet")
            continue
        frames.append(a4.read_run(p, "exp05"))
    new = pd.concat(frames, ignore_index=True)
    new["condition"] = new["request_id"].map(lambda i: requests[i]["condition"])
    # exp05's no-opening rows are the new models' base rows (d0); effort rows are told apart by their model name.
    new["dose"] = new["request_id"].map(lambda i: requests[i].get("dose") or
                                        ("d0" if requests[i]["condition"] == "none" else None))
    new["opening_tokens_original"] = new["request_id"].map(lambda i: requests[i].get("opening_tokens_original"))
    new["prefill_matches_request"] = [p == requests[i]["prefill"] for i, p in zip(new["request_id"], new["prefill"])]
    new = score(new, empty_is_violation)
    new["effort"] = new["model"].map(EFFORTS)

    old = a4.analysis_rows(a4.load(empty_is_violation))
    old = old[old["model"].isin(models + list(EFFORTS)) & old["mode"].isin(OPENERS)].copy()
    old["dose"] = np.where(old["condition"] == "none", "d0", "generic")
    old["effort"] = old["model"].map(EFFORTS)
    keep = (((old["condition"] == "none") & (old["prompt"].isin(["baseline", "stacked"])))
            | ((old["condition"] == "prefill_compliant") & (old["prompt"] == "baseline"))
            | (old["condition"] == "prefill_no_rule"))
    return pd.concat([new, old.loc[keep]], ignore_index=True)


def dose_rows(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """The dose cells (exp05) plus the reference points: d0 = no opening with the baseline prompt, generic arms."""
    subject = df["model"].isin(models) & df["mode"].isin(OPENERS)
    ref_none = (df["dose"] == "d0") & (df["prompt"] == "baseline")
    ref_generic = df["dose"] == "generic"
    return df[subject & (df["dose"].isin(DOSES) | ref_none | ref_generic)]


def cell(df: pd.DataFrame, model: str, condition: str, dose: str) -> pd.DataFrame:
    return df[(df["model"] == model) & (df["condition"] == condition) & (df["dose"] == dose)]


# --- Tables ----------------------------------------------------------------------------------------------------------
def dose_table(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap) -> tuple[list[dict], dict]:
    rows, draws = [], {}
    for model in models:
        for dose in X_ORDER:
            for condition in (["none"] if dose == "d0" else ["prefill_compliant", "prefill_no_rule"]):
                d = cell(df, model, condition, dose)
                if d.empty:
                    continue
                est = a4.macro_km(d, OPENERS, T_STAR, boot)
                short = a4.macro_km(d, OPENERS, T_SHORT, boot)
                lenient = a4.macro_km(d.assign(time=d["time_lenient"], event=d["event_lenient"]), a4.CASE_RULES,
                                      T_STAR, boot)
                draws[(model, condition, dose)] = est
                full = d[d["full_trace_cell"].astype(bool)]
                rows.append({"model": model, "dose": dose, "condition": condition, "n": len(d),
                             "opening_tokens_median": float(d["opening_tokens_original"].median())
                             if d["opening_tokens_original"].notna().any() else None,
                             "S_t_star": ca.stat(*est) if est else None, "S_short": ca.stat(*short) if short else None,
                             "P1": ca.stat(*ca.pooled(d, "compliant_final", boot)),
                             "S_t_star_case_lenient": ca.stat(*lenient) if lenient else None,
                             "accuracy_full_traces": ca.stat(*ca.pooled(full, "correct", boot)) if len(full) else None,
                             "continuation_tokens_full_median": float(full["reasoning_tokens"].median())
                             if len(full) else None,
                             "empty_trace_share": float(100 * d["empty_trace"].mean()),
                             "aborted_share": float(100 * d["aborted"].mean())})
    return rows, draws


def primary_contrasts(draws: dict, models: list[str]) -> list[dict]:
    """R_d1, R_d3, R_trend per model; Holm over all of them. p_holm_manifest: manifest.json's Holm over the 9
    contrasts of cfg.EXP05_MODELS (only when all three are analyzed)."""
    out = []
    for model in models:
        diffs = {}
        for name, dose in PRIMARY:
            (pa, da), (pb, db) = draws[(model, "prefill_compliant", dose)], draws[(model, "prefill_no_rule", dose)]
            diffs[name] = (pa - pb, da - db)
        diffs["R_trend"] = (diffs["R_d3"][0] - diffs["R_d1"][0], diffs["R_d3"][1] - diffs["R_d1"][1])
        for name, (p, d) in diffs.items():
            out.append({"model": model, "contrast": name, "difference": ca.stat(p, d), "p": bootstrap_p(d)})
    adjusted = holm({(c["model"], c["contrast"]): c["p"] for c in out})
    manifest_family = [c for c in out if c["model"] in cfg.EXP05_MODELS]
    adjusted_manifest = (holm({(c["model"], c["contrast"]): c["p"] for c in manifest_family})
                         if set(cfg.EXP05_MODELS) <= set(models) else {})
    for c in out:
        c["p_holm"] = adjusted[(c["model"], c["contrast"])]
        c["holm_n"] = len(out)
        c["p_holm_manifest"] = adjusted_manifest.get((c["model"], c["contrast"]))
    return out


def dose_effect(draws: dict, models: list[str]) -> list[dict]:
    """Secondary: rule-arm S(t*) at d3 minus d1 (does more compliant own text help?), and the same for no-rule."""
    out = []
    for model in models:
        for condition in ("prefill_compliant", "prefill_no_rule"):
            (p3, d3), (p1, d1) = draws[(model, condition, "d3")], draws[(model, condition, "d1")]
            out.append({"model": model, "condition": condition, "d3_minus_d1": ca.stat(p3 - p1, d3 - d1),
                        "p": bootstrap_p(d3 - d1)})
    return out


def per_rule_table(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap) -> pd.DataFrame:
    out = []
    for model in models:
        for dose in X_ORDER:
            for condition in (["none"] if dose == "d0" else ["prefill_compliant", "prefill_no_rule"]):
                d = cell(df, model, condition, dose)
                for mode in OPENERS:
                    c = d[d["mode"] == mode]
                    if c.empty:
                        continue
                    point, dr = a4.km_at(c, T_STAR, boot)
                    lo, hi = ca.ci(dr)
                    out.append({"model": model, "dose": dose, "condition": condition, "mode": mode, "n": len(c),
                                "S_t_star": 100 * point, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                                "S_short": 100 * a4.km_at(c, T_SHORT, boot)[0],
                                "first_violation_lt5": 100 * (c["event"] & (c["time"] < 5)).mean()})
    return pd.DataFrame(out)


def effort_table(df: pd.DataFrame, boot: ca.Bootstrap) -> tuple[list[dict], list[dict]]:
    rows, draws = [], {}
    for condition, prompt in cfg.EXP05_EFFORT_ARMS:
        for model, effort in EFFORTS.items():
            d = df[(df["model"] == model) & (df["condition"] == condition) & (df["prompt"] == prompt)
                   & df["mode"].isin(OPENERS) & ~df["dose"].isin(DOSES)]
            if d.empty:
                continue
            est = a4.macro_km(d, OPENERS, T_STAR, boot)
            draws[(condition, prompt, effort)] = est
            rows.append({"condition": condition, "prompt": prompt, "effort": effort, "n": len(d),
                         "S_t_star": ca.stat(*est), "S_short": ca.stat(*a4.macro_km(d, OPENERS, T_SHORT, boot)),
                         "P1": ca.stat(*ca.pooled(d, "compliant_final", boot)),
                         "median_reasoning_tokens_full": float(d.loc[d["full_trace_cell"].astype(bool),
                                                                     "reasoning_tokens"].median()),
                         "empty_trace_share": float(100 * d["empty_trace"].mean())})
    contrasts = []
    for condition, prompt in cfg.EXP05_EFFORT_ARMS:
        for effort in ("xhigh", "low"):
            a, b = draws.get((condition, prompt, effort)), draws.get((condition, prompt, "medium"))
            if a and b:
                contrasts.append({"arm": f"{condition} / {prompt}", "contrast": f"{effort} - medium",
                                  "difference": ca.stat(a[0] - b[0], a[1] - b[1]), "p": bootstrap_p(a[1] - b[1])})
    adjusted = holm({(c["arm"], c["contrast"]): c["p"] for c in contrasts})
    for c in contrasts:
        c["p_holm"] = adjusted[(c["arm"], c["contrast"])]
    return rows, contrasts


def checks(df: pd.DataFrame) -> dict:
    new = df[df["source_run"] == "exp05"]
    aborted = new[new["aborted"]]
    return {"rows_per_model_and_file": {f"{m} | {f}": int(n) for (m, f), n in
                                        new.groupby(["model", "generation_file"]).size().items()},
            "aborted_only_where_allowed": bool(aborted["abort_on_violation"].astype(bool).all()),
            "aborted_all_non_compliant": bool((aborted["compliant"] == 0).all()),
            "prefill_stored_as_requested": bool(new["prefill_matches_request"].all()),
            "empty_trace_rows": int(new["empty_trace"].sum()),
            "truncated_rows": int((new["think_status"] == "truncated").sum()),
            "no_think_close_rows": int((new["think_status"] == "no_think_close").sum())}


# --- Figures ---------------------------------------------------------------------------------------------------------
def fig_dose(rows: list[dict], models: list[str], fig_dir, metric: str, name: str, what: str) -> str:
    """G1 / G1b: S(t*) by opening, rule vs no rule, one panel per model (a grid of GRID_COLS columns)."""
    n_cols = min(GRID_COLS, len(models))
    n_rows = -(-len(models) // n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_yaxes=True, vertical_spacing=0.28 / n_rows,
                        horizontal_spacing=0.04, subplot_titles=models)
    for k, model in enumerate(models):
        row, col = k // n_cols + 1, k % n_cols + 1
        for condition, style in ARM_STYLE.items():
            pts = [r for x in X_ORDER for r in rows
                   if (r["model"], r["dose"], r["condition"]) == (model, x, condition) and r[metric]]
            if not pts:
                continue
            fig.add_trace(go.Scatter(
                x=[X_LABEL[r["dose"]] for r in pts], y=[r[metric]["value"] for r in pts],
                mode="markers" if condition == "none" else "lines+markers", name=style["label"],
                legendgroup=condition, showlegend=k == 0,
                marker={"color": style["color"], "size": 9}, line={"color": style["color"]},
                error_y={"type": "data", "symmetric": False,
                         "array": [r[metric]["ci"][1] - r[metric]["value"] for r in pts],
                         "arrayminus": [r[metric]["value"] - r[metric]["ci"][0] for r in pts]},
                hovertemplate="%{x}: %{y:.1f}%<extra>" + style["label"] + "</extra>"), row=row, col=col)
        fig.update_xaxes(categoryorder="array", categoryarray=[X_LABEL[x] for x in X_ORDER], tickangle=0,
                         tickfont={"size": 9}, row=row, col=col)
    fig.update_yaxes(range=[0, 100])
    fig.update_yaxes(title_text=f"% no violation in<br>the first {T_STAR} tokens", col=1)
    fig.update_layout(title=f"{name.split('_')[0]}. {what} (KM S({T_STAR}) of the continuation, 95% CIs)<br><sup>Own "
                            "opening = the model's own no-rule reasoning for the item, rewritten to obey the rule. The "
                            "gap between the green and blue lines is what the rule adds beyond copying the opening."
                            "</sup>",
                      legend={"orientation": "h", "y": -0.12 / n_rows, "x": 0.5, "xanchor": "center", "yanchor": "top"},
                      margin={"t": 120, "b": 90})
    return a4.save(fig, fig_dir, name, 400 * n_cols, 150 + 380 * n_rows)


def fig_rule_gap(rows: list[dict], draws: dict, models: list[str], fig_dir) -> str:
    """G2: what the rule adds beyond copying (rule arm - no-rule arm) at each opening, per model."""
    fig = go.Figure()
    for model in models:
        color = MODEL_COLORS[model]
        xs, ys, lo, hi = [], [], [], []
        for dose in ["generic"] + DOSES:
            a, b = draws.get((model, "prefill_compliant", dose)), draws.get((model, "prefill_no_rule", dose))
            if not (a and b):
                continue
            s = ca.stat(a[0] - b[0], a[1] - b[1])
            xs.append(X_LABEL[dose]); ys.append(s["value"])
            lo.append(s["value"] - s["ci"][0]); hi.append(s["ci"][1] - s["value"])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=model, marker={"color": color, "size": 9},
                                 line={"color": color},
                                 error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo}))
    fig.add_hline(y=0, line={"color": "black", "width": 1})
    fig.update_xaxes(categoryorder="array", categoryarray=[X_LABEL[x] for x in ["generic"] + DOSES])
    fig.update_yaxes(title_text=f"rule arm − no-rule arm, S({T_STAR}) points")
    fig.update_layout(title="G2. What the rule adds beyond copying the opening (95% CIs)<br><sup>Above 0: the "
                            "continuation keeps the rule more when the prompt states it than when it only sees "
                            "compliant text</sup>", legend={"orientation": "h", "y": -0.2}, margin={"t": 100})
    return a4.save(fig, fig_dir, "G2_rule_minus_copying", 1000, 520)


def fig_per_rule(per_rule: pd.DataFrame, models: list[str], fig_dir) -> str:
    """G3: per rule, rule arm (solid) and no-rule arm (dashed) across openings, one color per model."""
    fig = make_subplots(rows=1, cols=len(OPENERS), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[a4.MODE_LABEL[m] for m in OPENERS])
    colors = MODEL_COLORS
    for col, mode in enumerate(OPENERS, start=1):
        for model in models:
            for condition, dash in (("prefill_compliant", "solid"), ("prefill_no_rule", "dash")):
                d = per_rule[(per_rule["model"] == model) & (per_rule["mode"] == mode)
                             & (per_rule["condition"] == condition) & per_rule["dose"].isin(DOSES)]
                fig.add_trace(go.Scatter(x=[X_LABEL[x] for x in d["dose"]], y=d["S_t_star"], mode="lines+markers",
                                         name=f"{model}, {ARM_STYLE[condition]['label']}",
                                         legendgroup=f"{model}{condition}", showlegend=col == 1,
                                         line={"color": colors[model], "dash": dash},
                                         marker={"color": colors[model]}), row=1, col=col)
        fig.update_xaxes(categoryorder="array", categoryarray=[X_LABEL[x] for x in DOSES], tickfont={"size": 9},
                         col=col)
    fig.update_yaxes(range=[0, 100], title_text=f"S({T_STAR}), %", col=1)
    fig.update_layout(title=f"G3. Per rule: S({T_STAR}) by own-opening length (solid: rule in prompt; dashed: no rule)",
                      legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center"},
                      margin={"t": 90, "b": 150 + 25 * len(models)})
    return a4.save(fig, fig_dir, "G3_per_rule", 1500, 560 + 25 * len(models))


def fig_effort(rows: list[dict], fig_dir) -> str:
    """G4: Qwen3.8 at xhigh / medium / low, per arm."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=[f"S({T_STAR}): no violation in the first {T_STAR} tokens",
                                        f"S({T_SHORT})"])
    colors = {"xhigh": "#D55E00", "medium": "#999999", "low": "#56B4E9"}
    arms = [f"{c} / {p}" for c, p in cfg.EXP05_EFFORT_ARMS]
    for effort, color in colors.items():
        for col, metric in ((1, "S_t_star"), (2, "S_short")):
            pts = [next((r for r in rows if f"{r['condition']} / {r['prompt']}" == arm and r["effort"] == effort), None)
                   for arm in arms]
            fig.add_trace(go.Bar(x=["no opening, stacked prompt", "compliant generic sentence, baseline prompt"],
                                 y=[p[metric]["value"] if p else None for p in pts], name=f"effort {effort}",
                                 legendgroup=effort, showlegend=col == 1, marker={"color": color},
                                 error_y={"type": "data", "symmetric": False,
                                          "array": [p[metric]["ci"][1] - p[metric]["value"] if p else None for p in pts],
                                          "arrayminus": [p[metric]["value"] - p[metric]["ci"][0] if p else None
                                                         for p in pts]}), row=1, col=col)
    fig.update_yaxes(range=[0, 100], title_text="% of traces (4 opener rules)", col=1)
    fig.update_layout(barmode="group", title="G4. Qwen3.8-27B-FP8: does reasoning effort change rule-keeping? (95% CIs)",
                      legend={"orientation": "h", "y": -0.18}, margin={"t": 90})
    return a4.save(fig, fig_dir, "G4_effort", 1200, 520)


# --- Secondary: does answer correctness go with rule-keeping? ---------------------------------------------------------
# Only full-trace cells (exp03's ~25%, the same (item, rule) cells in every arm) run to the end whatever the rule
# violations, so only there is correctness observed without selecting on compliance. Wrong answers come with longer
# traces (more room to violate), so the length-matched comparison is the one to read: among rollouts that reached t
# tokens, the share with no violation in the first t.
# arm label -> (plot color, row selector)
CORRECTNESS_ARMS = {
    "no opening": (ARM_STYLE["none"]["color"], lambda d: (d["condition"] == "none") & (d["dose"] == "d0")),
    "rule in prompt": (ARM_STYLE["prefill_compliant"]["color"],
                       lambda d: (d["condition"] == "prefill_compliant") & d["dose"].isin(DOSES)),
    "no rule (imitation only)": (ARM_STYLE["prefill_no_rule"]["color"],
                                 lambda d: (d["condition"] == "prefill_no_rule") & d["dose"].isin(DOSES))}
CORRECTNESS_MIN_N = 10  # rollouts per correct / wrong group, else no estimate


def correctness_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full-trace rollouts with a graded answer, the 4 opener rules, with obeyed_throughout and clean_first_<t>."""
    d = df[df["full_trace_cell"].astype(bool) & df["mode"].isin(OPENERS) & df["correct"].notna()].copy()
    d["correct"] = d["correct"].astype(bool)
    d["obeyed_throughout"] = d["compliant_final"]
    for t in (T_SHORT, T_STAR):
        reached = d["reasoning_tokens"] >= t
        d[f"clean_first_{t}"] = np.where(reached, (~(d["event"] & (d["time"] < t))).astype(float), np.nan)
    return d


def correctness_table(df: pd.DataFrame, models: list[str], boot: ca.Bootstrap,
                      arms: dict = CORRECTNESS_ARMS) -> list[dict]:
    """Per model (and all models pooled) and arm: right - wrong on each measure, question bootstrap."""
    d = correctness_frame(df[df["model"].isin(models)])
    out = []
    for model in models + ["all models"]:
        m = d if model == "all models" else d[d["model"] == model]
        for arm, (_, selects) in arms.items():
            a = m[selects(m)]
            right, wrong = a[a["correct"]], a[~a["correct"]]
            row = {"model": model, "arm": arm, "n_right": len(right), "n_wrong": len(wrong),
                   "median_tokens_right": float(right["reasoning_tokens"].median()) if len(right) else None,
                   "median_tokens_wrong": float(wrong["reasoning_tokens"].median()) if len(wrong) else None}
            for measure in ["obeyed_throughout", f"clean_first_{T_SHORT}", f"clean_first_{T_STAR}"]:
                r, w = right[right[measure].notna()], wrong[wrong[measure].notna()]
                row[f"n_{measure}"] = [len(r), len(w)]
                if min(len(r), len(w)) < CORRECTNESS_MIN_N:
                    row[measure] = row[f"{measure}_right_minus_wrong"] = None
                    continue
                (pr, dr), (pw, dw) = ca.pooled(r, measure, boot), ca.pooled(w, measure, boot)
                row[measure] = {"right": ca.stat(pr, dr), "wrong": ca.stat(pw, dw)}
                row[f"{measure}_right_minus_wrong"] = ca.stat(pr - pw, dr - dw)
            out.append(row)
    return out


def fig_correctness(rows: list[dict], models: list[str], fig_dir, arms: dict = CORRECTNESS_ARMS,
                    name: str = "G5_correctness_check",
                    note: str = "Opening doses pooled; compare the right panel's gaps with the arm gaps in G1.") -> str:
    """G5: right - wrong, raw (whole trace) vs length-matched (first t* tokens), per model and arm."""
    panels = [("obeyed_throughout", "raw: obeyed the rule for the whole trace"),
              (f"clean_first_{T_STAR}", f"length-matched: no violation in the first {T_STAR} tokens<br>"
                                        f"(rollouts that reached {T_STAR} tokens)")]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[title for _, title in panels])
    x_order = models + ["all models"]
    for col, (measure, _) in enumerate(panels, start=1):
        for arm, (color, _) in arms.items():
            pts = {r["model"]: r[f"{measure}_right_minus_wrong"] for r in rows if r["arm"] == arm}
            xs = [x for x in x_order if pts.get(x)]
            fig.add_trace(go.Scatter(x=xs, y=[pts[x]["value"] for x in xs], mode="markers", name=arm,
                                     legendgroup=arm, showlegend=col == 1,
                                     marker={"color": color, "size": 9},
                                     error_y={"type": "data", "symmetric": False,
                                              "array": [pts[x]["ci"][1] - pts[x]["value"] for x in xs],
                                              "arrayminus": [pts[x]["value"] - pts[x]["ci"][0] for x in xs]}),
                          row=1, col=col)
        fig.add_hline(y=0, line={"color": "black", "width": 1}, row=1, col=col)
        fig.update_xaxes(categoryorder="array", categoryarray=x_order, tickangle=-30, row=1, col=col)
    fig.update_yaxes(title_text="right − wrong answers, % points", col=1)
    fig.update_layout(scattermode="group",
                      title=f"{name.split('_')[0]}. Do rollouts that answer correctly keep the rule more? "
                            "(full-trace cells only; 95% CIs)<br><sup>Wrong answers come with longer traces, so the "
                            f"raw gap mixes correctness with length. {note} 0 with no CI: nobody in either group kept "
                            "the rule (floor)</sup>",
                      legend={"orientation": "h", "y": -0.35}, margin={"t": 140, "b": 150})
    return a4.save(fig, fig_dir, name, 1300, 560)


# --- Report ----------------------------------------------------------------------------------------------------------
def report(s: dict, per_rule: pd.DataFrame, figures: list[str], fig_rel: str) -> str:
    f = ca.fmt
    lines = ["# exp05_dose: automated report (UNVERIFIED)", "",
             f"Run {s['run']}. Models: {', '.join(s['models'])}"
             + (f" (skipped, no graded rows: {', '.join(s['skipped_models'])})" if s["skipped_models"] else "") + ". "
             f"Scoring: empty thinking traces {'censored' if s['censor_empty'] else 'are violations at token 0 (manifest)'}. "
             "Every number is grader-scored (no LLM judge).", "",
             "## Primary: what the rule adds beyond copying (rule arm - no-rule arm, S(%d), 4 opener rules)" % T_STAR, "",
             f"Holm over all {len(s['primary_contrasts'])} contrasts (deviations_one_analysis.json).", "",
             "| model | contrast | difference | p | p (Holm) |", "|---|---|---|---|---|"]
    lines += [f"| {c['model']} | {c['contrast']} | {f(c['difference'])} | {c['p']:.3f} | {c['p_holm']:.3f} |"
              for c in s["primary_contrasts"]]
    lines += ["", "## Secondary: d3 - d1 within each arm (no multiplicity correction)", "",
              "| model | condition | d3 - d1 | p |", "|---|---|---|---|"]
    lines += [f"| {c['model']} | {c['condition']} | {f(c['d3_minus_d1'])} | {c['p']:.3f} |" for c in s["dose_effect"]]
    lines += ["", "## Per cell (d0 and generic are exp04 reference rows; the new models' d0 is exp05's base rows)", "",
              f"| model | opening | condition | n | opening tokens (median) | S({T_STAR}) | S({T_SHORT}) | P1 | "
              f"case rules S({T_STAR}), lenient | accuracy (full) | continuation tokens (full, median) | empty % | "
              "aborted % |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["cells"]:
        lines.append(f"| {r['model']} | {r['dose']} | {r['condition']} | {r['n']} | {r['opening_tokens_median']} | "
                     f"{f(r['S_t_star'])} | {f(r['S_short'])} | {f(r['P1'])} | {f(r['S_t_star_case_lenient'])} | "
                     f"{f(r['accuracy_full_traces'])} | {r['continuation_tokens_full_median']} | "
                     f"{r['empty_trace_share']:.1f} | {r['aborted_share']:.1f} |")
    lines += ["", "## Effort check (Qwen3.8; exp04's requests; Holm over the 4 contrasts)", "",
              "| arm | effort | n | S(%d) | S(%d) | P1 | median reasoning tokens (full) | empty %% |" % (T_STAR, T_SHORT),
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['condition']} / {r['prompt']} | {r['effort']} | {r['n']} | {f(r['S_t_star'])} | "
              f"{f(r['S_short'])} | {f(r['P1'])} | {r['median_reasoning_tokens_full']} | {r['empty_trace_share']:.1f} |"
              for r in s["effort"]]
    lines += ["", "| arm | contrast | difference | p | p (Holm) |", "|---|---|---|---|---|"]
    lines += [f"| {c['arm']} | {c['contrast']} | {f(c['difference'])} | {c['p']:.3f} | {c['p_holm']:.3f} |"
              for c in s["effort_contrasts"]]
    lines += ["", "## Secondary: answer correctness vs rule-keeping (full-trace cells; opening doses pooled)", "",
              "Only full-trace cells observe correctness without selecting on compliance. Wrong answers are longer, "
              f"so read the length-matched column (rollouts that reached {T_STAR} tokens). Right − wrong in % points; "
              f"n/a: fewer than {CORRECTNESS_MIN_N} rollouts in a group; 0.0 [0.0, 0.0]: nobody in either group kept "
              "the rule (a floor, not evidence of no link).", "",
              "| model | arm | n right / wrong | median tokens right / wrong | obeyed throughout: right − wrong | "
              f"no violation in first {T_SHORT}: right − wrong | no violation in first {T_STAR}: right − wrong "
              f"(n right / wrong) |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['arm']} | {r['n_right']} / {r['n_wrong']} | {r['median_tokens_right']} / "
              f"{r['median_tokens_wrong']} | {f(r['obeyed_throughout_right_minus_wrong'])} | "
              f"{f(r[f'clean_first_{T_SHORT}_right_minus_wrong'])} | {f(r[f'clean_first_{T_STAR}_right_minus_wrong'])} "
              f"({' / '.join(map(str, r[f'n_clean_first_{T_STAR}']))}) |" for r in s["correctness"]]
    lines += ["", f"## Per rule S({T_STAR})", "", ca.markdown_table(per_rule.round(1)), "", "## Figures", ""]
    lines += [f"- [{name}]({fig_rel}/{name}.html) ![{name}]({fig_rel}/{name}.png)" for name in figures]
    lines += ["", "## Checks", "", "```", json.dumps(s["checks"], indent=1), "```", "",
              "Status: UNVERIFIED until a human adds it to VERIFIED.md"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="analysis run name")
    parser.add_argument("--censor-empty-traces", action="store_true",
                        help="sensitivity: censor empty thinking traces at token 0 instead of scoring a violation")
    parser.add_argument("--skip-models", nargs="*", default=[], help="leave these models out")
    parser.add_argument("--skip-missing", action="store_true",
                        help="leave out ungraded generation files and models without both dose arms (reported)")
    parser.add_argument("--scratch", action="store_true", help="write under /tmp/exp05_scratch (tests)")
    args = parser.parse_args()
    if args.scratch:
        out_dir = fig_dir = Path("/tmp/exp05_scratch") / args.run
    else:
        out_dir, fig_dir = EXP.results / "analysis" / args.run, EXP.figure_dir(args.run)
    for d in {out_dir, fig_dir}:
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")

    wanted = [m for m in cfg.EXP05_ALL_MODELS if m not in args.skip_models]
    everything = a4.add_lenient_case_times(load(not args.censor_empty_traces, wanted, args.skip_missing))
    df = dose_rows(everything, wanted)
    have = {m for m in wanted if all(len(cell(df, m, c, dose)) for c in ("prefill_compliant", "prefill_no_rule")
                                     for dose in DOSES)}
    if not args.skip_missing and have != set(wanted):
        raise SystemExit(f"no complete dose rows for {sorted(set(wanted) - have)} (use --skip-missing to leave out); "
                         "nothing written")
    for d in {out_dir, fig_dir}:
        d.mkdir(parents=True)
    models = [m for m in wanted if m in have]
    skipped = [m for m in wanted if m not in have]
    items = [json.loads(line) for line in cfg.EXP03_ITEMS_PATH.open()]
    boot = ca.Bootstrap(items, cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    cells, draws = dose_table(df, models, boot)
    effort_rows, effort_contrasts = effort_table(everything, boot)
    per_rule = per_rule_table(df, models, boot)
    summary = {"exp_id": EXP.exp_id, "run": args.run, "models": models, "skipped_models": skipped,
               "censor_empty": args.censor_empty_traces, "t_star": T_STAR,
               "primary_contrasts": primary_contrasts(draws, models), "dose_effect": dose_effect(draws, models),
               "cells": cells, "correctness": correctness_table(df, models, boot), "effort": effort_rows,
               "effort_contrasts": effort_contrasts, "checks": checks(everything)}
    figures = [fig_dose(cells, models, fig_dir, "S_t_star", "G1_dose_response",
                        "Does a longer compliant start help, and is it the rule or copying? 4 opener rules, strict"),
               fig_dose(cells, models, fig_dir, "S_t_star_case_lenient", "G1b_dose_response_case_lenient",
                        "The 2 case rules with notation let through (lenient grader)"),
               fig_rule_gap(cells, draws, models, fig_dir), fig_per_rule(per_rule, models, fig_dir),
               fig_correctness(summary["correctness"], models, fig_dir)]
    if effort_rows:
        figures.append(fig_effort(effort_rows, fig_dir))
    per_rule.to_csv(out_dir / "per_cell.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    text = report(summary, per_rule, figures, os.path.relpath(fig_dir, out_dir))
    (out_dir / "REPORT_auto.md").write_text(text)
    print(f"wrote {out_dir}")
    print(text)


if __name__ == "__main__":
    main()
