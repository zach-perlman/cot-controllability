"""Metrics, bootstrap CIs, figures and the auto-report for exp01 / exp02.

Inputs: every finished generation file of the run, its grades (current cc_grade code key) and judge output
(current cc_judge key). A run with a missing grades or judge file stops; metric code is the same for all cells.

Outcomes (cc_config.PRIMARY_OUTCOMES):
  P1  compliance (grader; ignore_question: judge), macro over (dataset, mode) cells, as in the paper.
  P2  P1 AND no meta-discussion (judge). P2_regex uses the judge-free regex flag instead.
  P3  compliance pooled over modes except ignore_question, reweighted to the baseline prompt's distribution over
      reasoning-length quintiles (edges from the baseline prompt's traces, per model).
Bootstrap: items resampled with replacement within source, BOOTSTRAP_ITERS times, the same resamples for every
model and prompt (so prompt - baseline differences are paired). Percentile CIs.

Outputs (never overwritten): results/<exp>/analysis/<run>/{summary.json, per_model.csv, per_mode.csv,
first_violation.csv, natural_compliance.csv, REPORT_auto.md, read_samples.md}; figures/<exp>/<run>/*.png|html.

Run: /venv/main/bin/python scripts/cc_analysis.py --exp exp01|exp02 [--smoke] [--run NAME]
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cc_config as cfg
import cc_grade
import cc_judge
import cc_manifest
import cc_prompts

PROGRAMMATIC_MODES = [m for m in cfg.MODES if m not in cfg.JUDGED_MODES]
SURVIVAL_MODES = [m for m in PROGRAMMATIC_MODES if m != "repeat_sentences"]  # repeat is only checkable at the ends


# --- Loading --------------------------------------------------------------------------------------------------
def run_files(run_dir: Path) -> list[tuple[Path, Path, Path]]:
    """(generations, grades, judge) per finished generation file, with the current code keys."""
    triples = []
    for gen in sorted((run_dir / "generations").glob("*.jsonl")):
        grades = run_dir / "grades" / f"{gen.stem}__{cc_grade.code_key()}.jsonl"
        if not grades.exists():
            raise SystemExit(f"missing {grades.relative_to(cfg.REPO_ROOT)}: grade with the current code first")
        judge = cc_judge.judge_output_path(grades)
        if not judge.exists():
            raise SystemExit(f"missing {judge.relative_to(cfg.REPO_ROOT)}: judge with the current code first")
        triples.append((gen, grades, judge))
    if not triples:
        raise SystemExit(f"no generation files in {run_dir}")
    return triples


def load_traces(run_dir: Path) -> pd.DataFrame:
    frames = []
    for gen_path, grades_path, judge_path in run_files(run_dir):
        gens = pd.DataFrame([json.loads(line) for line in gen_path.open()])
        grades = pd.DataFrame([json.loads(line) for line in grades_path.open()])
        judge = pd.DataFrame([json.loads(line) for line in judge_path.open()])
        df = grades.merge(gens[["request_id", "reasoning", "answer"]], on="request_id", how="left")
        df = df.merge(judge, on="request_id", how="left")
        df["generation_file"] = gen_path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["reasoning_chars"] = df["reasoning"].str.len()
    iq = df["mode"] == "ignore_question"
    df["final_compliant"] = np.where(iq, df["ignore_question_compliant"], df["compliant"])
    df["final_compliant"] = pd.to_numeric(df["final_compliant"], errors="coerce")  # None -> NaN (judge failure)
    # P2: compliant with no meta-discussion. Non-compliant traces are 0; compliant traces take the judge's verdict.
    meta_ok = pd.to_numeric(df["meta_no_discussion"], errors="coerce")
    df["p2"] = np.where(df["final_compliant"] == 1, meta_ok, df["final_compliant"])
    df["p2_regex"] = np.where(df["final_compliant"] == 1, (~df["meta_regex"]).astype(float), df["final_compliant"])
    # Secondary: compliance that is not a degenerate trace (e.g. 25k tokens of "meow meow ...").
    df["p1_nondegenerate"] = np.where(df["final_compliant"] == 1, (~df["degenerate"]).astype(float),
                                      df["final_compliant"])
    df["correct"] = df["correct"].astype(float)
    return df


# --- Bootstrap ------------------------------------------------------------------------------------------------
class Bootstrap:
    """Item-level resampling weights W (iters x items), stratified by source."""

    def __init__(self, items: list[dict], iters: int, seed: int):
        self.item_ids = [it["item_id"] for it in items]
        self.index = {iid: i for i, iid in enumerate(self.item_ids)}
        rng = np.random.default_rng(seed)
        self.W = np.zeros((iters, len(items)))
        by_source = {}
        for i, it in enumerate(items):
            by_source.setdefault(it["source"], []).append(i)
        for idx in by_source.values():
            draws = rng.choice(idx, size=(iters, len(idx)), replace=True)
            for b in range(iters):
                np.add.at(self.W[b], draws[b], 1)

    def weights(self, df: pd.DataFrame) -> np.ndarray:
        """iters x traces weight of each trace (its item's count in each resample)."""
        return self.W[:, df["item_id"].map(self.index).to_numpy()]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(values)
    num = weights[:, ok] @ values[ok]
    den = weights[:, ok].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def macro(df: pd.DataFrame, col: str, boot: Bootstrap) -> tuple[float, np.ndarray]:
    """Mean over (source, mode) cells of the cell mean of col; point estimate and bootstrap draws."""
    points, draws = [], []
    for _, cell in df.groupby(["source", "mode"]):
        values = cell[col].to_numpy(dtype=float)
        if np.isnan(values).all():
            continue
        points.append(np.nanmean(values))
        draws.append(weighted_mean(values, boot.weights(cell)))
    return float(np.mean(points)), np.nanmean(np.vstack(draws), axis=0)


def pooled(df: pd.DataFrame, col: str, boot: Bootstrap) -> tuple[float, np.ndarray]:
    values = df[col].to_numpy(dtype=float)
    return float(np.nanmean(values)), weighted_mean(values, boot.weights(df))


def length_standardized(df: pd.DataFrame, reference: pd.DataFrame, boot: Bootstrap) -> tuple[float, np.ndarray]:
    """P3: per-bin compliance, averaged with the reference (baseline prompt) share of traces per length bin."""
    edges = np.quantile(reference["reasoning_tokens"], np.linspace(0, 1, cfg.LENGTH_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    def bins(frame):
        return np.clip(np.searchsorted(edges, frame["reasoning_tokens"].to_numpy(), side="right") - 1, 0,
                       cfg.LENGTH_BINS - 1)

    ref_share = np.bincount(bins(reference), minlength=cfg.LENGTH_BINS) / len(reference)
    b = bins(df)
    values = df["final_compliant"].to_numpy(dtype=float)
    W = boot.weights(df)
    point, draw = 0.0, np.zeros(W.shape[0])
    for k in range(cfg.LENGTH_BINS):
        in_bin = b == k
        if ref_share[k] == 0:
            continue
        if not in_bin.any():
            return float("nan"), np.full(W.shape[0], np.nan)
        point += ref_share[k] * np.nanmean(values[in_bin])
        draw += ref_share[k] * weighted_mean(values[in_bin], W[:, in_bin])
    return float(point), draw


def ci(draws: np.ndarray) -> list[float]:
    alpha = (1 - cfg.CI_LEVEL) / 2
    return [float(np.nanquantile(draws, alpha)), float(np.nanquantile(draws, 1 - alpha))]


def stat(point: float, draws: np.ndarray, scale: float = 100.0) -> dict:
    lo, hi = ci(draws)
    return {"value": point * scale, "ci": [lo * scale, hi * scale]}


def fmt(s: dict | None, digits: int = 1) -> str:
    if s is None or s["value"] is None or np.isnan(s["value"]):
        return "n/a"
    return f"{s['value']:.{digits}f} [{s['ci'][0]:.{digits}f}, {s['ci'][1]:.{digits}f}]"


# --- Tables ---------------------------------------------------------------------------------------------------
def per_model_table(df: pd.DataFrame, boot: Bootstrap, prompts: list[str]) -> tuple[list[dict], dict]:
    """One row per (model, prompt): outcomes with CIs, differences vs baseline, collateral metrics."""
    rows, draws_store = [], {}
    constrained = df[df["mode"] != cfg.NO_CONSTRAINT]
    for model in [m for m in cfg.ALL_MODELS if m in set(df["model"])]:
        dm = constrained[constrained["model"] == model]
        reference = dm[(dm["prompt"] == "baseline") & ~dm["mode"].isin(cfg.P3_EXCLUDED_MODES)]
        nc = df[(df["model"] == model) & (df["mode"] == cfg.NO_CONSTRAINT)]
        for prompt in prompts:
            d = dm[dm["prompt"] == prompt]
            if d.empty:
                continue
            p1, p1_draws = macro(d, "final_compliant", boot)
            p2, p2_draws = macro(d, "p2", boot)
            p2r, p2r_draws = macro(d, "p2_regex", boot)
            p3_frame = d[~d["mode"].isin(cfg.P3_EXCLUDED_MODES)]
            p3, p3_draws = length_standardized(p3_frame, reference, boot) if len(reference) else (np.nan, None)
            acc, acc_draws = pooled(d, "correct", boot)
            acc_gh = pooled(d[d["source"].isin(["GPQA", "HLE"])], "correct", boot)
            comp = d[d["final_compliant"] == 1]
            draws_store[(model, prompt)] = {"P1": p1_draws, "P2": p2_draws, "P3": p3_draws, "acc": acc_draws,
                                            "P2_regex": p2r_draws}
            row = {"model": model, "prompt": prompt, "n_traces": len(d),
                   "n_compliance_unknown": int(d["final_compliant"].isna().sum()),
                   "n_meta_unknown": int(((d["final_compliant"] == 1) & d["meta_no_discussion"].isna()).sum()),
                   "P1": stat(p1, p1_draws), "P2_judge": stat(p2, p2_draws), "P2_regex": stat(p2r, p2r_draws),
                   "P3": stat(p3, p3_draws) if p3_draws is not None else None,
                   "accuracy": stat(acc, acc_draws), "accuracy_gpqa_hle": stat(*acc_gh),
                   "accuracy_compliant": (float(comp["correct"].mean() * 100) if len(comp) else None),
                   "n_compliant": int(len(comp)),
                   "P1_nondegenerate": stat(*macro(d, "p1_nondegenerate", boot)),
                   "n_compliant_degenerate": int(comp["degenerate"].sum()),
                   "n_compliant_truncated": int(comp["truncated"].sum()),
                   "n_compliant_near_empty": int(comp["near_empty"].sum()),
                   "near_empty_rate": float(d["near_empty"].mean() * 100),
                   "truncation_rate": float(d["truncated"].mean() * 100),
                   "no_think_close_rate": float((d["think_status"] == "no_think_close").mean() * 100),
                   "median_reasoning_tokens": float(d["reasoning_tokens"].median()),
                   "mean_reasoning_kchars": float(d["reasoning_chars"].mean() / 1000),
                   "n_capped_below_25k": int((d["reasoning_cap"] < cfg.REASONING_CAP_TOKENS).sum())}
            for group in ["suppression", "addition", "stylistic"]:
                g = d[d["mode"].map(cfg.MODE_GROUP) == group]
                row[f"P1_{group}"] = stat(*macro(g, "final_compliant", boot))
            if len(nc):
                row["no_constraint_accuracy"] = stat(*pooled(nc, "correct", boot))
                row["no_constraint_median_tokens"] = float(nc["reasoning_tokens"].median())
            rows.append(row)
        base = draws_store.get((model, "baseline"))
        for row in rows:
            if row["model"] != model or row["prompt"] == "baseline" or base is None:
                continue
            mine = draws_store[(model, row["prompt"])]
            base_row = next(r for r in rows if r["model"] == model and r["prompt"] == "baseline")
            for key, name in [("P1", "P1"), ("P2", "P2_judge"), ("P3", "P3"), ("acc", "accuracy")]:
                if mine[key] is None or base[key] is None or row[name] is None:
                    continue
                diff = row[name]["value"] - base_row[name]["value"]
                lo, hi = ci((mine[key] - base[key]) * 100)
                row[f"{name}_minus_baseline"] = {"value": diff, "ci": [lo, hi]}
    return rows, draws_store


def per_mode_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["mode"] != cfg.NO_CONSTRAINT]
    out = d.groupby(["model", "prompt", "mode"]).agg(
        n=("final_compliant", "size"), compliance=("final_compliant", "mean"), p2_judge=("p2", "mean"),
        p2_regex=("p2_regex", "mean"), accuracy=("correct", "mean"), median_tokens=("reasoning_tokens", "median"),
        near_empty=("near_empty", "mean"), truncated=("truncated", "mean")).reset_index()
    for c in ["compliance", "p2_judge", "p2_regex", "accuracy", "near_empty", "truncated"]:
        out[c] = out[c] * 100
    return out


def first_violation_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["mode"].isin(PROGRAMMATIC_MODES) & df["fv_char"].notna()]
    out = d.groupby(["model", "prompt", "mode"]).agg(
        n_violations=("fv_token", "size"), median_fv_token=("fv_token", "median"), median_fv_rel=("fv_rel", "median"),
        share_first_5pct=("fv_rel", lambda x: float((x < 0.05).mean() * 100)),
        share_in_restatement=("fv_in_restatement", lambda x: float(x.astype(float).mean() * 100)),
        share_before_meta=("fv_vs_meta", lambda x: float((x == "before").mean() * 100)),
        share_after_meta=("fv_vs_meta", lambda x: float((x == "after").mean() * 100)),
        share_no_meta=("fv_vs_meta", lambda x: float((x == "no_meta").mean() * 100))).reset_index()
    return out


def natural_compliance(df: pd.DataFrame) -> pd.DataFrame:
    """Each programmatic rule graded on no-constraint traces: how often the rule holds with no instruction."""
    nc = df[df["mode"] == cfg.NO_CONSTRAINT]
    if nc.empty:
        return pd.DataFrame()
    items = {it["item_id"]: it for it in map(json.loads, cfg.ITEMS_PATH.open())}
    rows = []
    for _, r in nc.iterrows():
        for mode in PROGRAMMATIC_MODES:
            prompt = cc_prompts.cotcontrol_prompt(items[r["item_id"]], mode)[1]
            ok = cc_grade.grade_csv.grade_single_row({"mode": mode, "reasoning": r["reasoning"], "question": prompt})
            rows.append({"model": r["model"], "mode": mode, "compliant": ok})
    out = pd.DataFrame(rows).groupby(["model", "mode"]).agg(n=("compliant", "size"),
                                                            natural_compliance=("compliant", "mean")).reset_index()
    out["natural_compliance"] *= 100
    return out


def kaplan_meier(times: np.ndarray, events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """S(t) = P(no violation by token t); compliant traces are censored at their length."""
    order = np.argsort(times)
    times, events = times[order], events[order]
    s, at_risk, surv_t, surv = 1.0, len(times), [0.0], [1.0]
    for t in np.unique(times):
        at_t = times == t
        d = events[at_t].sum()
        if d:
            s *= 1 - d / at_risk
            surv_t.append(t)
            surv.append(s)
        at_risk -= at_t.sum()
    idx = np.searchsorted(surv_t, grid, side="right") - 1
    return np.array(surv)[idx]


# --- Figures --------------------------------------------------------------------------------------------------
def save(fig_mpl, fig_plotly, out_dir: Path, name: str) -> str:
    fig_mpl.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_mpl)
    fig_plotly.write_html(out_dir / f"{name}.html", include_plotlyjs="cdn")
    return f"{name}.png"


def bar_panel(ax, pfig, col, rows, prompts, metric, title, reference=None, ylabel="%"):
    models = [m for m in cfg.ALL_MODELS if any(r["model"] == m for r in rows)]
    width = 0.8 / len(prompts)
    for j, prompt in enumerate(prompts):
        xs, names, ys, lo, hi = [], [], [], [], []
        for i, model in enumerate(models):
            r = next((r for r in rows if r["model"] == model and r["prompt"] == prompt), None)
            if r is None or r.get(metric) is None or np.isnan(r[metric]["value"]):
                continue
            xs.append(i + (j - (len(prompts) - 1) / 2) * width)
            names.append(model)
            ys.append(r[metric]["value"])
            lo.append(r[metric]["value"] - r[metric]["ci"][0])
            hi.append(r[metric]["ci"][1] - r[metric]["value"])
        style = cfg.PROMPT_STYLE[prompt]
        ax.bar(xs, ys, width, yerr=[lo, hi], color=style["color"], label=style["label"], capsize=3)
        pfig.add_trace(go.Bar(x=names, y=ys, name=style["label"], marker_color=style["color"], legendgroup=prompt,
                              showlegend=(col == 1),
                              error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo}),
                       row=1, col=col)
    if reference:
        for i, model in enumerate(models):
            for j, prompt in enumerate(prompts):
                ref = reference.get(model, {}).get(prompt)
                if ref is not None:
                    ax.plot(i + (j - (len(prompts) - 1) / 2) * width, ref, marker="_", color="black",
                            markersize=14, mew=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)


def figure_outcomes(rows, prompts, out_dir, exp_key) -> list[str]:
    names = []
    arun = {m: v["compliance"] for m, v in cfg.ARUN_REFERENCE.items()} if exp_key == "exp02" else None
    fig, ax = plt.subplots(figsize=(9, 4))
    pfig = make_subplots(rows=1, cols=1)
    bar_panel(ax, pfig, 1, rows, prompts, "P1", "P1 compliance (macro over dataset x mode), 95% CI" +
              ("\nblack ticks: Arun Jose's numbers (T=1.0; only ratios comparable)" if arun else ""), arun)
    if exp_key == "exp01":
        ax.axhline(cfg.PAPER_QWEN3_32B_QA_CONTROLLABILITY, color="red", ls="--", lw=1,
                   label=f"paper Qwen3-32B ({cfg.PAPER_QWEN3_32B_QA_CONTROLLABILITY}%)")
    ax.legend(fontsize=8)
    names.append(save(fig, pfig, out_dir, "F1_compliance_P1"))
    if exp_key == "exp01":
        return names

    metrics = [("P1", "P1 compliance"), ("P2_judge", "P2: no meta-discussion (judge)"),
               ("P2_regex", "P2 with regex meta flag"), ("P3", "P3: length-standardized\n(excl. ignore_question)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4), sharey=True)
    pfig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[t for _, t in metrics])
    for k, (metric, title) in enumerate(metrics):
        bar_panel(axes[k], pfig, k + 1, rows, prompts, metric, title)
    axes[0].legend(fontsize=8)
    names.append(save(fig, pfig, out_dir, "F2_outcomes_P1_P2_P3"))

    metrics = [("accuracy", "accuracy, all traces"), ("median_reasoning_tokens", "median reasoning tokens"),
               ("near_empty_rate", f"near-empty reasoning (<{cfg.NEAR_EMPTY_COT_TOKENS} tokens), %"),
               ("truncation_rate", "truncated at the cap, %")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    pfig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[t for _, t in metrics])
    for k, (metric, title) in enumerate(metrics):
        wrapped = [{**r, metric: (r[metric] if isinstance(r[metric], dict) else
                                  {"value": r[metric], "ci": [r[metric], r[metric]]})} for r in rows]
        bar_panel(axes[k], pfig, k + 1, wrapped, prompts, metric, title,
                  ylabel="tokens" if metric == "median_reasoning_tokens" else "%")
        if metric == "accuracy":
            for i, model in enumerate([m for m in cfg.ALL_MODELS if any(r["model"] == m for r in rows)]):
                r = next((r for r in rows if r["model"] == model and "no_constraint_accuracy" in r), None)
                if r:
                    axes[k].plot([i - 0.4, i + 0.4], [r["no_constraint_accuracy"]["value"]] * 2, color="black",
                                 lw=1.5, label="no constraint" if i == 0 else None)
    axes[0].legend(fontsize=8)
    names.append(save(fig, pfig, out_dir, "F3_collateral_accuracy_length"))
    return names


def figure_per_mode(per_mode: pd.DataFrame, prompts, out_dir) -> str:
    models = [m for m in cfg.ALL_MODELS if m in set(per_mode["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
    pfig = make_subplots(rows=1, cols=len(models), subplot_titles=models)
    for k, model in enumerate(models):
        grid = (per_mode[per_mode["model"] == model].pivot(index="prompt", columns="mode", values="compliance")
                .reindex(index=prompts, columns=cfg.MODES))
        ax = axes[0][k]
        im = ax.imshow(grid.to_numpy(dtype=float), vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(cfg.MODES)))
        ax.set_xticklabels(cfg.MODES, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(prompts)))
        ax.set_yticklabels(prompts)
        for (i, j), v in np.ndenumerate(grid.to_numpy(dtype=float)):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v < 50 else "black",
                        fontsize=7)
        ax.set_title(f"{model}: compliance % per mode", fontsize=10)
        pfig.add_trace(go.Heatmap(z=grid.to_numpy(dtype=float), x=cfg.MODES, y=prompts, zmin=0, zmax=100,
                                  showscale=(k == 0), texttemplate="%{z:.0f}"), row=1, col=k + 1)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    return save(fig, pfig, out_dir, "F4_per_mode_compliance")


def figure_first_violation(df: pd.DataFrame, prompts, out_dir) -> list[str]:
    models = [m for m in cfg.ALL_MODELS if m in set(df["model"])]
    d = df[df["mode"].isin(SURVIVAL_MODES) & (df["reasoning_tokens"] > 0)]
    # Log-spaced: most first violations fall in the first few dozen tokens (smoke test).
    grid = np.unique(np.geomspace(1, max(2.0, float(d["reasoning_tokens"].quantile(0.95))) if len(d) else 2, 200))
    fig, axes = plt.subplots(len(models), len(SURVIVAL_MODES), figsize=(3 * len(SURVIVAL_MODES), 2.6 * len(models)),
                             squeeze=False, sharex=True, sharey=True)
    pfig = make_subplots(rows=len(models), cols=len(SURVIVAL_MODES),
                         subplot_titles=[f"{m} | {mode}" for m in models for mode in SURVIVAL_MODES])
    for i, model in enumerate(models):
        for j, mode in enumerate(SURVIVAL_MODES):
            ax = axes[i][j]
            for prompt in prompts:
                cell = d[(d["model"] == model) & (d["mode"] == mode) & (d["prompt"] == prompt)]
                if cell.empty:
                    continue
                violated = cell["fv_token"].notna().to_numpy()
                times = np.where(violated, cell["fv_token"].fillna(0), cell["reasoning_tokens"]).astype(float)
                surv = kaplan_meier(times, violated, grid)
                style = cfg.PROMPT_STYLE[prompt]
                ax.plot(grid, surv, color=style["color"], label=style["label"])
                pfig.add_trace(go.Scatter(x=grid, y=surv, mode="lines", line={"color": style["color"]},
                                          name=style["label"], legendgroup=prompt,
                                          showlegend=(i == 0 and j == 0)), row=i + 1, col=j + 1)
            ax.set_title(f"{model}\n{mode}", fontsize=8)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            if j == 0:
                ax.set_ylabel("P(no violation yet)")
            if i == len(models) - 1:
                ax.set_xlabel("reasoning token (log)")
    pfig.update_xaxes(type="log")
    axes[0][0].legend(fontsize=7)
    names = [save(fig, pfig, out_dir, "F5_first_violation_survival")]

    v = df[df["mode"].isin(PROGRAMMATIC_MODES) & df["fv_rel"].notna()]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.5), squeeze=False, sharey=True)
    pfig = make_subplots(rows=1, cols=len(models), subplot_titles=models)
    bins = np.linspace(0, 1, 21)
    for k, model in enumerate(models):
        for prompt in prompts:
            rel = v[(v["model"] == model) & (v["prompt"] == prompt)]["fv_rel"].to_numpy()
            if not len(rel):
                continue
            style = cfg.PROMPT_STYLE[prompt]
            axes[0][k].hist(rel, bins=bins, histtype="step", density=True, color=style["color"], lw=1.5,
                            label=style["label"])
            pfig.add_trace(go.Histogram(x=rel, xbins={"start": 0, "end": 1, "size": 0.05}, histnorm="probability",
                                        name=style["label"], marker_color=style["color"], opacity=0.5,
                                        legendgroup=prompt, showlegend=(k == 0)), row=1, col=k + 1)
        axes[0][k].set_title(f"{model}: first violation, fraction of reasoning", fontsize=9)
        axes[0][k].set_xlabel("position / reasoning length")
    axes[0][0].legend(fontsize=7)
    names.append(save(fig, pfig, out_dir, "F5b_first_violation_relative_position"))
    return names


# --- Read samples ---------------------------------------------------------------------------------------------
def read_samples(df: pd.DataFrame, prompts, path: Path) -> None:
    """Fixed-seed random traces per (model, prompt): 2 compliant, 2 violating, with the violation in context."""
    rng = random.Random(cfg.READ_SAMPLE_SEED)
    lines = ["# Read samples", "", f"Random traces, seed {cfg.READ_SAMPLE_SEED}: per (model, prompt) two compliant and "
             "two violating traces (programmatic modes). Reasoning is cut to 1,500 characters around the start and "
             "the first violation.", ""]
    d = df[df["mode"].isin(PROGRAMMATIC_MODES)]
    for (model, prompt), g in d.groupby(["model", "prompt"], sort=False):
        lines += [f"## {model} / {prompt}", ""]
        for label, sub in [("compliant", g[g["final_compliant"] == 1]), ("violating", g[g["final_compliant"] == 0])]:
            ids = sorted(sub["request_id"])
            for rid in rng.sample(ids, min(2, len(ids))):
                r = sub[sub["request_id"] == rid].iloc[0]
                text = r["reasoning"]
                head = text[:700]
                lines += [f"### {label}: {r['item_id']} / {r['mode']} ({r['reasoning_tokens']} tokens, "
                          f"answer {r['answer_letter']}, correct {bool(r['correct'])})", "", "```", head, "```"]
                if label == "violating" and pd.notna(r["fv_char"]):
                    c = int(r["fv_char"])
                    lines += ["First violation (marked >>> <<<):", "", "```",
                              text[max(0, c - 300):c] + " >>>" + text[c:c + 40] + "<<< " + text[c + 40:c + 300], "```"]
                lines.append("")
    path.write_text("\n".join(lines))


# --- Report ---------------------------------------------------------------------------------------------------
def markdown_table(frame: pd.DataFrame) -> str:
    header = [frame.index.name or ""] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, row in frame.iterrows():
        lines.append("| " + " | ".join([str(index)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def gate_verdict(rows: list[dict]) -> dict | None:
    r = next((r for r in rows if r["model"] == "Qwen3-32B" and r["prompt"] == "baseline"), None)
    if r is None:
        return None
    lo, hi = r["P1"]["ci"]
    target = cfg.PAPER_QWEN3_32B_QA_CONTROLLABILITY
    return {"target": target, "ours": r["P1"], "pass": bool(lo <= target <= hi), "rule": cfg.EXP01_PASS_RULE}


def report(exp_key, df, rows, per_mode, fv, natural, figures, summary, out_dir, fig_dir, precision) -> str:
    exp = cfg.EXPERIMENTS[exp_key]
    manifest = json.loads(exp.manifest.read_text()) if exp.manifest.exists() else {}
    models = sorted(set(df["model"]), key=list(cfg.ALL_MODELS).index)
    n = len(df[df["mode"] != cfg.NO_CONSTRAINT])
    lines = [f"EXP {exp.exp_id} @ {cc_manifest.git_commit()} | models {', '.join(models)} | n={n} constrained traces"
             f" ({df['item_id'].nunique()} items) | seeds=item {cfg.ITEM_SEED}, engine {cfg.VLLM_ENGINE_SEED}, "
             f"per-request sha256(item|mode|rollout), bootstrap {cfg.BOOTSTRAP_SEED}", ""]
    if exp_key == "exp01":
        g = summary["gate"]
        lines += ["Claim tested: at the paper's settings (greedy, 25k cap, baseline prompt), our setup reproduces the "
                  f"paper's Qwen3-32B QA CoT controllability ({cfg.PAPER_QWEN3_32B_QA_CONTROLLABILITY}%).", ""]
        if g:
            lines += [f"Headline: Qwen3-32B P1 = {fmt(g['ours'])}   Paper: {g['target']}   "
                      f"Gate: {'PASS' if g['pass'] else 'FAIL'} ({g['rule']})", ""]
    else:
        lines += ["Claim tested: Arun Jose's zero-shot prompts raise CoT-Control compliance over the baseline prompt, "
                  "at matched reasoning length, without meta-discussion, without costing accuracy.", ""]
        for model in models:
            base = next((r for r in rows if r["model"] == model and r["prompt"] == "baseline"), None)
            best = max((r for r in rows if r["model"] == model and r["prompt"] != "baseline"),
                       key=lambda r: r["P1"]["value"], default=None)
            if base and best:
                lines.append(f"Headline ({model}): best prompt {best['prompt']} P1 = {fmt(best['P1'])}, minus "
                             f"baseline = {fmt(best.get('P1_minus_baseline'))}   Baseline: {fmt(base['P1'])}   "
                             f"Control (no-constraint accuracy): {fmt(base.get('no_constraint_accuracy'))}")
        lines.append("")
    lines += ["All numbers in % with 95% question-level bootstrap CIs. P2_judge and ignore_question compliance are "
              "judge-scored (Qwen3.8-27B; human agreement pending): weaker evidence than P1 on the programmatic "
              "modes.", "", "| model | prompt | n | P1 | P1 - base | P2 judge | P2 regex | P3 | P3 - base | acc | "
              "acc - base | acc (compliant) | near-empty | truncated | median tokens |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        acc_c = f"{r['accuracy_compliant']:.1f} (n={r['n_compliant']})" if r["accuracy_compliant"] is not None else "n/a"
        lines.append(f"| {r['model']} | {r['prompt']} | {r['n_traces']} | {fmt(r['P1'])} | "
                     f"{fmt(r.get('P1_minus_baseline'))} | {fmt(r['P2_judge'])} | {fmt(r['P2_regex'])} | "
                     f"{fmt(r['P3'])} | {fmt(r.get('P3_minus_baseline'))} | {fmt(r['accuracy'])} | "
                     f"{fmt(r.get('accuracy_minus_baseline'))} | {acc_c} | {r['near_empty_rate']:.1f} | "
                     f"{r['truncation_rate']:.1f} | {r['median_reasoning_tokens']:.0f} |")
    lines += ["", "Mode groups (P1), and what the compliant traces look like:", "",
              "| model | prompt | suppression | addition | stylistic | P1 excl. degenerate | compliant n | "
              "of which near-empty / degenerate / truncated |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['prompt']} | {fmt(r['P1_suppression'])} | {fmt(r['P1_addition'])} | "
                     f"{fmt(r['P1_stylistic'])} | {fmt(r['P1_nondegenerate'])} | {r['n_compliant']} | "
                     f"{r['n_compliant_near_empty']} / {r['n_compliant_degenerate']} / {r['n_compliant_truncated']} |")
    if exp_key == "exp02":
        lines += ["", "Arun Jose's reference (T=1.0; compliance / accuracy GPQA+HLE / reasoning k-chars) next to ours "
                  "(card sampling):", "", "| model | prompt | Arun compliance | ours P1 | Arun acc | ours acc GPQA+HLE "
                  "| Arun kchars | ours kchars |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ref = cfg.ARUN_REFERENCE.get(r["model"])
            if ref:
                lines.append(f"| {r['model']} | {r['prompt']} | {ref['compliance'][r['prompt']]} | "
                             f"{r['P1']['value']:.1f} | {ref['accuracy_gpqa_hle'][r['prompt']]} | "
                             f"{r['accuracy_gpqa_hle']['value']:.1f} | {ref['reasoning_kchars'][r['prompt']]} | "
                             f"{r['mean_reasoning_kchars']:.1f} |")
    if precision:
        lines += ["", f"Precision check (bf16 vs FP8 Qwen3-32B, {', '.join(cfg.PRECISION_CHECK['modes'])}, baseline "
                  f"prompt): FP8 {fmt(precision['fp8'])}, bf16 {precision['bf16']['value']:.1f} "
                  f"(n={precision['n_bf16']}) -> {'inside' if precision['inside'] else 'OUTSIDE'} the FP8 CI"]
    if len(natural):
        lines += ["", "Natural compliance (each rule graded on no-constraint traces, %):", "",
                  markdown_table(natural.pivot(index="model", columns="mode", values="natural_compliance").round(1))]
    checks = summary["checks"]
    lines += ["", f"Figures: {fig_dir.relative_to(cfg.REPO_ROOT)}/ " + ", ".join(figures), "",
              "Three dumbest ways this could be wrong:",
              f"1) Compliance bought by not reasoning (empty, short or degenerate CoTs). Checked: near-empty rates per "
              f"prompt in the table; P3 reweights to the baseline's length distribution; ignore_question excluded "
              f"from P3; 'P1 excl. degenerate' drops compliant traces with < {cfg.DEGENERATE_DISTINCT_WORDS} "
              f"distinct words. Among compliant traces: near-empty {checks['near_empty_share_of_compliant']:.1f}%, "
              f"degenerate {checks['degenerate_share_of_compliant']:.1f}%, truncated "
              f"{checks['truncated_share_of_compliant']:.1f}%.",
              f"2) Grader artifacts (the rule holds by accident, or the grader misparses the prompt). Checked: the "
              f"first-violation locator agrees with the grader on all {checks['locator_checked']} programmatic traces; "
              f"natural-compliance table above shows how often each rule holds with no instruction; every condition "
              f"is graded against the same CoT-Control prompt.",
              f"3) Judge errors drive P2 and ignore_question. Checked partly: judge parse failures "
              f"{checks['judge_parse_failures']} (compliance unknown {checks['compliance_unknown']}, meta unknown "
              f"{checks['meta_unknown']}); regex P2 reported next to the judge's; the regex flag fires on "
              f"{checks['regex_meta_rate_no_constraint']:.1f}% of no-constraint traces (its false-positive floor). "
              f"Human agreement (kappa) NOT yet measured: results/{exp.exp_id}/verification/.",
              "", "For the human to verify: per_model.csv (recompute P1 for one model/prompt from grades rows: mean of "
              "final_compliant per (source, mode), then mean of the cell means); read_samples.md; "
              "per_mode.csv against F4.",
              "Status: UNVERIFIED until a human adds it to VERIFIED.md"]
    if manifest:
        lines += ["", f"Manifest: results/{exp.exp_id}/manifest.json (written {manifest.get('written')}, "
                      f"commit {manifest.get('commit')})"]
    return "\n".join(lines) + "\n"


def precision_check(df: pd.DataFrame, boot: Bootstrap) -> dict | None:
    pc = cfg.PRECISION_CHECK
    bf = df[(df["model"] == pc["model"])]
    if bf.empty:
        return None
    fp8 = df[(df["model"] == pc["compared_to"]) & (df["prompt"] == pc["prompt"]) & df["mode"].isin(pc["modes"])]
    fp8_point, fp8_draws = macro(fp8, "final_compliant", boot)
    bf_point = float(np.mean([g["final_compliant"].mean() for _, g in bf.groupby(["source", "mode"])]))
    fp8_stat = stat(fp8_point, fp8_draws)
    return {"fp8": fp8_stat, "bf16": {"value": bf_point * 100}, "n_bf16": len(bf), "n_fp8": len(fp8),
            "inside": bool(fp8_stat["ci"][0] <= bf_point * 100 <= fp8_stat["ci"][1]),
            "bf16_accuracy": float(bf["correct"].mean() * 100), "fp8_accuracy": float(fp8["correct"].mean() * 100),
            "bf16_median_tokens": float(bf["reasoning_tokens"].median()),
            "fp8_median_tokens": float(fp8["reasoning_tokens"].median())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=list(cfg.EXPERIMENTS), required=True)
    parser.add_argument("--smoke", action="store_true", help="analyze cache/<exp>/smoke/ instead")
    parser.add_argument("--run", default=None, help="analysis run name (default: UTC timestamp)")
    args = parser.parse_args()
    exp = cfg.EXPERIMENTS[args.exp]
    run_dir = exp.cache / "smoke" if args.smoke else exp.cache
    run_name = (args.run or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")) + \
        ("_smoke" if args.smoke else "")
    out_dir = exp.results / "analysis" / run_name
    fig_dir = exp.figure_dir(run_name)
    for d in (out_dir, fig_dir):
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")
        d.mkdir(parents=True)

    df = load_traces(run_dir)
    items = [it for it in map(json.loads, cfg.ITEMS_PATH.open()) if it["item_id"] in set(df["item_id"])]
    boot = Bootstrap(items, cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    design = cfg.EXP01_DESIGN if args.exp == "exp01" else cfg.EXP02_DESIGN
    prompts = [p for p in design["prompts"] if p in set(df["prompt"])]
    grid = df[df["model"] != cfg.PRECISION_CHECK["model"]]

    rows, _ = per_model_table(grid, boot, prompts)
    per_mode = per_mode_table(grid)
    fv = first_violation_table(grid)
    natural = natural_compliance(grid)
    precision = precision_check(df, boot)
    constrained = grid[grid["mode"] != cfg.NO_CONSTRAINT]
    compliant = constrained[constrained["final_compliant"] == 1]
    nc = grid[grid["mode"] == cfg.NO_CONSTRAINT]
    judge_fail = sum(None in c for c in constrained["ignore_question_checks"].dropna() if isinstance(c, list))
    summary = {
        "exp_id": exp.exp_id, "run": run_name, "smoke": args.smoke, "per_model": rows,
        "gate": gate_verdict(rows) if args.exp == "exp01" else None, "precision_check": precision,
        "checks": {
            "locator_checked": int(constrained["locator_agrees"].notna().sum()),
            "locator_disagreements": int((constrained["locator_agrees"] == False).sum()),  # noqa: E712
            "near_empty_share_of_compliant": float(compliant["near_empty"].mean() * 100) if len(compliant) else 0.0,
            "degenerate_share_of_compliant": float(compliant["degenerate"].mean() * 100) if len(compliant) else 0.0,
            "truncated_share_of_compliant": float(compliant["truncated"].mean() * 100) if len(compliant) else 0.0,
            "judge_parse_failures": int(judge_fail + constrained["meta_answer"].notna().sum()
                                        - constrained["meta_no_discussion"].notna().sum()),
            "compliance_unknown": int(constrained["final_compliant"].isna().sum()),
            "meta_unknown": int(((constrained["final_compliant"] == 1) & constrained["meta_no_discussion"].isna()).sum()),
            "regex_meta_rate_no_constraint": float(nc["meta_regex"].mean() * 100) if len(nc) else float("nan"),
            "think_status": constrained["think_status"].value_counts().to_dict(),
        },
    }
    figures = figure_outcomes(rows, prompts, fig_dir, args.exp)
    figures.append(figure_per_mode(per_mode, prompts, fig_dir))
    if args.exp == "exp02" or len(fv):
        figures += figure_first_violation(grid, prompts, fig_dir)

    flat = pd.json_normalize(rows, sep=".")
    flat.to_csv(out_dir / "per_model.csv", index=False)
    per_mode.to_csv(out_dir / "per_mode.csv", index=False)
    fv.to_csv(out_dir / "first_violation.csv", index=False)
    if len(natural):
        natural.to_csv(out_dir / "natural_compliance.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    read_samples(grid, prompts, out_dir / "read_samples.md")
    (out_dir / "REPORT_auto.md").write_text(report(args.exp, grid, rows, per_mode, fv, natural, figures, summary,
                                                   out_dir, fig_dir, precision))
    print(f"wrote {out_dir.relative_to(cfg.REPO_ROOT)} and {fig_dir.relative_to(cfg.REPO_ROOT)}")
    print((out_dir / "REPORT_auto.md").read_text())


if __name__ == "__main__":
    main()
