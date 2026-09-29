"""exp03 analysis: survival to SURVIVAL_T_STAR reasoning tokens without a rule violation, per the exp03 manifest.

Per trace of an abortable mode: event = the first violation (grader's locator, token fv_token, 0-based); a trace
without one is censored at its length (reasoning_tokens). Aborted traces always have an event, since generation
stopped only after a confirmed violation (cc_abort).

Primary: Kaplan-Meier S(t*) = P(no violation in the first t* tokens) per (model, prompt, mode), averaged over the
  7 abortable modes; each Arun prompt minus baseline per model, with the question-level bootstrap paired across
  prompts (cc_analysis.Bootstrap, stratified by source); two-sided bootstrap p-values, Holm over the 9 contrasts.
Robustness: reached-t* rate = share of traces whose first t* tokens are observed and violation-free (time >= t*).
Secondary: P1 / P2 (judge) / P2_regex as exp02; accuracy on the requests that never abort (the same cells for
  every prompt); abort, censoring, near-empty and truncation rates; survival curves; S(t*) and P1 pooled with exp02.
Audit: the abort rule replayed on the full-trace cells (check_abort_rule.replay), and P1 in full-trace cells vs
  aborting cells (the cells are a random draw, so the two should agree).

Outputs (never overwritten): results/exp03_abort_survival/analysis/<run>/{summary.json, per_cell.csv, REPORT_auto.md}
and figures/exp03/<run>/S1_survival.{png,html}.

Run: /venv/main/bin/python scripts/cc_survival.py [--run NAME] [--no-pooled]
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cc_analysis as ca
import cc_config as cfg
import check_abort_rule

T_STAR = cfg.SURVIVAL_T_STAR
ARUN_PROMPTS = [p for p in cfg.PROMPTS if p != "baseline"]


# --- Loading ----------------------------------------------------------------------------------------------------
def with_survival_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event"] = df["fv_token"].notna()
    df["time"] = np.where(df["event"], df["fv_token"].fillna(0), df["reasoning_tokens"]).astype(float)
    df["reached_t_star"] = (df["time"] >= T_STAR).astype(float)
    return df


def load_exp03() -> pd.DataFrame:
    df = ca.load_traces(cfg.EXP03.cache)
    flags = pd.DataFrame([{k: r[k] for k in ("request_id", "abort_on_violation", "full_trace_cell")}
                          for r in map(json.loads, cfg.EXP03.requests.open())])
    df = df.merge(flags, on="request_id", how="left", validate="many_to_one")
    df["aborted"] = df["think_status"] == "aborted"
    df.loc[df["aborted"], "correct"] = np.nan  # no answer phase; accuracy is read on full traces only
    return with_survival_columns(df)


def load_exp02() -> pd.DataFrame:
    df = ca.load_traces(cfg.EXP02.cache)
    df = df[df["model"].isin(cfg.GRID_MODELS) & df["prompt"].isin(cfg.PROMPTS + [cfg.NO_CONSTRAINT])]
    return with_survival_columns(df)


# --- Estimators ---------------------------------------------------------------------------------------------------
def kaplan_meier(times: np.ndarray, events: np.ndarray, weights: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    """Weighted KM S(t) = P(no violation in the first t tokens), for each weight row (B x traces) and t.

    A violation at token u counts against the traces still at risk at u: those whose first violation is at u or
    later, and violation-free traces that reasoned past u (length > u)."""
    out = np.ones((weights.shape[0], len(t_values)))
    s = np.ones(weights.shape[0])
    for u in np.unique(times[events]):
        if u >= t_values.max():
            break
        at_risk = (events & (times >= u)) | (~events & (times > u))
        d = weights[:, events & (times == u)].sum(axis=1)
        n = weights[:, at_risk].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = s * np.where(n > 0, 1 - d / n, 1.0)
        out[:, t_values > u] = s[:, None]
    return out


def survival_at_t_star(cell: pd.DataFrame, boot: ca.Bootstrap) -> tuple[float, np.ndarray]:
    times, events = cell["time"].to_numpy(), cell["event"].to_numpy()
    weights = np.vstack([np.ones(len(cell)), boot.weights(cell)])  # row 0: the point estimate
    s = kaplan_meier(times, events, weights, np.array([T_STAR]))[:, 0]
    return float(s[0]), s[1:]


def macro_over_modes(df: pd.DataFrame, estimator) -> tuple[float, np.ndarray]:
    """Mean over the abortable modes of estimator(cell) -> (point, draws)."""
    results = [estimator(df[df["mode"] == mode]) for mode in cfg.ABORT_MODES if (df["mode"] == mode).any()]
    return float(np.mean([p for p, _ in results])), np.mean(np.vstack([d for _, d in results]), axis=0)


def reached_rate(cell: pd.DataFrame, boot: ca.Bootstrap) -> tuple[float, np.ndarray]:
    return ca.pooled(cell, "reached_t_star", boot)


def bootstrap_p(diff_draws: np.ndarray) -> float:
    """Two-sided: twice the share of paired resamples on the far side of zero (floor: 1 / resamples)."""
    far_side = max(1, min(np.sum(diff_draws <= 0), np.sum(diff_draws >= 0)))
    return float(min(1.0, 2 * far_side / len(diff_draws)))


def holm(p_values: dict) -> dict:
    order = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


# --- Tables -------------------------------------------------------------------------------------------------------
def outcome_table(df: pd.DataFrame, boot: ca.Bootstrap) -> tuple[list[dict], list[dict]]:
    """Per (model, prompt) outcomes, and the 9 prompt - baseline contrasts of the primary and robustness outcomes."""
    rows, contrasts = [], []
    for model in [m for m in cfg.GRID_MODELS if m in set(df["model"])]:
        draws = {}
        for prompt in cfg.PROMPTS:
            d = df[(df["model"] == model) & (df["prompt"] == prompt)]
            if d.empty:
                continue
            abortable = d[d["mode"].isin(cfg.ABORT_MODES)]
            s_point, s_draws = macro_over_modes(abortable, lambda c: survival_at_t_star(c, boot))
            r_point, r_draws = macro_over_modes(abortable, lambda c: reached_rate(c, boot))
            draws[prompt] = {"survival": (s_point, s_draws), "reached": (r_point, r_draws)}
            full = d[~d["abort_on_violation"].astype(bool)]  # requests that never abort: same cells for every prompt
            row = {"model": model, "prompt": prompt, "n": len(d), "n_abortable": len(abortable),
                   "S_t_star": ca.stat(s_point, s_draws), "reached_t_star": ca.stat(r_point, r_draws),
                   "P1": ca.stat(*ca.macro(d, "final_compliant", boot)),
                   "P2_judge": ca.stat(*ca.macro(d, "p2", boot)),
                   "P2_regex": ca.stat(*ca.macro(d, "p2_regex", boot)),
                   "accuracy_full_traces": ca.stat(*ca.pooled(full, "correct", boot)),
                   "n_full_traces": len(full),
                   "aborted_share_of_abortable": float(abortable["aborted"].mean() * 100),
                   "censored_before_t_star_share": float(((~abortable["event"])
                                                          & (abortable["time"] < T_STAR)).mean() * 100),
                   "near_empty_share_full": float(full["near_empty"].mean() * 100),
                   "truncated_share": float(d["truncated"].mean() * 100),
                   "median_reasoning_tokens_full": float(full["reasoning_tokens"].median())}
            rows.append(row)
        for prompt in ARUN_PROMPTS:
            if prompt not in draws or "baseline" not in draws:
                continue
            for outcome in ("survival", "reached"):
                point = draws[prompt][outcome][0] - draws["baseline"][outcome][0]
                diff = draws[prompt][outcome][1] - draws["baseline"][outcome][1]
                contrasts.append({"model": model, "prompt": prompt, "outcome": outcome,
                                  "difference": ca.stat(point, diff), "p": bootstrap_p(diff)})
    for outcome in ("survival", "reached"):
        subset = [c for c in contrasts if c["outcome"] == outcome]
        adjusted = holm({(c["model"], c["prompt"]): c["p"] for c in subset})
        for c in subset:
            c["p_holm"] = adjusted[(c["model"], c["prompt"])]
    return rows, contrasts


def per_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Point estimates per (model, prompt, mode) of the abortable modes."""
    out = []
    for (model, prompt, mode), cell in df[df["mode"].isin(cfg.ABORT_MODES)].groupby(["model", "prompt", "mode"]):
        s = kaplan_meier(cell["time"].to_numpy(), cell["event"].to_numpy(), np.ones((1, len(cell))),
                         np.array([T_STAR]))[0, 0]
        out.append({"model": model, "prompt": prompt, "mode": mode, "n": len(cell), "S_t_star": 100 * s,
                    "reached_t_star": 100 * cell["reached_t_star"].mean(),
                    "P1": 100 * cell["final_compliant"].mean(), "aborted": int(cell["aborted"].sum())})
    return pd.DataFrame(out)


# --- Audit and checks -----------------------------------------------------------------------------------------------
def abort_audit(df: pd.DataFrame) -> dict:
    """Replay the abort rule on the full-trace cells: would it have fired, and only on the true first violation?"""
    items = {it["item_id"]: it for it in map(json.loads, cfg.EXP03_ITEMS_PATH.open())}
    requests = {r["request_id"]: r for r in map(json.loads, cfg.EXP03.requests.open())}
    full = df[df["full_trace_cell"].astype(bool) & df["mode"].isin(cfg.ABORT_MODES)]
    tasks = [({"request_id": r.request_id, "model": r.model, "mode": r.mode, "reasoning": r.reasoning},
              items[r.item_id], requests[r.request_id]["grading_prompt"]) for r in full.itertuples()]
    with mp.Pool(max(1, min(len(tasks), mp.cpu_count() - 16))) as pool:
        replays = pool.map(check_abort_rule.replay, tasks, chunksize=4)
    violating = [r for r in replays if r["compliant"] == 0]
    caught = [r for r in violating if r["fired_at"] is not None]
    errors = [r for r in replays if r["error"]]
    # Randomization check: the full-trace cells are a random quarter of the items per mode, so compliance there
    # should match the aborting cells of the same (model, prompt).
    by_arm = (df[df["mode"].isin(cfg.ABORT_MODES)].groupby(["model", "prompt", "full_trace_cell"])["final_compliant"]
              .mean().mul(100).unstack().rename(columns={False: "P1_aborting_cells", True: "P1_full_trace_cells"}))
    return {"full_trace_traces": len(replays), "violating": len(violating), "caught": len(caught),
            "false_or_unstable": len(errors), "errors": errors[:10],
            "P1_by_arm": by_arm.reset_index().to_dict("records")}


def dumb_checks(df: pd.DataFrame) -> dict:
    requests = [json.loads(line) for line in cfg.EXP03.requests.open()]
    per_model = df.groupby("model")["request_id"].agg(["count", "nunique"])
    aborted = df[df["aborted"]]
    return {
        "every_request_once_per_model": bool((per_model["count"] == len(requests)).all()
                                             and (per_model["nunique"] == len(requests)).all()),
        "rows_per_model": per_model["count"].to_dict(),
        "aborted_only_where_allowed": bool(aborted["abort_on_violation"].astype(bool).all()),
        "aborted_all_non_compliant": bool((aborted["compliant"] == 0).all()),
        "aborted_violation_inside_trace": bool((aborted["fv_token"] < aborted["reasoning_tokens"]).all()),
        "aborted_no_answer": bool((aborted["answer"].fillna("") == "").all()),
        "aborted_rows": len(aborted),
        "abortable_rows_not_aborted_but_violating": int((df["abort_on_violation"].astype(bool) & ~df["aborted"]
                                                         & (df["compliant"] == 0)).sum()),
    }


# --- Figures and report -----------------------------------------------------------------------------------------
def figure_survival(df: pd.DataFrame, fig_dir) -> str:
    grid = np.arange(0, cfg.SURVIVAL_CURVE_MAX_TOKENS + 1, 20, dtype=float)
    models = [m for m in cfg.GRID_MODELS if m in set(df["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.6), squeeze=False, sharey=True)
    pfig = make_subplots(rows=1, cols=len(models), subplot_titles=models)
    for k, model in enumerate(models):
        for prompt in cfg.PROMPTS:
            d = df[(df["model"] == model) & (df["prompt"] == prompt) & df["mode"].isin(cfg.ABORT_MODES)]
            if d.empty:
                continue
            curves = [kaplan_meier(c["time"].to_numpy(), c["event"].to_numpy(), np.ones((1, len(c))), grid)[0]
                      for _, c in d.groupby("mode")]
            s = 100 * np.mean(curves, axis=0)
            style = cfg.PROMPT_STYLE[prompt]
            axes[0][k].plot(grid, s, color=style["color"], label=style["label"])
            pfig.add_trace(go.Scatter(x=grid, y=s, mode="lines", line={"color": style["color"]},
                                      name=style["label"], legendgroup=prompt, showlegend=k == 0), row=1, col=k + 1)
        axes[0][k].axvline(T_STAR, color="k", lw=0.5, ls=":")
        axes[0][k].set_title(model, fontsize=9)
        axes[0][k].set_xlabel("reasoning token")
    axes[0][0].set_ylabel("% with no violation yet (KM, mean over 7 modes)")
    axes[0][0].legend(fontsize=7)
    return ca.save(fig, pfig, fig_dir, "S1_survival")


def report(summary: dict, figure: str, fig_dir) -> str:
    f = ca.fmt
    lines = ["# exp03_abort_survival: automated report (UNVERIFIED)", "",
             f"Run {summary['run']}. Primary: {cfg.EXP03_PRIMARY}.", "",
             "## Primary: S(t*) and prompt - baseline (points, 95% CI)", "",
             "| model | prompt | S(t*) | reached t* | P1 | P2 (judge) | P2_regex | accuracy, full traces | "
             "aborted % | censored < t* % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["per_model"]:
        lines.append(f"| {r['model']} | {r['prompt']} | {f(r['S_t_star'])} | {f(r['reached_t_star'])} | "
                     f"{f(r['P1'])} | {f(r['P2_judge'])} | {f(r['P2_regex'])} | {f(r['accuracy_full_traces'])} | "
                     f"{r['aborted_share_of_abortable']:.1f} | {r['censored_before_t_star_share']:.1f} |")
    lines += ["", "| model | prompt | outcome | difference | p | p (Holm, 9) |", "|---|---|---|---|---|---|"]
    for c in summary["contrasts"]:
        lines.append(f"| {c['model']} | {c['prompt']} | {c['outcome']} | {f(c['difference'])} | {c['p']:.3f} | "
                     f"{c['p_holm']:.3f} |")
    lines += ["", f"![survival]({os.path.relpath(fig_dir, summary['_results_root'])}/{figure})", "",
              "P2 uses the LLM judge (Qwen3.8-27B) and is weaker evidence than the grader-based S(t*), reached-t* "
              "and P1.", ""]
    pooled = summary.get("pooled_with_exp02")
    lines += ["## Secondary: pooled with exp02 (150 items)", ""]
    if isinstance(pooled, dict):
        lines += ["| model | prompt | S(t*) | P1 |", "|---|---|---|---|"]
        lines += [f"| {r['model']} | {r['prompt']} | {f(r['S_t_star'])} | {f(r['P1'])} |" for r in pooled["per_model"]]
        lines += ["", "| model | prompt | outcome | difference | p (Holm, 9) |", "|---|---|---|---|---|"]
        lines += [f"| {c['model']} | {c['prompt']} | {c['outcome']} | {f(c['difference'])} | {c['p_holm']:.3f} |"
                  for c in pooled["contrasts"]]
    else:
        lines.append(f"not computed: {pooled}")
    a = summary["abort_audit"]
    lines += ["", "## Abort audit (full-trace cells)", "",
              f"{a['full_trace_traces']} full traces, {a['violating']} violating, {a['caught']} would have been "
              f"aborted, {a['false_or_unstable']} false or unstable firings.", "",
              "| model | prompt | P1 aborting cells | P1 full-trace cells |", "|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['prompt']} | {r.get('P1_aborting_cells', float('nan')):.1f} | "
              f"{r.get('P1_full_trace_cells', float('nan')):.1f} |" for r in a["P1_by_arm"]]
    lines += ["", "## Dumb checks", "", "```", json.dumps(summary["checks"], indent=1), "```", "",
              "Status: UNVERIFIED until a human adds it to VERIFIED.md"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="analysis run name (default: UTC timestamp)")
    parser.add_argument("--no-pooled", action="store_true", help="skip the secondary analysis pooled with exp02")
    args = parser.parse_args()
    exp = cfg.EXP03
    run_name = args.run or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir, fig_dir = exp.results / "analysis" / run_name, exp.figure_dir(run_name)
    for d in (out_dir, fig_dir):
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")
        d.mkdir(parents=True)

    df = load_exp03()
    items = [json.loads(line) for line in cfg.EXP03_ITEMS_PATH.open()]
    boot = ca.Bootstrap(items, cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    rows, contrasts = outcome_table(df, boot)
    summary = {"exp_id": exp.exp_id, "run": run_name, "t_star": T_STAR, "per_model": rows, "contrasts": contrasts,
               "checks": dumb_checks(df), "abort_audit": abort_audit(df)}

    if args.no_pooled:
        summary["pooled_with_exp02"] = "skipped (--no-pooled)"
    else:
        try:
            old = load_exp02()
            old = old.assign(aborted=False, abort_on_violation=False, full_trace_cell=False)
            both = pd.concat([df, old[df.columns.intersection(old.columns)]], ignore_index=True)
            both_items = items + [json.loads(line) for line in cfg.ITEMS_PATH.open()]
            prow, pcontrasts = outcome_table(both, ca.Bootstrap(both_items, cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED))
            summary["pooled_with_exp02"] = {"per_model": [{k: r[k] for k in ("model", "prompt", "n", "S_t_star", "P1")}
                                                          for r in prow], "contrasts": pcontrasts}
        except SystemExit as e:  # an exp02 file without grades or judge output
            summary["pooled_with_exp02"] = f"not computed: {e}"

    per_cell_table(df).to_csv(out_dir / "per_cell.csv", index=False)
    figure = figure_survival(df, fig_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    summary["_results_root"] = out_dir
    (out_dir / "REPORT_auto.md").write_text(report(summary, figure, fig_dir))
    print(f"wrote {out_dir.relative_to(cfg.REPO_ROOT)}")
    print((out_dir / "REPORT_auto.md").read_text())


if __name__ == "__main__":
    main()
