"""CPU checks of cc_survival's estimator and tables before exp03 data exist.

1. No censoring: KM S(t*) equals the share of traces whose first violation is at token t* or later.
2. Integer bootstrap weights equal duplicating rows.
3. Agrees with lifelines on real exp02 cells (8B and 32B grades), with the same convention: a violation-free trace
   of length L is known violation-free on tokens 0..L-1 (lifelines: censored at L - 0.5); S(t*) = P(first
   violation >= t*) (lifelines: survival at t* - 0.5).
4. outcome_table runs on exp02 grades dressed as exp03 (a random 75% of abortable-mode violating traces cut just
   after their violation), and the cut changes neither S(t*) nor P1: aborting after a violation loses nothing the
   primary outcome needs.

Run: /venv/main/bin/python scripts/check_survival.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import cc_analysis as ca
import cc_config as cfg
import cc_survival as sv

T = sv.T_STAR


def exp02_grades() -> pd.DataFrame:
    rows = [json.loads(line) for p in sorted(cfg.EXP02.grades.glob("*.jsonl")) for line in p.open()]
    df = pd.DataFrame(rows)
    df = df[df["model"].isin(["Qwen3-8B", "Qwen3-32B"]) & df["prompt"].isin(cfg.PROMPTS)].copy()
    df["final_compliant"] = pd.to_numeric(df["compliant"], errors="coerce")
    df["p2"] = df["p2_regex"] = df["final_compliant"]
    df["correct"] = df["correct"].astype(float)
    return sv.with_survival_columns(df)


def main() -> None:
    rng = np.random.default_rng(0)

    times = rng.integers(0, 3000, 500).astype(float)
    events = np.ones(500, dtype=bool)
    s = sv.kaplan_meier(times, events, np.ones((1, 500)), np.array([T]))[0, 0]
    assert np.isclose(s, np.mean(times >= T)), (s, np.mean(times >= T))
    print(f"1 ok: uncensored KM {s:.4f} = empirical share")

    times = rng.integers(0, 3000, 200).astype(float)
    events = rng.random(200) < 0.7
    w = rng.integers(0, 4, 200)
    weighted = sv.kaplan_meier(times, events, w[None, :].astype(float), np.array([T]))[0, 0]
    dup = sv.kaplan_meier(np.repeat(times, w), np.repeat(events, w), np.ones((1, w.sum())), np.array([T]))[0, 0]
    assert np.isclose(weighted, dup), (weighted, dup)
    print(f"2 ok: weighted {weighted:.4f} = duplicated rows")

    df = exp02_grades()
    worst = 0.0
    cells = df[df["mode"].isin(cfg.ABORT_MODES)].groupby(["model", "prompt", "mode"])
    for _, cell in cells:
        ev = cell["event"].to_numpy()
        t = cell["time"].to_numpy()
        ours = sv.kaplan_meier(t, ev, np.ones((1, len(cell))), np.array([T]))[0, 0]
        kmf = KaplanMeierFitter().fit(np.where(ev, t, t - 0.5), ev)
        ref = float(kmf.survival_function_at_times(T - 0.5).iloc[0])
        worst = max(worst, abs(ours - ref))
    assert worst < 1e-9, worst
    print(f"3 ok: {len(cells)} exp02 cells agree with lifelines (max |diff| {worst:.1e})")

    full = df.assign(abort_on_violation=False, full_trace_cell=False, aborted=False, near_empty=False)
    cut = full.copy()
    violating = cut["mode"].isin(cfg.ABORT_MODES) & cut["event"]
    chosen = violating & (rng.random(len(cut)) < 0.75)
    cut.loc[chosen, "reasoning_tokens"] = cut.loc[chosen, "fv_token"] + 50
    cut.loc[chosen, ["aborted", "abort_on_violation"]] = True
    cut.loc[chosen, "correct"] = np.nan
    cut = sv.with_survival_columns(cut)
    items = [json.loads(line) for line in cfg.ITEMS_PATH.open()]
    boot = ca.Bootstrap(items, 200, cfg.BOOTSTRAP_SEED)
    rows_full, _ = sv.outcome_table(full, boot)
    rows_cut, contrasts = sv.outcome_table(cut, boot)
    for a, b in zip(rows_full, rows_cut):
        for key in ("S_t_star", "reached_t_star", "P1"):
            assert np.isclose(a[key]["value"], b[key]["value"]), (a["model"], a["prompt"], key, a[key], b[key])
    assert len(contrasts) == 2 * 2 * len(sv.ARUN_PROMPTS) and all(0 <= c["p_holm"] <= 1 for c in contrasts)
    print("4 ok: cutting violating traces after the violation leaves S(t*), reached-t* and P1 unchanged")
    for r in rows_cut:
        print(f"   {r['model']:10s} {r['prompt']:13s} S(t*) {ca.fmt(r['S_t_star'])}  P1 {ca.fmt(r['P1'])}  "
              f"censored<t* {r['censored_before_t_star_share']:.1f}%")
    for c in contrasts:
        if c["outcome"] == "survival":
            print(f"   {c['model']:10s} {c['prompt']:13s} diff {ca.fmt(c['difference'])} p {c['p']:.3f} "
                  f"holm {c['p_holm']:.3f}")
    print("all survival checks passed")


if __name__ == "__main__":
    main()
