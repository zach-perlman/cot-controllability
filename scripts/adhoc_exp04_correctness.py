"""Ad hoc (human request, 2026-09-30; not in exp04's manifest): exp05's answer-correctness check on exp04's rows.

Do rollouts that answer correctly keep the rule more? Full-trace cells only (the ~25% that run to the end whatever
the violations), the 4 opener rules, question bootstrap; raw (whole trace) and length-matched (no violation in the
first t tokens, among rollouts that reached t) right - wrong. Scoring as exp04's v2 run: an empty thinking trace is a
violation at token 0. Same functions as exp05's G5 (cc_exp05_analysis.correctness_table / fig_correctness).

Outputs (never overwritten): results/exp04_prefill/analysis/adhoc_correctness/ and
figures/exp04_prefill/adhoc_correctness/.
Run: /venv/main/bin/python scripts/adhoc_exp04_correctness.py
"""

from __future__ import annotations

import json
import os

import cc_analysis as ca
import cc_config as cfg
import cc_exp04_analysis as a4
import cc_exp05_analysis as a5

RUN = "adhoc_correctness"
FIGURE_ARMS = [("none", "baseline"), ("prefill_compliant", "baseline"), ("prefill_noncompliant", "baseline"),
               ("prefill_no_rule", cfg.NO_CONSTRAINT)]  # the arms of exp04's primary contrasts
TABLE_ARMS = FIGURE_ARMS + [("none", "stacked"), ("prefill_compliant", "stacked"), ("prefill_noncompliant", "stacked")]


def arms(pairs: list[tuple[str, str]]) -> dict:
    """label -> (color, row selector), in exp05's correctness_table format."""
    return {a4.arm_label(c, p): (cfg.CONDITION_STYLE[c]["color"],
                                 lambda d, c=c, p=p: (d["condition"] == c) & (d["prompt"] == p)) for c, p in pairs}


def report(rows: list[dict], figure: str, fig_rel: str) -> str:
    f = ca.fmt
    lines = ["# exp04_prefill: ad hoc answer-correctness check (UNVERIFIED)", "",
             "Not pre-registered (human request, 2026-09-30). Full-trace cells, 4 opener rules; empty thinking traces "
             "are violations at token 0 (v2 scoring). Wrong answers come with longer traces, so read the "
             f"length-matched columns. Right - wrong in % points; n/a: fewer than {a5.CORRECTNESS_MIN_N} rollouts in a "
             "group; 0.0 [0.0, 0.0]: nobody in either group kept the rule (a floor, not evidence of no link). "
             "Every number is grader-scored (no LLM judge).", "",
             "| model | arm | n right / wrong | median tokens right / wrong | obeyed throughout: right - wrong | "
             f"no violation in first {a5.T_SHORT}: right - wrong | no violation in first {a5.T_STAR}: right - wrong "
             "(n right / wrong) |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['arm']} | {r['n_right']} / {r['n_wrong']} | {r['median_tokens_right']} / "
              f"{r['median_tokens_wrong']} | {f(r['obeyed_throughout_right_minus_wrong'])} | "
              f"{f(r[f'clean_first_{a5.T_SHORT}_right_minus_wrong'])} | "
              f"{f(r[f'clean_first_{a5.T_STAR}_right_minus_wrong'])} "
              f"({' / '.join(map(str, r[f'n_clean_first_{a5.T_STAR}']))}) |" for r in rows]
    lines += ["", f"- [{figure}]({fig_rel}/{figure}.html) ![{figure}]({fig_rel}/{figure}.png)", "",
              "Status: UNVERIFIED until a human adds it to VERIFIED.md"]
    return "\n".join(lines) + "\n"


def main() -> None:
    out_dir, fig_dir = cfg.EXP04.results / "analysis" / RUN, cfg.EXP04.figure_dir(RUN)
    for d in {out_dir, fig_dir}:
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")
    df = a4.analysis_rows(a4.load(empty_trace_is_violation=True))
    models = a4.models_in(df)
    items = [json.loads(line) for line in cfg.EXP03_ITEMS_PATH.open()]
    boot = ca.Bootstrap(items, cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    rows = a5.correctness_table(df, models, boot, arms(TABLE_ARMS))
    for d in {out_dir, fig_dir}:
        d.mkdir(parents=True)
    figure = a5.fig_correctness([r for r in rows if r["arm"] in arms(FIGURE_ARMS)], models, fig_dir,
                                arms(FIGURE_ARMS), name="F7_correctness_check",
                                note="Baseline-prompt arms (exp04's primary contrasts); stacked arms in the report.")
    (out_dir / "correctness.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    text = report(rows, figure, os.path.relpath(fig_dir, out_dir))
    (out_dir / "REPORT_auto.md").write_text(text)
    print(f"wrote {out_dir.relative_to(cfg.REPO_ROOT)}")
    print(text)


if __name__ == "__main__":
    main()
