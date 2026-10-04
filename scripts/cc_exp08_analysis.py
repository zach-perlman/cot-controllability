"""exp08 analysis (results/exp08_openings_channels/manifest.json): the openings, channels and accuracy contrasts.

Every cell is (arm label "<arm>|<opening>|<channel>", model, rule). A contrast is the mean over its model x rule
cells of (cell a - cell b), with a paired bootstrap over the 160 questions (2000 multinomial draws, one set of
question weights shared by every cell) and a two-sided bootstrap p; Holm over the 4 primary contrasts.
  openings  S(1000) of the graded text (the continuation after an opening)
  channels  S(200), thinking on and off
  accuracy  % correct; the no-rule reference row of a question serves every rule
Extension (manifest_extension_exp06_rules.json): the openings arms on exp06's five rules (cc_exp08_ext), loaded when
their grades exist; contrasts O1x-O4x, outside the Holm family. Also reported: the thinking-off contrasts on rows with
a closed tag block only (sensitivity), and A vs exp07's upgraded test rows (exploratory, cross-run).

Run: /venv/main/bin/python scripts/cc_exp08_analysis.py --run NAME [--skip-missing]
Writes results/exp08_openings_channels/analysis/<run>/ and figures/exp08_openings_channels/<run>/ (never overwritten).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import cc_config as cfg
import cc_exp04_analysis as a4
import cc_exp07 as e7
import cc_exp08 as e
import cc_exp08_ext as x
from cc_exp07_analysis import BOOT_SEED, N_BOOT
from cc_survival import holm, kaplan_meier

A_RULES = ["alternating_case", "no_digits", "no_first_person"]  # A runs off / in the accuracy part
# name: (metric, cell a, cell b, rules, primary)
CONTRASTS = {
    "O1 compliant opening - none (A)": ("S_1000", "A|compliant|on", "A|none|on", e.OPENING_RULES, True),
    "O3 commitment - compliant opening (A)": ("S_1000", "A|commitment|on", "A|compliant|on", e.OPENING_RULES, True),
    "C2 thinking off - on (A)": ("S_200", "A|none|off", "A|none|on", A_RULES, True),
    "ACC1 A - no rule (accuracy)": ("accuracy", "A|none|full", "no_rule|none|full", A_RULES, True),
    "O2 rule + opening - opening only": ("S_1000", "A|compliant|on", "no_rule|compliant|on", e.OPENING_RULES, False),
    "O4 non-compliant opening - none (A)": ("S_1000", "A|noncompliant|on", "A|none|on", e.OPENING_RULES, False),
    "C1 thinking off - on (CoT-Control prompt)": ("S_200", "baseline|none|off", "baseline|none|on", e.RULES, False),
    "C3 A - CoT-Control prompt, thinking on": ("S_200", "A|none|on", "baseline|none|on", A_RULES, False),
    "C4 A - CoT-Control prompt, thinking off": ("S_200", "A|none|off", "baseline|none|off", A_RULES, False),
    "ACC2 CoT-Control prompt - no rule (accuracy)": ("accuracy", "baseline|none|full", "no_rule|none|full", e.RULES,
                                                     False),
    "ACC3 A - CoT-Control prompt (accuracy)": ("accuracy", "A|none|full", "baseline|none|full", A_RULES, False),
}
EXTENSION_NONCOMPLIANT_RULES = [r for r in x.RULES if r != "word_suppression"]
EXTENSION_CONTRASTS = {
    "O1x compliant opening - none (A), exp06 rules": ("S_1000", "A|compliant|on", "A|none|on", x.RULES, False),
    "O2x rule + opening - opening only, exp06 rules": ("S_1000", "A|compliant|on", "no_rule|compliant|on", x.RULES,
                                                       False),
    "O3x commitment - compliant opening (A), exp06 rules": ("S_1000", "A|commitment|on", "A|compliant|on", x.RULES,
                                                            False),
    "O4x non-compliant opening - none (A), exp06 rules": ("S_1000", "A|noncompliant|on", "A|none|on",
                                                          EXTENSION_NONCOMPLIANT_RULES, False),
}
CONTRASTS.update(EXTENSION_CONTRASTS)
TAGGED_ONLY = ["C1 thinking off - on (CoT-Control prompt)", "C2 thinking off - on (A)",
               "C4 A - CoT-Control prompt, thinking off"]
ALL_RULES = e.RULES + x.RULES
GUARDS = ["empty", "degenerate", "meta_regex", "ended_clean_early"]


# --- Data -----------------------------------------------------------------------------------------------------------
def load(skip_missing: bool) -> tuple[pd.DataFrame, list[str]]:
    """exp08's grades, the extension's where they exist, and exp07's CoT-Control thinking-on rows for the reused
    models, with survival columns."""
    frames, missing = [], []
    for model in e.MODELS:
        path = x.grades_path(model)
        if path is not None and path.exists():
            frames.append(pd.DataFrame([json.loads(line) for line in path.open()]).assign(model=model))
    for model in e.MODELS:
        try:
            path = e.grades_path(model)
        except SystemExit:
            path = None
        if path is None or not path.exists():
            if not skip_missing:
                raise SystemExit(f"{model}: no grades yet")
            missing.append(model)
            continue
        frames.append(pd.DataFrame([json.loads(line) for line in path.open()]).assign(model=model))
    reused = [m for m in e.REUSED_BASELINE if m not in missing]
    if reused:
        old = e7.load_grades("test", reused)
        old = old[(old["prompt"] == "baseline") & old["mode"].isin(e.RULES)]
        frames.append(old.assign(cell="baseline|none|on", channel="on", part="channels", condition="none"))
    df = pd.concat(frames, ignore_index=True)
    return e7.survival(df), missing


def question_weights(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    items = sorted({it["item_id"] for it in e.test_items()})
    weights = np.random.default_rng(BOOT_SEED).multinomial(len(items), np.ones(len(items)) / len(items),
                                                           size=N_BOOT).astype(float)
    return {it: i for i, it in enumerate(items)}, weights


def cell_values(df: pd.DataFrame) -> dict:
    """(metric, cell, model, rule) -> (point, bootstrap draws). The no-rule accuracy rows (rule no_constraint) are
    copied to every rule."""
    index, weights = question_weights(df)
    out = {}
    for (cell, model, mode), c in df.groupby(["cell", "model", "mode"]):
        w = weights[:, c["item_id"].map(index).to_numpy()]
        times, events = c["time"].to_numpy(), c["event"].to_numpy().astype(bool)
        channel = cell.rsplit("|", 1)[1]
        if channel == "full":
            y = c["correct"].astype(float).to_numpy()
            values = {"accuracy": (100 * y.mean(), 100 * (w @ y) / w.sum(axis=1))}
        else:
            ts = np.array([200, 1000])
            point = kaplan_meier(times, events, np.ones((1, len(c))), ts)[0]
            draws = kaplan_meier(times, events, w, ts)
            values = {"S_200": (100 * point[0], 100 * draws[:, 0]), "S_1000": (100 * point[1], 100 * draws[:, 1])}
        rules_of = e.RULES if mode == cfg.NO_CONSTRAINT else [mode]
        for metric, v in values.items():
            for rule in rules_of:
                out[(metric, cell, model, rule)] = v
    return out


def contrast(values: dict, metric: str, a: str, b: str, cells: list[tuple[str, str]]) -> dict | None:
    cells = [(m, r) for m, r in cells if (metric, a, m, r) in values and (metric, b, m, r) in values]
    if not cells:
        return None
    point = float(np.mean([values[(metric, a, m, r)][0] - values[(metric, b, m, r)][0] for m, r in cells]))
    d = np.mean([values[(metric, a, m, r)][1] - values[(metric, b, m, r)][1] for m, r in cells], axis=0)
    return {"value": point, "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))), "n_cells": len(cells)}


def contrasts(values: dict, models: list[str]) -> dict:
    out = {}
    for name, (metric, a, b, rule_set, primary) in CONTRASTS.items():
        cells = [(m, r) for m in models for r in rule_set]
        res = contrast(values, metric, a, b, cells)
        if res is None:
            continue
        family = "primary" if primary else "extension" if name in EXTENSION_CONTRASTS else "secondary"
        res.update(metric=metric, a=a, b=b, primary=primary, family=family,
                   per_model={m: contrast(values, metric, a, b, [(m, r) for r in rule_set]) for m in models},
                   per_rule={r: contrast(values, metric, a, b, [(m, r) for m in models]) for r in rule_set})
        out[name] = res
    adjusted = holm({n: v["p_two_sided"] for n, v in out.items() if v["primary"]})
    for n, p in adjusted.items():
        out[n]["p_holm"] = p
    return out


def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Per rule set (exp08's held-out rules / the extension's exp06 rules) and cell, pooled over models and rules: n,
    guards, length of the graded text, no-tag share (off), truncation and full-trace compliance (accuracy part)."""
    d = df.copy()
    d[GUARDS] = d[GUARDS].astype(float)
    d["rule_set"] = np.where(d["mode"].isin(x.RULES), "exp06 rules", "held-out rules")
    rows = []
    for (rule_set, cell), c in d.groupby(["rule_set", "cell"]):
        q = c["reasoning_tokens"].quantile([0.25, 0.5, 0.75]).to_numpy()
        row = {"rules": rule_set, "cell": cell, "n": len(c), **(100 * c[GUARDS].mean()).round(1).to_dict(),
               "tokens_p25": int(q[0]), "tokens_median": int(q[1]), "tokens_p75": int(q[2])}
        if cell.endswith("|off"):
            row["no_tag_block"] = round(100 * float((c["external_blocks"].fillna(0) == 0).mean()), 1) \
                if "external_blocks" in c else None
        if cell.endswith("|full"):
            row["truncated"] = round(100 * float(c["truncated"].astype(float).mean()), 1)
            if not cell.startswith("no_rule"):
                row["compliant_full_trace"] = round(100 * float(c["compliant"].astype(float).mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# --- Sensitivity and exploratory ------------------------------------------------------------------------------------
def with_breakdowns(values: dict, metric: str, a: str, b: str, models: list[str], rule_set: list[str]) -> dict | None:
    res = contrast(values, metric, a, b, [(m, r) for m in models for r in rule_set])
    if res is not None:
        res.update(metric=metric, a=a, b=b,
                   per_model={m: contrast(values, metric, a, b, [(m, r) for r in rule_set]) for m in models},
                   per_rule={r: contrast(values, metric, a, b, [(m, r) for m in models]) for r in rule_set})
    return res


def tagged_only(df: pd.DataFrame, models: list[str]) -> dict:
    """C1, C2, C4 with thinking-off rows restricted to those with a closed <output_reasoning> block (otherwise a
    no-tag row is a violation at token 0, and the no-tag share differs between prompts)."""
    keep = (df["channel"] != "off") | (df["external_blocks"].fillna(0) > 0)
    values = cell_values(df[keep])
    out = {}
    for name in TAGGED_ONLY:
        metric, a, b, rule_set, _ = CONTRASTS[name]
        res = with_breakdowns(values, metric, a, b, models, rule_set)
        if res is not None:
            out[name] = res
    return out


def a_vs_upgraded(values: dict, models: list[str]) -> dict:
    """Exploratory, cross-run: A|none|on (exp08) - exp07's upgraded thinking-on test rows, on the held-out rules,
    for the models exp07 tested (same questions, rules, seeds, stop and engine)."""
    reused = [m for m in models if m in e.REUSED_BASELINE]
    if not reused:
        return {}
    upgraded = e7.load_grades("test", reused)
    upgraded = upgraded[(upgraded["prompt"] == "upgraded") & upgraded["mode"].isin(e.RULES)]
    both = {**values, **cell_values(e7.survival(upgraded.assign(cell="upgraded|none|on", channel="on")))}
    out = {}
    for metric in ("S_200", "S_1000"):
        res = with_breakdowns(both, metric, "A|none|on", "upgraded|none|on", reused, e.RULES)
        if res is not None:
            out[f"A - upgraded (exp07 test rows), {metric}"] = res
    return out


# --- Figure and report ----------------------------------------------------------------------------------------------
FAMILY_COLOR = {"primary": "#B5532A", "secondary": "#0072B2", "extension": "#009E73"}


def fig_contrasts(results: dict, models: list[str], fig_dir) -> str:
    """X1. Every contrast: mean over its cells (95% CI) and per model."""
    names = list(results)
    fig = go.Figure()
    vals = [results[n] for n in names]
    fig.add_trace(go.Scatter(
        x=[v["value"] for v in vals], y=names, mode="markers", name="mean over its cells",
        marker={"size": 12, "color": [FAMILY_COLOR[v["family"]] for v in vals]},
        error_x={"type": "data", "symmetric": False, "thickness": 1.5, "color": "#444444",
                 "array": [v["ci"][1] - v["value"] for v in vals],
                 "arrayminus": [v["value"] - v["ci"][0] for v in vals]}))
    for model, symbol in zip(models, ("circle-open", "square-open", "diamond-open", "triangle-up-open")):
        pts = [(n, results[n]["per_model"][model]["value"]) for n in names if results[n]["per_model"].get(model)]
        fig.add_trace(go.Scatter(x=[x for _, x in pts], y=[n for n, _ in pts], mode="markers", name=model,
                                 marker={"symbol": symbol, "size": 9, "color": "#666666"}))
    fig.add_vline(x=0, line={"color": "#999999", "width": 1})
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="difference (S points for O/C contrasts, accuracy points for ACC)")
    fig.update_layout(title="X1. exp08: openings, channels and accuracy with exp07b's prompt A, 160 test questions"
                            "<br><sup>Large markers: mean over the contrast's model x rule cells, 95% CI from a "
                            "bootstrap over questions (orange: pre-registered primary, Holm over 4; blue: secondary; "
                            "green: extension on exp06's rules).<br>Open markers: per model. O: S(1000) of the text "
                            "after the opening; C: S(200); ACC: % correct (40 questions).</sup>",
                      legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"},
                      margin={"l": 380, "t": 120, "b": 110})
    return a4.save(fig, fig_dir, "X1_contrasts", 1500, 300 + 32 * len(names))


def fmt(s: dict | None) -> str:
    return "-" if s is None else f"{s['value']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]"


def report(summary: dict) -> str:
    models = summary["models"]
    lines = [f"# exp08 analysis: {summary['run']} (UNVERIFIED until a human adds it to VERIFIED.md)", "",
             f"Models: {', '.join(models)}." + (f" Missing: {', '.join(summary['missing'])}."
                                                 if summary["missing"] else ""), "",
             "## Contrasts (mean over model x rule cells, 95% CI over questions)", "",
             "| contrast | metric | cells | difference | p | Holm p | " + " | ".join(models) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(models)]
    for name, v in summary["contrasts"].items():
        holm_p = f"{v['p_holm']:.3f}" if "p_holm" in v else "-"
        lines.append(f"| {'**' + name + '**' if v['primary'] else name} | {v['metric']} | {v['n_cells']} | "
                     f"{fmt(v)} | {v['p_two_sided']:.3f} | {holm_p} | "
                     + " | ".join(fmt(v["per_model"].get(m)) for m in models) + " |")
    rules_shown = [r for r in ALL_RULES if any(v["per_rule"].get(r) for v in summary["contrasts"].values())]
    lines += ["", "## Per rule", "", "| contrast | " + " | ".join(rules_shown) + " |",
              "|---|" + "---|" * len(rules_shown)]
    for name, v in summary["contrasts"].items():
        lines.append(f"| {name} | " + " | ".join(f"{v['per_rule'][r]['value']:+.1f}" if v["per_rule"].get(r)
                                                 else "-" for r in rules_shown) + " |")
    for title, key, note in [
            ("Sensitivity: thinking off, rows with a closed tag block only", "tagged_only",
             "Same contrasts as above with thinking-off rows that have no closed <output_reasoning> block dropped "
             "(above, they count as a violation at token 0)."),
            ("Exploratory: A vs exp07's upgraded prompt", "a_vs_upgraded",
             "Cross-run (exp07's upgraded test rows: same questions, held-out rules, seeds, stop and engine); not "
             "pre-registered, no multiplicity correction.")]:
        lines += ["", f"## {title}", "", note, "",
                  "| contrast | metric | cells | difference | p | " + " | ".join(models) + " |",
                  "|---|---|---|---|---|" + "---|" * len(models)]
        for name, v in summary[key].items():
            lines.append(f"| {name} | {v['metric']} | {v['n_cells']} | {fmt(v)} | {v['p_two_sided']:.3f} | "
                         + " | ".join(fmt(v["per_model"].get(m)) for m in models) + " |")
    lines += ["", "## Cells (pooled over models and the rules of each rule set)", ""]
    desc = pd.DataFrame(summary["descriptives"]).fillna("-")
    lines += ["| " + " | ".join(desc.columns) + " |", "|" + "---|" * len(desc.columns)]
    lines += ["| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |" for r in desc.itertuples(index=False)]
    lines += ["", "## Figures", ""]
    lines += [f"- [{f}](../../../../figures/exp08_openings_channels/{summary['run']}/{f}.html)"
              for f in summary["figures"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    out_dir = e.EXP.results / "analysis" / args.run
    fig_dir = e.EXP.figure_dir(args.run)
    for d in (out_dir, fig_dir):
        if d.exists():
            raise SystemExit(f"{d} exists; analysis runs are never overwritten")
    df, missing = load(args.skip_missing)
    models = [m for m in e.MODELS if m not in missing]
    values = cell_values(df)
    results = contrasts(values, models)
    out_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)
    pd.DataFrame([{"metric": k[0], "cell": k[1], "model": k[2], "mode": k[3], "value": p,
                   "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)}
                  for k, (p, d) in values.items()]).round(2).to_csv(out_dir / "per_cell.csv", index=False)
    summary = {"exp_id": e.EXP.exp_id, "run": args.run, "models": models, "missing": missing, "n_boot": N_BOOT,
               "boot_seed": BOOT_SEED, "contrasts": results, "tagged_only": tagged_only(df, models),
               "a_vs_upgraded": a_vs_upgraded(values, models), "descriptives": descriptives(df).to_dict("records"),
               "figures": [fig_contrasts(results, models, fig_dir)]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "REPORT_auto.md").write_text(report(summary))
    print(report(summary))


if __name__ == "__main__":
    main()
