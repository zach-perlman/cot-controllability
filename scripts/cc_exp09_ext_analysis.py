"""Analysis of exp09's extension (results/exp09_final_test/manifest_extension_exp08_parts.json): exp08's openings
and thinking on/off contrasts on exp09's questions and 6 of its models.

Every cell is (arm label "<arm>|<opening>|<channel>", model, rule). The thinking-on |none cells are exp09's short
rows (A|short -> A|none|on, baseline|short -> baseline|none|on). A contrast is the mean over its model x rule cells of
(cell a - cell b), with exp09's paired bootstrap over questions (stratified by source; cc_exp09_analysis
.question_draws) and a two-sided bootstrap p; Holm over O1, O3, C2.
  openings  S(1000) of the graded continuation after the opening
  channels  S(200), thinking on and off
Also reported: the thinking-off contrasts on rows with a closed tag block only (sensitivity, as exp08), and per-cell
descriptives (guards, graded-text length, the share of thinking-off responses without a closed tag block).

Run: /venv/main/bin/python scripts/cc_exp09_ext_analysis.py --run NAME [--skip-missing]
Writes results/exp09_final_test/analysis_ext/<run>/ (never overwritten).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import cc_exp07 as e7
import cc_exp08 as e8
import cc_exp09 as e9
import cc_exp09_ext as x
from cc_exp08_analysis import GUARDS, contrast, fmt
from cc_exp09_analysis import question_draws
from cc_survival import holm, kaplan_meier

OUT_ROOT = e9.EXP.results / "analysis_ext"
T_VALUES = np.array([200, 1000])
# name: (metric, cell a, cell b, rules, primary, only where A runs thinking off)
CONTRASTS = {
    "O1 compliant opening - none (A)": ("S_1000", "A|compliant|on", "A|none|on", e8.OPENING_RULES, True, False),
    "O3 commitment - compliant opening (A)": ("S_1000", "A|commitment|on", "A|compliant|on", e8.OPENING_RULES, True,
                                              False),
    "C2 thinking off - on (A)": ("S_200", "A|none|off", "A|none|on", e8.RULES, True, True),
    "O2 rule + opening - opening only": ("S_1000", "A|compliant|on", "no_rule|compliant|on", e8.OPENING_RULES, False,
                                         False),
    "O4 non-compliant opening - none (A)": ("S_1000", "A|noncompliant|on", "A|none|on", e8.OPENING_RULES, False,
                                            False),
    "C1 thinking off - on (CoT-Control prompt)": ("S_200", "baseline|none|off", "baseline|none|on", e8.RULES, False,
                                                  False),
    "C3 A - CoT-Control prompt, thinking on": ("S_200", "A|none|on", "baseline|none|on", e8.RULES, False, True),
    "C4 A - CoT-Control prompt, thinking off": ("S_200", "A|none|off", "baseline|none|off", e8.RULES, False, True),
}
TAGGED_ONLY = ["C1 thinking off - on (CoT-Control prompt)", "C2 thinking off - on (A)",
               "C4 A - CoT-Control prompt, thinking off"]


# --- Data -----------------------------------------------------------------------------------------------------------
def load(skip_missing: bool) -> tuple[pd.DataFrame, list[str]]:
    """The extension's grades and exp09's short rows of the extension's rules, with survival columns."""
    frames, present = [], []
    for model in x.MODEL_ORDER:
        try:
            ext, main = x.grades_path(model), e9.grades_path(model)
        except SystemExit:
            ext = main = None
        if ext is None or not ext.exists() or not main.exists():
            if skip_missing:
                continue
            raise SystemExit(f"{model}: no extension or exp09 grades (pass --skip-missing)")
        frames.append(pd.DataFrame([json.loads(line) for line in ext.open()]).assign(model=model))
        short = pd.DataFrame([json.loads(line) for line in main.open()]).assign(model=model)
        short = short[(short["channel"] == "short") & short["mode"].isin(e8.RULES)].copy()
        short["cell"] = short["prompt"] + "|none|on"
        short["channel"] = "on"
        frames.append(short)
        present.append(model)
    if not frames:
        raise SystemExit("no graded model")
    return e7.survival(pd.concat(frames, ignore_index=True)), present


def cell_values(df: pd.DataFrame) -> dict:
    """(metric, cell, model, rule) -> (point, bootstrap draws)."""
    index, weights = question_draws()
    out = {}
    for (cell, model, mode), c in df.groupby(["cell", "model", "mode"]):
        w = weights[:, c["item_id"].map(index).to_numpy()]
        times, events = c["time"].to_numpy(), c["event"].to_numpy().astype(bool)
        point = kaplan_meier(times, events, np.ones((1, len(c))), T_VALUES)[0]
        draws = kaplan_meier(times, events, w, T_VALUES)
        out[("S_200", cell, model, mode)] = (100 * point[0], 100 * draws[:, 0])
        out[("S_1000", cell, model, mode)] = (100 * point[1], 100 * draws[:, 1])
    return out


def cells_of(values: dict, models: list[str], rules: list[str], only_a_off: bool) -> list[tuple[str, str]]:
    cells = [(m, r) for m in models for r in rules]
    return [(m, r) for m, r in cells if ("S_200", "A|none|off", m, r) in values] if only_a_off else cells


def with_breakdowns(values: dict, name: str, models: list[str]) -> dict | None:
    metric, a, b, rules, primary, only_a_off = CONTRASTS[name]
    cells = cells_of(values, models, rules, only_a_off)
    res = contrast(values, metric, a, b, cells)
    if res is not None:
        res.update(metric=metric, a=a, b=b, primary=primary,
                   per_model={m: contrast(values, metric, a, b, [c for c in cells if c[0] == m]) for m in models},
                   per_rule={r: contrast(values, metric, a, b, [c for c in cells if c[1] == r]) for r in rules})
    return res


def contrasts(values: dict, models: list[str]) -> dict:
    out = {n: r for n in CONTRASTS if (r := with_breakdowns(values, n, models)) is not None}
    for n, p in holm({n: v["p_two_sided"] for n, v in out.items() if v["primary"]}).items():
        out[n]["p_holm"] = p
    return out


def tagged_only(df: pd.DataFrame, models: list[str]) -> dict:
    """C1, C2, C4 with thinking-off rows restricted to those with a closed <output_reasoning> block."""
    keep = (df["channel"] != "off") | (df["external_blocks"].fillna(0) > 0)
    values = cell_values(df[keep])
    return {n: r for n in TAGGED_ONLY if (r := with_breakdowns(values, n, models)) is not None}


def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d[GUARDS] = d[GUARDS].astype(float)
    rows = []
    for (model, cell), c in d.groupby(["model", "cell"]):
        q = c["reasoning_tokens"].quantile([0.25, 0.5, 0.75]).to_numpy()
        row = {"model": model, "cell": cell, "n": len(c), **(100 * c[GUARDS].mean()).round(1).to_dict(),
               "tokens_p25": int(q[0]), "tokens_median": int(q[1]), "tokens_p75": int(q[2])}
        if cell.endswith("|off"):
            row["no_tag_block"] = round(100 * float((c["external_blocks"].fillna(0) == 0).mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# --- Report ---------------------------------------------------------------------------------------------------------
def contrast_table(results: dict, models: list[str], with_holm: bool) -> list[str]:
    head = "| contrast | metric | cells | difference | p | " + ("Holm p | " if with_holm else "")
    lines = [head + " | ".join(models) + " |", "|---|---|---|---|---|" + ("---|" if with_holm else "")
             + "---|" * len(models)]
    for name, v in results.items():
        holm_p = (f"{v['p_holm']:.3f} | " if "p_holm" in v else "- | ") if with_holm else ""
        label = f"**{name}**" if v["primary"] and with_holm else name
        lines.append(f"| {label} | {v['metric']} | {v['n_cells']} | {fmt(v)} | {v['p_two_sided']:.3f} | {holm_p}"
                     + " | ".join(fmt(v["per_model"].get(m)) for m in models) + " |")
    return lines


def report(summary: dict) -> str:
    models = summary["models"]
    lines = [f"# exp09 extension (exp08's openings and channels): {summary['run']} "
             "(UNVERIFIED until a human adds it to VERIFIED.md)", "",
             f"Models: {', '.join(models)}." + (f" Missing: {', '.join(summary['missing'])}." if summary["missing"]
                                                 else ""),
             "Not part of exp09's pre-registered test; Holm over O1, O3, C2 only.", "",
             "## Contrasts (mean over model x rule cells, 95% CI over questions)", ""]
    lines += contrast_table(summary["contrasts"], models, with_holm=True)
    rules = e8.RULES
    lines += ["", "## Per rule", "", "| contrast | " + " | ".join(rules) + " |", "|---|" + "---|" * len(rules)]
    for name, v in summary["contrasts"].items():
        lines.append(f"| {name} | " + " | ".join(fmt(v["per_rule"].get(r)) for r in rules) + " |")
    lines += ["", "## Sensitivity: thinking off, rows with a closed tag block only", "",
              "Thinking-off rows without a closed <output_reasoning> block dropped (above, they are a violation at "
              "token 0).", ""]
    lines += contrast_table(summary["tagged_only"], models, with_holm=False)
    lines += ["", "## Cells", ""]
    desc = pd.DataFrame(summary["descriptives"]).fillna("-")
    lines += ["| " + " | ".join(desc.columns) + " |", "|" + "---|" * len(desc.columns)]
    lines += ["| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |" for r in desc.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    out_dir = OUT_ROOT / args.run
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; analysis runs are never overwritten")
    df, models = load(args.skip_missing)
    values = cell_values(df)
    results = contrasts(values, models)
    out_dir.mkdir(parents=True)
    pd.DataFrame([{"metric": k[0], "cell": k[1], "model": k[2], "mode": k[3], "value": p,
                   "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)}
                  for k, (p, d) in values.items()]).round(2).to_csv(out_dir / "per_cell.csv", index=False)
    summary = {"run": args.run, "models": models, "missing": [m for m in x.MODEL_ORDER if m not in models],
               "contrasts": results, "tagged_only": tagged_only(df, models),
               "descriptives": descriptives(df).to_dict("records")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "REPORT_auto.md").write_text(report(summary))
    print(report(summary))


if __name__ == "__main__":
    main()
