"""Analysis of exp09's extension 2 (results/exp09_final_test/manifest_extension2.json): thinking on vs off on all 12
rules and 7 models, and exp08's openings on 7 rules.

Every cell is (arm label "<arm>|<opening>|<channel>", model, rule). The thinking-on |none cells are exp09's short
rows (A|short -> A|none|on, baseline|short -> baseline|none|on). A contrast is the mean over its model x rule cells of
(cell a - cell b), with exp09's paired bootstrap over questions (stratified by source; cc_exp09_analysis
.question_draws) and a two-sided bootstrap p; Holm over C1, C2, O1, O3.
  channels   S(200), thinking on and off
  openings   S(1000) of the graded continuation after the opening
Thinking-off responses are capped at 1200 tokens (deviations.json; request files requests_ext2c_<M>), so thinking-off
accuracy is not estimable and its contrasts (ACC_A, ACC_baseline) are dropped. A tag block that never closes (cut at
the cap, or ended without its closing tag) is graded on its text so far (cc_exp09_ext2.unclosed_block_reasoning).
Every cell is restricted to the 80-question subset the extension-2 rows cover (cc_exp09_ext2.subset_items;
deviations.json), with exp09's bootstrap (same seed, stratified by source) over those 80.
Sensitivity: C2 and C4 on the cells where A off has all of exp09's A examples (same_examples); C1, C2, C4 on
thinking-off rows whose tag blocks all close (tagged_only), with unclosed blocks read as empty, a violation at token 0
(strict_tags: cc_exp06's reading), and without rows with a hidden reasoning block (no_hidden_reasoning); every
contrast without exp06's 5 rules (held_out_only).

Run: /venv/main/bin/python scripts/cc_exp09_ext2_analysis.py --run NAME [--skip-missing]
Writes results/exp09_final_test/analysis_ext2/<run>/ (never overwritten).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import cc_exp07 as e7
import cc_exp09 as e9
import cc_exp09_ext2 as x
import exp07_rules as r7
import exp09_rules as R
from cc_exp08_analysis import GUARDS, contrast, fmt
from cc_exp09_analysis import BOOT_SEED, N_BOOT, question_draws
from cc_survival import holm, kaplan_meier

OUT_ROOT = e9.EXP.results / "analysis_ext2"
T_VALUES = np.array([200, 1000])
RULE_SETS = {"exp06": r7.SEARCH_RULES, "held_out_exp07": r7.HELDOUT_RULES, "new_exp09": R.NEW_RULES}
NO_NONCOMPLIANT = ["word_suppression"]
# name: (metric, cell a, cell b, rules, primary)
CONTRASTS = {
    "C1 thinking off - on (CoT-Control prompt)": ("S_200", "baseline|none|off", "baseline|none|on", R.ALL_RULES, True),
    "C2 thinking off - on (A)": ("S_200", "A|none|off", "A|none|on", R.ALL_RULES, True),
    "O1 compliant opening - none (A)": ("S_1000", "A|compliant|on", "A|none|on", x.OPENING_RULES, True),
    "O3 commitment - compliant opening (A)": ("S_1000", "A|commitment|on", "A|compliant|on", x.OPENING_RULES, True),
    "C3 A - CoT-Control prompt, thinking on": ("S_200", "A|none|on", "baseline|none|on", R.ALL_RULES, False),
    "C4 A - CoT-Control prompt, thinking off": ("S_200", "A|none|off", "baseline|none|off", R.ALL_RULES, False),
    "O2 rule + opening - opening only": ("S_1000", "A|compliant|on", "no_rule|compliant|on", x.OPENING_RULES, False),
    "O4 non-compliant opening - none (A)": ("S_1000", "A|noncompliant|on", "A|none|on",
                                            [r for r in x.OPENING_RULES if r not in NO_NONCOMPLIANT], False),
}
SAME_EXAMPLES = ["C2 thinking off - on (A)", "C4 A - CoT-Control prompt, thinking off"]
TAGGED_ONLY = ["C1 thinking off - on (CoT-Control prompt)", "C2 thinking off - on (A)",
               "C4 A - CoT-Control prompt, thinking off"]


# --- Data -----------------------------------------------------------------------------------------------------------
def subset_ids() -> set[str]:
    return {it["item_id"] for it in x.subset_items()}


def subset_question_draws() -> tuple[dict, np.ndarray]:
    """question_draws over the 80-question subset (same seed, questions resampled within each source)."""
    items = x.subset_items()
    index = {it["item_id"]: k for k, it in enumerate(items)}
    rng = np.random.default_rng(BOOT_SEED)
    weights = np.zeros((N_BOOT, len(items)))
    for source in sorted({it["source"] for it in items}):
        cols = [k for k, it in enumerate(items) if it["source"] == source]
        weights[:, cols] = rng.multinomial(len(cols), np.ones(len(cols)) / len(cols), size=N_BOOT)
    return index, weights


def extension2_grades(model: str):
    """The model's capped extension-2 grades (80-question file), or None."""
    try:
        path = x.grades_path(model, x.CAPPED_STEM + x.SUBSET_SUFFIX)
    except SystemExit:
        return None
    return path if path.exists() else None


def load(skip_missing: bool, subset: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Extension 2's grades and exp09's short rows (all 12 rules), with survival columns; subset: only the 80
    questions every model's extension-2 rows cover."""
    frames, present = [], []
    for model in x.MODEL_ORDER:
        try:
            ext, main = extension2_grades(model), e9.grades_path(model)
        except SystemExit:
            ext = main = None
        if ext is None or not ext.exists() or not main.exists():
            if skip_missing:
                continue
            raise SystemExit(f"{model}: no extension-2 or exp09 grades (pass --skip-missing)")
        frames.append(pd.DataFrame([json.loads(line) for line in ext.open()]).assign(model=model))
        short = pd.DataFrame([json.loads(line) for line in main.open()]).assign(model=model)
        short = short[short["channel"] == "short"].copy()
        short["cell"] = short["prompt"] + "|none|on"
        short["channel"] = "on"
        frames.append(short)
        present.append(model)
    if not frames:
        raise SystemExit("no graded model")
    df = pd.concat(frames, ignore_index=True)
    if subset:
        df = df[df["item_id"].isin(subset_ids())]
    return e7.survival(df), present


def same_example_cells(models: list[str]) -> set[tuple[str, str]]:
    """(model, rule) cells where A thinking off has all of exp09's thinking-on examples (the request records)."""
    out = set()
    for m in models:
        record = json.loads((e9.EXP.results / f"requests_record_{x.CAPPED_STEM}_{m}.json").read_text())
        out |= {(m, rule) for rule, r in record["rules"].items() if r["same_examples"]}
    return out


def cell_values(df: pd.DataFrame, subset: bool = True) -> dict:
    """(metric, cell, model, rule) -> (point, bootstrap draws); subset: draws over the 80 questions (df holds only
    them), else over exp09's 120."""
    index, weights = subset_question_draws() if subset else question_draws()
    out = {}
    for (cell, model, mode), c in df.groupby(["cell", "model", "mode"]):
        w = weights[:, c["item_id"].map(index).to_numpy()]
        times, events = c["time"].to_numpy(), c["event"].to_numpy().astype(bool)
        point = kaplan_meier(times, events, np.ones((1, len(c))), T_VALUES)[0]
        draws = kaplan_meier(times, events, w, T_VALUES)
        out[("S_200", cell, model, mode)] = (100 * point[0], 100 * draws[:, 0])
        out[("S_1000", cell, model, mode)] = (100 * point[1], 100 * draws[:, 1])
    return out


def unclosed(df: pd.DataFrame) -> pd.Series:
    """Thinking-off rows graded on a tag block that never closed."""
    return (df["channel"] == "off") & df["unclosed"].notna()


def strict_tags(df: pd.DataFrame) -> pd.DataFrame:
    """cc_exp06's reading: an unclosed tag block has no text, a violation at token 0."""
    d = df.copy()
    d.loc[unclosed(d), "reasoning_tokens"] = 0
    return e7.survival(d)


def with_breakdowns(values: dict, name: str, cells: list[tuple[str, str]], models: list[str]) -> dict | None:
    metric, a, b, rules, primary = CONTRASTS[name]
    res = contrast(values, metric, a, b, cells)
    if res is not None:
        res.update(metric=metric, a=a, b=b, primary=primary,
                   per_model={m: contrast(values, metric, a, b, [c for c in cells if c[0] == m]) for m in models},
                   per_rule={r: contrast(values, metric, a, b, [c for c in cells if c[1] == r]) for r in rules},
                   per_rule_set={s: contrast(values, metric, a, b, [c for c in cells if c[1] in rs])
                                 for s, rs in RULE_SETS.items()},
                   per_model_group={g: contrast(values, metric, a, b, [c for c in cells if c[0] in ms])
                                    for g, ms in (("design", e9.DESIGN_MODELS), ("fresh", e9.FRESH_MODELS))})
    return res


def all_cells(name: str, models: list[str]) -> list[tuple[str, str]]:
    return [(m, r) for m in models for r in CONTRASTS[name][3]]


def contrasts(values: dict, models: list[str]) -> dict:
    out = {n: r for n in CONTRASTS if (r := with_breakdowns(values, n, all_cells(n, models), models)) is not None}
    for n, p in holm({n: v["p_two_sided"] for n, v in out.items() if v["primary"]}).items():
        out[n]["p_holm"] = p
    return out


def sensitivity(df: pd.DataFrame, values: dict, models: list[str]) -> dict:
    same = same_example_cells(models)
    tagged_values = cell_values(df[~unclosed(df)])
    strict_values = cell_values(strict_tags(df))
    held_out = set(r7.HELDOUT_RULES) | set(R.NEW_RULES)
    no_hidden = (df["channel"] != "off") | (df["hidden_reasoning_tokens"].fillna(0) == 0)
    no_hidden_values = cell_values(df[no_hidden])
    return {
        "strict_tags": {n: r for n in TAGGED_ONLY if (r := with_breakdowns(
            strict_values, n, all_cells(n, models), models)) is not None},
        "no_hidden_reasoning": {n: r for n in TAGGED_ONLY if (r := with_breakdowns(
            no_hidden_values, n, all_cells(n, models), models)) is not None},
        "same_examples": {n: r for n in SAME_EXAMPLES if (r := with_breakdowns(
            values, n, [c for c in all_cells(n, models) if c in same], models)) is not None},
        "tagged_only": {n: r for n in TAGGED_ONLY if (r := with_breakdowns(
            tagged_values, n, all_cells(n, models), models)) is not None},
        "held_out_only": {n: r for n in CONTRASTS if (r := with_breakdowns(
            values, n, [c for c in all_cells(n, models) if c[1] in held_out], models)) is not None},
    }


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
            row["unclosed_at_cap"] = round(100 * float((c["unclosed"] == "cap").mean()), 1)
            row["unclosed_at_stop"] = round(100 * float((c["unclosed"] == "stop").mean()), 1)
            row["hidden_reasoning_block"] = round(100 * float((c["hidden_reasoning_tokens"].fillna(0) > 0).mean()), 1)
            row["outside_tag_chars_median"] = int(c["outside_tag_chars"].median())
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


def breakdown_table(results: dict, key: str, columns: list[str]) -> list[str]:
    lines = ["| contrast | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for name, v in results.items():
        lines.append(f"| {name} | " + " | ".join(fmt(v[key].get(c)) for c in columns) + " |")
    return lines


def report(summary: dict) -> str:
    models = summary["models"]
    lines = [f"# exp09 extension 2 (thinking on/off everywhere, openings): {summary['run']} "
             "(UNVERIFIED until a human adds it to VERIFIED.md)", "",
             f"Models: {', '.join(models)}." + (f" Missing: {', '.join(summary['missing'])}." if summary["missing"]
                                                 else ""),
             "Not part of exp09's pre-registered test; Holm over C1, C2, O1, O3 only. Survival is a rule-grader "
             "metric (no LLM judge). Thinking-off responses are capped at 1200 tokens (no thinking-off accuracy).", "",
             "## Contrasts (mean over model x rule cells, 95% CI over questions)", ""]
    lines += contrast_table(summary["contrasts"], models, with_holm=True)
    lines += ["", "## Per rule set", ""] + breakdown_table(summary["contrasts"], "per_rule_set", list(RULE_SETS))
    lines += ["", "## Per model group", ""] + breakdown_table(summary["contrasts"], "per_model_group",
                                                              ["design", "fresh"])
    lines += ["", "## Per rule", ""] + breakdown_table(summary["contrasts"], "per_rule", R.ALL_RULES)
    for name, title in (("strict_tags", "an unclosed tag block read as empty (violation at token 0)"),
                        ("no_hidden_reasoning", "thinking-off rows without a hidden reasoning block only"),
                        ("same_examples", "A thinking off with all of exp09's examples only"),
                        ("tagged_only", "thinking-off rows whose tag blocks all close only"),
                        ("held_out_only", "without exp06's 5 rules (A was chosen on them)")):
        lines += ["", f"## Sensitivity: {title}", ""] + contrast_table(summary["sensitivity"][name], models, False)
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
    out_dir.mkdir(parents=True)
    pd.DataFrame([{"metric": k[0], "cell": k[1], "model": k[2], "mode": k[3], "value": p,
                   "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)}
                  for k, (p, d) in values.items()]).round(2).to_csv(out_dir / "per_cell.csv", index=False)
    summary = {"run": args.run, "models": models, "missing": [m for m in x.MODEL_ORDER if m not in models],
               "contrasts": contrasts(values, models), "sensitivity": sensitivity(df, values, models),
               "descriptives": descriptives(df).to_dict("records")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "REPORT_auto.md").write_text(report(summary))
    print(report(summary))


if __name__ == "__main__":
    main()
