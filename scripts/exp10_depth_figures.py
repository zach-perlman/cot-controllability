"""Figures of exp10's depth stage (dev runs: dev_c2 + depth_pilot grades), for an outside reader:
  1 hidden_vs_latent   per model: the depth it reaches while hiding its reasoning vs the depth it reaches with no
                       reasoning at all (y = x: hiding reaches exactly as deep as no CoT)
  2 curves             accuracy by depth for a standard reasoner and for Ouro, per task: visible CoT, no CoT, best
                       filler, and the two hidden-reasoning conditions
  3 loops              Ouro's loop count vs accuracy on chain items of 4+ steps
  4 filler             no-CoT accuracy vs the length of dot filler: padding buys no latent depth
Depth d: the fit P(h) = c + (1-c) sigma(s (d - h)) of cc_exp10.fit_depth (c = the bank's chance), the step count
at which accuracy is halfway between chance and perfect; 95% intervals by a paired bootstrap over items (resampled
within each h, the same draw for every condition). Exploratory: dev items only (about 10-15 per depth).
Figures carry no item text.
  /venv/main/bin/python scripts/exp10_depth_figures.py [--bootstrap 200]
Output: results/exp10_hide_what_you_need/analysis/depth_dev_figures_<UTC stamp>/ (never overwritten).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import time
from pathlib import Path

import numpy as np

import cc_exp10 as E
import cc_exp10_tasks as X

TASKS = ("chain", "arithmetic")
TASK_LABEL = {"chain": "Chain (state machine: apply h steps to a number)",
              "arithmetic": "Arithmetic (evaluate an expression with h operations)"}
STEP_LABEL = {"chain": "steps h", "arithmetic": "operations h"}
NAME = re.compile(r"^(?P<model>.+?)__card__stream_abort_(?P<set>dev_c2|depth_pilot)_(?P=model)__\w+\.jsonl$")
FILLERS = ["F100", "F500", "F2000", "F8000"]
LATENT = ["C0", *FILLERS]  # no reasoning: plain, or with dot filler
HIDDEN = ["C4", "C4off", "Code"]  # reasoning under the number ban (primary reading: hidden_correct)
MODELS = {  # the full-loop models of the headline figure: display name, group
    "Gemma-4-31B-FP8": ("Gemma-4 31B", "standard"), "Qwen3.8-27B-FP8": ("Qwen3.8 27B", "standard"),
    "Qwen3-32B": ("Qwen3 32B", "standard"), "IQuest-40B-Loop-Thinking": ("IQuest 40B Loop", "looped"),
    "Nanbeige4.2-3B": ("Nanbeige4.2 3B", "looped"), "Ouro-1.4B-Thinking": ("Ouro 1.4B", "looped"),
    "Ouro-2.6B-Thinking": ("Ouro 2.6B", "looped")}
OURO_LOOPS = {"Ouro-2.6B-Thinking": "Ouro 2.6B", "Ouro-1.4B-Thinking": "Ouro 1.4B"}
TRAINED_LOOPS = 4
# Okabe-Ito, one colour per model in every figure; the marker shape gives the group
MODEL_COLOR = {"Gemma-4-31B-FP8": "#0072B2", "Qwen3.8-27B-FP8": "#56B4E9", "Qwen3-32B": "#009E73",
               "IQuest-40B-Loop-Thinking": "#CC79A7", "Nanbeige4.2-3B": "#8C6D31", "Ouro-1.4B-Thinking": "#E69F00",
               "Ouro-2.6B-Thinking": "#D55E00"}
GROUP_MARKER = {"standard": "o", "looped": "D"}
# axis range of figure 1: every depth fits inside it except Ouro 2.6B's chain point, drawn at the edge as ">="
AXIS_TOP = {"chain": 8.0, "arithmetic": 4.0}
COND_STYLE = {
    "C2": {"label": "visible CoT (no rule)", "color": "#000000", "ls": "-", "marker": "o"},
    "C0": {"label": "no CoT", "color": "#0072B2", "ls": "-", "marker": "s"},
    "best_filler": {"label": "no CoT + best dot filler", "color": "#56B4E9", "ls": "--", "marker": "s"},
    "C4": {"label": "hidden: reasoning on, numbers banned", "color": "#CC79A7", "ls": "-", "marker": "^"},
    "C4off": {"label": "hidden: scratchpad, numbers banned", "color": "#D55E00", "ls": "-", "marker": "D"},
}


# --- Data -----------------------------------------------------------------------------------------------------------
def load() -> dict:
    """{(model, task, condition): [grade rows]} from every dev_c2 and depth_pilot grade file."""
    rows = collections.defaultdict(list)
    for task in TASKS:
        for path in sorted(glob.glob(str(E.EXP.cache / task / "grades" / "*.jsonl"))):
            m = NAME.match(Path(path).name)
            if m:
                for g in map(json.loads, open(path)):
                    rows[(m["model"], task, g["condition"])].append(g)
    return rows


def outcome(condition: str, g: dict) -> bool:
    """C2 has no rule: plain accuracy. Every other condition: correct with no state leak (no-CoT rows: correct)."""
    return bool(g["correct"] if condition == "C2" else g["hidden_correct"])


def chance(task: str) -> float:
    return X.evals(task, "dev")[0]["chance"]


def max_h(task: str) -> int:
    return max(it["h"] for it in X.evals(task, "dev"))


def depth(rows: list[dict], condition: str, task: str) -> float:
    return E.fit_depth([g["h"] for g in rows], [outcome(condition, g) for g in rows], chance(task))["d"]


def bootstrap_draws(rows_by_cond: dict, n: int, seed: int) -> list[dict]:
    """n paired resamples: item ids drawn with replacement within each h (the same draw for every condition)."""
    rng = np.random.default_rng(seed)
    by_item = {c: {g["item_id"]: g for g in rows} for c, rows in rows_by_cond.items()}
    items = {}
    for rows in rows_by_cond.values():
        for g in rows:
            items.setdefault(g["h"], set()).add(g["item_id"])
    items = {h: sorted(ids) for h, ids in items.items()}
    draws = []
    for _ in range(n):
        picked = [i for h, ids in items.items() for i in rng.choice(ids, size=len(ids), replace=True)]
        draws.append({c: [by_item[c][i] for i in picked if i in by_item[c]] for c in rows_by_cond})
    return draws


def best_depth(rows_by_cond: dict, conditions: list[str], task: str) -> tuple[float, str]:
    """The largest fitted depth among `conditions` (the condition that gives the model its best chance)."""
    fits = {c: depth(rows_by_cond[c], c, task) for c in conditions if rows_by_cond.get(c)}
    best = max(fits, key=fits.get)
    return fits[best], best


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# --- Figures --------------------------------------------------------------------------------------------------------
def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    return plt


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.pdf")


def model_legend(plt, fig, **kw) -> None:
    handles = [plt.Line2D([], [], marker=GROUP_MARKER[group], ls="", ms=6, color=MODEL_COLOR[m],
                          label=f"{label} ({group})") for m, (label, group) in MODELS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5, **kw)


def fig_hidden_vs_latent(rows: dict, out: Path, n_boot: int) -> dict:
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    record = {}
    for ax, task in zip(axes, TASKS):
        top = AXIS_TOP[task]
        ax.plot([0, top], [0, top], color="#999999", ls="--", lw=1, zorder=0)
        ax.text(top * 0.97, top * 0.9, "y = x: hiding reaches\nexactly as deep as no CoT", ha="right", va="top",
                fontsize=8, color="#777777", rotation=0)
        ax.fill_between([0, top], [0, top], [top, top], color="#D55E00", alpha=0.05, zorder=0)
        ax.text(top * 0.3, top * 0.62, "hiding goes deeper\nthan no CoT", fontsize=8, color="#D55E00", va="top",
                ha="center")
        for model, (label, group) in MODELS.items():
            by_cond = {c: rows.get((model, task, c), []) for c in LATENT + HIDDEN}
            if not all(by_cond[c] for c in ["C0", "C4", "C4off"]):
                continue
            x, x_cond = best_depth(by_cond, LATENT, task)
            y, y_cond = best_depth(by_cond, HIDDEN, task)
            boots = [(best_depth(d, LATENT, task)[0], best_depth(d, HIDDEN, task)[0])
                     for d in bootstrap_draws(by_cond, n_boot, seed=hash((model, task)) % 2**32)]
            bx, by = np.array(boots).T
            x_lo, x_hi, y_lo, y_hi = *np.percentile(bx, [2.5, 97.5]), *np.percentile(by, [2.5, 97.5])
            clip = lambda v: min(v, top)
            ax.errorbar(clip(x), clip(y), xerr=[[clip(x) - clip(x_lo)], [clip(x_hi) - clip(x)]],
                        yerr=[[clip(y) - clip(y_lo)], [clip(y_hi) - clip(y)]], fmt=GROUP_MARKER[group], ms=7,
                        color=MODEL_COLOR[model], elinewidth=1, capsize=2, alpha=0.9, clip_on=False, zorder=3)
            if y > top:
                ax.annotate(f"{label}: ≥ {top:g}\n(deepest tested)", (clip(x), top), textcoords="offset points",
                            xytext=(10, -22), fontsize=8, color=MODEL_COLOR[model])
            record[f"{model} | {task}"] = {"latent_d": x, "latent_condition": x_cond, "latent_ci": [x_lo, x_hi],
                                           "hidden_d": y, "hidden_condition": y_cond, "hidden_ci": [y_lo, y_hi]}
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_title(TASK_LABEL[task].split(" (")[0] + (" (zoomed; items go to 12 operations)"
                                                         if task == "arithmetic" else ""))
        ax.set_xlabel(f"no-CoT depth ({STEP_LABEL[task]})\nbest of: no CoT, no CoT + dot filler")
        ax.set_ylabel(f"hidden-reasoning depth ({STEP_LABEL[task]})\nbest of: reasoning on, scratchpad, own code")
    model_legend(plt, fig, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Hidden reasoning reaches no deeper than no reasoning at all, except Ouro 2.6B on chain",
                 y=1.11, fontweight="bold")
    fig.text(0.5, -0.06, "Depth = number of steps at which accuracy is halfway between chance and perfect (fitted). "
             "Hidden = correct with no intermediate number written.\nWith visible CoT and no rule, every model stays "
             "at 57% or more on the three deepest depths tested. Bars: 95% bootstrap intervals; dev items only "
             "(~14 per depth).",
             ha="center", fontsize=8, color="#555555")
    save(fig, out, "1_hidden_vs_latent")
    return record


def curve(ax, rows: list[dict], condition: str, style_key: str, band: bool) -> None:
    by_h = collections.defaultdict(list)
    for g in rows:
        by_h[g["h"]].append(outcome(condition, g))
    hs = sorted(by_h)
    acc = [np.mean(by_h[h]) for h in hs]
    s = COND_STYLE[style_key]
    ax.plot(hs, acc, color=s["color"], ls=s["ls"], marker=s["marker"], ms=4, lw=1.6, label=s["label"])
    if band:
        lo, hi = zip(*[wilson(sum(by_h[h]), len(by_h[h])) for h in hs])
        ax.fill_between(hs, lo, hi, color=s["color"], alpha=0.12, lw=0)


def fig_curves(rows: dict, out: Path) -> None:
    plt = style()
    shown = ["Gemma-4-31B-FP8", "Ouro-2.6B-Thinking"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for i, task in enumerate(TASKS):
        for j, model in enumerate(shown):
            ax = axes[i, j]
            by_cond = {c: rows.get((model, task, c), []) for c in ["C2", *LATENT, "C4", "C4off"]}
            _, filler = best_depth(by_cond, FILLERS, task)
            curve(ax, by_cond["C2"], "C2", "C2", band=False)
            curve(ax, by_cond["C0"], "C0", "C0", band=True)
            curve(ax, by_cond[filler], filler, "best_filler", band=False)
            curve(ax, by_cond["C4"], "C4", "C4", band=False)
            curve(ax, by_cond["C4off"], "C4off", "C4off", band=True)
            ax.axhline(chance(task), color="#999999", ls=":", lw=1)
            ax.text(max_h(task), chance(task) + 0.02, "chance", ha="right", fontsize=7, color="#777777")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{MODELS[model][0]} · {task}")
            ax.set_xlabel(STEP_LABEL[task])
            if j == 0:
                ax.set_ylabel("accuracy (hidden: correct and\nno intermediate number written)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle("Visible CoT holds up with depth; hidden reasoning falls like no CoT, except Ouro's scratchpad "
                 "on chain", y=1.08, fontweight="bold")
    fig.text(0.5, -0.04, "~14 dev items per point; shaded: 95% Wilson intervals for no CoT and the hidden scratchpad. "
             "Ouro's scratchpad writes the procedure with every number\nreplaced by a placeholder, but still states "
             "which branch each step takes (odd/even, above/below). Chain h = 1 has a wording trap that small models "
             "fall for.", ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    save(fig, out, "2_accuracy_by_depth")


def fig_loops(rows: dict, out: Path) -> dict:
    plt = style()
    fig, axes = plt.subplots(1, len(OURO_LOOPS), figsize=(9, 3.8), sharey=True)
    record = {}
    for ax, (base, label) in zip(axes, OURO_LOOPS.items()):
        for cond, key in (("C2", "C2"), ("C4off", "C4off"), ("C0", "C0")):
            xs, ys, los, his = [], [], [], []
            for n in range(1, TRAINED_LOOPS + 1):
                model = base if n == TRAINED_LOOPS else f"{base}-loop{n}"
                deep = [g for g in rows.get((model, "chain", cond), []) if g["h"] >= 4]
                if not deep:
                    continue
                k = sum(outcome(cond, g) for g in deep)
                lo, hi = wilson(k, len(deep))
                xs.append(n), ys.append(k / len(deep)), los.append(lo), his.append(hi)
                record[f"{model} | {cond}"] = {"k": k, "n": len(deep)}
            s = COND_STYLE[key]
            ax.errorbar(xs, ys, yerr=[np.array(ys) - los, np.array(his) - ys], color=s["color"], marker=s["marker"],
                        ls=s["ls"], lw=1.6, ms=5, capsize=2, label=s["label"])
        ax.axhline(chance("chain"), color="#999999", ls=":", lw=1)
        ax.set_xticks(range(1, TRAINED_LOOPS + 1))
        ax.set_xlabel("recurrent passes per token (trained: 4)")
        ax.set_title(label)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("accuracy, chain items of 4-8 steps")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle("Cutting Ouro's recurrent passes lowers visible CoT and the hidden scratchpad alike; no CoT stays "
                 "at chance",
                 y=1.13, fontweight="bold")
    fig.text(0.5, -0.06, "Same checkpoint run with fewer loop passes. 1-2 passes often degenerate (repetitive text). "
             "Bars: 95% Wilson intervals; dev items only.", ha="center", fontsize=8, color="#555555")
    save(fig, out, "3_loops")
    return record


def fig_filler(rows: dict, out: Path) -> dict:
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    doses = [0, 100, 500, 2000, 8000]
    record = {}
    for ax, task in zip(axes, TASKS):
        for model, (label, group) in MODELS.items():
            ds = []
            for cond in LATENT:
                r = rows.get((model, task, cond), [])
                ds.append(depth(r, cond, task) if r else np.nan)
            if np.all(np.isnan(ds)):
                continue
            record[f"{model} | {task}"] = dict(zip(LATENT, ds))
            ax.plot(range(len(doses)), ds, marker=GROUP_MARKER[group], ms=4, lw=1.4, color=MODEL_COLOR[model],
                    ls="-" if group == "standard" else "--")
        ax.set_xticks(range(len(doses)), ["none", "100", "500", "2000", "8000"])
        ax.set_xlabel("dot filler before the answer (tokens)")
        ax.set_ylabel(f"no-CoT depth ({STEP_LABEL[task]})")
        ax.set_title(task.capitalize())
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    model_legend(plt, fig, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Dot filler buys no extra depth: no-CoT depth stays flat or falls as the filler grows",
                 y=1.2, fontweight="bold")
    fig.text(0.5, -0.06, "Solid: standard; dashed: looped. Depth fitted as in figure 1 (0 = at chance even for "
             "one step); dev items only. 8000 dots push the small looped models to chance.",
             ha="center", fontsize=8, color="#555555")
    save(fig, out, "4_filler")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    out = E.EXP.results / "analysis" / f"depth_dev_figures_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    out.mkdir(parents=True, exist_ok=False)
    rows = load()
    record = {"bootstrap": args.bootstrap, "hidden_vs_latent": fig_hidden_vs_latent(rows, out, args.bootstrap)}
    fig_curves(rows, out)
    record["loops"] = fig_loops(rows, out)
    record["filler"] = fig_filler(rows, out)
    (out / "plotted.json").write_text(json.dumps(record, indent=1, default=float) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
