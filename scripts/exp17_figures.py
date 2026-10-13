"""exp17 figures (no model). Every number is read from results/exp17_jspace_keys/<run>/ or cache/exp17/<run>/trials_*.

  fig_layers.png     readable vs used, per model (a row each): (left) top-1 rate among the 10 colours at the final
                     position, per layer: the J++ lens reading the key s1 and the answer, and the key direction's own
                     cross-fitted readout of s1 (readout.json); (right) per band (drawn at its middle layer), dev
                     success of the key-direction swap, the J++ key swap and the J++ answer swap at each band's best
                     alpha, with the matched random null of the key-direction swap (trials_dev; the answer swap
                     from trials_answer_early and trials_late where they exist). Gemma's swaps: run gemma_occupied
  fig_controls.png   the factual positive control per band and alpha: J++ intermediate swap, the same with the answer
                     pair's span removed (matched size, Qwen), and the random null (trials_posctl, trials_posctl_orth)
  fig_lens.png       the share of each J++ colour atom's norm inside the subspace the residual stream occupies, per
                     layer, Qwen vs Gemma (occupied_subspace.json)
  fig_test.png       the locked decomposition arms on test items, the dose-response and its colour mass
                     (trials_test); fig_preview the same arms on dev items at the dev selection (trials_preview)
  fig_patch.png      interchange patching on dev per band, window and subspace (trials_patch_<window>); fig_test_patch
                     the same on test items once they exist

  python exp17_figures.py --out dev_v1 [--models qwen gemma]
-> figures/exp17_jspace_keys/<out>/ (folder must be new)
"""
from __future__ import annotations

import argparse
import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cc_config as cfg  # noqa: E402
import exp15_score as S  # noqa: E402

RESULTS = cfg.REPO_ROOT / "results" / "exp17_jspace_keys"
CACHE = cfg.REPO_ROOT / "cache" / "exp17"
FIGURES = cfg.REPO_ROOT / "figures" / "exp17_jspace_keys"
LABEL = {"qwen": "Qwen3.6-27B", "gemma": "Gemma-4-31B-it"}
SLIDING_BANDS = {"qwen": ("b24_36", "b30_42", "b36_48", "b42_54", "b48_60", "b54_62"),
                 "gemma": ("b22_34", "b28_40", "b34_46", "b40_52", "b46_58", "b52_58")}
KEY, JPP, ANSWER, NULL = "#1b6ca8", "#e08214", "#4daf4a", "#7f7f7f"


def trials(run: str, phase: str) -> dict[str, dict[str, dict]] | None:
    """arm -> item_id -> record, on the items whose in-batch clean pass gives gold (brew) or the answer (factual)."""
    path = CACHE / run / f"trials_{phase}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    def clean_correct(r: dict) -> bool:
        return r["outcome"] == "unchanged" if "outcome" in r else r["kept"]

    keep = {r["item_id"] for r in rows if r["arm"] == "clean" and clean_correct(r)}
    out = collections.defaultdict(dict)
    for r in rows:
        if r["arm"] != "clean" and r["item_id"] in keep:
            out[r["arm"]][r["item_id"]] = r
    return out


def success(recs: dict) -> list[float]:
    return [float(r["outcome"] == "success") if "outcome" in r else float(r["success"]) for r in recs.values()]


def rate_ci(recs: dict) -> tuple[float, float, float]:
    m, lo, hi, _ = S.boot_ci(success(recs))
    return m, lo, hi


def band_mid(band: str) -> float:
    lo, hi = (int(x) for x in band[1:].split("_"))
    return (lo + hi) / 2


def best_alpha(arms: dict, prefix: str, band: str) -> str | None:
    """The arm name prefix|band|alpha with the highest success rate (dev: a summary, not a selection)."""
    names = [a for a in arms if a.startswith(f"{prefix}|{band}|")]
    return max(names, key=lambda a: sum(success(arms[a])) / len(arms[a])) if names else None


def causal_run(model: str) -> str:
    """The run whose swap and patch trials stand for the model: Gemma's raw J++ vectors lie mostly outside the
    occupied subspace (fig_lens), so its J++ arms are the occupied-subspace run's."""
    return {"qwen": "qwen", "gemma": "gemma_occupied"}[model]


def draw_rates(ax, xs, cis, colour, label, marker="o"):
    xs, cis = list(xs), list(cis)
    if not xs:
        return
    ax.errorbar(xs, [c[0] for c in cis], yerr=[[c[0] - c[1] for c in cis], [c[2] - c[0] for c in cis]],
                color=colour, marker=marker, ms=5, capsize=2, lw=1.4, label=label)


# --- fig_layers -----------------------------------------------------------------------------------------------------
def fig_layers(models: list[str], out) -> None:
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.9 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        ax_r, ax_c = axes[row]
        ro = json.loads((RESULTS / model / "readout.json").read_text())
        layers = sorted(int(l) for l in ro["by_layer"])
        series = [("dm_cv_s1", KEY, "key s1: key direction's own readout (cross-fitted)"),
                  ("jpp_s1", JPP, "key s1: J++ lens"), ("jpp_gold", ANSWER, "answer: J++ lens")]
        for name, colour, label in series:
            pts = [(l, ro["by_layer"][str(l)][name]) for l in layers if name in ro["by_layer"][str(l)]]
            ax_r.plot([p[0] for p in pts], [p[1][0] for p in pts], color=colour, lw=1.6, label=label)
            ax_r.fill_between([p[0] for p in pts], [p[1][1] for p in pts], [p[1][2] for p in pts], color=colour,
                              alpha=0.15, lw=0)
        ax_r.axhline(ro["chance"], color=NULL, ls=":", lw=1, label="chance (1 of 10)")
        ax_r.set_ylim(0, 1.02)
        ax_r.set_xlabel("layer", fontsize=9)
        ax_r.set_ylabel("top-1 among the 10 colours", fontsize=9)
        ax_r.set_title(f"{LABEL[model]}: what is readable at the final position (dev, n={ro['n_eval']})",
                       fontsize=9, loc="left")
        ax_r.legend(fontsize=7, frameon=False, loc="upper left")

        run = causal_run(model)
        dev = trials(run, "dev")
        if dev is None:
            ax_c.text(0.5, 0.5, f"{LABEL[model]}: J++ swaps pending\n(occupied-subspace rerun)", ha="center",
                      va="center", fontsize=10, transform=ax_c.transAxes, color=NULL)
            ax_c.set_axis_off()
            continue
        # the answer swap's exp17_late trials (alpha 1-8, with nulls) where they exist; else the dev grid's (alpha 1-2)
        answer_sources = [t for t in (trials(run, "answer_early"), trials(run, "late")) if t] + [dev]
        for prefix, colour, label in (("dm", KEY, "key-direction swap s1 -> s1'"),
                                      ("jpp", JPP, "J++ key swap s1 -> s1'"),
                                      ("jpp_answer", ANSWER, "J++ answer swap gold -> a'")):
            pts = []
            for band in SLIDING_BANDS[model]:
                sources = answer_sources if prefix == "jpp_answer" else [dev]
                source = next((s for s in sources if best_alpha(s, prefix, band)), None)
                if source is not None:
                    pts.append((band_mid(band), rate_ci(source[best_alpha(source, prefix, band)])))
            draw_rates(ax_c, [p[0] for p in pts], [p[1] for p in pts], colour, label)
        nulls = [(band_mid(b), rate_ci(dev[f"null_dm|{b}|{best_alpha(dev, 'dm', b).split('|')[-1]}"]))
                 for b in SLIDING_BANDS[model] if best_alpha(dev, "dm", b)]
        draw_rates(ax_c, [p[0] for p in nulls], [p[1] for p in nulls], NULL, "random direction, key swap's size",
                   marker="s")
        ax_c.set_ylim(-0.02, 1.02)
        ax_c.set_xlim(*ax_r.get_xlim())
        n = len(dev[f"dm|{SLIDING_BANDS[model][0]}|1"])
        ax_c.set_xlabel("edited band (drawn at its middle layer)", fontsize=9)
        ax_c.set_ylabel("success rate (best alpha per band)", fontsize=9)
        lens = "\nJ++ vectors projected onto the occupied subspace" if run.endswith("_occupied") else ""
        ax_c.set_title(f"{LABEL[model]}: what changes the output when swapped (dev, n={n}){lens}", fontsize=9,
                       loc="left")
        ax_c.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("exp17: the key is readable but its used copy is outside J++; the answer enters J++ late "
                 "(dev items; UNVERIFIED)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig_layers.png", dpi=150)
    plt.close(fig)


# --- fig_controls ---------------------------------------------------------------------------------------------------
def band_label(band: str) -> str:
    return band.replace("b", "L").replace("_", "-")


def bars(ax, groups: list[str], series: list[tuple[dict, str, str]], width: float) -> None:
    """Grouped bars with 95% CIs: per (arms by group, colour, label) one bar per group; a 0 is written on empty bars
    so that a zero rate is not mistaken for a missing arm."""
    for k, (by_group, colour, label) in enumerate(series):
        offset = (k - (len(series) - 1) / 2) * width
        xs = [i + offset for i, g in enumerate(groups) if g in by_group]
        cis = [rate_ci(by_group[g]) for g in groups if g in by_group]
        ax.bar(xs, [c[0] for c in cis], width, color=colour, edgecolor="black", lw=0.5, label=label)
        ax.errorbar(xs, [c[0] for c in cis], yerr=[[c[0] - c[1] for c in cis], [c[2] - c[0] for c in cis]],
                    fmt="none", color="black", capsize=2, lw=0.8)
        for x, c in zip(xs, cis):
            if c[0] == 0:
                ax.text(x, 0.01, "0", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(groups)), [band_label(g) for g in groups], fontsize=8)


def fig_controls(models: list[str], out) -> None:
    runs = [r for r in ("qwen", "gemma_occupied") if r.split("_")[0] in models and trials(r, "posctl")]
    alphas = ("1", "2", "4")
    fig, axes = plt.subplots(len(runs), len(alphas), figsize=(12, 3.4 * len(runs)), squeeze=False, sharey=True)
    for row, run in enumerate(runs):
        pos, orth = trials(run, "posctl"), trials(run, "posctl_orth") or {}
        bands = sorted({a.split("|")[1] for a in pos if a.startswith("inter|")}, key=band_mid)
        for col, alpha in enumerate(alphas):
            ax = axes[row][col]
            series = [({b: pos[f"inter|{b}|{alpha}"] for b in bands}, JPP, "J++ intermediate swap"),
                      ({b: orth[f"inter_orth_m|{b}|{alpha}"] for b in bands if f"inter_orth_m|{b}|{alpha}" in orth},
                       ANSWER, "same, answer pair's span removed (matched size)"),
                      ({b: pos[f"null|{b}|{alpha}"] for b in bands}, NULL, "random direction, same size")]
            bars(ax, bands, [s for s in series if s[0]], width=0.27)
            n = len(pos[f"inter|{bands[0]}|{alpha}"])
            prompt = "chat template" if run == "gemma_occupied" else "raw text"
            ax.set_title(f"{LABEL[run.split('_')[0]]}, alpha {alpha} ({prompt}, n={n} prompts)", fontsize=9,
                         loc="left")
            if col == 0:
                ax.set_ylabel("answer becomes swap_answer", fontsize=9)
                ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("exp17 positive control: the paper's two-hop factual prompts, clamped J++ swap at every position "
                 "(UNVERIFIED)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_controls.png", dpi=150)
    plt.close(fig)


# --- fig_lens -------------------------------------------------------------------------------------------------------
def fig_lens(models: list[str], out) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for model, colour in zip(models, (KEY, JPP)):
        occ = json.loads((RESULTS / model / "occupied_subspace.json").read_text())["by_layer"]
        layers = sorted(int(l) for l in occ)
        ax.plot(layers, [occ[str(l)]["kept_colour_mean"] for l in layers], color=colour, marker="o", ms=3,
                label=f"{LABEL[model]}: J++ colour atoms")
        ax.plot(layers, [occ[str(l)]["brew_energy"] for l in layers], color=colour, ls="--", lw=1,
                label=f"{LABEL[model]}: brew residuals (energy share)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("layer", fontsize=9)
    ax.set_ylabel("share inside the occupied subspace", fontsize=9)
    ax.set_title("Occupied subspace: top eigenvectors holding 99.99% of WikiText residual energy", fontsize=9,
                 loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_lens.png", dpi=150)
    plt.close(fig)


# --- fig_test -------------------------------------------------------------------------------------------------------
TEST_GROUPS = (("key direction (dm) band", (("dm_full", KEY, "dm, full"), ("dm_rem", "#92c5de", "dm remainder"),
                                             ("dm_jpart", "#fdb863", "dm J++ part"),
                                             ("jpp_matched", JPP, "J++ colour vectors"),
                                             ("null_dm", NULL, "random"))),
               ("J++ band", (("jpp_raw", JPP, "J++ colour vectors"), ("null_jpp", NULL, "random"))),
               ("NLA reconstructor layer", (("ar_full", KEY, "NLA, full"), ("ar_rem", "#92c5de", "NLA remainder"),
                                            ("ar_jpart", "#fdb863", "NLA J++ part"), ("null_ar", NULL, "random"))),
               ("in-task control band", (("jpp_answer_ctl", ANSWER, "J++ answer swap"),
                                         ("null_answer_ctl", NULL, "random"))))
DOSE_ARMS = (("dm_jpart", "#fdb863", "dm J++ part"), ("jpp_matched", JPP, "J++ colour vectors"),
             ("null_dm", NULL, "random"))


def disrupted(recs: dict) -> dict:
    """The same records with outcome 'success' when the answer is neither gold nor the target: collateral damage."""
    return {i: {"outcome": "success" if r["outcome"] not in ("unchanged", "success") else "no"}
            for i, r in recs.items()}


def fig_test(models: list[str], out, phase: str = "test") -> None:
    """Per run: (left) success of each locked arm, grouped by where it edits, at the locked settings (all arms in a
    group at the same edit size; the in-task control's success is the answer swap's: a' said); (middle)
    dose-response: key swap success at 1x, 2x, 4x the full key-direction swap's edit size; (right) collateral damage
    at the same doses: the answer changes, but not to T(s1')."""
    runs = [causal_run(m) for m in models if trials(causal_run(m), phase)]
    if not runs:
        return
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 4.4 * len(runs)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.4, 1, 1]})
    for row, run in enumerate(runs):
        arms = trials(run, phase)
        ax = axes[row][0]
        x, ticks, groups = 0.0, [], []
        for group, members in TEST_GROUPS:
            present = [m for m in members if m[0] in arms]
            if not present:
                continue
            start = x
            for name, colour, label in present:
                m, lo, hi = rate_ci(arms[name])
                ax.bar(x, m, 0.8, color=colour, edgecolor="black", lw=0.5)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", color="black", capsize=2, lw=0.8)
                ticks.append((x, label))
                x += 1
            groups.append(((start + x - 1) / 2, group))
            x += 0.8
        ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7, rotation=35, ha="right")
        for gx, group in groups:
            ax.text(gx, 1.04, group, ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("swap success", fontsize=9)
        n = len(arms["dm_full"])
        ax.set_title(f"{LABEL[run.split('_')[0]]}: locked arms ({phase} items, n={n})", fontsize=9, loc="left")
        for col, (transform, ylabel) in enumerate(((None, "key swap success: answer T(s1')"),
                                                   (disrupted, "answer changed, not to T(s1')")), 1):
            axd = axes[row][col]
            for name, colour, label in DOSE_ARMS:
                pts = [(s, arms[name if s == 1 else f"{name}_x{s}"]) for s in (1, 2, 4)
                       if (name if s == 1 else f"{name}_x{s}") in arms]
                draw_rates(axd, [p[0] for p in pts], [rate_ci(transform(p[1]) if transform else p[1]) for p in pts],
                           colour, label)
            full = arms["dm_full"]
            axd.axhline(rate_ci(transform(full) if transform else full)[0], color=KEY, ls="--", lw=1,
                        label="dm, full (1x)")
            axd.set_xscale("log", base=2)
            axd.set_xticks([1, 2, 4], ["1x", "2x", "4x"])
            axd.set_ylim(-0.02, 1.02)
            axd.set_xlabel("edit size (x the full dm swap's)", fontsize=9)
            axd.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                axd.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"exp17 locked decomposition ({phase}; UNVERIFIED)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"fig_{phase}.png", dpi=150)
    plt.close(fig)


# --- fig_patch ------------------------------------------------------------------------------------------------------
PATCH_SERIES = (("full", "#404040", "whole residual"), ("dm", KEY, "key directions (dm, 10-d)"),
                ("dm_jpart", "#fdb863", "their J++ parts (10-d)"), ("dm_rem", "#92c5de", "their remainders (10-d)"),
                ("jpp", JPP, "J++ colour vectors (10-d)"), ("random", NULL, "random subspace (10-d)"))


def patch_trials(run: str, phase: str) -> dict[str, dict[str, dict]] | None:
    """trials() restricted further to items whose counterfactual is answered T(s1') clean (arm cf_clean)."""
    arms = trials(run, phase)
    if arms is None:
        return None
    cf_ok = {i for i, r in arms.pop("cf_clean").items() if r["outcome"] == "success"}
    return {a: {i: r for i, r in recs.items() if i in cf_ok} for a, recs in arms.items()}


def fig_patch(models: list[str], out, prefix: str = "patch") -> None:
    runs = [causal_run(m) for m in models if trials(causal_run(m), f"{prefix}_start")]
    if not runs:
        return
    fig, axes = plt.subplots(len(runs), 2, figsize=(13, 3.6 * len(runs)), squeeze=False, sharey=True)
    for row, run in enumerate(runs):
        for col, window in enumerate(("start", "after")):
            ax = axes[row][col]
            arms = patch_trials(run, f"{prefix}_{window}")
            if arms is None:
                ax.set_axis_off()
                continue
            bands = sorted({a.split("|")[1] for a in arms}, key=band_mid)
            series = [({b: arms[f"patch_{span}|{b}"] for b in bands if f"patch_{span}|{b}" in arms}, colour, label)
                      for span, colour, label in PATCH_SERIES]
            bars(ax, bands, series, width=0.13)
            n = len(arms[f"patch_full|{bands[0]}"])
            where = {"start": "from the start colour", "after": "from the token after the start colour"}[window]
            ax.set_title(f"{LABEL[run.split('_')[0]]}: patched {where} to the end (n={n})", fontsize=9, loc="left")
            if col == 0:
                ax.set_ylabel("answer becomes T(s1')", fontsize=9)
    handles, labels = next(ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0])
    fig.legend(handles, labels, loc="lower center", ncol=len(PATCH_SERIES), fontsize=8, frameon=False)
    bank = "dev" if prefix == "patch" else "test"
    fig.suptitle("exp17 interchange patching: the counterfactual prompt's start colour is T^-1(s1'); its residual is "
                 f"copied in one subspace at every layer of a band ({bank} items; UNVERIFIED)", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.06 / len(runs), 1, 0.95))
    fig.savefig(out / f"fig_{prefix}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="+", default=["qwen", "gemma"])
    a = p.parse_args()
    out = FIGURES / a.out
    out.mkdir(parents=True, exist_ok=False)
    fig_layers(a.models, out)
    fig_controls(a.models, out)
    fig_lens(a.models, out)
    fig_test(a.models, out, "preview")
    fig_test(a.models, out, "test")
    fig_patch(a.models, out)
    fig_patch(a.models, out, prefix="test_patch")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
