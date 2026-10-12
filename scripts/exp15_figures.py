"""exp15 figures (no item text) from the report JSONs, into figures/exp15_wrong_keys/<out>/ (must be new).

  python exp15_figures.py --read <read tag> --write <write tag> --dev-grid <dev writes tag> --out <run name>

  fig_reads.png          per reader: P1 key AUC (h=2, h=3), P2 key AUC, P2 direction (key vs T(a')), P2 key AUC
                         with the answer-leak filter; 95% bootstrap CIs, chance at 0.5
  fig_decomposition.png  per reader, on the consistent off-cycle errors: share where w (H_key), the true s_{h-1}
                         (H_override) or neither (H_slip) ranks top
  fig_writes.png         P3 key swap - random null at each method's locked setting and in the layer-42 head-to-head,
                         the answer-swap gate, and P4 repair - null where run
  fig_dev_grid.png       the dev key swap - null grid (layer x alpha) per method
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp15 import EXP


def bars(ax, labels, entries, colours, names, null=None):
    """Grouped bars of {mean, lo, hi} entries: entries[k][j] is series k at label j (None = missing)."""
    width = 0.8 / len(entries)
    x = np.arange(len(labels))
    for k, (series, colour, name) in enumerate(zip(entries, colours, names)):
        mean = np.array([e["mean"] if e and e["n"] else np.nan for e in series])
        lo = np.array([e["lo"] if e and e["n"] else np.nan for e in series])
        hi = np.array([e["hi"] if e and e["n"] else np.nan for e in series])
        ax.bar(x + (k - (len(entries) - 1) / 2) * width, mean, width, yerr=[mean - lo, hi - mean], color=colour,
               label=name, capsize=2, error_kw={"lw": 0.8})
    if null is not None:
        ax.axhline(null, c="grey", ls="--", lw=1)
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def fig_reads(read: dict, out) -> None:
    readers = list(read["readers"])
    series = [("P1_correct_h2", "key_auc", "P1 correct h=2: s1 vs controls"),
              ("P1_correct_h3", "key_auc", "P1 correct h=3: s2 vs controls"),
              ("P2_offcycle", "key_auc", "P2 off-cycle: w vs controls"),
              ("P2_offcycle", "direction", "P2 off-cycle: w vs T(answer)"),
              ("P2_offcycle", "key_auc_leak_filtered", "P2 off-cycle: w vs controls, leak-filtered")]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    bars(ax, readers, [[read["readers"][r][g].get(m) for r in readers] for g, m, _ in series],
         [f"C{k}" for k in range(len(series))], [s[2] for s in series], null=0.5)
    ax.set_ylabel("AUC (items, 95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    n = read["group_sizes"]
    ax.set_title(f"exp15 reads, {read['bank']} bank (n: correct h=2 {n['P1_correct_h2']}, h=3 {n['P1_correct_h3']}, "
                 f"consistent off-cycle {n['P2_offcycle']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_reads.png", dpi=150)


def fig_decomposition(read: dict, out) -> None:
    readers = list(read["readers"])
    shares = {m: np.array([read["readers"][r]["P2_offcycle"][m]["mean"] for r in readers])
              for m in ("top_key", "top_true", "top_neither")}
    fig, ax = plt.subplots(figsize=(9, 3.4))
    bottom = np.zeros(len(readers))
    for m, colour, name in (("top_key", "C3", "w ranks top (H_key)"), ("top_true", "C0", "true s_{h-1} ranks top "
                            "(H_override)"), ("top_neither", "C7", "a control ranks top (H_slip)")):
        ax.bar(readers, shares[m], bottom=bottom, color=colour, label=name)
        bottom += shares[m]
    ax.set_ylabel("share of consistent off-cycle errors")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Which candidate each reader ranks top ({read['bank']} bank, n={read['group_sizes']['P2_offcycle']})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_decomposition.png", dpi=150)


def fig_writes(write: dict, out) -> None:
    methods = list(write["P3"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [3, 3, 2]})
    bars(axes[0], [f"{m}\nL{write['P3'][m]['layer']} a{write['P3'][m]['alpha']}" for m in methods],
         [[write["P3"][m]["key_minus_null"] for m in methods]], ["C0"], ["key swap - random null"], null=0.0)
    failed = [m for m in methods if not write["P3"][m]["gate"]["pass"]]
    gate_note = f"answer-swap gate failed: {', '.join(failed)}" if failed else "all passed the answer-swap gate"
    axes[0].set_title(f"P3 at each method's dev-locked setting ({gate_note})", fontsize=9)
    axes[0].set_ylabel("success rate difference")
    l42 = list(write["P3_layer42"])
    bars(axes[1], l42, [[write["P3_layer42"][m]["key_minus_null"] for m in l42]], ["C1"], ["L42"], null=0.0)
    axes[1].set_title("P3 head-to-head at layer 42", fontsize=9)
    if write["P4"]:
        p4 = list(write["P4"])
        bars(axes[2], p4, [[write["P4"][m]["repair_minus_null"] for m in p4], [write["P4"][m]["redirect"] for m in p4]],
             ["C2", "C4"], ["repair (c = w -> s2) - null", "redirect success"], null=0.0)
        axes[2].legend(fontsize=7)
        axes[2].set_title("P4 repair of consistent off-cycle h=3 errors", fontsize=9)
    else:
        axes[2].text(0.5, 0.5, "P4 untestable (K4):\nno method reached\ndev key swap - null >= 0.10",
                     ha="center", va="center", fontsize=9, transform=axes[2].transAxes)
        axes[2].set_axis_off()
    fig.suptitle(f"exp15 writes, {write['bank']} bank (95% paired bootstrap CIs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_writes.png", dpi=150)


def fig_dev_grid(grid: dict, out) -> None:
    methods = list(grid["grid"])
    fig, axes = plt.subplots(1, len(methods), figsize=(2.2 * len(methods), 2.8), sharey=False)
    for ax, m in zip(np.atleast_1d(axes), methods):
        cells = grid["grid"][m]
        layers = sorted({int(k.split("|")[0][1:]) for k in cells})
        alphas = sorted({float(k.split("|")[1][1:]) for k in cells})
        values = [[cells.get(f"L{l}|a{a}", {}).get("key_minus_null") for a in alphas] for l in layers]
        z = np.array([[np.nan if v is None else v for v in row] for row in values])
        im = ax.imshow(z, vmin=-0.1, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(alphas)), [str(a) for a in alphas], fontsize=7)
        ax.set_yticks(range(len(layers)), [str(l) for l in layers], fontsize=7)
        ax.set_title(m, fontsize=8)
        ax.set_xlabel("alpha", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("layer", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="dev key swap - null")
    fig.suptitle("exp15 dev write grid", fontsize=9)
    fig.savefig(out / "fig_dev_grid.png", dpi=150, bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--read")
    p.add_argument("--write")
    p.add_argument("--dev-grid")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = EXP.figure_dir(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.read:
        read = json.loads((EXP.results / a.read / "read_report.json").read_text())
        fig_reads(read, out)
        fig_decomposition(read, out)
    if a.write:
        fig_writes(json.loads((EXP.results / a.write / "write_report.json").read_text()), out)
    if a.dev_grid:
        fig_dev_grid(json.loads((EXP.results / a.dev_grid / "dev_write_grid.json").read_text()), out)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
