"""exp15 exploratory: when, by layer, does each token lens read the key, compared with when it reads the answer?

The answer-leak worry for P2: a reader that picks up the model's answer a' together with the rule line w -> a' that
produced it would score w above the controls without the model ever holding w as its key. If w is read at layers
where the lens does not yet rank a' first, the w signal comes before the answer. Per token lens, primary cell and
layer 0..62 (each (cell, layer) pair scored on its own, colour ranks within the pair):
  P1 correct h=3:      AUC of the true s2 vs controls
  P2 off-cycle (0.3):  AUC of w vs controls, of the true s_{h-1} vs controls, and of w vs T(a')
  answer_top1:         share of items where the lens ranks the model's answer first (on the same items)

  python exp15_layer_course.py --bank test --tag test_layer_course_v1
      -> results/exp15_wrong_keys/<tag>/layer_course.json and figures/exp15_wrong_keys/<tag>/fig_layer_course.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import torch

import exp15 as E
import exp15_report as R
import exp15_score as S
from exp15 import EXP, PRIMARY_CELLS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def course(x: torch.Tensor, members, row: dict, index: dict, target, others, prior: torch.Tensor | None = None
           ) -> list[float]:
    """Per layer: mean over items of AUC(S(target) vs S(others)), S = colour ranks within the (cell, layer) pair,
    minus each colour's mean rank over the bank at that layer if prior [n_layers, 10] is given.
    x [n_items, n_layers, 10]; target(it) -> colour; others(it) -> colours."""
    r = S.ranks(x[[row[it.item_id] for it in members]])               # [n, L, 10]
    if prior is not None:
        r = r - prior
    out = []
    for layer in range(r.shape[1]):
        out.append(round(sum(S.item_auc(r[k, layer], index[target(it)], [index[c] for c in others(it)])
                             for k, it in enumerate(members)) / len(members), 4))
    return out


def answer_top1(x: torch.Tensor, members, row: dict, index: dict) -> list[float]:
    s = x[[row[it.item_id] for it in members]]                         # [n, L, 10]
    pred = torch.tensor([index[it.pred] for it in members])
    return [round(float((s[:, layer].argmax(-1) == pred).float().mean()), 4) for layer in range(s.shape[1])]


def layer_course(bank: str, tag: str) -> None:
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    fig_dir = EXP.figure_dir(tag)
    fig_dir.mkdir(parents=True, exist_ok=False)
    table = E.item_table(bank)
    metas, _ = E.load_capture(bank, with_hidden=False)
    groups = R.read_groups(table, metas)
    lr = E.load_lens_reads(bank)
    colours = list(lr["labels"][:10])
    index = {c: k for k, c in enumerate(colours)}
    row = {i: n for n, i in enumerate(lr["item_ids"])}
    p1, p2 = groups["P1_correct_h3"], groups["P2_offcycle"]
    p2_dir = [it for it in p2 if S.direction_eligible(it)]
    record = {"bank": bank, "exploratory": True, "n": {"P1_correct_h3": len(p1), "P2_offcycle": len(p2),
                                                        "P2_direction_eligible": len(p2_dir)},
              "layers": list(lr["layers"]), "readers": {}}
    for kind in R.TOKEN_READERS:
        record["readers"][kind] = {}
        for cell in PRIMARY_CELLS:
            x = lr["scores"][kind][:, list(lr["cells"]).index(cell), :, :10].float()   # [items, L, 10]
            prior = S.ranks(x).mean(0)                                                   # [L, 10], all items
            entry = {"P2_answer_top1": answer_top1(x, p2, row, index), "P1_answer_top1": answer_top1(x, p1, row, index)}
            for suffix, pr in (("", None), ("_prior_corrected", prior)):
                entry |= {
                    "P1_true_s2_vs_controls" + suffix: course(x, p1, row, index, lambda it: it.true_prev,
                                                              lambda it: it.controls, pr),
                    "P2_w_vs_controls" + suffix: course(x, p2, row, index, lambda it: it.key, lambda it: it.controls,
                                                        pr),
                    "P2_true_vs_controls" + suffix: course(x, p2, row, index, lambda it: it.true_prev,
                                                           lambda it: it.controls, pr),
                    "P2_w_vs_forward" + suffix: course(x, p2_dir, row, index, lambda it: it.key,
                                                       lambda it: [it.forward], pr)}
            record["readers"][kind][cell] = entry
        print(kind, "done", flush=True)
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "layer_course.json")
    plot(record, fig_dir, "")
    plot(record, fig_dir, "_prior_corrected")


def plot(record: dict, fig_dir, suffix: str) -> None:
    """One panel per (lens, cell); suffix '' plots raw AUCs, '_prior_corrected' the colour-prior-corrected ones."""
    readers = list(record["readers"])
    cells = list(next(iter(record["readers"].values())))
    fig, axes = plt.subplots(len(readers), len(cells), figsize=(4 * len(cells), 2.4 * len(readers)), sharex=True,
                             sharey=True)
    layers = record["layers"]
    for i, kind in enumerate(readers):
        for j, cell in enumerate(cells):
            ax, e = axes[i][j], record["readers"][kind][cell]
            ax.plot(layers, e["P2_w_vs_controls" + suffix], c="C3", label="P2: w vs controls")
            ax.plot(layers, e["P2_true_vs_controls" + suffix], c="C0", label="P2: true s_{h-1} vs controls")
            ax.plot(layers, e["P2_w_vs_forward" + suffix], c="C3", ls=":", label="P2: w vs T(answer)")
            ax.plot(layers, e["P1_true_s2_vs_controls" + suffix], c="C0", ls="--", lw=0.8,
                    label="P1 h=3: s2 vs controls")
            ax.plot(layers, e["P2_answer_top1"], c="k", lw=0.8, label="P2: answer ranked first (share)")
            ax.axhline(0.5, c="grey", lw=0.5)
            ax.axvspan(36, 48, color="grey", alpha=0.1)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(cell, fontsize=9)
            if j == 0:
                ax.set_ylabel(kind, fontsize=9)
    axes[-1][0].set_xlabel("layer (shaded: primary window 36-48)")
    axes[0][0].legend(fontsize=6, loc="upper left")
    n = record["n"]
    kind_of_auc = "colour-prior-corrected AUCs" if suffix else "raw AUCs"
    fig.suptitle(f"exp15 layer course, {record['bank']} bank, {kind_of_auc} (exploratory; P2 n={n['P2_offcycle']}, "
                 f"P1 h=3 n={n['P1_correct_h3']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_layer_course{suffix}.png", dpi=130)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", choices=E.BANKS, required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    layer_course(a.bank, a.tag)


if __name__ == "__main__":
    main()
