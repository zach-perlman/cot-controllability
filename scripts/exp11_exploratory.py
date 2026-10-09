"""exp11 exploratory follow-ups on the test bank, NOT pre-registered (decided after reading the test addendum's
secondary lens margins, which showed s1 above the decoy at the first ingredient's token on shortcut items).

  stir1   the colour probe (the addendum's settings) and the J/R-lens s1-minus-decoy margins at the first
          ingredient's token, per h=2 group, with h=1 (the same lookup as the answer) as the reference
  checks  CPU-only checks on saved outputs: whether the h=2 correct group beats chance, and whether the twin
          patch's stir-count setting is the donor's count or generic disruption (two-stir donors as the contrast)
"""
from __future__ import annotations

import sys
from collections import defaultdict

import torch

import exp11 as X
import exp11_explore as E
import exp11_q23 as Q


@torch.no_grad()
def lens_by_group(bank: str, position: str = "stir1", layers=range(8, 61, 4)) -> None:
    import exp11_model as M
    tokenizer, model = M.load()
    metas = [m for m in E.load(bank) if m["format"] == "three_first" and m["h"] in (1, 2)]
    colours = metas[0]["colours"]
    ids = list(M.colour_ids(tokenizer, colours).values())
    g = torch.Generator().manual_seed(0)
    for name in ("j-lens", "r-lens"):
        lens = M.Lens(name, model, ids)
        rows = defaultdict(list)
        for m in metas:
            lab = Q.three_labels(m)
            if lab is None:
                continue
            target = lab["s1"] if m["h"] == 2 else lab["answer"]
            at = lens.logits(m["hidden"])[:, list(m["positions"]).index(position)]
            rows[m["group"]].append(at[:, colours.index(target)] - at[:, colours.index(lab["decoy"])])
        print(f"\n## {name}, {bank}, position {position}: s1 (h=1: answer) minus decoy [95% CI], layers {list(layers)}")
        for grp, xs in sorted(rows.items()):
            x = torch.stack(xs)
            cells = []
            for l in layers:
                mu, lo, hi = Q.mean_ci(x[:, lens.layers.index(l)], g)
                cells.append(f"{mu:+.2f}[{lo:+.2f},{hi:+.2f}]")
            print(f"  {grp:>11} n={len(x):>3} | " + " ".join(cells))
        del lens
        torch.cuda.empty_cache()


def correct_vs_chance(metas: list[dict]) -> None:
    """Three columns, h=2: among answers that are not the shortcut, how often the answer is the gold (a random
    choice among the 9 non-shortcut colours: 1/9), and the mean probability of the gold on shortcut items."""
    h2 = [m for m in metas if m["format"] == "three_first" and m["h"] == 2]
    shortcut_of = lambda m: m["brew"].apply(m["brew"].start, m["brew"].stirs[-1:])
    non_shortcut = [m for m in h2 if m["pred"] != shortcut_of(m)]
    hits = sum(m["pred"] == m["gold"] for m in non_shortcut)
    print(f"\n## h=2 correct vs chance: {hits}/{len(non_shortcut)} non-shortcut answers are the gold "
          f"({hits / len(non_shortcut):.2f}; chance 0.11)")
    on_shortcut = [m for m in h2 if m["group"] == "shortcut"]
    p_gold = sum(m["p_gold"] for m in on_shortcut) / len(on_shortcut)
    p_other = sum((1 - m["p_pred"] - m["p_gold"]) / 8 for m in on_shortcut) / len(on_shortcut)
    print(f"   shortcut items (n={len(on_shortcut)}): mean p(gold) {p_gold:.3f} vs mean p(each other wrong colour) "
          f"{p_other:.3f}")


def setting_vs_disruption(metas: list[dict]) -> None:
    """Shortcut recipients, answer position: the fraction answering their own s1 or their own shortcut when the
    donor is a one-stir state (the twin patch) vs a two-stir state (the mode transplant). If any foreign state made
    the model fall back to the first stir, two-stir donors would move recipients to s1 as well."""
    by_id = {m["item_id"]: m for m in metas}
    runs = [("one-stir donor: own twin", "q23_twinpatch_test_three_first_final.pt", "treatment"),
            ("one-stir donor: another item's twin", "q23_twinpatch_test_three_first_final.pt", "control"),
            ("two-stir donor: a correct item", "q23_modepatch_test_three_first_h2.pt", "treatment"),
            ("two-stir donor: a shortcut item", "q23_modepatch_test_three_first_h2.pt", "control")]
    print("\n## stir-count setting vs disruption, shortcut recipients: own s1 / own shortcut answer rate by layer")
    for name, file, arm in runs:
        blob = torch.load(X.EXP.cache / file, weights_only=False)
        colours = blob["colours"]
        res = [r for r in blob["results"] if by_id[r["recipient"]]["group"] == "shortcut"]
        s1 = torch.tensor([colours.index(by_id[r["recipient"]]["brew"].path()[1]) for r in res])
        cut = torch.tensor([colours.index(by_id[r["recipient"]]["pred"]) for r in res])
        answer = torch.stack([r[arm] for r in res]).argmax(-1)                  # [items, layers]
        cells = [f"L{l} {float((answer[:, k] == s1).float().mean()):.2f}/{float((answer[:, k] == cut).float().mean()):.2f}"
                 for k, l in enumerate(blob["layers"]) if l in (32, 36, 40, 44, 48)]
        print(f"  {name:>36} n={len(res)} | " + "  ".join(cells))
    base = sum(by_id[r["recipient"]]["brew"].path()[1] == by_id[r["recipient"]]["pred"] for r in res) / len(res)
    print(f"  baseline: s1 equals the shortcut answer on {base:.2f} of these items")


def main() -> None:
    if sys.argv[1:] == ["checks"]:
        metas = E.load("test")
        correct_vs_chance(metas)
        setting_vs_disruption(metas)
        return
    Q.probe(["test"], k_pca=64, wd=1e-3, positions=("stir1",), save=False)
    lens_by_group("test")


if __name__ == "__main__":
    main()
