"""exp11 exploratory follow-ups on the test bank, NOT pre-registered (decided after reading the test addendum's
secondary lens margins, which showed s1 above the decoy at the first ingredient's token on shortcut items).

  stir1   the colour probe (the addendum's settings) and the J/R-lens s1-minus-decoy margins at the first
          ingredient's token, per h=2 group, with h=1 (the same lookup as the answer) as the reference
  checks  CPU-only checks on saved outputs: whether the h=2 correct group beats chance, and whether the twin
          patch's stir-count setting is the donor's count or generic disruption (two-stir donors as the contrast)
  columns (added after the J++ run, whose column-order check split the J-lens first-ingredient margin unevenly)
          the first-ingredient s1-minus-decoy contrast split by whether s1's column is printed before the decoy's:
          the saved lens margins (J-lens, R-lens, J++; bank test) and the stir1 probe (banks test and dev)
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


def column_order() -> None:
    """A real lookup at the first ingredient's token makes s1 readable whichever column is printed first; a reader
    that favours the earlier-printed entry of the start colour's rule line makes a margin of opposite sign in the two
    halves. Lookup part = the mean of the two halves; column part = half their difference."""
    import exp11_jpp as J
    g = torch.Generator().manual_seed(0)
    halves = lambda x, first: (Q.mean_ci(x[first], g), Q.mean_ci(x[~first], g))
    show = lambda c: f"{c[0]:+.3f}[{c[1]:+.3f},{c[2]:+.3f}]"
    blob = torch.load(X.EXP.cache / "jpp_three_test.pt", weights_only=False)
    first, groups = torch.tensor(blob["column_first"]), blob["groups"]
    print("## lens margins at the first ingredient's token (bank test): s1 (h=1: answer) minus decoy, band means, "
          "split by column order [95% CI]")
    for name in J.LENSES:
        x = blob["margins"][name][:, :, J.POSITIONS.index("stir1")]
        for grp in ("h1", "shortcut", "correct"):
            idx = torch.tensor([i for i, g_ in enumerate(groups) if g_ == grp])
            for band in ("early", "mid"):
                (a, b) = halves(x[idx][:, list(J.BANDS[band])].mean(1), first[idx])
                print(f"  {name:>6} {grp:>8} {band:>5}: s1 column first {show(a)} (n={int(first[idx].sum())}), "
                      f"decoy column first {show(b)} (n={int((~first[idx]).sum())}); "
                      f"lookup {(a[0] + b[0]) / 2:+.3f}, column {(a[0] - b[0]) / 2:+.3f}")
    probe_column_split(["test"])
    probe_column_split(["dev", "fmt_dev"])


def probe_column_split(banks: list[str]) -> None:
    """The stir1 probe (pooled over `banks`, as in the dev replication: 'dev' alone is too small for the probe to
    read even the start colour) split by column order."""
    import exp11_jpp as J
    g = torch.Generator().manual_seed(0)
    show = lambda c: f"{c[0]:+.3f}[{c[1]:+.3f},{c[2]:+.3f}]"
    metas = {m["item_id"]: m for bank in banks for m in E.load(bank)}
    res = Q.probe(banks, k_pca=64, wd=1e-3, positions=("stir1",), save=False)["results"]
    print(f"\n## stir1 probe (banks {'+'.join(banks)}): s1 (h=1: answer) minus decoy held-out accuracy, band "
          f"{Q.BAND.start}..{Q.BAND.stop - 1}, split by column order [95% CI]")
    for h, target in ((1, "answer"), (2, "s1")):
        r = res[(h, "stir1", target)]
        d = (r["correct"].float() - res[(h, "stir1", "decoy")]["correct"].float())[:, list(Q.BAND)].mean(1)
        col = torch.tensor([J.column_first(metas[i]) for i in r["item_ids"]])
        for grp in sorted(set(r["groups"])):
            idx = torch.tensor([i for i, g_ in enumerate(r["groups"]) if g_ == grp])
            a, b = Q.mean_ci(d[idx][col[idx]], g), Q.mean_ci(d[idx][~col[idx]], g)
            print(f"  {grp:>11}: s1 column first {show(a)} (n={int(col[idx].sum())}), decoy column first "
                  f"{show(b)} (n={int((~col[idx]).sum())}); lookup {(a[0] + b[0]) / 2:+.3f}, "
                  f"column {(a[0] - b[0]) / 2:+.3f}")


def main() -> None:
    if sys.argv[1:] == ["columns"]:
        column_order()
        return
    if sys.argv[1:] == ["columns_dev"]:
        probe_column_split(["dev", "fmt_dev"])
        return
    if sys.argv[1:] == ["checks"]:
        metas = E.load("test")
        correct_vs_chance(metas)
        setting_vs_disruption(metas)
        return
    Q.probe(["test"], k_pca=64, wd=1e-3, positions=("stir1",), save=False)
    lens_by_group("test")


if __name__ == "__main__":
    main()
