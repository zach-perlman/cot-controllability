"""exp11: the test captures re-read with the J++ lens (manifest jpp_addendum, written before any J++ readout).

  gate     the paper's Figure-1 check of our loader: a 'heart' token in the J++ top 10 at layer 24 or 32
           (exits non-zero if it fails, so nothing after it runs)
  three    three columns (bank test): target-minus-decoy margins per item, layer and position for the J-lens, R-lens
           and J++ (saved once), then the band means, the addendum's primary readings, onsets and column-order split
  single   single table (bank single_test): exp11_report.q2_single with J++ (band 36..48; secondary 24..32)

Layers are block outputs (layer l = hidden[l + 1]). Targets: h=1 the answer (the positive control), h=2 s1; the decoy
is the start colour's entry under an unused ingredient, so the null margin is 0 by construction.
"""
from __future__ import annotations

import sys

import torch

import exp11 as X
import exp11_explore as E
import exp11_q23 as Q
import exp11_report as R

LENSES = ("j-lens", "r-lens", "jpp")
POSITIONS = ("start", "stir1", "stir2", "question_end", "final")
GROUPS = ("h1", "correct", "shortcut", "other_wrong")
BANDS = {"early": range(8, 21), "mid": range(24, 53)}
PRIMARY = (("stir1", "early"), ("stir1", "mid"), ("final", "early"), ("final", "mid"))
ALPHA = 0.05 / len(PRIMARY)                   # Bonferroni over the primary cells
GRID = range(8, 61, 4)
N_LAYERS = 63                                 # lens source layers 0..62
N_BOOT = 1000
GATE_PROMPT = "Fact: In humans, the organ that pumps blood through the body has this many chambers: "
GATE_WORDS = ("heart", "cardiac", "心脏")
GATE_LAYERS = (24, 32)


def ci(x: torch.Tensor, g: torch.Generator, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean over items of x [n] with a bootstrap (1 - alpha) CI."""
    boot = x[torch.randint(len(x), (N_BOOT, len(x)), generator=g)].mean(1)
    return float(x.mean()), float(boot.quantile(alpha / 2)), float(boot.quantile(1 - alpha / 2))


def fmt(c: tuple[float, float, float]) -> str:
    return f"{c[0]:+.2f}[{c[1]:+.2f},{c[2]:+.2f}]"


# --- gate ------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def gate() -> None:
    import exp11_model as M
    tokenizer, model = M.load()
    ids = tokenizer.encode(GATE_PROMPT, add_special_tokens=False)
    hidden, logits = M.forward(model, ids, [len(ids) - 1])
    vocab = M.text_parts(model)[2].shape[0]
    words = tokenizer.batch_decode([[i] for i in range(len(tokenizer))]) + [""] * (vocab - len(tokenizer))
    dropped = torch.tensor([not any(ch.isalnum() for ch in w) for w in words])      # Readout Filtering
    lens = M.Lens("jpp", model, list(range(vocab)))
    readout = lens.logits(hidden)[:, 0].masked_fill(dropped, float("-inf"))         # [layers, vocab]
    print(f"## gate: J++ top 10 at the final position of {GATE_PROMPT!r}, Readout Filtering on")
    print(f"  model's own top 10: {[words[i] for i in logits.topk(10).indices]}")
    hit = {}
    for l in range(8, 57, 8):
        top = [words[i] for i in readout[lens.layers.index(l)].topk(10).indices]
        hit[l] = any(w in t.lower() for t in top for w in GATE_WORDS)
        print(f"  layer {l:>2} {'*' if hit[l] else ' '} {top}")
    passed = any(hit[l] for l in GATE_LAYERS)
    print(f"  gate {'PASSED' if passed else 'FAILED'}: a heart token in the top 10 at layer 24 or 32")
    if not passed:
        sys.exit(1)


# --- part A: three columns -------------------------------------------------------------------------------------------
def margins_path(bank: str):
    path = X.EXP.cache / f"jpp_three_{bank}.pt"
    if path.exists():
        raise SystemExit(f"{path} exists; analysis outputs are written once")
    return path


def column_first(m: dict) -> bool:
    """Whether the target's column (the first stir's ingredient) is printed before the decoy's (the first unused
    ingredient, as in exp11_q23.three_labels)."""
    b = m["brew"]
    unused = next(i for i in b.ingredients if i not in b.stirs)
    return b.ingredients.index(b.stirs[0]) < b.ingredients.index(unused)


@torch.no_grad()
def three(bank: str = "test") -> None:
    import exp11_model as M
    path = margins_path(bank)
    tokenizer, model = M.load()
    metas = [m for m in E.load(bank) if m["format"] == "three_first" and m["h"] in (1, 2)]
    cases = [(m, lab) for m in metas if (lab := Q.three_labels(m)) is not None]
    colours = metas[0]["colours"]
    ids = list(M.colour_ids(tokenizer, colours).values())
    margins = {}
    for name in LENSES:
        lens = M.Lens(name, model, ids)
        out = torch.full((len(cases), N_LAYERS, len(POSITIONS)), float("nan"))     # nan: layer or position absent
        for k, (m, lab) in enumerate(cases):
            at = lens.logits(m["hidden"])                                          # [lens layers, positions, 10]
            target = lab["s1"] if m["h"] == 2 else lab["answer"]
            d = at[..., colours.index(target)] - at[..., colours.index(lab["decoy"])]
            for p, pos in enumerate(POSITIONS):
                if pos in m["positions"]:
                    out[k, lens.layers, p] = d[:, list(m["positions"]).index(pos)]
        margins[name] = out
        del lens
        torch.cuda.empty_cache()
    blob = {"bank": bank, "lenses": LENSES, "positions": POSITIONS, "item_ids": [m["item_id"] for m, _ in cases],
            "groups": [m["group"] for m, _ in cases], "column_first": [column_first(m) for m, _ in cases],
            "margins": margins}
    torch.save(blob, path)
    report_three(blob)


def onset(x: torch.Tensor, g: torch.Generator) -> int | None:
    """x [items, layers]: the first grid layer at which the 95% CI lies above 0 there and at the next grid layer."""
    grid = list(GRID)
    above = [ci(x[:, l], g)[1] > 0 for l in grid]
    return next((grid[k] for k in range(len(grid) - 1) if above[k] and above[k + 1]), None)


def report_three(blob: dict) -> None:
    g = torch.Generator().manual_seed(0)
    members = {grp: [i for i, x in enumerate(blob["groups"]) if x == grp] for grp in GROUPS}
    col = torch.tensor(blob["column_first"])

    def at(name: str, pos: str, grp: str) -> torch.Tensor:
        """[items in grp, layers] margins at one position."""
        return blob["margins"][name][members[grp], :, POSITIONS.index(pos)]

    def band_mean(name: str, pos: str, grp: str, band: str) -> torch.Tensor:
        return at(name, pos, grp)[:, list(BANDS[band])].mean(1)

    print(f"## bank {blob['bank']}, three_first: n per group " + ", ".join(f"{k}={len(v)}" for k, v in members.items()))
    for name in LENSES:
        print(f"\n## {name}: band-mean target minus decoy [95% CI]; early {BANDS['early'][0]}..{BANDS['early'][-1]}, "
              f"mid {BANDS['mid'][0]}..{BANDS['mid'][-1]}")
        for pos in POSITIONS:
            for grp in GROUPS:
                if at(name, pos, grp).isnan().all():                              # stir2 at h=1
                    continue
                cells = [f"{band} {fmt(ci(band_mean(name, pos, grp, band), g))}" for band in BANDS]
                print(f"  {pos:>12} {grp:>11} n={len(members[grp]):>3} | " + "  ".join(cells))

    for name in LENSES:
        print(f"\n## {name}: per-layer target minus decoy [95% CI], layers {list(GRID)}")
        for pos in POSITIONS:
            for grp in GROUPS:
                x = at(name, pos, grp)
                if x.isnan().all():
                    continue
                print(f"  {pos:>12} {grp:>11} | " + " ".join(fmt(ci(x[:, l], g)) for l in GRID))

    print(f"\n## primary cells: shortcut s1 minus decoy and the h=1 guard, band means, Bonferroni "
          f"{100 * (1 - ALPHA):.2f}% CIs; readings for J++ only (J-lens, R-lens shown for comparison)")
    sc_col = col[members["shortcut"]]
    for pos, band in PRIMARY:
        for name in LENSES:
            guard = ci(band_mean(name, pos, "h1", band), g, ALPHA)
            x = band_mean(name, pos, "shortcut", band)
            sc = ci(x, g, ALPHA)
            split = (float(x[sc_col].mean()), float(x[~sc_col].mean()))
            line = (f"  ({pos}, {band}) {name:>6}: shortcut {fmt(sc)}  h1 guard {fmt(guard)}  "
                    f"column-order split: s1 column first {split[0]:+.2f} (n={int(sc_col.sum())}), "
                    f"decoy column first {split[1]:+.2f} (n={int((~sc_col).sum())})")
            if name == "jpp":
                if guard[1] <= 0:
                    reading = "no_reading (the h=1 control is not readable)"
                elif sc[1] > 0:
                    reading = "present" if min(split) > 0 else "present, but does not stand (column-order split)"
                elif sc[2] < 0:
                    reading = "below"
                else:
                    reading = f"absent (CI upper end = {sc[2] / guard[0]:.2f} of the h=1 control)"
                line += f"  -> {reading}"
            print(line)

    print("\n## onset of the h=1 control (first grid layer with the 95% CI above 0 there and at the next grid layer)")
    onsets = {(name, pos): onset(at(name, pos, "h1"), g) for name in LENSES for pos in ("stir1", "final")}
    for (name, pos), l in onsets.items():
        print(f"  {name:>6} {pos:>6}: {l}")
    jpp, others = onsets[("jpp", "final")], [onsets[(n, "final")] for n in ("j-lens", "r-lens")]
    earlier = jpp is not None and all(o is None or jpp < o for o in others)
    print(f"  J++ reads earlier at the final position: {'yes' if earlier else 'no'}")


# --- part B: single table --------------------------------------------------------------------------------------------
def single(bank: str = "single_test") -> None:
    R.q2_single(bank, "jpp")
    print("\n(secondary, no reading)")
    R.q2_single(bank, "jpp", band=range(24, 33))


if __name__ == "__main__":
    {"gate": gate, "three": three, "single": single}[sys.argv[1]]()
