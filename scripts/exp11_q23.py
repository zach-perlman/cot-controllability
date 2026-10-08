"""exp11 Q2 colour probes and Q3 patching (manifest + test addendum), on any captured bank. Layers are block outputs
(the lenses' numbering: layer l = hidden[l + 1]); every run writes its raw outputs once under cache/exp11.

  probe      --bank B [--k-pca K] [--wd W]   three-ingredient items: held-out colour-probe accuracy per layer and
                                             group for the labels s1, decoy, s0, answer (h=2) and answer, decoy (h=1)
  backpatch  --bank B                        h=2 shortcut items: the item's own final-position residual from a later
                                             layer written into an earlier one, vs another shortcut item's
  twinpatch  --bank B --site final|count     the residual at `site` taken from the item's twin (same table and
                                             start, last stir removed) vs from another item's twin
  modepatch  --bank B --format F --h H       wrong items: the final-position residual from an unrelated correct item
                                             at the same depth vs from an unrelated wrong item
"""
from __future__ import annotations

import argparse
import json
import random

import torch

import exp11 as X
import exp11_brew as B
import exp11_explore as E

BAND = range(24, 53)                     # the Q2 layer band (fixed on dev; see the test addendum)
BACKPATCH_PAIRS = [(l1, l1 + d) for l1 in (8, 16, 24, 32, 40, 48) for d in (4, 8, 16) if l1 + d <= 60]
TWIN_LAYERS = list(range(0, 61, 4))
MODE_LAYERS = list(range(28, 53, 4))
# the wrong-answer groups the mode transplant rescues: (format, h) -> (group name, membership test)
MODE_CELLS = {("three_first", 2): ("shortcut", lambda m: m["group"] == "shortcut"),
              ("single_first", 3): ("one_short", lambda m: m["pred"] == m["brew"].path()[-2]),
              ("single_last", 3): ("one_short", lambda m: m["pred"] == m["brew"].path()[-2])}
N_BOOT = 1000


# --- shared ---------------------------------------------------------------------------------------------------------
def partner(n: int, seed: str) -> list[int]:
    """A fixed derangement of range(n): item i's control donor is item partner[i]."""
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def requests_by_id(bank: str) -> dict[str, dict]:
    return {r["item_id"]: r for r in map(json.loads, X.requests_path(bank).open())}


def out_path(kind: str, bank: str, tag: str = "") -> "X.EXP.cache":
    path = X.EXP.cache / f"q23_{kind}_{bank}{('_' + tag) if tag else ''}.pt"
    if path.exists():
        raise SystemExit(f"{path} exists; analysis outputs are written once")
    return path


def three_labels(m: dict) -> dict[str, str] | None:
    """Probe labels of a three-ingredient item; None if the item is excluded from the s1-vs-decoy contrast."""
    b = m["brew"]
    if m["h"] == 1:
        unused = [i for i in b.ingredients if i not in b.stirs]
        return {"answer": b.path()[1], "decoy": b.table[b.start][unused[0]]}
    if m["h"] != 2:
        return None
    mates = E.row_mates(b)
    if len(set(mates.values())) < 3:
        return None
    return {"s1": mates["s1"], "decoy": mates["decoy"], "s0": b.start, "answer": m["pred"]}


def mean_ci(x: torch.Tensor, g: torch.Generator) -> tuple[float, float, float]:
    """Mean over items of x [n] with a bootstrap 95% CI."""
    boot = x[torch.randint(len(x), (N_BOOT, len(x)), generator=g)].mean(1)
    return float(x.mean()), float(boot.quantile(0.025)), float(boot.quantile(0.975))


# --- Q2: probes -----------------------------------------------------------------------------------------------------
def probe(banks: list[str], k_pca: int | None, wd: float, positions=("final",), save: bool = True) -> dict:
    """Held-out correctness [n, 63] per (h, position, label) over the three_first items of `banks`, with the items'
    groups; printed per group as the band mean."""
    import exp11_probe as P
    metas = [m for bank in banks for m in E.load(bank) if m["format"] == "three_first"]
    out = {"banks": banks, "k_pca": k_pca, "wd": wd, "band": list(BAND), "results": {}}
    g = torch.Generator().manual_seed(0)
    for h in (1, 2):
        ms = [(m, lab) for m in metas if m["h"] == h and (lab := three_labels(m)) is not None]
        groups = [m["group"] for m, _ in ms]
        folds = P.stratified_folds(groups)
        colours = ms[0][0]["colours"]
        for pos in positions:
            x = torch.stack([m["hidden"][1:64, list(m["positions"]).index(pos)] for m, _ in ms])   # [n, 63, d]
            for label in ms[0][1]:
                y = [colours.index(lab[label]) for _, lab in ms]
                correct = P.cv_correct(x, y, folds, k_pca=k_pca, weight_decay=wd)
                out["results"][(h, pos, label)] = {"correct": correct, "groups": groups,
                                                   "item_ids": [m["item_id"] for m, _ in ms]}
            print(f"\n## probe h={h} position={pos} k_pca={k_pca} wd={wd}: band {BAND.start}..{BAND.stop - 1} "
                  f"mean held-out accuracy (chance 0.10) [95% CI]")
            for grp in sorted(set(groups)):
                idx = [i for i, x in enumerate(groups) if x == grp]
                cells = []
                for label in ms[0][1]:
                    c = out["results"][(h, pos, label)]["correct"][idx][:, list(BAND)].float().mean(1)
                    cells.append(f"{label}={mean_ci(c, g)[0]:.2f}")
                target = "s1" if h == 2 else "answer"          # at h=1 the answer is the same lookup as s1
                d = (out["results"][(h, pos, target)]["correct"][idx].float()
                     - out["results"][(h, pos, "decoy")]["correct"][idx].float())[:, list(BAND)].mean(1)
                mu, lo, hi = mean_ci(d, g)
                cells.append(f"{target}-decoy={mu:+.3f}[{lo:+.3f},{hi:+.3f}]")
                print(f"  {grp:>11} n={len(idx):>3} " + " ".join(cells))
    if save:
        torch.save(out, out_path("probe", "+".join(banks), f"k{k_pca}_wd{wd}"))
    return out


# --- Q3: patching ---------------------------------------------------------------------------------------------------
def _setup(bank: str):
    import cc_exp10_render as R
    import exp11_model as M
    tokenizer, model = M.load()
    return tokenizer, model, R, M


def backpatch(bank: str) -> None:
    """h=2 shortcut items, final position: treatment writes the item's own layer-l2 residual into layer l1 (l1 < l2,
    Biran et al.'s back-patching); control writes another shortcut item's layer-l2 residual into layer l1."""
    import exp11_patch as PT
    path = out_path("backpatch", bank)
    tokenizer, model, R, M = _setup(bank)
    metas = [m for m in E.load(bank) if m["format"] == "three_first" and m["h"] == 2 and m["group"] == "shortcut"]
    rows = requests_by_id(bank)
    colours = metas[0]["colours"]
    cids = list(M.colour_ids(tokenizer, colours).values())
    other = partner(len(metas), f"backpatch:{bank}")
    results = []
    for i, m in enumerate(metas):
        ids = R.prompt_ids(tokenizer, M.FAMILY, rows[m["item_id"]])
        fin, fin_other = list(m["positions"]).index("final"), list(metas[other[i]]["positions"]).index("final")
        patches = [None]
        for l1, l2 in BACKPATCH_PAIRS:
            patches.append(PT.Patch(l1, len(ids) - 1, m["hidden"][l2 + 1, fin]))
            patches.append(PT.Patch(l1, len(ids) - 1, metas[other[i]]["hidden"][l2 + 1, fin_other]))
        logits = torch.cat([PT.colour_logits(model, ids, patches[j:j + 18], cids) for j in range(0, len(patches), 18)])
        results.append({"item_id": m["item_id"], "gold": m["gold"], "shortcut": m["pred"],
                        "s1": m["brew"].path()[1], "unpatched": logits[0],
                        "treatment": logits[1::2], "control": logits[2::2]})
        if i % 20 == 0:
            print(f"{i}/{len(metas)}", flush=True)
    torch.save({"bank": bank, "pairs": BACKPATCH_PAIRS, "colours": colours, "results": results}, path)
    summarise_backpatch(path)


def summarise_backpatch(path) -> None:
    blob = torch.load(path, weights_only=False)
    colours, res = blob["colours"], blob["results"]
    g = torch.Generator().manual_seed(0)
    gold = torch.tensor([colours.index(r["gold"]) for r in res])
    base = torch.stack([r["unpatched"] for r in res]).argmax(-1)
    print(f"\n## back-patching, {len(res)} shortcut items; unpatched argmax = gold {float((base == gold).float().mean()):.2f}")
    print("pair (l1<-l2): rescued to gold, treatment vs control, paired difference [95% CI]")
    for k, (l1, l2) in enumerate(blob["pairs"]):
        t = (torch.stack([r["treatment"][k] for r in res]).argmax(-1) == gold).float()
        c = (torch.stack([r["control"][k] for r in res]).argmax(-1) == gold).float()
        mu, lo, hi = mean_ci(t - c, g)
        print(f"  {l1:>2}<-{l2:>2}: {t.mean():.3f} vs {c.mean():.3f}  diff {mu:+.3f} [{lo:+.3f},{hi:+.3f}]")


def twinpatch(bank: str, site: str, fmt: str, reverse: bool = False) -> None:
    """Patch the residual at `site` ('final', or 'count' = the count word's last token, single tables only) from the
    item's twin (treatment) and from another item's twin (control), at every layer in TWIN_LAYERS, over the parents
    of format `fmt`. With `reverse`, the twins are the recipients and the parents the donors."""
    import exp11_patch as PT
    twin_bank = next(t for t, p in X.TWIN_BANKS.items() if p == bank)
    path = out_path("twinpatch", bank, f"{fmt}_{site}" + ("_reverse" if reverse else ""))
    tokenizer, model, R, M = _setup(bank)
    pos_name = {"final": "final", "count": "stirs"}[site]
    twins = {m["item_id"].removesuffix(":twin"): m for m in E.load(twin_bank)}       # parent id -> twin
    parents = [m for m in E.load(bank) if m["item_id"] in twins and m["format"] == fmt]
    pairs = [(twins[p["item_id"]], p) if reverse else (p, twins[p["item_id"]]) for p in parents]
    rows = {**requests_by_id(bank), **requests_by_id(twin_bank)}
    colours = parents[0]["colours"]
    cids = list(M.colour_ids(tokenizer, colours).values())
    other = partner(len(pairs), f"twinpatch:{bank}:{site}:{reverse}")
    results = []
    for i, (rec, donor) in enumerate(pairs):
        ctrl = pairs[other[i]][1]
        ids = R.prompt_ids(tokenizer, M.FAMILY, rows[rec["item_id"]])
        at = rec["positions"][pos_name]
        d_at, c_at = list(donor["positions"]).index(pos_name), list(ctrl["positions"]).index(pos_name)
        patches = [None] + [PT.Patch(l, at, donor["hidden"][l + 1, d_at]) for l in TWIN_LAYERS] \
                         + [PT.Patch(l, at, ctrl["hidden"][l + 1, c_at]) for l in TWIN_LAYERS]
        logits = torch.cat([PT.colour_logits(model, ids, patches[j:j + 17], cids) for j in range(0, len(patches), 17)])
        n = len(TWIN_LAYERS)
        results.append({"recipient": rec["item_id"], "donor": donor["item_id"], "control_donor": ctrl["item_id"],
                        "h": rec["h"], "gold": rec["gold"], "donor_gold": donor["gold"],
                        "control_donor_gold": ctrl["gold"], "path": rec["brew"].path(),
                        "unpatched": logits[0], "treatment": logits[1:1 + n], "control": logits[1 + n:]})
        if i % 20 == 0:
            print(f"{i}/{len(pairs)}", flush=True)
    torch.save({"bank": bank, "format": fmt, "site": site, "reverse": reverse, "layers": TWIN_LAYERS,
                "colours": colours, "results": results}, path)
    summarise_twinpatch(path)


def summarise_twinpatch(path) -> None:
    """Per layer, the fraction of recipients whose argmax is the donor's gold (treatment) or the recipient's own
    colour one stir from its gold in the donor's direction (control: an unrelated item's twin carries the same count
    word but another table)."""
    blob = torch.load(path, weights_only=False)
    colours, res = blob["colours"], blob["results"]
    g = torch.Generator().manual_seed(0)
    gold = torch.tensor([colours.index(r["gold"]) for r in res])
    donor_gold = torch.tensor([colours.index(r["donor_gold"]) for r in res])
    ctrl_gold = torch.tensor([colours.index(r["control_donor_gold"]) for r in res])
    base = torch.stack([r["unpatched"] for r in res]).argmax(-1)
    print(f"\n## twin patch, site={blob['site']} reverse={blob['reverse']}, {len(res)} recipients; unpatched: "
          f"argmax = own gold {float((base == gold).float().mean()):.2f}, = donor's gold "
          f"{float((base == donor_gold).float().mean()):.2f}")
    print("layer: treatment -> donor's gold | control -> recipient's donor-gold colour, -> control donor's gold "
          "| own gold kept (treatment, control)")
    for k, l in enumerate(blob["layers"]):
        t = torch.stack([r["treatment"][k] for r in res]).argmax(-1)
        c = torch.stack([r["control"][k] for r in res]).argmax(-1)
        print(f"  {l:>2}: {float((t == donor_gold).float().mean()):.2f} | {float((c == donor_gold).float().mean()):.2f}, "
              f"{float((c == ctrl_gold).float().mean()):.2f} | {float((t == gold).float().mean()):.2f}, "
              f"{float((c == gold).float().mean()):.2f}")


def modepatch(bank: str, fmt: str, h: int) -> None:
    """Wrong items of MODE_CELLS[(fmt, h)]: the final-position residual at each layer in MODE_LAYERS taken from an
    unrelated CORRECT item at the same depth (treatment) or from an unrelated item of the same wrong group (control).
    The twin patches show the final position carrying an item-independent stir-count setting at layers 36-44; if
    the error lives in that setting, the treatment rescues the recipient's own gold and the control does not."""
    import exp11_patch as PT
    group_name, is_wrong = MODE_CELLS[(fmt, h)]
    path = out_path("modepatch", bank, f"{fmt}_h{h}")
    tokenizer, model, R, M = _setup(bank)
    cell = [m for m in E.load(bank) if m["format"] == fmt and m["h"] == h]
    correct = [m for m in cell if m["pred"] == m["gold"]]
    wrong = [m for m in cell if m["pred"] != m["gold"] and is_wrong(m)]
    random.Random(f"modepatch:{bank}:{fmt}:{h}").shuffle(correct)
    other = partner(len(wrong), f"modepatch:{bank}:{fmt}:{h}")
    rows = requests_by_id(bank)
    colours = cell[0]["colours"]
    cids = list(M.colour_ids(tokenizer, colours).values())
    results = []
    for i, m in enumerate(wrong):
        donor, ctrl = correct[i % len(correct)], wrong[other[i]]
        ids = R.prompt_ids(tokenizer, M.FAMILY, rows[m["item_id"]])
        fin_d, fin_c = list(donor["positions"]).index("final"), list(ctrl["positions"]).index("final")
        patches = [None] + [PT.Patch(l, len(ids) - 1, donor["hidden"][l + 1, fin_d]) for l in MODE_LAYERS] \
                         + [PT.Patch(l, len(ids) - 1, ctrl["hidden"][l + 1, fin_c]) for l in MODE_LAYERS]
        logits = PT.colour_logits(model, ids, patches, cids)
        n = len(MODE_LAYERS)
        results.append({"recipient": m["item_id"], "donor": donor["item_id"], "control_donor": ctrl["item_id"],
                        "gold": m["gold"], "wrong_answer": m["pred"], "donor_gold": donor["gold"],
                        "control_donor_answer": ctrl["pred"], "unpatched": logits[0],
                        "treatment": logits[1:1 + n], "control": logits[1 + n:]})
    torch.save({"bank": bank, "format": fmt, "h": h, "group": group_name, "layers": MODE_LAYERS,
                "n_correct_donors": len(correct), "colours": colours, "results": results}, path)
    summarise_modepatch(path)


def summarise_modepatch(path) -> None:
    blob = torch.load(path, weights_only=False)
    colours, res = blob["colours"], blob["results"]
    g = torch.Generator().manual_seed(0)
    idx = lambda key: torch.tensor([colours.index(r[key]) for r in res])
    gold, wrong_answer, donor_gold = idx("gold"), idx("wrong_answer"), idx("donor_gold")
    print(f"\n## mode transplant, {blob['bank']} {blob['format']} h={blob['h']}: {len(res)} {blob['group']} recipients, "
          f"{blob['n_correct_donors']} correct donors")
    print("layer: own gold, treatment vs control, paired diff [95% CI] | own wrong answer kept (t, c) | "
          "donor's gold (t)")
    for k, l in enumerate(blob["layers"]):
        t = torch.stack([r["treatment"][k] for r in res]).argmax(-1)
        c = torch.stack([r["control"][k] for r in res]).argmax(-1)
        mu, lo, hi = mean_ci((t == gold).float() - (c == gold).float(), g)
        print(f"  {l:>2}: {float((t == gold).float().mean()):.2f} vs {float((c == gold).float().mean()):.2f} "
              f"diff {mu:+.2f} [{lo:+.2f},{hi:+.2f}] | {float((t == wrong_answer).float().mean()):.2f}, "
              f"{float((c == wrong_answer).float().mean()):.2f} | {float((t == donor_gold).float().mean()):.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["probe", "backpatch", "twinpatch", "modepatch"])
    p.add_argument("--bank", required=True, help="for probe: one or more banks joined by '+'")
    p.add_argument("--k-pca", type=int, default=None)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--positions", nargs="+", default=["final"])
    p.add_argument("--site", choices=["final", "count"], default="final")
    p.add_argument("--format", default="three_first", choices=B.FORMATS)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--h", type=int, default=2)
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()
    if a.cmd == "probe":
        probe(a.bank.split("+"), a.k_pca, a.wd, tuple(a.positions), save=not a.no_save)
    elif a.cmd == "backpatch":
        backpatch(a.bank)
    elif a.cmd == "twinpatch":
        twinpatch(a.bank, a.site, a.format, a.reverse)
    else:
        modepatch(a.bank, a.format, a.h)


if __name__ == "__main__":
    main()
