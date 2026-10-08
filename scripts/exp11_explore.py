"""exp11 dev exploration (manifest: the dev phase may choose the layer band, primary position, probe settings).

Prints, for one captured bank:
  1. behaviour: accuracy per h, colour-token mass, h=2/h=3 error families vs the permutation null;
  2. lens readouts at h=2 (logit lens, J-lens, R-lens), per captured position and layer, per group: the fraction of
     items whose target colour is the top colour among the colours other than the model's own answer. Targets:
     s0 (start), s1 (first intermediate), decoy (start's colour under the unused ingredient), and, for correct items,
     the shortcut colour; chance is 1/9.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch

import exp11 as X
import exp11_brew as B

FAMILIES = ("stop_early", "skip_one_stir", "reverse_order", "last_stir_only", "wrong_column", "inverse_lookup")


def load(bank: str):
    items = {r["item_id"]: r for r in X.load_bank(bank)}
    metas = [json.loads(l) for l in X.capture_paths(bank)[1].open()]
    blob = torch.load(X.capture_paths(bank)[0], weights_only=False)
    for m, hidden in zip(metas, blob["hidden"]):
        it = items[m["item_id"]]
        brew = B.parse(it["problem"])
        probs = torch.softmax(torch.tensor(m["colour_logits"]), -1)
        m.update(brew=brew, gold=it["answer"], hidden=hidden, pred=m["colours"][int(probs.argmax())],
                 p_pred=float(probs.max()), confident=float(probs.max()) >= 0.5, format=it.get("format", "three_first"),
                 p_gold=float(probs[m["colours"].index(it["answer"])]))
        m["group"] = group(m)
    return metas


def row_mates(brew: B.Brew) -> dict[str, str]:
    """The start colour's rule-line entries for an h=2 item: s1, the shortcut answer, and the decoy."""
    a, b = brew.stirs[0], brew.stirs[-1]
    c = next(i for i in brew.ingredients if i not in brew.stirs)
    row = brew.table[brew.start]
    return {"s1": row[a], "shortcut": row[b], "decoy": row[c]}


def group(m: dict) -> str:
    if m["h"] < 2:
        return "h1"
    if m["pred"] == m["gold"]:
        return "correct"
    shortcut = m["brew"].apply(m["brew"].start, m["brew"].stirs[-1:])
    return "shortcut" if m["pred"] == shortcut else "other_wrong"


def behaviour(metas: list[dict]) -> None:
    for fmt in sorted({m.get("format", "three_first") for m in metas}):
        print(f"\n## behaviour, format {fmt} (bf16, exact answer distribution)")
        behaviour_one([m for m in metas if m.get("format", "three_first") == fmt])


def behaviour_one(metas: list[dict]) -> None:
    rng = random.Random(0)
    for h in sorted({m["h"] for m in metas}):
        ms = [m for m in metas if m["h"] == h]
        acc = sum(m["pred"] == m["gold"] for m in ms) / len(ms)
        mass = min(m["colour_mass"] for m in ms)
        groups = {g: sum(m["group"] == g for m in ms) for g in ("correct", "shortcut", "other_wrong")}
        conf = sum(m["confident"] for m in ms) / len(ms)
        print(f"h={h}: n={len(ms)} acc={acc:.2f} min_colour_mass={mass:.4f} confident={conf:.2f} groups={groups}")
        wrong = [m for m in ms if m["pred"] != m["gold"]]
        if h == 1 or not wrong:
            continue
        sets = {m["item_id"]: {f: s - {m["gold"]} for f, s in B.hypotheses(m["brew"]).items()} for m in ms}
        cells = []
        for f in FAMILIES:
            obs = sum(m["pred"] in sets[m["item_id"]][f] for m in wrong) / len(wrong)
            perm = []
            for _ in range(200):
                hits = []
                for m in wrong:
                    o = rng.choice([x for x in ms if x is not m])
                    if m["pred"] != o["gold"]:
                        hits.append(m["pred"] in sets[o["item_id"]][f])
                perm.append(sum(hits) / max(1, len(hits)))
            cells.append(f"{f}={obs:.2f}(p{sum(perm) / len(perm):.2f})")
        print("   wrong:", len(wrong), " ".join(cells))


@torch.no_grad()
def readouts(metas: list[dict], lens_names=("logit", "j-lens", "r-lens")) -> dict:
    import exp11_model as M
    tokenizer, model = M.load()
    colours = metas[0]["colours"]
    ids = list(M.colour_ids(tokenizer, colours).values())
    n_layers = len(M.text_parts(model)[0])
    h2 = [m for m in metas if m["h"] == 2]
    out = {}
    for name in lens_names:
        lens = M.LogitLens(model, ids, n_layers) if name == "logit" else M.Lens(name, model, ids)
        # tallies[(position, group, target)][layer] -> [hits, n]
        tallies = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for m in h2:
            mates = row_mates(m["brew"])
            if len(set(mates.values())) < 3:
                continue
            targets = {"s0": m["brew"].start, "s1": mates["s1"], "decoy": mates["decoy"]}
            if m["group"] == "correct":
                targets["shortcut"] = mates["shortcut"]
            logits = lens.logits(m["hidden"])                      # [layers, positions, 10]
            others = [i for i, c in enumerate(colours) if c != m["pred"]]
            top = logits[:, :, others].argmax(-1)                 # [layers, positions] index into `others`
            for p, pos in enumerate(m["positions"]):
                for t, colour in targets.items():
                    hit = (top[:, p] == others.index(colours.index(colour))) if colour != m["pred"] else None
                    if hit is None:
                        continue
                    for layer, v in zip(lens.layers, hit.tolist()):
                        tallies[(pos, m["group"], t)][layer][0] += v
                        tallies[(pos, m["group"], t)][layer][1] += 1
        out[name] = {k: {l: v[0] / v[1] for l, v in d.items()} for k, d in tallies.items()}
        out[name + "_n"] = {k: next(iter(d.values()))[1] for k, d in tallies.items()}
        del lens
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def paired_margins(metas: list[dict], lens_names=("logit", "j-lens", "r-lens"), n_boot: int = 1000) -> None:
    """Per item, the lens logit of a target minus the decoy's (both entries of the start colour's rule line, so the
    null is 0 by construction); mean over items with a bootstrap 95% CI, per lens, position and layer.
      h=2 shortcut items: target s1 (is the first hop computed when the model takes the shortcut?)
      h=1 items (positive control): target the answer table[s0][stir1], the same lookup made as the final answer"""
    import exp11_model as M
    tokenizer, model = M.load()
    colours = metas[0]["colours"]
    ids = list(M.colour_ids(tokenizer, colours).values())
    n_layers = len(M.text_parts(model)[0])
    cases = []
    for m in metas:
        b = m["brew"]
        unused = [i for i in b.ingredients if i not in b.stirs]
        if m["h"] == 1:
            cases.append(("h1 answer", m, b.table[b.start][b.stirs[0]], b.table[b.start][unused[0]]))
        elif m["h"] == 2 and m["group"] == "shortcut":
            mates = row_mates(b)
            if len(set(mates.values())) == 3:
                cases.append(("h2 shortcut s1", m, mates["s1"], mates["decoy"]))
    g = torch.Generator().manual_seed(0)
    for name in lens_names:
        lens = M.LogitLens(model, ids, n_layers) if name == "logit" else M.Lens(name, model, ids)
        diffs = defaultdict(list)                                  # (case, position) -> [items] of [layers]
        for case, m, target, decoy in cases:
            logits = lens.logits(m["hidden"])                      # [layers, positions, 10]
            d = logits[:, :, colours.index(target)] - logits[:, :, colours.index(decoy)]
            for p, pos in enumerate(m["positions"]):
                if pos in ("start", "stir1", "stir2", "question_end", "final"):
                    diffs[(case, pos)].append(d[:, p])
        print(f"\n## {name}: mean logit(target) - logit(decoy) [95% CI], layers 8..60 step 8")
        for (case, pos), rows in sorted(diffs.items()):
            x = torch.stack(rows)                                  # [items, layers]
            boot = x[torch.randint(len(x), (n_boot, len(x)), generator=g)].mean(1)
            lo, hi = boot.quantile(0.025, 0), boot.quantile(0.975, 0)
            cells = [f"{x[:, l].mean():+.2f}[{lo[l]:+.2f},{hi[l]:+.2f}]" for l in range(8, 61, 8)]
            print(f"{case:>15} {pos:>12} n={len(x):>3} | " + " ".join(cells))
        del lens
        torch.cuda.empty_cache()


@torch.no_grad()
def paired_single(metas: list[dict], lens_names=("j-lens", "r-lens"), positions=("question_end", "final"),
                  layers=range(24, 61, 4), n_boot: int = 1000) -> None:
    """Single-table items: per item, the lens logit of each trajectory state minus the mean logit of the off-path
    colours (not on the trajectory and not the model's answer); mean over items with a bootstrap 95% CI, per
    captured position, split into correct / one_short (answer = the state one stir before the end, so that state
    is the answer) / other_wrong, at h=2 and h=3."""
    import exp11_model as M
    tokenizer, model = M.load()
    colours = metas[0]["colours"]
    ids = list(M.colour_ids(tokenizer, colours).values())
    single = [m for m in metas if m["format"].startswith("single") and m["h"] >= 2]
    g = torch.Generator().manual_seed(0)
    for name in lens_names:
        lens = M.Lens(name, model, ids)
        rows = defaultdict(list)                                   # (position, h, group, target) -> [items] of [layers]
        for m in single:
            path = m["brew"].path()
            off = [colours.index(c) for c in colours if c not in path and c != m["pred"]]
            grp = ("correct" if m["pred"] == m["gold"] else
                   "one_short" if m["pred"] == path[-2] else "other_wrong")
            logits = lens.logits(m["hidden"])                      # [layers, positions, 10]
            for pos in positions:
                at = logits[:, list(m["positions"]).index(pos)]    # [layers, 10]
                base = at[:, off].mean(-1)
                for k in range(1, m["h"] + 1):
                    rows[(pos, m["h"], grp, f"s{k}")].append(at[:, colours.index(path[k])] - base)
        print(f"\n## {name}, single table: logit(state) - mean logit(off-path) [95% CI], layers {list(layers)}")
        for key, xs in sorted(rows.items()):
            x = torch.stack(xs)
            boot = x[torch.randint(len(x), (n_boot, len(x)), generator=g)].mean(1)
            lo, hi = boot.quantile(0.025, 0), boot.quantile(0.975, 0)
            print(f"{key[0]:>12} h={key[1]} {key[2]:>11} {key[3]} n={len(x):>3} | "
                  + " ".join(f"{x[:, l].mean():+.1f}[{lo[l]:+.1f},{hi[l]:+.1f}]" for l in layers))
        del lens
        torch.cuda.empty_cache()


@torch.no_grad()
def answer_line_mate(metas: list[dict], n_boot: int = 1000, layers=range(24, 61, 4)) -> None:
    """Line-mate check, single-table items, final position, J-lens: the logit of the model's answer and of its
    inverse image f^-1(answer) (the input colour of the rule line that outputs the answer) minus the mean logit of
    the colours that are neither on the trajectory nor either of those two. If f^-1(answer) rises together with the
    answer in other_wrong items too, a visible predecessor marks reading the answer's rule line, not a held state."""
    import exp11_model as M
    tokenizer, model = M.load()
    colours = metas[0]["colours"]
    lens = M.Lens("j-lens", model, list(M.colour_ids(tokenizer, colours).values()))
    rows = defaultdict(list)
    g = torch.Generator().manual_seed(0)
    for m in metas:
        if not m["format"].startswith("single") or m["h"] < 2:
            continue
        b, path = m["brew"], m["brew"].path()
        pre = b.inverse(m["pred"], B.SINGLE)
        grp = ("correct" if m["pred"] == m["gold"] else "one_short" if m["pred"] == path[-2] else
               "other_wrong_offpath_pre" if pre not in path else "other_wrong_onpath_pre")
        at = lens.logits(m["hidden"])[:, list(m["positions"]).index("final")]
        base = at[:, [colours.index(c) for c in colours if c not in path and c not in (m["pred"], pre)]].mean(-1)
        rows[(m["h"], grp, "answer")].append(at[:, colours.index(m["pred"])] - base)
        rows[(m["h"], grp, "f^-1(answer)")].append(at[:, colours.index(pre)] - base)
    print(f"\n## j-lens line-mate check, single table, final position, layers {list(layers)}")
    for key, xs in sorted(rows.items()):
        x = torch.stack(xs)
        boot = x[torch.randint(len(x), (n_boot, len(x)), generator=g)].mean(1)
        lo, hi = boot.quantile(0.025, 0), boot.quantile(0.975, 0)
        print(f"h={key[0]} {key[1]:>23} {key[2]:>12} n={len(x):>3} | "
              + " ".join(f"{x[:, l].mean():+.1f}[{lo[l]:+.1f},{hi[l]:+.1f}]" for l in layers))


def show(out: dict, band=range(0, 63, 4)) -> None:
    for name in ("logit", "j-lens", "r-lens"):
        print(f"\n## {name}: fraction with the target as top non-answer colour (chance 0.11); layers {list(band)}")
        for key in sorted(out[name]):
            pos, grp, tgt = key
            curve = out[name][key]
            print(f"{pos:>12} {grp:>11} {tgt:>8} n={out[name + '_n'][key]:>3} | "
                  + " ".join(f"{curve[l]:.2f}" for l in band if l in curve))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev")
    p.add_argument("--no-lens", action="store_true")
    p.add_argument("--paired", action="store_true", help="only the paired target-minus-decoy margins")
    p.add_argument("--single", action="store_true", help="only the single-table state margins")
    p.add_argument("--line-mate", action="store_true", help="answer vs f^-1(answer) at the final position")
    p.add_argument("--single-positions", action="store_true",
                   help="single-table state margins at the start-colour and stir-count tokens, J-lens, layers 8..60")
    a = p.parse_args()
    metas = load(a.bank)
    behaviour(metas)
    if a.single:
        paired_single(metas)
    elif a.line_mate:
        answer_line_mate(metas)
    elif a.single_positions:
        paired_single(metas, lens_names=("j-lens",), positions=("start", "stirs"), layers=range(8, 61, 4))
    elif a.paired:
        paired_margins(metas)
    elif not a.no_lens:
        out = readouts(metas)
        show(out)
        path = X.EXP.cache / f"explore_{a.bank}_readouts.json"
        path.write_text(json.dumps({n: {"|".join(k): v for k, v in d.items()} for n, d in out.items()}))


if __name__ == "__main__":
    main()
