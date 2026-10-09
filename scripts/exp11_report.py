"""exp11 test-phase readouts and the pre-registered reading rules (manifest + test addendum).

  q1_single --bank B           Q1, single table: wrong answers by the number of stirs applied along the start's cycle
  q2_three  --probe-file F     Q2, three ingredients: apply the reading rules to a saved probe run
  q2_single --bank B           Q2, single table: J-lens / R-lens answer-relative readouts at the final position
                               (held-state contrast D and the correct-vs-one_short AUC, with the answer-contrast control)
  q3        --bank B           Q3: apply the addendum's reading rules to the saved patching runs of a bank

Layers are block outputs (layer l = hidden[l + 1]).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch

import exp11_brew as B
import exp11_explore as E

N_BOOT = 1000
THRESHOLD = 0.05                     # manifest reading_rules_q2 (unchanged)
SINGLE_BAND = range(36, 49)          # single-table mid band: before the answer's onset (layer 52 on dev)


def boot_mean(x: torch.Tensor, g: torch.Generator) -> tuple[float, float, float]:
    b = x[torch.randint(len(x), (N_BOOT, len(x)), generator=g)].mean(1)
    return float(x.mean()), float(b.quantile(0.025)), float(b.quantile(0.975))


# --- Q2, three ingredients: probe reading rules ---------------------------------------------------------------------
def onset(curve: torch.Tensor) -> int:
    """First layer at which an accuracy curve exceeds half of its own peak."""
    return int((curve > 0.5 * curve.max()).nonzero()[0])


def q2_three(probe_file: str, position: str = "final", confident_ids: set[str] | None = None) -> dict:
    """The reading rules on a saved probe run; with `confident_ids`, every group is restricted to those items (the
    secondary confident-subset read; the h=1 guard is kept on all h=1 items)."""
    blob = torch.load(probe_file, weights_only=False)
    band = blob["band"]
    res = blob["results"]
    g = torch.Generator().manual_seed(0)
    keep = (lambda i: True) if confident_ids is None else \
        (lambda i: res[(2, position, "s1")]["item_ids"][i] in confident_ids)

    def contrast(h, target, members, pos=position):
        c_t = res[(h, pos, target)]["correct"][members].float()
        c_d = res[(h, pos, "decoy")]["correct"][members].float()
        return (c_t - c_d)[:, band].mean(1)

    out = {}
    h1 = list(range(len(res[(1, "final", "answer")]["groups"])))
    out["guard_h1"] = boot_mean(contrast(1, "answer", h1, pos="final"), g)        # the guard is read at final
    groups = res[(2, position, "s1")]["groups"]
    members = {grp: [i for i, x in enumerate(groups) if x == grp and keep(i)]
               for grp in ("correct", "shortcut", "other_wrong")}
    for grp, idx in members.items():
        out[grp] = {"n": len(idx), "s1_minus_decoy": boot_mean(contrast(2, "s1", idx), g) if idx else None}
    # onsets of the s1 accuracy curve, shortcut minus correct, bootstrapped within groups
    s1 = res[(2, position, "s1")]["correct"].float()
    diffs = []
    for _ in range(N_BOOT):
        pick = {grp: torch.tensor(idx)[torch.randint(len(idx), (len(idx),), generator=g)]
                for grp, idx in members.items() if grp != "other_wrong"}
        diffs.append(onset(s1[pick["shortcut"]].mean(0)) - onset(s1[pick["correct"]].mean(0)))
    diffs = torch.tensor(diffs, dtype=torch.float32)
    out["onset_shortcut_minus_correct"] = (onset(s1[members["shortcut"]].mean(0)) - onset(s1[members["correct"]].mean(0)),
                                           float(diffs.quantile(0.025)), float(diffs.quantile(0.975)))
    out["reading"] = read_three(out)
    subset = "" if confident_ids is None else ", confident subset (argmax p >= 0.5)"
    print(f"## Q2 three ingredients, {probe_file}, position {position}, band {band[0]}..{band[-1]}{subset}")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


def above(x) -> bool:
    return x is not None and x[0] > THRESHOLD and x[1] > 0


def null(x) -> bool:
    return x is not None and x[0] <= THRESHOLD and x[1] <= 0 <= x[2]


def read_three(out: dict) -> str:
    """The manifest's rules, with the test addendum's guard: the h=1 positive control (the same lookup made as the
    final answer) must be readable, otherwise nothing is read."""
    if not above(out["guard_h1"]):
        return "no_reading (the h=1 positive control fails)"
    sc, co = out["shortcut"]["s1_minus_decoy"], out["correct"]["s1_minus_decoy"]
    if null(sc):
        return "never_computed" + ("" if above(co) else " (correct group not itself above threshold)")
    if above(sc) and above(co):
        lo, hi = out["onset_shortcut_minus_correct"][1:]
        return "too_late" if lo > 0 else "unused" if lo <= 0 <= hi else "earlier_in_shortcut"
    if above(sc):
        return "present_in_shortcut (correct group not above threshold: too_late vs unused not readable)"
    return "indeterminate (shortcut contrast neither above threshold nor a null)"


# --- Q2, single table: answer-relative lens readouts -----------------------------------------------------------------
def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(a positive scores above a negative), ties counted half (Mann-Whitney)."""
    d = pos[:, None] - neg[None, :]
    return float((d > 0).float().mean() + 0.5 * (d == 0).float().mean())


@torch.no_grad()
def q2_single(bank: str, lens_name: str = "j-lens", band=SINGLE_BAND) -> dict:
    """Per single-table item at h in {2, 3}, final position, lens margins over `band` of the colours f^d(answer)
    (d = -1, 0) and of the trajectory states, each minus the mean of the colours that are neither on the trajectory
    nor within two stirs of the answer.
      D   = margin(f^-1(answer)) - margin(answer): a held state that precedes the answer (> 0)
      R   = margin(s2) - margin(s1), h=3: which state is held; AUC for correct (answer s3) vs one_short (answer s2)
      A   = margin(s3) - margin(s2), h=3: the answer contrast in the same band (the control: if AUC(A) >= AUC(R),
            the band already reads the answer and R is not evidence of a held state)"""
    import exp11_model as M
    tokenizer, model = M.load()
    metas = [m for m in E.load(bank) if m["format"].startswith("single") and m["h"] >= 2]
    colours = metas[0]["colours"]
    lens = M.Lens(lens_name, model, list(M.colour_ids(tokenizer, colours).values()))
    band = list(band)
    per = defaultdict(lambda: defaultdict(list))
    for m in metas:
        b, path = m["brew"], m["brew"].path()
        orbit = {0: m["pred"]}
        for d in (1, 2):
            orbit[d] = b.apply(orbit[d - 1], [B.SINGLE])
            orbit[-d] = b.inverse(orbit[-d + 1], B.SINGLE)
        grp = ("correct" if m["pred"] == m["gold"] else "one_short" if m["pred"] == path[-2] else
               "one_extra" if orbit[-1] == m["gold"] else "other_wrong")
        at = lens.logits(m["hidden"])[:, list(m["positions"]).index("final")][band]          # [band, 10]
        base = at[:, [colours.index(c) for c in colours if c not in path and c not in orbit.values()]].mean(-1)
        margin = {c: float((at[:, colours.index(c)] - base).mean()) for c in set(path) | set(orbit.values())}
        per[(m["h"], grp)]["D"].append(margin[orbit[-1]] - margin[orbit[0]])
        if m["h"] == 3:
            per[(3, grp)]["R"].append(margin[path[2]] - margin[path[1]])
            per[(3, grp)]["A"].append(margin[path[3]] - margin[path[2]])
    g = torch.Generator().manual_seed(0)
    out = {"bank": bank, "lens": lens_name, "band": band, "D": {}, "auc": {}}
    print(f"## Q2 single table, {bank}, {lens_name}, final position, band {band[0]}..{band[-1]}")
    for key, v in sorted(per.items()):
        out["D"][key] = {"n": len(v["D"]), "D": boot_mean(torch.tensor(v["D"]), g)}
        print(f"  h={key[0]} {key[1]:>11} n={len(v['D']):>3} D = {out['D'][key]['D']}")
    pos, neg = per[(3, "correct")], per[(3, "one_short")]
    for stat in ("R", "A"):
        p, n = torch.tensor(pos[stat]), torch.tensor(neg[stat])
        out["auc"][stat] = auc(p, n)
    boots = []
    for _ in range(N_BOOT):
        idx_p = torch.randint(len(pos["R"]), (len(pos["R"]),), generator=g)
        idx_n = torch.randint(len(neg["R"]), (len(neg["R"]),), generator=g)
        r = auc(torch.tensor(pos["R"])[idx_p], torch.tensor(neg["R"])[idx_n])
        a = auc(torch.tensor(pos["A"])[idx_p], torch.tensor(neg["A"])[idx_n])
        boots.append((r, a, r - a))
    boots = torch.tensor(boots)
    out["auc_ci"] = {k: (float(boots[:, i].quantile(0.025)), float(boots[:, i].quantile(0.975)))
                     for i, k in enumerate(("R", "A", "R_minus_A"))}
    print(f"  h=3 correct (n={len(pos['R'])}) vs one_short (n={len(neg['R'])}): AUC(R) = {out['auc']['R']:.3f} "
          f"{out['auc_ci']['R']}, AUC(A) = {out['auc']['A']:.3f} {out['auc_ci']['A']}, "
          f"AUC(R) - AUC(A) CI {out['auc_ci']['R_minus_A']}")
    return out


# --- Q1, single table -----------------------------------------------------------------------------------------------
def q1_single(bank: str) -> None:
    """Each wrong answer by the number of stirs applied along the start colour's cycle, nearest to h when the cycle
    is short (a colour can sit at several cycle positions); 'off_cycle' if it is on none of 0..h+3."""
    import json
    import exp11 as X
    items = {r["item_id"]: r for r in X.load_bank(bank)}
    tally = defaultdict(lambda: defaultdict(int))
    for m in map(json.loads, X.capture_paths(bank)[1].open()):
        it = items[m["item_id"]]
        b = B.parse(it["problem"])
        if B.SINGLE not in b.stirs:
            continue
        pred = m["colours"][int(torch.tensor(m["colour_logits"]).argmax())]
        hops = [k for k in range(it["h"] + 4) if b.apply(b.start, [B.SINGLE] * k) == pred]
        label = ("correct" if pred == it["answer"] else
                 f"{min(hops, key=lambda k: abs(k - it['h'])) - it['h']:+d} stirs" if hops else "off_cycle")
        tally[(it.get("format"), it["h"])][label] += 1
    print(f"## Q1 single table, {bank}: answers by stirs applied (a random wrong colour: 1/9 per cycle position)")
    for (fmt, h), c in sorted(tally.items()):
        n = sum(c.values())
        print(f"  {fmt} h={h} n={n} acc={c['correct'] / n:.3f} "
              + " ".join(f"{k}={v}" for k, v in sorted(c.items())))


# --- Q3: the addendum's reading rules on saved patching runs ---------------------------------------------------------
def _argmax(rows, key, k=None):
    return torch.stack([r[key] if k is None else r[key][k] for r in rows]).argmax(-1)


def _paired(a: torch.Tensor, b: torch.Tensor, g) -> tuple[float, float, float]:
    """Mean of the paired difference of two indicator vectors, with a bootstrap 95% CI."""
    return boot_mean(a.float() - b.float(), g)


def _rescue(name: str, blob: dict, layer_index: int, label: str, g) -> None:
    """Treatment minus control rate of argmax = the recipient's own gold at one patch setting; rescue iff > 0.05
    with the CI excluding 0."""
    colours, res = blob["colours"], blob["results"]
    gold = torch.tensor([colours.index(r["gold"]) for r in res])
    t, c = _argmax(res, "treatment", layer_index) == gold, _argmax(res, "control", layer_index) == gold
    d = _paired(t, c, g)
    verdict = "rescue" if d[0] > THRESHOLD and d[1] > 0 else "no rescue"
    print(f"  {name} ({label}, n={len(res)}): gold {float(t.float().mean()):.3f} vs control "
          f"{float(c.float().mean()):.3f}, diff {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] -> {verdict}")


def _twin_readings(blob: dict, g, setting_band=range(36, 49)) -> None:
    """Twin patches: the composition / setting-window / colour-arrival readings (Q3b), and for the count site the
    read-out layer and the item-independence check (part B)."""
    colours, res, layers = blob["colours"], blob["results"], blob["layers"]
    idx = lambda key: torch.tensor([colours.index(r[key]) for r in res])
    gold, donor_gold, ctrl_gold = idx("gold"), idx("donor_gold"), idx("control_donor_gold")
    base = _argmax(res, "unpatched")
    tag = f"site={blob['site']} reverse={blob['reverse']} {blob.get('format', '')} n={len(res)}"
    if blob["site"] == "final" and not blob["reverse"] and blob.get("format", "three_first") == "three_first":
        best = max(((l, _paired(_argmax(res, "treatment", k) == gold, base == gold, g))
                    for k, l in enumerate(layers) if l in setting_band), key=lambda x: x[1][0])
        comp = best[1][0] > THRESHOLD and best[1][1] > 0
        print(f"  [{tag}] composition after write-in: best layer {best[0]} gold minus unpatched "
              f"{best[1][0]:+.3f} [{best[1][1]:+.3f},{best[1][2]:+.3f}] -> {'composition' if comp else 'no composition'}")
    window, arrival = [], None
    for k, l in enumerate(layers):
        c = _argmax(res, "control", k)
        d = _paired(c == donor_gold, c == ctrl_gold, g)
        if d[0] > 0.2 and d[1] > 0:
            window.append(l)
        if arrival is None and float((c == ctrl_gold).float().mean()) > 0.5:
            arrival = l
    print(f"  [{tag}] control: item-independent setting window {window or 'none'}; colour arrival layer {arrival}")
    if blob["site"] == "count":
        inc = [float((_argmax(res, "treatment", k) == donor_gold).float().mean() - (base == donor_gold).float().mean())
               for k in range(len(layers))]
        k_max = max(range(len(layers)), key=lambda k: inc[k])
        readout = max(layers[k] for k in range(len(layers)) if k >= k_max and inc[k] >= 0.5 * inc[k_max])
        d = _paired(_argmax(res, "treatment", k_max) == donor_gold, _argmax(res, "control", k_max) == donor_gold, g)
        indep = -0.1 <= d[1] and d[2] <= 0.1
        print(f"  [{tag}] count: max increase {inc[k_max]:+.3f} at layer {layers[k_max]}; read-out layer {readout}; "
              f"treatment minus control there {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] -> "
              f"{'item-independent count' if indep else 'not shown item-independent'}")


def q3(bank: str) -> None:
    import exp11 as X
    g = torch.Generator().manual_seed(0)
    print(f"## Q3 readings, {bank}")
    for path in sorted(X.EXP.cache.glob("q23_*.pt")):
        kind = path.name.split("_")[1]
        if kind == "probe":
            continue
        blob = torch.load(path, weights_only=False)
        if blob["bank"] != bank:
            continue
        if kind == "backpatch":
            k = blob["pairs"].index((32, 48))
            _rescue("back-patching", blob, k, "primary pair 32<-48", g)
        elif kind == "modepatch":
            _rescue(f"mode transplant {blob['format']} h={blob['h']}", blob, blob["layers"].index(40),
                    "primary layer 40", g)
        elif kind == "twinpatch":
            _twin_readings(blob, g)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["q1_single", "q2_three", "q2_single", "q3"])
    p.add_argument("--probe-file")
    p.add_argument("--bank")
    p.add_argument("--lens", default="j-lens")
    p.add_argument("--confident", action="store_true", help="q2_three on the confident subsets")
    p.add_argument("--position", default="final", help="q2_three: the probe position (h=1 guard read at final)")
    a = p.parse_args()
    if a.cmd == "q1_single":
        q1_single(a.bank)
    elif a.cmd == "q2_three":
        ids = None
        if a.confident:
            bank = torch.load(a.probe_file, weights_only=False)["banks"][0]
            ids = {m["item_id"] for m in E.load(bank) if m["confident"]}
        q2_three(a.probe_file, position=a.position, confident_ids=ids)
    elif a.cmd == "q2_single":
        q2_single(a.bank, a.lens)
    else:
        q3(a.bank)


if __name__ == "__main__":
    main()
