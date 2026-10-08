"""exp11 test-phase readouts and the pre-registered reading rules (manifest + test addendum).

  q2_three  --probe-file F     Q2, three ingredients: apply the reading rules to a saved probe run
  q2_single --bank B           Q2, single table: J-lens / R-lens answer-relative readouts at the final position
                               (held-state contrast D and the correct-vs-one_short AUC, with the answer-contrast control)

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


def q2_three(probe_file: str, position: str = "final") -> dict:
    blob = torch.load(probe_file, weights_only=False)
    band = blob["band"]
    res = blob["results"]
    g = torch.Generator().manual_seed(0)

    def contrast(h, target, members):
        c_t = res[(h, position, target)]["correct"][members].float()
        c_d = res[(h, position, "decoy")]["correct"][members].float()
        return (c_t - c_d)[:, band].mean(1)

    out = {}
    h1 = list(range(len(res[(1, position, "answer")]["groups"])))
    out["guard_h1"] = boot_mean(contrast(1, "answer", h1), g)
    groups = res[(2, position, "s1")]["groups"]
    members = {grp: [i for i, x in enumerate(groups) if x == grp] for grp in ("correct", "shortcut", "other_wrong")}
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
    print(f"## Q2 three ingredients, {probe_file}, position {position}, band {band[0]}..{band[-1]}")
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
def q2_single(bank: str, lens_name: str = "j-lens") -> dict:
    """Per single-table item at h in {2, 3}, final position, lens margins over SINGLE_BAND of the colours f^d(answer)
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
    band = list(SINGLE_BAND)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["q2_three", "q2_single"])
    p.add_argument("--probe-file")
    p.add_argument("--bank")
    p.add_argument("--lens", default="j-lens")
    a = p.parse_args()
    if a.cmd == "q2_three":
        q2_three(a.probe_file)
    else:
        q2_single(a.bank, a.lens)


if __name__ == "__main__":
    main()
