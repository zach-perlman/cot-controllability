"""exp15 exploratory confound check: how much of each read AUC is colour prior rather than item information?

Every reader has fixed preferences among the 10 colour words (the layer-0 lens reads of w are already above 0.5 at
question_end, where nothing item-specific has been computed). If the key w is not uniformly distributed over
colours on errors, AUC(w vs controls) exceeds 0.5 with no item information. Two corrections, per reader and group:
  shuffled null   S from another item of the same group, targets and controls from this item (n_perm
                  permutations): keeps every colour prior, removes the item-specific signal. Reported: the null mean,
                  its 2.5-97.5 percentiles, the observed AUC minus the null mean, and the permutation p
  prior-corrected each colour's mean S over all items of the bank that the reader scored is subtracted before the
                  AUC (item-level bootstrap CI as in the main report)
for P1 (key vs controls on correct h=2 and h=3 items) and P2 (w vs controls, w vs T(a') on consistent off-cycle
errors); and the P2 hypothesis decomposition (which of w, the true s_{h-1}, a control ranks top) on raw and
prior-corrected scores, next to the shares colour preferences alone produce.

  python exp15_shuffle_null.py --bank test --tag test_shuffle_null_v1
"""
from __future__ import annotations

import argparse
import json

import torch

import exp15 as E
import exp15_report as R
import exp15_score as S
from exp15 import EXP

N_PERM = 2000
SEED = 0


def group_tensors(rs: S.ReaderScores, members, index: dict, target, others):
    """Scores [n, 10], target index [n], others mask [n, 10] for the members the reader scored."""
    kept = [it for it in members if rs.score(it.item_id) is not None]
    scores = torch.stack([rs.score(it.item_id).float() for it in kept])
    tgt = torch.tensor([index[target(it)] for it in kept])
    mask = torch.zeros(len(kept), 10, dtype=torch.bool)
    for k, it in enumerate(kept):
        mask[k, [index[c] for c in others(it)]] = True
    return scores, tgt, mask


def item_aucs(scores: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per item: P(S(target) > S(other)) over its others, ties half."""
    t = scores[torch.arange(len(scores)), tgt][:, None]
    win = (t > scores).float() + 0.5 * (t == scores).float()
    return (win * mask).sum(1) / mask.sum(1)


def shuffled_null(scores, tgt, mask, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    observed = float(item_aucs(scores, tgt, mask).mean())
    g = torch.Generator().manual_seed(seed)
    null = torch.tensor([float(item_aucs(scores[torch.randperm(len(scores), generator=g)], tgt, mask).mean())
                         for _ in range(n_perm)])
    return {"observed": round(observed, 4), "null_mean": round(float(null.mean()), 4),
            "null_lo": round(float(null.quantile(0.025)), 4), "null_hi": round(float(null.quantile(0.975)), 4),
            "excess": round(observed - float(null.mean()), 4),
            "p_perm": round(float(((null >= observed).sum() + 1) / (n_perm + 1)), 5), "n": len(scores)}


def top_shares(scores: torch.Tensor, members, index: dict) -> dict:
    """Shares of items where w, the true s_{h-1}, or a control ranks top among {w, true, controls} (ties split)."""
    rows = [S.read_metrics(s, it, index) for s, it in zip(scores, members)]
    return {m: round(sum(r[m] for r in rows) / len(rows), 4) for m in ("top_key", "top_true", "top_neither")}


def decomposition(rs: S.ReaderScores, members, index: dict, prior: torch.Tensor, n_perm: int = 200) -> dict:
    """The P2 hypothesis decomposition on raw S, on prior-corrected S, and its expectation with S from a shuffled
    item (raw S; the shares a reader with only colour preferences would produce)."""
    kept = [it for it in members if rs.score(it.item_id) is not None]
    scores = torch.stack([rs.score(it.item_id).float() for it in kept])
    g = torch.Generator().manual_seed(SEED)
    perms = [top_shares(scores[torch.randperm(len(kept), generator=g)], kept, index) for _ in range(n_perm)]
    return {"raw": top_shares(scores, kept, index), "prior_corrected": top_shares(scores - prior, kept, index),
            "shuffled_null": {m: round(sum(p[m] for p in perms) / n_perm, 4) for m in perms[0]}, "n": len(kept)}


def colour_means(rs: S.ReaderScores, item_ids) -> torch.Tensor:
    return torch.stack([s for i in item_ids if (s := rs.score(i)) is not None]).float().mean(0)


def shuffle_report(bank: str, tag: str) -> None:
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    table = E.item_table(bank)
    metas, _ = E.load_capture(bank, with_hidden=False)
    colours = metas[0]["colours"]
    index = {c: k for k, c in enumerate(colours)}
    groups = R.read_groups(table, metas)
    tests = {"P1_correct_h2|key_auc": ("P1_correct_h2", lambda it: it.key, lambda it: it.controls),
             "P1_correct_h3|key_auc": ("P1_correct_h3", lambda it: it.key, lambda it: it.controls),
             "P2_offcycle|key_auc": ("P2_offcycle", lambda it: it.key, lambda it: it.controls),
             "P2_offcycle|direction": ("P2_offcycle", lambda it: it.key, lambda it: [it.forward])}
    record = {"bank": bank, "exploratory": True, "n_perm": N_PERM, "readers": {}}
    for name, rs in R.reader_scores(bank).items():
        prior = colour_means(rs, [it.item_id for it in table])
        entry = {"colour_prior": {c: round(float(prior[k]), 3) for c, k in index.items()}}
        for test, (g, target, others) in tests.items():
            members = groups[g] if test != "P2_offcycle|direction" else \
                [it for it in groups[g] if S.direction_eligible(it)]
            if not any(rs.score(it.item_id) is not None for it in members):
                continue
            scores, tgt, mask = group_tensors(rs, members, index, target, others)
            corrected = item_aucs(scores - prior, tgt, mask)
            entry[test] = {"shuffled": shuffled_null(scores, tgt, mask),
                           "prior_corrected": S.summarise([{"x": float(v)} for v in corrected], "x", 0.5)}
        if any(rs.score(it.item_id) is not None for it in groups["P2_offcycle"]):
            entry["P2_decomposition"] = decomposition(rs, groups["P2_offcycle"], index, prior)
        record["readers"][name] = entry
        print(name, {t: (e["shuffled"]["observed"], e["shuffled"]["null_mean"], e["prior_corrected"]["mean"],
                         e["prior_corrected"]["lo"]) for t, e in entry.items() if t in tests}, flush=True)
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "shuffle_null.json")
    E.atomic_write_text(report_md(record), out_dir / "shuffle_null.md")


def report_md(record: dict) -> str:
    tests = ("P1_correct_h2|key_auc", "P1_correct_h3|key_auc", "P2_offcycle|key_auc", "P2_offcycle|direction")
    lines = [f"# exp15 colour-prior check: {record['bank']} (exploratory, UNVERIFIED)", "",
             "Each cell: observed AUC / shuffled-item null mean [2.5, 97.5 pct] / prior-corrected AUC [95% CI].", "",
             "| reader | " + " | ".join(tests) + " |", "|---|" + "---|" * len(tests)]
    for name, e in record["readers"].items():
        cells = []
        for t in tests:
            if t not in e:
                cells.append("-")
                continue
            sh, pc = e[t]["shuffled"], e[t]["prior_corrected"]
            cells.append(f"{sh['observed']:.3f} / {sh['null_mean']:.3f} [{sh['null_lo']:.3f}, {sh['null_hi']:.3f}] / "
                         f"{pc['mean']:.3f} [{pc['lo']:.3f}, {pc['hi']:.3f}] n={sh['n']}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "P2 hypothesis decomposition (share where w / the true s_{h-1} / a control ranks top): raw, "
              "prior-corrected, and what colour preferences alone give (shuffled items).", "",
              "| reader | raw | prior-corrected | shuffled null |", "|---|---|---|---|"]
    for name, e in record["readers"].items():
        if "P2_decomposition" in e:
            d = e["P2_decomposition"]
            lines.append(f"| {name} | " + " | ".join(
                f"{d[k]['top_key']:.2f} / {d[k]['top_true']:.2f} / {d[k]['top_neither']:.2f}"
                for k in ("raw", "prior_corrected", "shuffled_null")) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", choices=E.BANKS, required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    shuffle_report(a.bank, a.tag)


if __name__ == "__main__":
    main()
