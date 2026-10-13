"""exp17 readout per layer (no model; CPU): what the J++ lens reads at the final position of correct h=2 dev items,
next to what the key direction itself reads.

  jpp      W_U[colour] . (gain * J_l h): the J++ lens's colour scores (the final norm's scale does not change ranks);
           top-1 rate among the 10 colours of the start colour s0, the key s1 and the answer (gold)
  dm_cv    the difference-of-means key readout, 5-fold cross-fitted: directions from the training folds' correct
           h=2/h=3 items (key = s_{h-1}, as exp17 prep), scores (h - training mean) . direction_c, top-1 rate of s1 on
           the held-out h=2 items. Shows whether the key is linearly present at a layer at all.
Rates come with 95% Wilson intervals. -> results/exp17_jspace_keys/<model>/readout.json

  EXP17_MODEL=qwen|gemma python exp17_readout.py
"""
from __future__ import annotations

import math
import random

import torch

import exp15 as E15
import exp17 as E

N_FOLDS = 5
MIN_LAYER = 8


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(p, 4), round(centre - half, 4), round(centre + half, 4)


def top1(scores: torch.Tensor, target: torch.Tensor) -> list[bool]:
    return (scores.argmax(1) == target).tolist()


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    jac = E.jpp_jacobians()
    gain, w_u = E.gain_and_unembed()
    colours, table, _ = E.item_table("dev")
    cids = torch.tensor(E15.colour_ids(E.load_tokenizer(), colours))
    hidden = torch.load(E.baseline_path("dev"), weights_only=False)["hidden"]          # [n, n_layers - 1, d]
    w = w_u[cids].float() * gain                                                     # [10, d] final-norm basis
    colour = {c: k for k, c in enumerate(colours)}

    evals = [i for i, it in enumerate(table) if it.h == 2 and it.group == "correct"]
    train = [i for i, it in enumerate(table) if it.h in (2, 3) and it.group == "correct"]
    key_of = {i: colour[table[i].path[table[i].h - 1]] for i in train}
    order = list(train)
    random.Random(17).shuffle(order)
    fold = {i: k % N_FOLDS for k, i in enumerate(order)}
    s0 = torch.tensor([colour[table[i].path[0]] for i in evals])
    s1 = torch.tensor([colour[table[i].path[1]] for i in evals])
    gold = torch.tensor([colour[table[i].gold] for i in evals])

    by_layer = {}
    for l in range(MIN_LAYER, hidden.shape[1]):
        h = hidden[:, l].float()
        rec = {}
        if l in jac:
            scores = h[evals] @ jac[l].float().T @ w.T                               # [n_eval, 10]
            for name, target in (("s0", s0), ("s1", s1), ("gold", gold)):
                hits = top1(scores, target)
                rec[f"jpp_{name}"] = wilson(sum(hits), len(hits))
        hits = []
        for f in range(N_FOLDS):
            tr = [i for i in train if fold[i] != f]
            te = [i for i in evals if fold.get(i) == f]
            mean, dirs = E15.diffmeans_directions(h[tr][None], torch.tensor([key_of[i] for i in tr]))
            scores = (h[te] - mean[0]) @ dirs[0].T
            hits += top1(scores, torch.tensor([colour[table[i].path[1]] for i in te]))
        rec["dm_cv_s1"] = wilson(sum(hits), len(hits))
        by_layer[l] = rec
        print(f"layer {l}: {rec}", flush=True)
    E.write_once(E.RESULTS / "readout.json",
                 {"items": "correct h=2 dev items, final position", "n_eval": len(evals), "n_dm_train": len(train),
                  "folds": N_FOLDS, "chance": 0.1, "by_layer": by_layer})


if __name__ == "__main__":
    main()
