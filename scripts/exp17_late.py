"""exp17 in-task positive control: can a J++ colour swap change the brew answer at all, and from which layers?

In the dev grid the J++ answer swap (gold -> a') at alpha 1-2 in bands 24-54 almost never changes Qwen's answer
(<= 0.016), while the same kind of swap moves the factual answer in 0.22-0.74 of prompts at similar doses. If J++
swaps cannot steer this task anywhere, "the key is outside J++" is not specific to the key. Here, on dev's correct
h=2 items and exp17's window (start colour .. last token): the J++ answer swap, alpha 1-8, each with a
random-direction null at its recorded edit norms, in
  late    the latest dev band and the later control bands     -> cache/exp17/<run>/trials_late.jsonl
  early   the other multi-layer dev bands (the dev grid only    -> cache/exp17/<run>/trials_answer_early.jsonl
          tried alpha 1-2 there, without a null)
Dev only; the control bands' setting for the test is picked from "late" by exp17_manifest.answer_control.

  EXP17_MODEL=qwen|gemma [EXP17_LENS=occupied] python exp17_late.py late|early
"""
from __future__ import annotations

import argparse

import torch

import exp15 as E15
import exp17 as E

LATEST_DEV_BAND = {"qwen": "b42_54", "gemma": "b40_52"}[E.MODEL_KEY]
ALPHAS = (1.0, 2.0, 4.0, 8.0)


def bands(which: str) -> tuple[str, ...]:
    if which == "late":
        return (LATEST_DEV_BAND,) + tuple(E.SPEC.control_bands)
    multi_layer = [b for b, (lo, hi) in E.SPEC.bands.items() if lo < hi]
    return tuple(b for b in multi_layer if b != LATEST_DEV_BAND)


def arms(band_names) -> list[E.Arm]:
    out = []
    for band in band_names:
        for a in ALPHAS:
            swap = f"jpp_answer|{band}|{a:g}"
            out += [E.Arm(swap, "swap", "jpp_answer_atoms", None, band, a, pair="answer"),
                    E.Arm(f"null_answer|{band}|{a:g}", "null", None, swap, band, a, pair="answer")]
    return out


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["late", "early"])
    which = p.parse_args().which
    band_names = bands(which)
    targets, colours = E.brew_targets("dev")
    jac = E.jpp_jacobians()
    gain, w_u = E.gain_and_unembed()
    cids = torch.tensor(E15.colour_ids(E.load_tokenizer(), colours))
    layers = sorted({l for b in band_names for l in E.band_layers(b, jac)})
    families = {"jpp_answer_atoms": {l: E.jpp_vectors(jac, l, gain, w_u[cids].to("cuda")).float() for l in layers}}
    print(f"{which}: bands {[(b, E.band_layers(b, jac)) for b in band_names]}", flush=True)
    phase = {"late": "late", "early": "answer_early"}[which]
    E.run_arms(phase, targets, arms(band_names), families, jac, E.brew_scorer(colours))


if __name__ == "__main__":
    main()
