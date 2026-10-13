"""exp17 decomposition at larger k: does the key direction lie in J++ space at all, beyond k = 25?

Per layer of the selected dm band (and the NLA reconstructor's layer on Qwen): the share of each unit dm_full colour
vector explained by non-negative gradient pursuit over the readable vocabulary's unit J++ vectors at k = 25, 100,
400, against the same for random unit vectors (chance). -> results/exp17_jspace_keys/<model>/ksweep.json

  EXP17_MODEL=qwen|gemma python exp17_ksweep.py
"""
from __future__ import annotations

import json

import torch

import exp17 as E
import exp17_swaps as SW

KS = (25, 100, 400)
N_RANDOM = 10


@torch.no_grad()
def main() -> None:
    out_path = E.RESULTS / "ksweep.json"
    E.E15.require_absent(out_path)
    tokenizer = E.load_tokenizer()
    blob = torch.load(E.directions_path(), weights_only=False)
    band = json.loads((E.RESULTS / "dev_selection.json").read_text())["chosen"]["dm"]["band"]
    jac = E.jpp_jacobians()
    layers = E.band_layers(band, jac)
    layers = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
    gain, w_u = E.gain_and_unembed()
    keep = SW.readable_token_ids(tokenizer)
    w_keep = w_u[torch.tensor(keep)].to("cuda")
    g = torch.Generator().manual_seed(1717)
    record = {"band": band, "ks": KS, "by_layer": {}}
    jobs = [(l, "dm_full") for l in layers]
    if E.SPEC.ar_band is not None:
        jobs.append((E.SPEC.bands[E.SPEC.ar_band][0], "ar_full"))
    for l, family in jobs:
        atoms = E.jpp_vectors(jac, l, gain, w_keep)
        vecs = blob["families"][family][l].float()
        rand = E.random_unit(torch.randn(N_RANDOM, vecs.shape[-1], generator=g), l)
        row = {}
        for k in KS:
            ev = [SW.nn_gradient_pursuit(v, atoms, k).explained for v in vecs]
            er = [SW.nn_gradient_pursuit(v, atoms, k).explained for v in rand]
            row[k] = {"explained": round(sum(ev) / len(ev), 4), "random": round(sum(er) / len(er), 4),
                      "random_max": round(max(er), 4)}
            print(f"{family} layer {l} k={k}: {row[k]}", flush=True)
        record["by_layer"][f"{family}|{l}"] = row
        del atoms
    E.write_once(out_path, record)


if __name__ == "__main__":
    main()
