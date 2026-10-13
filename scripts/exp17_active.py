"""exp17 lens diagnostic: how much of each J++ atom lies in the subspace the residual stream actually occupies?

On Gemma, the brew residuals have ~0 variance (and ~0 mean) inside the top-200 right singular directions of every
exp12 J++ map at layers >= 38, so the unit J++ atoms J^T (gain * W_U[t]) point mostly into directions the model never
uses: their coordinate swaps move ~nothing (doses 10-150x below Qwen's) and pursuit explains nothing. For a residual h
in the occupied subspace (projector P), the lens readout <v_t, h> = <P v_t, h>: the unoccupied part of an atom never
enters a readout. This script measures that part, per model and layer:
  occupied subspace  top-r eigenvectors of the uncentred second moment of block outputs on WikiText (prompts 64-191 of
                     the J++ fit pool, disjoint from exp12's 64 fit prompts; 128 tokens each, positions >= 16 as in the
                     fit), r = the smallest rank holding a fraction 1 - 1e-4 of the energy (also reported: 1e-3, 1e-5)
  kept_colour        |P v_t| for the 10 colour atoms (mean, min)
  kept_readable      |P v_t| over a fixed sample of 4096 readable tokens (mean)
  brew_energy        the fraction of the dev final-position residuals' energy inside P (should be ~1)
-> cache/exp17/<model>/occupied_basis.pt ({layer: [r, d]}), results/.../<model>/occupied_subspace.json

  EXP17_MODEL=qwen|gemma python exp17_active.py
"""
from __future__ import annotations

import sys

import torch

import exp15 as E15
import exp17 as E
import exp17_swaps as SW

sys.path[:0] = [str(E.cfg.REPO_ROOT / "third_party" / "jpp_lens" / "src"),
                str(E.cfg.REPO_ROOT / "third_party" / "jpp_lens" / "scripts")]
import jpp_cli  # noqa: E402

PROMPTS = slice(64, 192)
N_TOKENS, SKIP = 128, 16
ENERGY_TAILS = (1e-3, 1e-4, 1e-5)
KEEP_TAIL = 1e-4
LATE = {"qwen": range(42, 63), "gemma": range(40, 59)}[E.MODEL_KEY]


def wikitext_ids(tokenizer) -> torch.Tensor:
    texts = jpp_cli.load_fit_prompts(PROMPTS.stop)[PROMPTS]
    rows = [tokenizer(t, add_special_tokens=True)["input_ids"][:N_TOKENS] for t in texts]
    rows = [r for r in rows if len(r) == N_TOKENS]
    return torch.tensor(rows)


@torch.no_grad()
def second_moments(layers: list[int], d: int, batch: int = 8) -> tuple[dict[int, torch.Tensor], int]:
    tokenizer, model = E.load_model()
    ids = wikitext_ids(tokenizer)
    moments = {l: torch.zeros(d, d, device="cuda") for l in layers}
    hooks = []
    for l in layers:
        def hook(mod, args, out, l=l):
            h = (out[0] if isinstance(out, tuple) else out)[:, SKIP:].float().flatten(0, 1)
            moments[l].addmm_(h.T, h)
        hooks.append(model.model.layers[l].register_forward_hook(hook))
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch].cuda()
        model.model(input_ids=chunk, attention_mask=torch.ones_like(chunk), use_cache=False)
    for h in hooks:
        h.remove()
    return moments, len(ids) * (N_TOKENS - SKIP)


def main() -> None:
    if E.LENS != "raw":
        raise SystemExit("exp17_active measures the raw J++ vectors: run it without EXP17_LENS")
    tokenizer = E.load_tokenizer()
    jac = E.jpp_jacobians()
    layers = sorted({l for l in E.all_band_layers(jac)} | {l for l in LATE if l in jac})
    moments, n_vectors = second_moments(layers, jac[layers[0]].shape[0])
    gain, w_u = E.gain_and_unembed()
    colours, _, _ = E.item_table("dev")
    cids = torch.tensor(E15.colour_ids(tokenizer, colours))
    readable = SW.readable_token_ids(tokenizer)
    sample = torch.tensor(readable)[torch.randperm(len(readable), generator=torch.Generator().manual_seed(17))[:4096]]
    hidden = torch.load(E.baseline_path("dev"), weights_only=False)["hidden"]
    bases, per_layer = {}, {}
    for l in layers:
        evals, evecs = torch.linalg.eigh(moments[l])                     # ascending
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
        tail = 1 - evals.cumsum(0) / evals.sum()
        ranks = {f"{t:g}": int((tail > t).sum()) + 1 for t in ENERGY_TAILS}
        P = evecs[:, :ranks[f"{KEEP_TAIL:g}"]].T.contiguous()             # [r, d] orthonormal rows
        bases[l] = P.cpu()
        v_col = SW.lens_atoms(jac[l], gain, w_u[cids].to("cuda")).float()
        v_read = SW.lens_atoms(jac[l], gain, w_u[sample].to("cuda")).float()
        kept_col, kept_read = (v_col @ P.T).norm(dim=-1), (v_read @ P.T).norm(dim=-1)
        h = hidden[:, l].float().cuda()
        per_layer[l] = {"rank": ranks, "kept_colour_mean": round(float(kept_col.mean()), 4),
                        "kept_colour_min": round(float(kept_col.min()), 4),
                        "kept_readable_mean": round(float(kept_read.mean()), 4),
                        "brew_energy": round(float((h @ P.T).pow(2).sum() / h.pow(2).sum()), 6)}
        print(f"layer {l}: {per_layer[l]}", flush=True)
    E15.atomic_save({"layers": layers, "keep_tail": KEEP_TAIL, "bases": bases}, E.CACHE / "occupied_basis.pt")
    E.write_once(E.RESULTS / "occupied_subspace.json",
                 {"prompts": f"jpp_cli.load_fit_prompts pool [{PROMPTS.start}:{PROMPTS.stop}]", "n_tokens": N_TOKENS,
                  "skip": SKIP, "n_vectors": n_vectors, "keep_tail": KEEP_TAIL, "by_layer": per_layer})


if __name__ == "__main__":
    main()
