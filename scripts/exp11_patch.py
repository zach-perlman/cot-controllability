"""exp11 activation patching: one prompt run as a batch in which batch element b has the residual stream at token
position p_b, layer l_b, replaced by a given vector.

Layer convention (the lenses' and every exp11 readout's): "layer l" is the OUTPUT of decoder block l, i.e. the
capture's hidden[l + 1]. Patching it replaces what blocks l+1..63 see at that position; earlier blocks and other
positions are untouched.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

import exp11_model as M


@dataclass(frozen=True)
class Patch:
    layer: int              # replace the output of block `layer` (0..62)
    position: int           # token index in the recipient prompt
    vector: torch.Tensor    # [d_model]; from a capture this is hidden[layer + 1][position_name]


@torch.no_grad()
def colour_logits(model, ids: list[int], patches: list[Patch | None], colour_token_ids: list[int]) -> torch.Tensor:
    """Run `ids` once per entry of `patches` (None = unpatched) in one batch; return the next-token logits of the
    colour tokens at the last position, [len(patches), n_colours], float32 on CPU."""
    layers, _, _ = M.text_parts(model)
    by_layer: dict[int, list[tuple[int, Patch]]] = defaultdict(list)
    for b, p in enumerate(patches):
        if p is not None:
            by_layer[p.layer].append((b, p))

    def hook_for(entries):
        def hook(module, args, output):
            hidden = (output[0] if isinstance(output, tuple) else output).clone()
            for b, p in entries:
                hidden[b, p.position] = p.vector.to(hidden.device, hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    handles = [layers[l].register_forward_hook(hook_for(entries)) for l, entries in by_layer.items()]
    try:
        x = torch.tensor([ids] * len(patches), device="cuda")
        out = model(input_ids=x, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return out.logits[:, -1, colour_token_ids].float().cpu()
