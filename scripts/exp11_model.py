"""exp11: Qwen3.6-27B (bf16, transformers) forward passes with residual capture, and the J-lens / R-lens readouts.

Residual convention: hidden[l + 1] is the output of decoder block l (hidden[0] is the embeddings). The J-lens and
R-lens files map the block-l output to the block-62 output (`target_layer` 62, whose row is the identity); the J++
lens maps it to the final block's (63). All are read as softmax(W_U . norm(J_l . h_l)), restricted here to the brew
colour tokens.
"""
from __future__ import annotations

import glob

import torch

import cc_config as cfg

MODEL = "Qwen3.6-27B"
FAMILY = "qwen3.6"
LENS_DIR = glob.glob(f"{cfg.HF_HUB_DIR}/models--camilablank--workspace-lenses/snapshots/*/qwen3.6-27b")[0]
JPP_FILE = (f"{cfg.HF_HUB_DIR}/models--koayon--jpp-lenses/snapshots/6c96867a91c43c14cec17cf0f6d157b933c5fee5"
            "/qwen3.6-27b/lens.pt")


def model_path() -> str:
    return str(cfg.model_dir(MODEL))


def load():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path())
    model = AutoModelForImageTextToText.from_pretrained(model_path(), dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tokenizer, model


def text_parts(model):
    """(decoder layers, final norm, unembedding weight) of the language model."""
    lm = model.model.language_model
    return lm.layers, lm.norm, model.lm_head.weight


def colour_ids(tokenizer, colours) -> dict[str, int]:
    """The token each colour is answered with after 'Answer:' (a leading space); every colour must be one token."""
    ids = {}
    for c in colours:
        toks = tokenizer.encode(" " + c, add_special_tokens=False)
        assert len(toks) == 1, f"{c!r} is {len(toks)} tokens"
        ids[c] = toks[0]
    return ids


@torch.no_grad()
def forward(model, ids: list[int], positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuals at `positions` for every layer ([n_layers + 1, len(positions), d], bf16 on CPU) and the next-token
    logits at the last position (float32 on CPU)."""
    out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False)
    hidden = torch.stack([h[0, positions] for h in out.hidden_states]).cpu()
    return hidden, out.logits[0, -1].float().cpu()


def load_jacobians(kind: str) -> dict[int, torch.Tensor]:
    """{layer: J_l} in layer order. 'j-lens', 'r-lens': camilablank/workspace-lenses; 'jpp': koayon/jpp-lenses."""
    if kind == "jpp":
        jacobians = torch.load(JPP_FILE, map_location="cpu", weights_only=True, mmap=True)["parameters"]["jacobians"]
        return {l: jacobians[l] for l in sorted(jacobians)}
    blob = torch.load(f"{LENS_DIR}/{kind}/lens.pt", map_location="cpu", weights_only=False)
    return {l: blob["J"][l] for l in blob["source_layers"]}


class Lens:
    def __init__(self, kind: str, model, readout_ids: list[int]):
        self.J = {l: J.to("cuda", torch.float32) for l, J in load_jacobians(kind).items()}
        self.layers = list(self.J)
        _, self.norm, w_u = text_parts(model)
        self.w_u = w_u[readout_ids].float()

    @torch.no_grad()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [n_layers + 1, n, d] -> readout logits [len(layers), n, len(readout_ids)] (float32, CPU)."""
        rows = []
        for l in self.layers:
            h = hidden[l + 1].to("cuda", torch.float32)
            mapped = h @ self.J[l].T
            rows.append(self.norm(mapped.to(torch.bfloat16)).float() @ self.w_u.T)
        return torch.stack(rows).cpu()


class LogitLens(Lens):
    """The identity map at every layer (the baseline every lens readout is compared against)."""

    def __init__(self, model, readout_ids: list[int], n_layers: int):
        self.layers = list(range(n_layers))
        self.J = None
        _, self.norm, w_u = text_parts(model)
        self.w_u = w_u[readout_ids].float()

    @torch.no_grad()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.norm(hidden[l + 1].to("cuda")).float() @ self.w_u.T for l in self.layers]).cpu()
