"""exp15 readers: token lenses and the template lens (offline, from captured residuals), and the text readers and
text->vector reconstructors (NLA AV / AR, ceselder's Oracle lens, agu18dec's activation oracle and ptag AR).

Layer l is the output of decoder block l = hidden_states[l + 1]. transformers stores hidden_states[64] AFTER the
final norm, so the raw block-63 output is never available here: token lenses and the template lens read layers
0..62, and the logit lens's layer 63 is the model's own (already normed) final residual.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

import cc_config as cfg

REPO = Path(__file__).resolve().parents[1]
HUB = cfg.HF_HUB_DIR
QWEN = "Qwen3.6-27B"
LENS_DIR = Path(glob.glob(f"{HUB}/models--camilablank--workspace-lenses/snapshots/*/qwen3.6-27b")[0])
JPP_FILE = (HUB / "models--koayon--jpp-lenses/snapshots/6c96867a91c43c14cec17cf0f6d157b933c5fee5/qwen3.6-27b/lens.pt")
NLA_DIR = Path(glob.glob(f"{HUB}/models--ceselder--qwen3.6-27b-nla-rl/snapshots/*")[0])
OLENS_DIR = Path(glob.glob(f"{HUB}/models--ceselder--olens-online-step40-qwen36-27b/snapshots/*")[0])
AGU_DIR = Path(glob.glob(f"{HUB}/models--agu18dec--olens_and_ar/snapshots/*")[0])
LENS_LAYERS = tuple(range(63))
TOKEN_LENSES = ("logit", "jlens", "rlens", "jpp")
COUNT_WORDS = ("one", "two", "three", "four", "five", "six")      # count k <-> COUNT_WORDS[k - 1]


def qwen_dir() -> Path:
    return cfg.model_dir(QWEN)


def load_checkpoint_tensor(name: str) -> torch.Tensor:
    """One tensor from Qwen3.6-27B's safetensors shards, without loading the model."""
    root = qwen_dir()
    shard = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"][name]
    with safe_open(str(root / shard), "pt") as f:
        return f.get_tensor(name)


class Readout:
    """W_U[ids] . norm(x): Qwen3.5's final RMSNorm (x * rsqrt(mean x^2 + eps) * (1 + w), in fp32) and the
    unembedding rows of `token_ids`."""

    def __init__(self, token_ids: list[int], eps: float = 1e-6, device: str = "cuda"):
        self.gain = (1.0 + load_checkpoint_tensor("model.language_model.norm.weight").float()).to(device)
        self.w_u = load_checkpoint_tensor("lm_head.weight")[token_ids].float().to(device)        # [K, d]
        self.eps = eps

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gain

    def __call__(self, x: torch.Tensor, normed: bool = False) -> torch.Tensor:
        return (x.float() if normed else self.norm(x)) @ self.w_u.T


def load_jacobians(kind: str) -> dict[int, torch.Tensor]:
    """{l: J_l} for l in 0..62: J-lens / R-lens (camilablank/workspace-lenses, target block 62) and J++
    (koayon/jpp-lenses, target block 63; one matrix per layer as shipped)."""
    if kind == "jpp":
        jac = torch.load(JPP_FILE, map_location="cpu", weights_only=True, mmap=True)["parameters"]["jacobians"]
        return {l: jac[l] for l in LENS_LAYERS}
    blob = torch.load(LENS_DIR / {"jlens": "j-lens", "rlens": "r-lens"}[kind] / "lens.pt", map_location="cpu",
                      weights_only=False, mmap=True)
    assert list(blob["source_layers"]) == list(LENS_LAYERS)
    return {l: blob["J"][l] for l in LENS_LAYERS}


class TokenLenses:
    """Readouts W_U[ids] . norm(J_l h_l) for the logit lens (J = I) and the three Jacobian lenses."""

    def __init__(self, token_ids: list[int], kinds=TOKEN_LENSES, device: str = "cuda"):
        self.readout = Readout(token_ids, device=device)
        self.kinds = kinds
        self.jac = {k: {l: J.to(device, torch.float32) for l, J in load_jacobians(k).items()}
                    for k in kinds if k != "logit"}

    @torch.no_grad()
    def scores(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """hidden [n_cells, 65, d] -> {kind: [n_cells, 63, K]} (layers 0..62)."""
        h = hidden.to(self.readout.w_u.device, torch.float32)
        out = {}
        for k in self.kinds:
            if k == "logit":
                out[k] = torch.stack([self.readout(h[:, l + 1]) for l in LENS_LAYERS], 1)
            else:
                out[k] = torch.stack([self.readout(h[:, l + 1] @ self.jac[k][l].T) for l in LENS_LAYERS], 1)
        return {k: v.cpu() for k, v in out.items()}

    def direction(self, kind: str, layer: int, token_index: int) -> torch.Tensor:
        """The residual direction whose readout raises token `token_index` most (to first order, ignoring the
        norm's 1/rms): J_l^T (g * W_U[c]), with g the final norm's gain; for the logit lens, g * W_U[c]."""
        row = self.readout.gain * self.readout.w_u[token_index]
        return row if kind == "logit" else self.jac[kind][layer].T @ row


class TemplateLens:
    """camilablank's template lens (tl-full): one direction per (layer, word); the score of a word is the cosine of
    the residual with its template (only the colour and count rows are loaded)."""

    FILE = LENS_DIR / "template-lens" / "templates.safetensors"
    WORDS = LENS_DIR / "template-lens" / "template_words.txt"

    def __init__(self, words: list[str], device: str = "cuda"):
        rows = {}
        for line in self.WORDS.read_text().splitlines():
            idx, word = line.split("\t", 1)
            rows.setdefault(word, int(idx))
        self.words = words
        index = [rows[w] for w in words]
        with safe_open(str(self.FILE), "pt") as f:
            layers = f.get_tensor("layers").tolist()
            assert layers == list(range(64))
            sl = f.get_slice("templates")
            t = torch.stack([sl[l][index] for l in LENS_LAYERS])                    # [63, K, d]
        self.unit = (t.float() / t.float().norm(dim=-1, keepdim=True)).to(device)
        self.raw = t.float().to(device)

    @torch.no_grad()
    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [n_cells, 65, d] -> cosines [n_cells, 63, K]."""
        h = hidden[:, 1:64].to(self.unit.device, torch.float32)                      # layers 0..62
        h = h / h.norm(dim=-1, keepdim=True)
        return torch.einsum("cld,lkd->clk", h, self.unit).cpu()

    def direction(self, layer: int, word_index: int) -> torch.Tensor:
        return self.raw[layer, word_index]


# --- text readers ---------------------------------------------------------------------------------------------------
MARKER = "\u321c"          # ㈜, token 158983 in Qwen3.6-27B's tokenizer, the injection slot of every reader below
MARKER_ID = 158983
AO_PROMPT_ONE_LINE = ("An activation vector from layer {layer} of a language model is enclosed in activation tags: "
                      "<activation>{char}</activation>. Produce distinct concepts that encode this activation, each as "
                      "a '- ' bullet on its own line.")
# The README's rendering, with its line breaks; day-0 check 6 scored both and kept this one.
AO_PROMPT = AO_PROMPT_ONE_LINE.replace("activation tags: ", "activation tags:\n").replace("each as a", "each as\na")


def _easynla():
    path = str(REPO / "third_party" / "easynla")
    if path not in sys.path:
        sys.path.insert(0, path)


def _yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def chat_ids(tokenizer, content: str) -> list[int]:
    enc = tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        enable_thinking=False, return_dict=True)
    return list(enc["input_ids"])


def marker_slot(ids: list[int]) -> int:
    slots = [i for i, t in enumerate(ids) if t == MARKER_ID]
    assert len(slots) == 1, f"marker appears {len(slots)} times in the rendered prompt"
    return slots[0]


def load_lora_merged(base_dir, adapter_dir, device="cuda"):
    """A CausalLM with a LoRA merged in, after checking the adapter changes the logits of a probe prompt (an
    adapter that matches no module is otherwise indistinguishable from a working one)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(qwen_dir()))
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map=device).eval()
    probe = torch.tensor([chat_ids(tokenizer, "Name three colours.")], device=device)
    with torch.no_grad():
        before = base(input_ids=probe, use_cache=False).logits[0, -1].float()
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().eval()
    with torch.no_grad():
        after = model(input_ids=probe, use_cache=False).logits[0, -1].float()
    drift = float((after - before).abs().max())
    assert drift > 1e-2, f"adapter {adapter_dir} looks inert (max logit change {drift})"
    return tokenizer, model, drift


class InjectedSampler:
    """Sample from a chat prompt whose marker slot carries an activation vector.

    inject = 'karvonen1': at the output of decoder block 1, h_slot <- h_slot + ||h_slot|| v / ||v|| (EasyNLA's
             karvonen_inject_in_residual; NLA AV and ceselder's Oracle lens).
    inject = 'embed_unit': the slot's input embedding <- alpha v / ||v|| (agu18dec's activation oracle).
    Prompts are identical across a batch, so there is no padding; injection happens on the prefill only."""

    def __init__(self, model, tokenizer, inject: str, alpha: float | None = None):
        self.model, self.tokenizer, self.inject, self.alpha = model, tokenizer, inject, alpha
        self.state = {"vectors": None, "ids": None, "slot": None}
        if inject == "karvonen1":
            _easynla()
            from nla.injection import karvonen_inject_in_residual
            self._karvonen = karvonen_inject_in_residual
            model.model.layers[1].register_forward_hook(self._block_hook)
        else:
            model.get_input_embeddings().register_forward_hook(self._embed_hook)

    def _block_hook(self, module, args, output):
        resid = output[0] if isinstance(output, tuple) else output
        if self.state["vectors"] is None or resid.shape[1] < 2:
            return output
        ids = self.state["ids"].to(resid.device)
        left, right = int(ids[0, self.state["slot"] - 1]), int(ids[0, self.state["slot"] + 1])
        new = self._karvonen(ids, resid, self.state["vectors"].to(resid.device), MARKER_ID, left, right)
        return (new, *output[1:]) if isinstance(output, tuple) else new

    def _embed_hook(self, module, args, output):
        if self.state["vectors"] is None or output.shape[1] < 2:
            return output
        v = self.state["vectors"].to(output.device, torch.float32)
        out = output.clone()
        out[:, self.state["slot"]] = (self.alpha * v / v.norm(dim=-1, keepdim=True)).to(out.dtype)
        return out

    @torch.no_grad()
    def sample(self, prompt_ids: list[int], vectors: torch.Tensor, n_samples: int, max_new_tokens: int,
               temperature: float, top_p: float, seed: int, batch: int = 192) -> list[list[str]]:
        """vectors [N, d] -> N lists of n_samples decoded continuations."""
        slot = marker_slot(prompt_ids)
        rows = vectors.repeat_interleave(n_samples, 0)
        texts = []
        for start in range(0, len(rows), batch):
            v = rows[start:start + batch]
            ids = torch.tensor([prompt_ids] * len(v), device="cuda")
            self.state.update(vectors=v, ids=ids, slot=slot)
            torch.manual_seed(seed + start)
            try:
                out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                          temperature=temperature, top_p=top_p, top_k=0,
                                          max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            finally:
                self.state["vectors"] = None
            texts += [self.tokenizer.decode(o[len(prompt_ids):], skip_special_tokens=True) for o in out]
        return [texts[i * n_samples:(i + 1) * n_samples] for i in range(len(vectors))]


# reader name -> (sampling settings, injection, read layers). The layer is the block whose output is read.
READERS = {
    "nla": {"layers": (42,), "n_samples": 4, "max_new_tokens": 300, "temperature": 1.0, "top_p": 1.0},
    "olens": {"layers": (42,), "n_samples": 4, "max_new_tokens": 120, "temperature": 1.0, "top_p": 1.0},
    "ao": {"layers": (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60), "n_samples": 4, "max_new_tokens": 128,
           "temperature": 0.8, "top_p": 0.95},
}
NLA_ADAPTER = "iter_000800"


class TextReader:
    """One of READERS, loaded with its own model: read(vectors, layer) -> samples per vector."""

    def __init__(self, name: str, nla_adapter: str = NLA_ADAPTER, ao_prompt: str = AO_PROMPT):
        self.name, self.spec = name, READERS[name]
        if name == "nla":
            self.tokenizer, self.model, self.drift = load_lora_merged(NLA_DIR / "av_base",
                                                                      NLA_DIR / "av_rl_adapters" / nla_adapter)
            meta = _yaml(NLA_DIR / "av_base" / "nla_meta.yaml")
            self.template = meta["prompt_templates"]["actor"]
            self.sampler = InjectedSampler(self.model, self.tokenizer, "karvonen1")
        elif name == "olens":
            self.tokenizer, self.model, self.drift = load_lora_merged(qwen_dir(), OLENS_DIR)
            self.template = _yaml(OLENS_DIR / "nla_meta.yaml")["prompt_templates"]["actor"]
            self.sampler = InjectedSampler(self.model, self.tokenizer, "karvonen1")
        else:
            self.tokenizer, self.model, self.drift = load_lora_merged(qwen_dir(), AGU_DIR / "olens_s3d_rl600")
            self.template = ao_prompt
            self.sampler = InjectedSampler(self.model, self.tokenizer, "embed_unit", alpha=16000.0)

    def prompt_ids(self, layer: int) -> list[int]:
        if self.name == "ao":
            content = self.template.format(layer=layer, char=MARKER)
        else:
            content = self.template.format(injection_char=MARKER)
        ids = chat_ids(self.tokenizer, content)
        marker_slot(ids)
        return ids

    def read(self, vectors: torch.Tensor, layer: int, seed: int = 0, n_samples: int | None = None,
             batch: int = 192) -> list[list[str]]:
        assert layer in self.spec["layers"], f"{self.name} is not trained at layer {layer}"
        s = self.spec
        return self.sampler.sample(self.prompt_ids(layer), vectors, n_samples or s["n_samples"],
                                   s["max_new_tokens"], s["temperature"], s["top_p"], seed, batch)


def explanation_text(name: str, sample: str) -> str:
    """The part of a readout that is scored: NLA's <explanation> payload when present, else the whole sample."""
    if name == "nla":
        _easynla()
        from nla.schema import extract_explanation
        return extract_explanation(sample) or sample
    return sample


# --- text -> vector reconstructors ----------------------------------------------------------------------------------
class NlaAR:
    """ceselder's NLA reconstructor: a 43-block truncated Qwen3.6-27B + value head; explanation text -> predicted
    layer-42 activation (L2 norm sqrt(d), EasyNLA's mse_scale)."""

    def __init__(self, device: str = "cuda"):
        _easynla()
        from nla.models import NLACriticModel
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(qwen_dir()))
        self.model = NLACriticModel.from_pretrained(str(NLA_DIR / "ar_reconstructor"), dtype=torch.bfloat16,
                                                    device_map=device).eval()
        meta = _yaml(NLA_DIR / "nla_meta.yaml")
        self.template = meta["prompt_templates"]["critic"]
        self.scale = 5120 ** 0.5

    @torch.no_grad()
    def predict(self, explanations: list[str], batch: int = 32) -> torch.Tensor:
        from nla.utils.critic import critic_predict
        out = []
        for start in range(0, len(explanations), batch):
            enc = [self.tokenizer.encode(self.template.format(explanation=e), add_special_tokens=False)
                   for e in explanations[start:start + batch]]
            width = max(map(len, enc))
            ids = torch.tensor([e + [0] * (width - len(e)) for e in enc], device="cuda")
            mask = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e in enc], device="cuda")
            out.append(critic_predict(self.model, ids, mask, self.scale).cpu())
        return torch.cat(out)


def merge_lora_by_hand(text_model, lora_dir: Path) -> int:
    """W += (alpha / r) B A for every LoRA pair in lora_dir, into text_model.layers. agu18dec's keys carry the
    torch.compile wrapper ('layers.N._orig_mod.<path>'), which PEFT would silently fail to match."""
    from safetensors.torch import load_file
    config = json.loads((lora_dir / "adapter_config.json").read_text())
    assert not config["use_rslora"]
    scale = config["lora_alpha"] / config["r"]
    weights = load_file(str(lora_dir / "adapter_model.safetensors"))
    merged = 0
    for key in weights:
        if not key.endswith("lora_A.weight"):
            continue
        path = key.removeprefix("base_model.model.").replace("._orig_mod", "").removesuffix(".lora_A.weight")
        module = text_model.get_submodule(path)
        a, b = weights[key].float(), weights[key.replace("lora_A", "lora_B")].float()
        delta = (b.to(module.weight.device) @ a.to(module.weight.device)) * scale
        module.weight.data = (module.weight.data.float() + delta).to(module.weight.dtype)
        merged += 1
    return merged


class PtagAR:
    """agu18dec's ptag reconstructor: '[Layer NN]' tag tokens + a text span -> its activation at layer NN, predicted
    in their (pooled, whitened) space. Backbone = Qwen3.6-27B + LoRA; read = the block-NN output at the span's last
    real token -> LayerNorm without affine -> one shared linear head."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModelForCausalLM
        root = AGU_DIR / "ar_ptag_pooled"
        self.model = AutoModelForCausalLM.from_pretrained(str(qwen_dir()), dtype=torch.bfloat16,
                                                          device_map=device).eval()
        self.text = self.model.model
        self.n_merged = merge_lora_by_hand(self.text, root / "lora")
        heads = torch.load(root / "heads.pt", map_location=device, weights_only=False)
        self.tag_ids = dict(zip(heads["layers"], heads["tag_ids"].tolist()))
        self.w, self.b = heads["head"]["linear.weight"].float(), heads["head"]["linear.bias"].float()

    @torch.no_grad()
    def predict(self, spans: list[list[int]], layer: int, batch: int = 64) -> torch.Tensor:
        """spans (token ids, without the tag) -> [n, d] predictions in the AR's whitened space."""
        tag = self.tag_ids[layer]
        out = []
        for start in range(0, len(spans), batch):
            rows = [tag + s for s in spans[start:start + batch]]
            width = max(map(len, rows))
            ids = torch.tensor([r + [0] * (width - len(r)) for r in rows], device="cuda")
            mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device="cuda")
            hs = self.text(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
            last = hs[layer + 1][torch.arange(len(rows)), mask.sum(1) - 1].float()
            normed = torch.nn.functional.layer_norm(last, (last.shape[-1],))
            out.append((normed @ self.w.T + self.b).cpu())
        return torch.cat(out)


class Whitener:
    """Our stand-in for agu18dec's gated pooled whitener: per-layer mean and ZCA from natural-text activations,
    with ridge c * mean eigenvalue. white(x) = (x - mu) S^-1/2; unwhiten_delta(d) = d S^1/2 (a direction)."""

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, ridge: float = 0.1):
        evals, evecs = torch.linalg.eigh(cov.double())
        evals = evals.clamp_min(0) + ridge * evals.clamp_min(0).mean()
        self.mean = mean.float()
        self.inv_sqrt = (evecs * evals.rsqrt()) @ evecs.T
        self.sqrt = (evecs * evals.sqrt()) @ evecs.T
        self.inv_sqrt, self.sqrt = self.inv_sqrt.float(), self.sqrt.float()

    def white(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) @ self.inv_sqrt

    def unwhiten_delta(self, d: torch.Tensor) -> torch.Tensor:
        return d.float() @ self.sqrt
