"""exp16's Gemma arm (results/exp16_wrong_keys_gemma/manifest.json): exp15's read and write pipeline on
google/gemma-4-31B-it, on exp16's test bank (the same 5300 puzzles as the Qwen run), with exp15's code run under the
Gemma overrides below. Its tests are exp16_report's, in their own Holm family.

What differs from the Qwen run, and lives here:
  loading     Gemma4ForConditionalGeneration (exp12's loader class; AutoModelForCausalLM would not load its weights),
              seen through GemmaText: calling it runs the full model, .model is the text stack (its .layers are what
              exp15_edits.AdditiveEdit hooks), .lm_head applies Gemma's final-logit softcap like the full model
  read cells  Gemma's chat tail: '?<turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>Answer:' (12 tokens)
  lenses      the logit lens, and exp12's own Gemma fits (J-lens 16 prompts; R-lens and J++ 64 prompts), each only at
              the layers it was fitted at. Gemma 4's final norm is x * rsqrt(mean x^2 + eps) * w (no 1 + w)
  writes      lens and diffmeans directions only (this model has no NLA, template lens or ptag AR); write layers scaled
              from exp15's (28-44 of 64) to 60 layers; each method's setting chosen on Gemma's own dev grid (exp11's
              single_test bank) by exp15's selection rule and locked before any test write
  layers      primary layers 34-45 (exp15's 36-48 of 64, by relative depth); the layer-42 head-to-head stays at 42
Not run for Gemma: the text readers, self-report, count repair, read-then-write, AR edit.

  python exp16_gemma.py bank | check_model | dev_check | lens_reads | directions | zero_edit | erase |
                        report --tag T | stage <exp15 module> <args>
EXP16_GEMMA_SMOKE=1 uses the exp16 smoke bank's items, under cache/exp16_gemma_smoke and results/exp16_gemma_smoke.
Run with /venv/main/bin/python from scripts/; HF_HOME must point at the model cache.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open

SMOKE = os.environ.get("EXP16_GEMMA_SMOKE") == "1"
if SMOKE:
    os.environ["EXP16_SMOKE"] = "1"
import cc_config as cfg  # noqa: E402
import exp16 as E16  # noqa: E402  (exp16's overrides first: its cells, text-row spec and erase control)
import exp15 as E15  # noqa: E402
import exp15_edits as ED  # noqa: E402

EXP = cfg.Exp("exp16_gemma_smoke", "exp16_gemma_smoke") if SMOKE else cfg.Exp("exp16_wrong_keys_gemma", "exp16_gemma")
SOURCE = E16.EXP                                   # the Qwen run whose test-bank items are reused
MODEL = "google/gemma-4-31B-it"
CHECKPOINT = Path(glob.glob(f"{cfg.HF_HUB_DIR}/models--google--gemma-4-31B-it/snapshots/*")[0])
N_LAYERS = 60
LENS_LAYERS = tuple(range(N_LAYERS - 1))           # block outputs 0..58; hidden_states[60] is after the final norm
NORM_EPS = 1e-6
TAIL_CELLS = ("question_end", "turn_end", "nl_user", "turn_start", "model", "nl_asst", "channel_open", "thought",
              "nl_thought", "channel_close", "answer_word", "final")
TAIL_TOKENS = ("?", "<turn|>", "\n", "<|turn>", "model", "\n", "<|channel>", "thought", "\n", "<channel|>", "Answer",
               ":")
PRIMARY_LAYERS = tuple(range(34, 46))
LENS_FILES = {"jlens": "jlens_16p", "rlens": "rlens_64p", "jpp": "jpp_64p"}
TOKEN_LENSES = ("logit", *LENS_FILES)
WRITE_LAYERS = (26, 30, 34, 38, 40, 42)
METHODS = (*TOKEN_LENSES, "diffmeans")
# The J-lens fit has 7 source layers (8, 15, 22, 30, 38, 45, 52): it writes at the three in the write range.
METHOD_LAYERS = {m: WRITE_LAYERS for m in METHODS} | {"jlens": (22, 30, 38)}
ALPHAS = (0.25, 0.5, 1.0)                          # exp15's grid without 0.1 (never selected there)
# exp12's brew gate on single_test (results/exp12_gemma_lenses/brew_gate/gate.txt): accuracy per h, 250 items each.
EXP12_DEV_ACCURACY = {1: 1.000, 2: 0.928, 3: 0.760}
DEV_TOLERANCE = 3 / 250

E15_OVERRIDES = {"EXP": EXP, "MODEL": MODEL, "FAMILY": "gemma4", "N_LAYERS": N_LAYERS, "TAIL_CELLS": TAIL_CELLS,
                 "TAIL_TOKENS": TAIL_TOKENS, "PRIMARY_LAYERS": PRIMARY_LAYERS, "AO_PRIMARY_LAYERS": ()}
ED_OVERRIDES = {"WRITE_LAYERS": WRITE_LAYERS, "ALPHAS": ALPHAS, "LENS_METHODS": TOKEN_LENSES, "METHODS": METHODS,
                "METHOD_LAYERS": METHOD_LAYERS}
for _module, _overrides in ((E15, E15_OVERRIDES), (ED, ED_OVERRIDES)):
    for _name, _value in _overrides.items():
        assert hasattr(_module, _name), _name
        setattr(_module, _name, _value)
E16.EXP = EXP                                      # exp16's erase, zero-edit check and report write here
# cell_layer_hidden binds PRIMARY_LAYERS as a default argument at definition time.
E15.cell_layer_hidden.__defaults__ = (E15.PRIMARY_CELLS, PRIMARY_LAYERS)
TEST_BANK = E15.TEST_BANK
DEV_BANK = E15.DEV_BANK


# --- model ----------------------------------------------------------------------------------------------------------
class GemmaText:
    """Gemma4ForConditionalGeneration as exp15's code uses a CausalLM. Calling it runs the full model (softcapped
    logits; hidden_states[l + 1] is the output of block l). .model is the text stack: its .layers are hooked by the
    edits, its last_hidden_state is after the final norm. .lm_head(x) is the full model's head with its softcap."""

    def __init__(self, full):
        self.full = full
        self.model = full.model.language_model
        self.config = full.config
        self.cap = full.config.get_text_config().final_logit_softcapping

    def __call__(self, **kwargs):
        return self.full(**kwargs)

    def lm_head(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.full.lm_head(x)
        return torch.tanh(logits / self.cap) * self.cap if self.cap else logits

    def get_input_embeddings(self):
        return self.full.get_input_embeddings()


def model_path() -> str:
    return MODEL


def load_base(device_map="cuda"):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if device_map == "cpu":
        # transformers' loader runs a CUDA allocator warm-up even for a CPU load (and fails without a GPU)
        import transformers.modeling_utils as MU
        MU.caching_allocator_warmup = lambda *args, **kwargs: None
    full = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                       device_map=device_map).eval()
    return tokenizer, GemmaText(full)


E15.model_path = model_path
E15.load_base = load_base


def checkpoint_tensor(name: str) -> torch.Tensor:
    shard = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"][name]
    with safe_open(str(CHECKPOINT / shard), "pt") as f:
        return f.get_tensor(name)


# --- lenses ---------------------------------------------------------------------------------------------------------
class GemmaLenses:
    """Readouts W_U[ids] . norm(J_l h_l): the logit lens (J = I) at every layer 0..58 and exp12's Gemma fits at their
    fitted layers. W_U is the tied embedding; norm is Gemma 4's final RMSNorm (x * rsqrt(mean x^2 + eps) * w, fp32)."""

    def __init__(self, token_ids: list[int], device: str = "cuda"):
        self.gain = checkpoint_tensor("model.language_model.norm.weight").float().to(device)
        self.w_u = checkpoint_tensor("model.language_model.embed_tokens.weight")[token_ids].float().to(device)
        self.jac = {}
        for kind, name in LENS_FILES.items():
            blob = torch.load(cfg.REPO_ROOT / "cache" / "exp12" / "lenses" / f"{name}.pt", map_location="cpu",
                              weights_only=False, mmap=True)
            assert blob["config"]["hf_model_name"] == MODEL, blob["config"]["hf_model_name"]
            self.jac[kind] = {int(l): J.to(device, torch.float32) for l, J in blob["parameters"]["jacobians"].items()}

    def fitted(self, kind: str) -> tuple[int, ...]:
        return LENS_LAYERS if kind == "logit" else tuple(sorted(self.jac[kind]))

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS) * self.gain) @ self.w_u.T

    @torch.no_grad()
    def scores(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """hidden [n_cells, 61, d] -> {kind: [n_cells, 59, K]}, NaN at the layers a lens was not fitted at."""
        h = hidden.to(self.w_u.device, torch.float32)
        out = {}
        for kind in TOKEN_LENSES:
            s = torch.full((h.shape[0], len(LENS_LAYERS), self.w_u.shape[0]), float("nan"), device=h.device)
            for l in self.fitted(kind):
                x = h[:, l + 1] if kind == "logit" else h[:, l + 1] @ self.jac[kind][l].T
                s[:, l] = self.readout(x)
            out[kind] = s.cpu()
        return out

    def direction(self, kind: str, layer: int, token_index: int) -> torch.Tensor:
        """The residual direction whose readout raises the token most, to first order: J_l^T (w * W_U[c])."""
        row = self.gain * self.w_u[token_index]
        return row if kind == "logit" else self.jac[kind][layer].T @ row


@torch.no_grad()
def lens_reads(bank: str) -> None:
    """exp15.lens_reads' file for Gemma: per item, cell and layer 0..58, the 10 colour and 6 count-word readouts of the
    four token lenses (NaN where a lens was not fitted), plus each lens's fitted layers."""
    import exp15_readers as RD
    from transformers import AutoTokenizer
    path = E15.lens_reads_path(bank)
    E15.require_absent(path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    metas, _ = E15.load_capture(bank, with_hidden=False)
    colours = metas[0]["colours"]
    count_ids = [tokenizer.encode(" " + w, add_special_tokens=False) for w in RD.COUNT_WORDS]
    assert all(len(t) == 1 for t in count_ids)
    lenses = GemmaLenses(E15.colour_ids(tokenizer, colours) + [t[0] for t in count_ids],
                         device="cuda" if torch.cuda.is_available() else "cpu")
    scores = {k: [] for k in TOKEN_LENSES}
    item_ids = []
    for chunk in sorted(E15.capture_dir(bank).glob("chunk_*.pt")):
        blob = torch.load(chunk, weights_only=False)
        for m, hidden in zip(blob["metas"], blob["hidden"]):
            for k, v in lenses.scores(hidden).items():
                scores[k].append(v.half())
            item_ids.append(m["item_id"])
        print(f"lens_reads {bank} {len(item_ids)}/{len(metas)}", flush=True)
    E15.atomic_save({"item_ids": item_ids, "cells": E15.CELLS, "layers": LENS_LAYERS,
                     "labels": colours + [f"count{k}" for k in range(1, 7)],
                     "fitted": {k: lenses.fitted(k) for k in TOKEN_LENSES},
                     "scores": {k: torch.stack(v) for k, v in scores.items()}}, path)
    print(f"-> {path}")


E15.lens_reads = lens_reads


def reader_scores(bank: str) -> dict:
    """exp15_report.reader_scores for Gemma: each token lens at its fitted primary layers; probe and diffmeans as in
    exp15 (no text readers)."""
    import exp15_score as S
    lr = E15.load_lens_reads(bank)
    colours = list(lr["labels"][:10])
    pc = [list(lr["cells"]).index(c) for c in E15.PRIMARY_CELLS]
    out = {}
    for kind in TOKEN_LENSES:
        layers = [l for l in PRIMARY_LAYERS if l in lr["fitted"][kind]]
        pairs = [(c, l) for c in E15.PRIMARY_CELLS for l in layers]
        x = lr["scores"][kind][:, pc][:, :, layers, :10].float()
        out[kind] = S.ReaderScores(kind, "rank", pairs, {i: x[n].reshape(-1, 10) for n, i in enumerate(lr["item_ids"])})
    pr = E15.load_probe_reads(bank)
    assert list(pr["labels"]) == colours and tuple(pr["layers"]) == PRIMARY_LAYERS
    pairs = [(c, l) for c in E15.PRIMARY_CELLS for l in PRIMARY_LAYERS]
    for kind in ("probe", "diffmeans"):
        x = pr["scores"][kind]
        out[kind] = S.ReaderScores(kind, "rank", pairs, {i: x[n].reshape(-1, 10) for n, i in enumerate(pr["item_ids"])})
    return out


# --- write directions -----------------------------------------------------------------------------------------------
@torch.no_grad()
def build_directions() -> None:
    """exp15.build_directions' lens and diffmeans parts for Gemma -> directions.pt: v_layer(colour) [10, d] per method
    and write layer; diffmeans from correct h=2/3 dev items at the final cell, as in exp15."""
    from transformers import AutoTokenizer
    path = E15.directions_path()
    E15.require_absent(path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    metas, _ = E15.load_capture(DEV_BANK, with_hidden=False)
    colours = metas[0]["colours"]
    lenses = GemmaLenses(E15.colour_ids(tokenizer, colours), device="cpu")
    out = {m: {l: torch.stack([lenses.direction(m, l, k) for k in range(10)]) for l in METHOD_LAYERS[m]}
           for m in TOKEN_LENSES}
    table = E15.item_table(DEV_BANK)
    train = [i for i, it in enumerate(table) if it.group == "correct" and it.h in (2, 3)]
    y = torch.tensor([colours.index(table[i].key) for i in train])
    layers = METHOD_LAYERS["diffmeans"]
    x = E15.cell_layer_hidden(DEV_BANK, cells=("final",), layers=layers)[train, 0].transpose(0, 1)      # [L, n, d]
    _, dirs = E15.diffmeans_directions(x, y)
    out["diffmeans"] = {l: dirs[k] for k, l in enumerate(layers)}
    out["colours"] = colours
    E15.atomic_save(out, path)
    print({k: sorted(v) for k, v in out.items() if isinstance(v, dict)}, f"diffmeans from {len(train)} dev items")


# --- bank, checks ---------------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return E16.sha256(path)


def copy_bank() -> None:
    """exp16's test-bank items and requests, copied (with sha256) after the Gemma manifest is locked; and the ptag
    check record exp15's write selection reads (no ptag AR for Gemma)."""
    if not SMOKE and (not EXP.manifest.exists() or json.loads(EXP.manifest.read_text()).get("status") != "locked"):
        raise SystemExit("the Gemma arm copies the bank only after its manifest is locked")
    EXP.cache.mkdir(parents=True, exist_ok=True)
    record = {"source": str(SOURCE.cache.relative_to(cfg.REPO_ROOT)), "files": {}}
    for name in (f"items_{TEST_BANK}.jsonl", f"requests_{TEST_BANK}.jsonl"):
        src, dst = SOURCE.cache / name, EXP.cache / name
        E15.require_absent(dst)
        shutil.copy2(src, dst)
        record["files"][name] = {"sha256": sha256(dst), "source_sha256": sha256(src)}
    E15.atomic_write_text(json.dumps(record, indent=1) + "\n", EXP.results / "bank_source.json")
    E15.write_check("ptag", {"pass": False, "note": "no ptag AR for Gemma: written so exp15's selection skips it"})
    print(json.dumps(record, indent=1))


@torch.no_grad()
def bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest |a - b| in units of the bf16 spacing at max(|a|, |b|) (bf16 keeps 7 mantissa bits: 0.125 at 16-32)."""
    a, b = a.float(), b.float()
    spacing = torch.exp2(torch.floor(torch.log2(torch.maximum(a.abs(), b.abs()).clamp_min(1e-30))) - 7)
    return float(((a - b).abs() / spacing).max())


def check_model(device: str, name: str) -> None:
    """Before any capture: (a) the text-stack path run_trials uses (GemmaText.model + .lm_head) gives the full model's
    logits up to bf16 rounding (same argmax, colour probabilities within 0.01, no logit off by >= 1.0; a missing
    softcap or norm moves logits by whole units; the CPU run differed by at most 2 bf16 steps);
    (b) hidden_states[l + 1] is block l's output and hidden_states[60] is the final norm of block 59's;
    (c) a zero edit changes nothing and a unit edit changes the logits; (d) the first dev prompt's tail tokens."""
    import exp11_brew as B
    tokenizer, model = load_base(device_map=device)
    rows = E15.load_requests(DEV_BANK)[:2]
    items = E15.load_items(DEV_BANK)
    record = {"device": device}
    for n, row in enumerate(rows):
        text, ids = E15.prompt(tokenizer, row)
        brew = B.parse(items[row["item_id"]]["problem"])
        cids = E15.colour_ids(tokenizer, sorted(brew.table))
        cells = E15.read_cells(tokenizer, text, ids, brew)
        x = torch.tensor([ids], device=device)
        outputs = {}
        hooks = [model.model.layers[l].register_forward_hook(
            lambda mod, a, o, l=l: outputs.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach()))
            for l in (0, 30, 59)]
        full = model(input_ids=x, output_hidden_states=True, use_cache=False)
        for h in hooks:
            h.remove()
        hs = full.hidden_states
        text_path = model.lm_head(model.model(input_ids=x, use_cache=False).last_hidden_state[:, -1])
        final_norm = model.model.norm(outputs[59])
        full_logits = full.logits[:, -1]
        record[f"row{n}"] = {
            "n_hidden_states": len(hs), "cells": cells,
            "logit_max_abs_diff_text_path": float((text_path.float() - full_logits.float()).abs().max()),
            "logit_max_bf16_steps_text_path": bf16_ulps(text_path, full_logits),
            "argmax_same_text_path": bool(text_path.argmax() == full_logits.argmax()),
            "colour_prob_max_diff_text_path": float((torch.softmax(text_path[0, cids].float(), -1)
                                                     - torch.softmax(full_logits[0, cids].float(), -1)).abs().max()),
            "block_l_is_hidden_l+1": {l: bool(torch.equal(outputs[l], hs[l + 1])) for l in (0, 30)},
            "hidden_60_is_final_norm": bool(torch.allclose(final_norm.float(), hs[60].float(), atol=1e-2)),
            "argmax_token": tokenizer.decode(int(full.logits[0, -1].argmax())),
        }
        if n == 0:
            edit = ED.AdditiveEdit(model, 40)
            u = torch.nn.functional.normalize(torch.randn(1, model.config.get_text_config().hidden_size), dim=-1)
            end, start = torch.tensor([len(ids) - 1]), torch.tensor([cells["start"]])
            changes = {}
            for alpha in (0.0, 1.0):
                edit.batch(u, torch.tensor([alpha]), start, end)
                logits = model.lm_head(model.model(input_ids=x, use_cache=False).last_hidden_state[:, -1])
                edit.clear()
                changes[alpha] = float((logits.float() - text_path.float()).abs().max())
            edit.remove()
            record["edit_logit_change"] = changes
    r0 = record["row0"]
    text_path_ok = all(r["logit_max_abs_diff_text_path"] < 1.0 and r["argmax_same_text_path"]
                       and r["colour_prob_max_diff_text_path"] < 0.01 for r in (record["row0"], record["row1"]))
    record["pass"] = bool(r0["n_hidden_states"] == N_LAYERS + 1 and text_path_ok
                          and all(r0["block_l_is_hidden_l+1"].values()) and r0["hidden_60_is_final_norm"]
                          and record["edit_logit_change"][0.0] == 0.0 and record["edit_logit_change"][1.0] > 0.1)
    E15.write_check(name, record)
    print(json.dumps(record, indent=1, default=str))
    if not record["pass"]:
        raise SystemExit("Gemma model check failed")


def dev_check() -> None:
    """Gemma's dev capture must reproduce exp12's brew-gate accuracy on single_test (same rendering) within 3 items
    per h: the loader and the prompt are the ones exp12 measured."""
    table = E15.item_table(DEV_BANK)
    record = {"tolerance": DEV_TOLERANCE, "by_h": {}}
    for h, expected in EXP12_DEV_ACCURACY.items():
        rows = [it for it in table if it.h == h]
        acc = sum(it.group == "correct" for it in rows) / len(rows)
        record["by_h"][h] = {"n": len(rows), "accuracy": round(acc, 4), "exp12": expected,
                             "offcycle": sum(it.group == "offcycle" for it in rows)}
    record["pass"] = all(abs(v["accuracy"] - v["exp12"]) <= DEV_TOLERANCE + 1e-9 for v in record["by_h"].values())
    E15.write_check("dev_vs_exp12", record)
    print(json.dumps(record, indent=1))
    if not record["pass"]:
        raise SystemExit("Gemma's dev accuracy does not reproduce exp12's brew gate")


# --- command line ---------------------------------------------------------------------------------------------------
def run_stage(module_name: str, argv: list[str]) -> None:
    """An exp15 module's own main() under the Gemma overrides; exp15_report.reader_scores is replaced first, so
    modules that import it by name get Gemma's."""
    if not module_name.startswith("exp15"):
        raise SystemExit(f"stage runs exp15 modules only, not {module_name}")
    import exp15_report as R
    R.reader_scores = reader_scores
    module = importlib.import_module(module_name)
    sys.argv = [f"{module_name}.py", *argv]
    module.main()


if SMOKE:
    # The smoke run's dev grid and test swaps use 20 items each (the dev grid alone is ~50k trials on all items).
    _swap_items = E15.swap_items
    E15.swap_items = lambda bank: _swap_items(bank)[:20]


def report(tag: str) -> None:
    import exp15_report as R
    R.reader_scores = reader_scores
    import exp16_report
    assert exp16_report.EXP == EXP
    exp16_report.confirmatory(tag)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("bank", "dev_check", "lens_reads", "directions", "zero_edit", "erase"):
        sub.add_parser(name)
    check = sub.add_parser("check_model")
    check.add_argument("--device", default="cuda")
    check.add_argument("--name", help="check file name (default model_<device>)")
    sub.add_parser("report").add_argument("--tag", required=True)
    sub.add_parser("stage").add_argument("rest", nargs=argparse.REMAINDER, help="<exp15 module> <its arguments>")
    a = p.parse_args()
    print(f"exp16 Gemma arm ({EXP.exp_id})", flush=True)
    if a.cmd == "bank":
        copy_bank()
    elif a.cmd == "check_model":
        check_model(a.device, a.name or f"model_{a.device}")
    elif a.cmd == "dev_check":
        dev_check()
    elif a.cmd == "lens_reads":
        lens_reads(TEST_BANK)
    elif a.cmd == "directions":
        build_directions()
    elif a.cmd == "zero_edit":
        E16.zero_edit_check()
    elif a.cmd == "erase":
        E16.erase(TEST_BANK)
    elif a.cmd == "report":
        report(a.tag)
    elif a.cmd == "stage":
        run_stage(a.rest[0], a.rest[1:])


if __name__ == "__main__":
    main()
