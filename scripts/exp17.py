"""exp17: is the brew lookup key carried by the J++ space? (results/exp17_jspace_keys/manifest.json)

The global-workspace paper's claim under test: the J-space part of an intermediate's probe direction carries most of
the probe's causal effect on multi-hop answers (61% vs 28% for the remainder). Here the intermediate is the key of a
no-CoT brew lookup (s1 on a correct h=2 item; the model answers T(s1)), written s1 -> s1' by the paper's clamped
coordinate swap (exp17_swaps) along: the raw J++ colour vectors; the difference-of-means key direction (exp15's best
writer); its J++ part (k = 25 J++ vectors, non-negative gradient pursuit); and its remainder. Outcomes tell a key from
an output: T(s1') = the written colour was looked up; s1' itself = it was just said.

One model per process (EXP17_MODEL = qwen | gemma); every path is under cache/exp17/<run> and
results/exp17_jspace_keys/<run>, except the shared test bank (cache/exp17/items_test.jsonl) and the model's clean
baselines and occupied basis (cache/exp17/<model>). The run is the model, or <model>_occupied with
EXP17_LENS=occupied: every J++ vector and random direction is projected onto the subspace the model's residual stream
occupies (exp17_active) and renormalised. A J++ readout of an occupied residual only sees that projection, but on
Gemma the raw J++ vectors lie 80-95% outside it, so raw swaps along them barely move the output (run "gemma").

Stages (several may be given; the model is loaded once per process):
  baseline:dev | baseline:test   one batched clean forward per row: colour logits (and, on dev, the final-position
                                 residual at every layer, for the difference of means)
  check      the trial path (equal-length batches, text stack) against the unbatched full model; hook numbering;
             baseline accuracy against exp11's (Qwen) or exp12's (Gemma) on the dev bank
  prep       readable vocabulary, colour vectors per family and layer, J++ decompositions
  posctl     positive control: the paper's 90 two-hop factual prompts (probe-swap.json), raw J++ swap
  dev        the selection grid on dev's correct h=2 items
  select     each family's setting by the manifest's rule -> dev_selection.json
  preview    the post-table test arms on dev items at the dev selection (before the lock)
  bank       the fresh shared h=2 test bank (after the manifest is locked; written once; no model)
  test       the locked arms (manifest.json, or a locked addendum) on the test bank's correct h=2 items

Layer l is the output of decoder block l (hidden_states[l + 1]), the lenses' numbering. Run with /venv/main/bin/python
from scripts/; HF_HOME must point at the model cache.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

MODEL_KEY = os.environ.get("EXP17_MODEL", "")
if MODEL_KEY not in ("qwen", "gemma"):
    raise SystemExit("set EXP17_MODEL=qwen or EXP17_MODEL=gemma")
if MODEL_KEY == "gemma":
    import exp16_gemma as G  # noqa: E402  (applies the Gemma overrides to exp15: loader, chat tail, cells)
import cc_config as cfg  # noqa: E402
import exp11 as X  # noqa: E402
import exp11_brew as B  # noqa: E402
import exp15 as E15  # noqa: E402
import exp15_edits as ED  # noqa: E402
import exp15_score as S  # noqa: E402
import exp17_swaps as SW  # noqa: E402

torch.backends.cuda.enable_cudnn_sdp(False)

LENS = os.environ.get("EXP17_LENS", "raw")
if LENS not in ("raw", "occupied"):
    raise SystemExit("EXP17_LENS must be raw (default) or occupied")
RUN_KEY = MODEL_KEY if LENS == "raw" else f"{MODEL_KEY}_{LENS}"

EXP = cfg.Exp("exp17_jspace_keys", "exp17")
MODEL_CACHE = EXP.cache / MODEL_KEY      # lens-independent: clean baselines, the occupied basis
CACHE = EXP.cache / RUN_KEY
RESULTS = EXP.results / RUN_KEY
DEV_BANK = "single_test"                 # exp11's single-table test bank: exp15's dev bank (and exp12's Gemma gate)
TEST_SPEC = {"seed": "exp17_test_20261012", "format": "single_first", "per_h": {2: 1200}}
PROBE_SWAP = EXP.cache / "probe-swap.json"
PROBE_SWAP_SOURCE = {"repo": "https://github.com/anthropics/jacobian-lens", "commit":
                     "581d398613e5602a5af361e1c34d3a92ea82ba8e", "path": "data/experiments/probe-swap.json",
                     "sha256": "a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796"}
PURSUIT_K = 25                           # the paper's k for multi-hop intermediate probes
N_RANDOM_DECOMP = 10                     # random unit vectors per layer: the pursuit's explained variance by chance
BAND_ALPHAS = (1.0, 2.0, 4.0, 8.0)
SINGLE_LAYER_ALPHAS = (2.0, 4.0, 8.0, 16.0)
ANSWER_ALPHAS = (1.0, 2.0)
POSCTL_ALPHAS = (1.0, 2.0, 4.0)
POSCTL_CHAT = RUN_KEY == "gemma_occupied"            # see posctl()
POSCTL_INSTRUCTION = "Complete the sentence with one word. Reply with that word only.\n\n"
CHECK_TOLERANCE = 3                      # items per h


@dataclass(frozen=True)
class ModelSpec:
    key: str
    n_layers: int
    bands: dict                          # name -> (first, last) block, inclusive: the dev grid's bands
    ar_band: str | None                  # the NLA reconstructor's layer (Qwen only)
    control_bands: dict = field(default_factory=dict)   # later bands for the in-task J++ answer-swap control only


SPECS = {
    "qwen": ModelSpec("qwen", 64, {"b24_36": (24, 36), "b30_42": (30, 42), "b36_48": (36, 48), "b42_54": (42, 54),
                                   "b24_48": (24, 48), "l42": (42, 42)}, "l42",
                      control_bands={"b48_60": (48, 60), "b54_62": (54, 62)}),
    # Gemma 4 31B (60 blocks): Qwen's bands scaled by depth; its J++ fit has only even layers (and 15, 45); no NLA
    "gemma": ModelSpec("gemma", 60, {"b22_34": (22, 34), "b28_40": (28, 40), "b34_46": (34, 46), "b40_52": (40, 52),
                                     "b22_46": (22, 46)}, None,
                       control_bands={"b46_58": (46, 58), "b52_58": (52, 58)}),
}
SPEC = SPECS[MODEL_KEY]


# --- model adapter --------------------------------------------------------------------------------------------------
@functools.cache
def load_model():
    """(tokenizer, model): model.model.layers are the decoder blocks, model.model(...) the text stack (its
    last_hidden_state after the final norm), model.lm_head the head (Gemma's with its logit softcap)."""
    return E15.load_base()


@functools.cache
def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(E15.model_path())


def jpp_jacobians() -> dict[int, torch.Tensor]:
    """J++ maps {layer: J [d, d]} (fp32, memory-mapped): Qwen koayon/jpp-lenses (layers 0-62); Gemma exp12's fit for
    the -it model (64 prompts; even layers and 15, 45)."""
    if MODEL_KEY == "qwen":
        import exp15_readers as RD
        return RD.load_jacobians("jpp")
    blob = torch.load(cfg.REPO_ROOT / "cache" / "exp12" / "lenses" / "jpp_64p.pt", map_location="cpu",
                      weights_only=False, mmap=True)
    assert blob["config"]["hf_model_name"] == G.MODEL, blob["config"]["hf_model_name"]
    return {int(l): J for l, J in blob["parameters"]["jacobians"].items()}


def gain_and_unembed() -> tuple[torch.Tensor, torch.Tensor]:
    """The final norm's per-dimension gain (Qwen 1 + w, Gemma w) and the unembedding [vocab, d] (bf16)."""
    if MODEL_KEY == "qwen":
        import exp15_readers as RD
        return (1.0 + RD.load_checkpoint_tensor("model.language_model.norm.weight").float(),
                RD.load_checkpoint_tensor("lm_head.weight"))
    return (G.checkpoint_tensor("model.language_model.norm.weight").float(),
            G.checkpoint_tensor("model.language_model.embed_tokens.weight"))


@functools.cache
def occupied_basis() -> dict[int, torch.Tensor]:
    """{layer: [r, d] orthonormal rows, cuda}: the eigenvectors holding 99.99% of the model's WikiText residual
    energy at that layer (exp17_active)."""
    blob = torch.load(MODEL_CACHE / "occupied_basis.pt", weights_only=False)
    return {l: P.to("cuda", torch.float32) for l, P in blob["bases"].items()}


def jpp_vectors(jac: dict, layer: int, gain: torch.Tensor, w_rows: torch.Tensor) -> torch.Tensor:
    """Unit J++ vectors J_l^T (gain * W_U[t]) [n, d] (bf16, cuda) for the unembedding rows w_rows; with
    EXP17_LENS=occupied, their renormalised projections onto the occupied subspace."""
    atoms = SW.lens_atoms(jac[layer], gain, w_rows)
    return SW.project_unit(atoms, occupied_basis()[layer]) if LENS == "occupied" else atoms


def random_unit(rows: torch.Tensor, layer: int) -> torch.Tensor:
    """Random unit directions rows [n, d] (any normalisation): unit, and with EXP17_LENS=occupied, drawn inside the
    occupied subspace (where the J++ vectors they are compared with now lie)."""
    rows = torch.nn.functional.normalize(rows.float(), dim=-1)
    return SW.project_unit(rows, occupied_basis()[layer]) if LENS == "occupied" else rows


def band_layers(band: str, jpp_layers) -> tuple[int, ...]:
    """The blocks of a band (a dev band or a control band) at which a J++ map exists."""
    lo, hi = {**SPEC.bands, **SPEC.control_bands}[band]
    return tuple(l for l in range(lo, hi + 1) if l in jpp_layers)


def all_band_layers(jpp_layers) -> tuple[int, ...]:
    return tuple(sorted({l for b in SPEC.bands for l in band_layers(b, jpp_layers)}))


# --- io -------------------------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, record: dict) -> None:
    E15.require_absent(path)
    E15.atomic_write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", path)


def manifest() -> dict:
    return json.loads(EXP.manifest.read_text()) if EXP.manifest.exists() else {}


def require_locked() -> None:
    if manifest().get("status") != "locked":
        raise SystemExit(f"{EXP.manifest} is not locked")


def locked_settings() -> dict:
    """Run key -> locked settings: manifest.json's, then each locked addendum's (manifest_addendum_<name>.json, each
    written once before the runs it covers; an addendum may not change a setting already locked)."""
    require_locked()
    settings = {run: dict(s) for run, s in manifest()["locked_settings"].items()}
    for path in sorted(EXP.results.glob("manifest_addendum_*.json")):
        add = json.loads(path.read_text())
        assert add["status"] == "locked", path
        for run, extra in add["locked_settings"].items():
            clash = set(extra) & set(settings.get(run, {}))
            assert not clash, f"{path.name} re-locks {run}: {sorted(clash)}"
            settings[run] = {**settings.get(run, {}), **extra}
    return settings


# --- banks ----------------------------------------------------------------------------------------------------------
def bank_items(bank: str) -> dict[str, dict]:
    if bank == "dev":
        return E15.load_items(DEV_BANK)
    return {r["item_id"]: r for r in map(json.loads, (EXP.cache / "items_test.jsonl").open())}


def bank_requests(bank: str) -> list[dict]:
    if bank == "dev":
        return E15.load_requests(DEV_BANK)
    return [json.loads(l) for l in (EXP.cache / "requests_test.jsonl").open()]


def write_test_bank() -> None:
    """exp16's bank writer with exp17's seed and size (h=2 only): one h=2 shot plus 1200 eval items, single_first
    format, every gold re-derived by the parser, no problem text shared with exp11's, exp15's or exp16's banks; and
    its C0 requests (exp10's rendering). Shared by both models."""
    import cc_exp10 as E10
    import exp10_conditions as C
    require_locked()
    items_path, req_path = EXP.cache / "items_test.jsonl", EXP.cache / "requests_test.jsonl"
    E15.require_absent(items_path)
    E15.require_absent(req_path)
    brew, _ = E10.nocot_import()
    fmt = TEST_SPEC["format"]
    rng = random.Random(f"{TEST_SPEC['seed']}:{fmt}")
    cells = [("shot", 2, 0)] + [("eval", h, k) for h, n in TEST_SPEC["per_h"].items() for k in range(n)]
    rows = []
    for split, h, k in cells:
        gen = B.generate(rng, h, B.FORMAT_SPEC[fmt][0], brew.COLORS, brew.INGREDIENTS)
        problem = B.render(gen, fmt, brew.COLORS)
        assert B.parse(problem).path()[-1] == gen["answer"]
        rows.append({"item_id": f"exp17_test:{fmt}:{split}:h{h}:{k}", "bank": "exp17_test", "format": fmt,
                     "split": split, "h": h, "problem": problem, "answer": gen["answer"],
                     "instruction": brew.INSTRUCTION})
    texts = collections.Counter(r["problem"] for r in rows)
    texts.update(r["problem"] for b in X.ALL_BANKS if X.bank_path(b).exists() for r in X.load_bank(b))
    others = [cfg.REPO_ROOT / "cache" / d / "items_test.jsonl" for d in ("exp15", "exp16")]
    for p in others:
        texts.update(json.loads(line)["problem"] for line in p.open())
    if max(texts.values()) > 1:
        raise SystemExit("a problem text repeats across exp11's, exp15's, exp16's and exp17's banks")
    items_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    shot = next(r for r in rows if r["split"] == "shot")
    reqs = []
    for it in (r for r in rows if r["split"] == "eval"):
        row = {"item_id": it["item_id"], "bank": "exp17_test", "h": it["h"], "source": "brew", "exp10_render": True,
               "seed": 0, "format": it["format"], **C.fields("C0", it, shot)}
        row["request_id"] = cfg.content_key(row)
        reqs.append(row)
    req_path.write_text("".join(json.dumps(r) + "\n" for r in reqs))
    write_once(EXP.results / "test_bank.json",
               {**TEST_SPEC, "n_eval": len(reqs), "content_key": cfg.content_key(rows),
                "items_sha256": sha256(items_path), "requests_sha256": sha256(req_path),
                "checked_unique_against": [*X.ALL_BANKS, *(str(p.relative_to(cfg.REPO_ROOT)) for p in others)]})
    print(f"{len(rows)} items, {len(reqs)} requests -> {EXP.cache}")


# --- baseline pass --------------------------------------------------------------------------------------------------
def baseline_path(bank: str) -> Path:
    return MODEL_CACHE / f"baseline_{bank}.pt"


def equal_length_batches(lengths: list[int], batch: int) -> list[list[int]]:
    """Indices grouped by sequence length, each group cut into batches of at most `batch`. No batch is padded: with a
    padded attention mask Gemma's logits move by up to 1.4 (log/exp17/gemma_batch_diag.log), while unpadded batches
    reproduce the unbatched model exactly."""
    groups = collections.defaultdict(list)
    for i, n in enumerate(lengths):
        groups[n].append(i)
    return [g[s:s + batch] for _, g in sorted(groups.items()) for s in range(0, len(g), batch)]


def stack_batch(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equal-length ids [b, T], an all-ones attention mask, and each row's last position."""
    assert len(set(map(len, seqs))) == 1, "exp17 batches are never padded"
    ids = torch.tensor(seqs, device="cuda")
    return ids, torch.ones_like(ids), torch.full((len(seqs),), ids.shape[1] - 1, device="cuda")


def prepared_rows(bank: str) -> tuple[list[str], list[tuple[dict, list[int], int]]]:
    """The bank's colours and, per request row, (row, prompt ids, token index of the start colour)."""
    tokenizer = load_tokenizer()
    items = bank_items(bank)
    rows = bank_requests(bank)
    colours = sorted(B.parse(items[rows[0]["item_id"]]["problem"]).table)
    out = []
    for row in rows:
        brew = B.parse(items[row["item_id"]]["problem"])
        assert sorted(brew.table) == colours
        text, ids = E15.prompt(tokenizer, row)
        out.append((row, ids, E15.read_cells(tokenizer, text, ids, brew)["start"]))
    return colours, out


@torch.no_grad()
def baseline(bank: str, batch: int = 16) -> None:
    """Per row: the 10 colour logits at the final position (the trials' batched text-stack path), the start colour's
    token index, and on dev the final-position residual at every block output 0..n_layers-2 (difference of means)."""
    path = baseline_path(bank)
    E15.require_absent(path)
    tokenizer, model = load_model()
    colours, prepared = prepared_rows(bank)
    cids = torch.tensor(E15.colour_ids(tokenizer, colours), device="cuda")
    keep_hidden = bank == "dev"
    metas, hidden = [None] * len(prepared), [None] * len(prepared)
    n_done = 0
    for part in equal_length_batches([len(p[1]) for p in prepared], batch):
        ids, mask, ends = stack_batch([prepared[i][1] for i in part])
        out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=keep_hidden)
        rows_b = torch.arange(len(part), device="cuda")
        logits = model.lm_head(out.last_hidden_state[rows_b, ends]).float()
        probs = torch.softmax(logits, -1)
        if keep_hidden:
            # hidden_states[l + 1] is block l's output for l < n_layers - 1; the last entry is after the final norm
            stack = torch.stack([h[rows_b, ends] for h in out.hidden_states[1:SPEC.n_layers]], 1).cpu()
        for k, i in enumerate(part):
            row, seq, start = prepared[i]
            metas[i] = {"item_id": row["item_id"], "h": row["h"], "n_tokens": len(seq), "start": start,
                        "colour_logits": [round(float(x), 4) for x in logits[k, cids]],
                        "colour_mass": round(float(probs[k, cids].sum()), 6)}
            if keep_hidden:
                hidden[i] = stack[k]
        n_done += len(part)
        print(f"baseline {bank} {n_done}/{len(prepared)}", flush=True)
    E15.atomic_save({"colours": colours, "metas": metas, "hidden": torch.stack(hidden) if keep_hidden else None}, path)
    print(f"-> {path}")


def item_table(bank: str) -> tuple[list[str], list[S.Item], list[dict]]:
    blob = torch.load(baseline_path(bank), weights_only=False)
    items = bank_items(bank)
    table = [S.analyse(m["item_id"], B.parse(items[m["item_id"]]["problem"]), blob["colours"], m["colour_logits"])
             for m in blob["metas"]]
    return blob["colours"], table, blob["metas"]


def reference_accuracy() -> dict[int, float]:
    """Dev-bank accuracy per h from the run that defined it: exp11's labels (Qwen) or exp12's brew gate (Gemma)."""
    if MODEL_KEY == "qwen":
        return {h: c["correct"] / 250 for h, c in E15.EXP11_SINGLE_TEST.items()}
    return dict(G.EXP12_DEV_ACCURACY)


CHECK_FILE = "check_v2.json"     # check.json: the first check, on padded batches (Gemma failed it; manifest deviation)


@torch.no_grad()
def check() -> None:
    """Before any trial: (a) on 8 dev rows of one length (an equal-length batch, as in every exp17 forward) the
    batched text-stack path gives the unbatched full model's colour argmax, colour probabilities within 0.02; (b) a
    hook on block l sees hidden_states[l + 1]; (c) the dev baseline's accuracy per h is the reference run's within 3
    items."""
    tokenizer, model = load_model()
    colours, prepared = prepared_rows("dev")
    cids = torch.tensor(E15.colour_ids(tokenizer, colours), device="cuda")
    picks = max(equal_length_batches([len(p[1]) for p in prepared], 8), key=len)
    ids, mask, ends = stack_batch([prepared[i][1] for i in picks])
    batched = model.lm_head(model.model(input_ids=ids, attention_mask=mask, use_cache=False)
                            .last_hidden_state[torch.arange(len(picks), device="cuda"), ends]).float()[:, cids]
    rows, seen = [], {}
    layer = SPEC.bands[next(iter(SPEC.bands))][0]
    hook = model.model.layers[layer].register_forward_hook(
        lambda mod, a, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    for k, i in enumerate(picks):
        full = model(input_ids=torch.tensor([prepared[i][1]], device="cuda"), use_cache=False,
                     output_hidden_states=k == 0)
        single = full.logits[0, -1].float()[cids]
        if k == 0:
            hook_ok = bool(torch.equal(seen["h"], full.hidden_states[layer + 1]))
            hook.remove()
        p_b, p_s = torch.softmax(batched[k], -1), torch.softmax(single, -1)
        rows.append({"n_tokens": len(prepared[i][1]), "argmax_same": bool(p_b.argmax() == p_s.argmax()),
                     "colour_prob_max_diff": round(float((p_b - p_s).abs().max()), 5),
                     "colour_logit_max_diff": round(float((batched[k] - single).abs().max()), 4)})
    _, table, _ = item_table("dev")
    accuracy = {}
    for h, ref in reference_accuracy().items():
        group = [it for it in table if it.h == h]
        acc = sum(it.group == "correct" for it in group) / len(group)
        accuracy[h] = {"n": len(group), "accuracy": round(acc, 4), "reference": ref,
                       "ok": abs(acc - ref) * len(group) <= CHECK_TOLERANCE + 1e-9}
    record = {"batched_vs_single": rows, "hook_layer": layer, "hook_is_hidden_l+1": hook_ok, "accuracy": accuracy}
    record["pass"] = bool(all(r["argmax_same"] and r["colour_prob_max_diff"] < 0.02 for r in rows) and hook_ok
                          and all(a["ok"] for a in accuracy.values()))
    write_once(RESULTS / CHECK_FILE, record)
    print(json.dumps(record, indent=1))
    if not record["pass"]:
        raise SystemExit("exp17 check failed")


# --- prep: colour vectors per family and their J++ decompositions ----------------------------------------------------
def directions_path() -> Path:
    return CACHE / "directions.pt"


def decompose(vectors: torch.Tensor, atoms: torch.Tensor, keep: list[int], labels: list[str], own_rows=None):
    """Pursuit decomposition of each row of vectors [n, d]: (J-parts [n, d], per-row records). own_rows[k] is the
    dictionary row of the vector's own colour token (its rank in the support is recorded)."""
    tokenizer = load_tokenizer()
    parts, info = [], []
    for k, label in enumerate(labels):
        d = SW.nn_gradient_pursuit(vectors[k], atoms, PURSUIT_K)
        parts.append(d.j_part)
        rec = {"label": label, "explained": round(d.explained, 4),
               "support": [tokenizer.decode([keep[j]]) for j in d.support[:10]],
               "coef": [round(float(c), 4) for c in d.coef[:10]]}
        if own_rows is not None:
            rec["own_token_rank"] = d.support.index(own_rows[k]) if own_rows[k] in d.support else None
        info.append(rec)
    return torch.stack(parts), info


@torch.no_grad()
def prep() -> None:
    """Colour vectors [10, d] per family and layer (every layer of every band):
      jpp       unit J++ vectors of the colour tokens (' red', ...)
      dm_full   unit difference of means of the final-position residual, key colour vs the rest (correct h=2/h=3
                dev items, key = T^-1(answer) = s_{h-1}), as exp15's diffmeans direction
      dm_jpart  its J++ part: non-negative gradient pursuit, k = 25, over the readable vocabulary's unit J++ vectors
      dm_rem    dm_full - dm_jpart
      ar_*      (Qwen, layer 42) exp15's NLA reconstructor colour vectors, centred over the 10 colours and unit, split
                the same way
    Recorded per layer: explained variance per colour, the same for random unit vectors (chance level), the cosine of
    dm_full with the colour's J++ vector, and the key-class separation along dm_full relative to |h|."""
    path = directions_path()
    E15.require_absent(path)
    tokenizer = load_tokenizer()
    colours, table, _ = item_table("dev")
    hidden = torch.load(baseline_path("dev"), weights_only=False)["hidden"]          # [n, n_layers - 1, d] bf16
    jac = jpp_jacobians()
    layers = all_band_layers(jac)
    gain, w_u = gain_and_unembed()
    keep = SW.readable_token_ids(tokenizer)
    row_of = {t: r for r, t in enumerate(keep)}
    cids = E15.colour_ids(tokenizer, colours)
    own_rows = [row_of[c] for c in cids]
    w_keep = w_u[torch.tensor(keep)].to("cuda")
    train = [i for i, it in enumerate(table) if it.group == "correct" and it.h in (2, 3)]
    y = torch.tensor([colours.index(table[i].key) for i in train])
    x = hidden[train][:, list(layers)].float().transpose(0, 1)                        # [L, n, d]
    _, dm = E15.diffmeans_directions(x, y)                                             # [L, 10, d] unit
    ar_layer, ar = None, None
    if SPEC.ar_band is not None:
        ar_layer = SPEC.bands[SPEC.ar_band][0]
        raw = torch.load(cfg.REPO_ROOT / "cache" / "exp15" / "directions.pt", weights_only=False)
        assert raw["colours"] == colours
        ar = raw["nla_ar"][ar_layer].float()
        ar = torch.nn.functional.normalize(ar - ar.mean(0, keepdim=True), dim=-1)
    families, decomp = collections.defaultdict(dict), collections.defaultdict(dict)
    stats = collections.defaultdict(dict)
    g = torch.Generator().manual_seed(17)
    for li, l in enumerate(layers):
        atoms = jpp_vectors(jac, l, gain, w_keep)                                     # [n_keep, d] unit, cuda
        families["jpp"][l] = atoms[own_rows].float().cpu()
        todo = {"dm": dm[li]} | ({"ar": ar} if l == ar_layer else {})
        for name, vecs in todo.items():
            jpart, info = decompose(vecs, atoms, keep, colours, own_rows)
            families[f"{name}_full"][l] = vecs.float()
            families[f"{name}_jpart"][l] = jpart
            families[f"{name}_rem"][l] = vecs.float() - jpart
            decomp[name][l] = info
            stats[f"{name}_cos_own_jpp"][l] = [round(float(c), 4) for c in
                                               torch.nn.functional.cosine_similarity(vecs.float(),
                                                                                     families["jpp"][l], -1)]
        rand = random_unit(torch.randn(N_RANDOM_DECOMP, x.shape[-1], generator=g), l)
        _, rinfo = decompose(rand, atoms, keep, [f"random{k}" for k in range(N_RANDOM_DECOMP)])
        stats["random_explained"][l] = [r["explained"] for r in rinfo]
        # separation along dm_full: mean projection of key-class rows minus the rest, over mean |h|
        proj = torch.einsum("nd,cd->nc", x[li], dm[li])
        on = torch.nn.functional.one_hot(y, 10).bool()
        sep = torch.stack([proj[on[:, c], c].mean() - proj[~on[:, c], c].mean() for c in range(10)])
        stats["dm_separation_over_norm"][l] = round(float(sep.mean() / x[li].norm(dim=-1).mean()), 4)
        del atoms
        print(f"prep layer {l}: dm explained {sum(i['explained'] for i in decomp['dm'][l]) / 10:.3f}, random "
              f"{sum(stats['random_explained'][l]) / N_RANDOM_DECOMP:.3f}", flush=True)
    E15.atomic_save({"colours": colours, "layers": layers, "families": dict(families)}, path)

    def mean(v):
        return round(sum(v) / len(v), 4)

    summary = {
        "layers": layers, "n_dictionary": len(keep), "n_dm_train": len(train), "pursuit_k": PURSUIT_K,
        "explained_mean": {name: {l: mean([i["explained"] for i in v]) for l, v in d.items()}
                           for name, d in decomp.items()},
        "random_explained_mean": {l: mean(v) for l, v in stats["random_explained"].items()},
        "own_colour_in_support": {name: {l: sum(i["own_token_rank"] is not None for i in v) for l, v in d.items()}
                                  for name, d in decomp.items()},
        "cos_with_own_jpp_mean": {k: {l: mean(v) for l, v in per.items()} for k, per in stats.items()
                                  if k.endswith("cos_own_jpp")},
        "dm_separation_over_norm": stats["dm_separation_over_norm"],
        "per_colour": dict(decomp)}
    write_once(RESULTS / "decomposition.json", summary)
    print(json.dumps({k: summary[k] for k in ("explained_mean", "random_explained_mean", "dm_separation_over_norm")},
                     indent=1))


# --- arms, targets and the trial loop -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    name: str
    kind: str                    # swap | matched | null | patch
    main: str | None             # vector family of the main pair (patch: see patch_bases)
    ref_arm: str | None          # matched, null: the earlier arm (same band) whose recorded edit norms are used
    band: str
    alpha: float
    pair: str = "key"            # which of the target's pairs is written (and whose outcome labels apply)
    clamp: str | None = None     # family whose pair coordinates are held at their clean values
    scale: float = 1.0           # matched, null: edit norms = scale x the reference arm's recorded norms


@dataclass
class Target:
    item_id: str
    ids: list[int]
    start: int                   # first edited position (the window runs to the last token)
    pairs: dict                  # pair name -> (row written away from, row written towards) in every family
    outcomes: dict = field(default_factory=dict)      # brew: pair name -> {label: colour}
    tokens: dict = field(default_factory=dict)        # factual: answer / swap_answer token ids
    cf_ids: list[int] | None = None                   # patch arms: the counterfactual prompt (same length as ids)


def noise_vectors(targets: list[Target], arm: Arm, layers, d: int) -> dict[int, torch.Tensor]:
    """A fixed random unit direction per (target, arm, layer)."""
    out = {}
    for l in layers:
        rows = []
        for t in targets:
            seed = int(hashlib.sha256(f"exp17:{t.item_id}:{arm.name}:{l}".encode()).hexdigest()[:8], 16)
            rows.append(torch.randn(d, generator=torch.Generator().manual_seed(seed)))
        out[l] = random_unit(torch.stack(rows), l).cuda()
    return out


def edit_spec(arm: Arm, targets: list[Target], families: dict, layers, norms: dict) -> SW.EditSpec:
    """norms: arm name -> the edit norms recorded in that arm's forward on this batch."""
    def pairs(family):
        if family is None:
            return None
        return {l: torch.stack([SW.pair(families[family][l], *t.pairs[arm.pair]) for t in targets]) for l in layers}
    ref_norms = None
    if arm.ref_arm is not None:
        ref_norms = {l: n * arm.scale for l, n in norms[arm.ref_arm].items()}
    if arm.kind == "patch":
        return SW.EditSpec("patch", arm.alpha, layers, basis=patch_bases(arm, targets, families, layers))
    noise = None
    if arm.kind == "null":
        d = next(iter(families.values()))[layers[0]].shape[-1]
        noise = noise_vectors(targets, arm, layers, d)
    return SW.EditSpec(arm.kind, arm.alpha, layers, main=pairs(arm.main), ref_norms=ref_norms, noise=noise,
                       clamp=pairs(arm.clamp))


PATCH_RANK = 10                          # one vector per colour


def patch_bases(arm: Arm, targets: list[Target], families: dict, layers) -> dict[int, torch.Tensor] | None:
    """Per layer [b, PATCH_RANK, d]: an orthonormal basis of the subspace a patch arm copies from the counterfactual
    forward. arm.main: the family whose 10 colour vectors span it; 'random', a fixed random subspace per (target,
    arm, layer); None, the whole residual (no basis)."""
    if arm.main is None:
        return None
    if arm.main != "random":
        return {l: SW.orthonormal_rows(families[arm.main][l]).expand(len(targets), -1, -1) for l in layers}
    d = families["jpp"][layers[0]].shape[-1]
    out = {}
    for l in layers:
        rows = []
        for t in targets:
            seed = int(hashlib.sha256(f"exp17:{t.item_id}:{arm.name}:{l}".encode()).hexdigest()[:8], 16)
            gauss = torch.randn(PATCH_RANK, d, generator=torch.Generator().manual_seed(seed)).cuda()
            rows.append(SW.orthonormal_rows(random_unit(gauss, l)))
        out[l] = torch.stack(rows)
    return out


def trials_path(phase: str) -> Path:
    return CACHE / f"trials_{phase}.jsonl"


@torch.no_grad()
def run_arms(phase: str, targets: list[Target], arms: list[Arm], families: dict, jpp_layers, score,
             batch: int = 24) -> None:
    """For each batch of targets: one clean forward (recording the block outputs at the edit window), with patch
    arms one forward of the counterfactual prompts (recorded the same way), then one forward per arm. Appends one
    json line per (target, arm), plus one per target for each recorded pass (arms 'clean', 'cf_clean'), batch by
    batch; targets already complete are skipped, so a killed run resumes."""
    path = trials_path(phase)
    needs_cf = any(a.kind == "patch" for a in arms)
    done = collections.Counter()
    if path.exists():
        for line in path.read_text().splitlines():
            done[json.loads(line)["item_id"]] += 1
    targets = [t for t in targets if done[t.item_id] < len(arms) + 1 + needs_cf]
    print(f"{phase}: {len(targets)} targets x {len(arms)} arms to run", flush=True)
    if not targets:
        return
    names = [a.name for a in arms]
    for a in arms:
        if a.ref_arm is not None:
            ref = arms[names.index(a.ref_arm)]
            assert names.index(a.ref_arm) < names.index(a.name) and ref.band == a.band, (a.name, a.ref_arm)
    _, model = load_model()
    layers = tuple(sorted({l for a in arms for l in band_layers(a.band, jpp_layers)}))
    edit = SW.BandEdit(model.model.layers, layers)
    n_done = 0
    for batch_idx in equal_length_batches([len(t.ids) for t in targets], batch):
        part = [targets[i] for i in batch_idx]
        ids, mask, ends = stack_batch([t.ids for t in part])
        idx, wmask = SW.window_index([t.start for t in part], [len(t.ids) - 1 for t in part])
        edit.window(idx.cuda(), wmask.cuda())
        rows_b = torch.arange(len(part), device="cuda")

        def forward(x=ids):
            hs = model.model(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state
            return model.lm_head(hs[rows_b, ends]).float()

        edit.record()
        clean = forward()
        edit.clear()
        records = [{"item_id": t.item_id, "arm": "clean", **score(t, clean[k], "key")} for k, t in enumerate(part)]
        if needs_cf:
            cf_ids, _, _ = stack_batch([t.cf_ids for t in part])
            edit.record("counterfactual")
            cf = forward(cf_ids)
            edit.clear()
            records += [{"item_id": t.item_id, "arm": "cf_clean", **score(t, cf[k], "key")}
                        for k, t in enumerate(part)]
        norms = {}
        for arm in arms:
            edit.apply(edit_spec(arm, part, families, band_layers(arm.band, jpp_layers), norms))
            logits = forward()
            dose = edit.dose.tolist()
            norms[arm.name] = edit.norms
            edit.clear()
            records += [{"item_id": t.item_id, "arm": arm.name, "dose": round(dose[k], 5), **score(t, logits[k], arm.pair)}
                        for k, t in enumerate(part)]
        with path.open("a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in records))
        n_done += len(part)
        print(f"{phase}: {n_done}/{len(targets)} targets", flush=True)
    edit.remove()


def load_trials(phase: str) -> list[dict]:
    return [json.loads(l) for l in trials_path(phase).read_text().splitlines()]


# --- brew targets and scoring ---------------------------------------------------------------------------------------
def brew_targets(bank: str) -> tuple[list[Target], list[str]]:
    """Correct h=2 items with exp15's swap targets (exp15_edits.swap_targets, same seeds). Pairs: 'key' = (s1, s1'),
    'answer' = (gold, a'), 'other' = an unrelated colour pair off the item's path. Outcome labels: key and other as
    exp15 (success T(s1'), start_swap T^2(s1'), echo s1', unchanged gold); answer: success a' (said), relookup T(a')
    (looked up again), unchanged gold."""
    colours, table, metas = item_table(bank)
    items = bank_items(bank)
    _, prepared = prepared_rows(bank)
    ids_of = {row["item_id"]: ids for row, ids, _ in prepared}
    out = []
    for it, m in zip(table, metas):
        if it.h != 2 or it.group != "correct":
            continue
        brew = B.parse(items[it.item_id]["problem"])
        t = ED.swap_targets(it, colours, brew)
        if t is None:
            continue
        s1, s1_new, a_new = it.path[1], t["s1_new"], t["a_new"]
        oc, on = t["other"]
        ids = ids_of[it.item_id]
        assert len(ids) == m["n_tokens"]
        key_out = {"success": S.step(brew, s1_new), "start_swap": S.step(brew, s1_new, 2), "echo": s1_new,
                   "unchanged": it.gold}
        out.append(Target(it.item_id, ids, m["start"],
                          pairs={"key": (colours.index(s1), colours.index(s1_new)),
                                 "answer": (colours.index(it.gold), colours.index(a_new)),
                                 "other": (colours.index(oc), colours.index(on))},
                          outcomes={"key": key_out, "other": key_out,
                                    "answer": {"success": a_new, "relookup": S.step(brew, a_new),
                                               "unchanged": it.gold}}))
    return out, colours


def brew_scorer(colours):
    tokenizer = load_tokenizer()
    cids = torch.tensor(E15.colour_ids(tokenizer, colours), device="cuda")

    def score(t: Target, logits: torch.Tensor, pair: str) -> dict:
        cp = torch.softmax(logits[cids], -1)
        k = int(cp.argmax())
        answer = colours[k]
        outcome = next((label for label, c in t.outcomes[pair].items() if c == answer), "other")
        return {"answer": answer, "p_answer": round(float(cp[k]), 4), "outcome": outcome,
                "colour_mass": round(float(torch.softmax(logits, -1)[cids].sum()), 5)}
    return score


def families_on_cuda(extra_jpp_layers=()) -> dict:
    """prep's colour vectors on the GPU; the J++ colour vectors also at `extra_jpp_layers` (control bands beyond the
    dev bands), computed as prep computes them."""
    blob = torch.load(directions_path(), weights_only=False)
    families = {f: {l: v.float().cuda() for l, v in per.items()} for f, per in blob["families"].items()}
    missing = [l for l in extra_jpp_layers if l not in families["jpp"]]
    if missing:
        jac = jpp_jacobians()
        gain, w_u = gain_and_unembed()
        cids = torch.tensor(E15.colour_ids(load_tokenizer(), blob["colours"]))
        for l in missing:
            families["jpp"][l] = jpp_vectors(jac, l, gain, w_u[cids].to("cuda")).float()
    return families


# --- positive control: the paper's two-hop factual prompts ----------------------------------------------------------
def posctl_arms() -> list[Arm]:
    arms = []
    for band, (lo, hi) in SPEC.bands.items():
        if lo == hi:
            continue
        for a in POSCTL_ALPHAS:
            inter = f"inter|{band}|{a:g}"
            arms += [Arm(inter, "swap", "tok", None, band, a, pair="inter"),
                     Arm(f"answer|{band}|{a:g}", "swap", "tok", None, band, a, pair="answer"),
                     Arm(f"null|{band}|{a:g}", "null", None, inter, band, a, pair="inter")]
    return arms


def answer_forms(tokenizer, word: str) -> set[int]:
    """First tokens a chat reply may start the word with: bare, capitalised, after a space."""
    word = word.strip()
    return {tokenizer.encode(w, add_special_tokens=False)[0] for w in (word, word.capitalize(), " " + word)}


@torch.no_grad()
def posctl() -> None:
    """The paper's 90 two-hop factual prompts: the clamped J++ coordinate swap of the intermediate (intermediate ->
    swap_to), of the answer (answer -> swap_answer: the paper's check that the effect is not just writing the
    answer), and a random direction at the intermediate swap's size, at every position after BOS, per band and alpha.
    Swapped words are their first token after a space. Success: the next-token argmax (full vocabulary) is a first
    token of swap_answer; analysed on items whose clean argmax is a first token of the answer.
    Prompt format (POSCTL_CHAT): raw text, the next token after a space (the paper's); or, for Gemma-it, which does
    not continue raw text (run "gemma": 0 of 76 answered), the chat template with POSCTL_INSTRUCTION, the first reply
    token bare, capitalised or after a space."""
    if not PROBE_SWAP.exists():
        raise SystemExit(f"copy probe-swap.json ({PROBE_SWAP_SOURCE}) to {PROBE_SWAP}")
    assert sha256(PROBE_SWAP) == PROBE_SWAP_SOURCE["sha256"]
    tokenizer = load_tokenizer()
    items = json.loads(PROBE_SWAP.read_text())["items"]
    vocab, targets, multi = [], [], {}

    def row(token: int) -> int:
        if token not in vocab:
            vocab.append(token)
        return vocab.index(token)

    for it in items:
        toks = {k: tokenizer.encode(" " + it[k].strip(), add_special_tokens=False)
                for k in ("intermediate", "swap_to", "answer", "swap_answer")}
        head = {k: v[0] for k, v in toks.items()}
        if head["intermediate"] == head["swap_to"] or head["answer"] == head["swap_answer"]:
            continue
        if POSCTL_CHAT:
            forms = {k: answer_forms(tokenizer, it[k]) for k in ("answer", "swap_answer")}
            if forms["answer"] & forms["swap_answer"]:
                continue
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": POSCTL_INSTRUCTION + it["prompt"].rstrip(" ")}],
                add_generation_prompt=True, tokenize=True)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        else:
            forms = {k: {head[k]} for k in ("answer", "swap_answer")}
            ids = tokenizer(it["prompt"].rstrip(" "), add_special_tokens=True)["input_ids"]
        start = 1 if ids[0] in tokenizer.all_special_ids else 0
        multi[it["name"]] = [k for k, v in toks.items() if len(v) > 1]
        targets.append(Target(it["name"], ids, start,
                              pairs={"inter": (row(head["intermediate"]), row(head["swap_to"])),
                                     "answer": (row(head["answer"]), row(head["swap_answer"]))},
                              tokens=forms))
    jac = jpp_jacobians()
    gain, w_u = gain_and_unembed()
    w_vocab = w_u[torch.tensor(vocab)].to("cuda")
    families = {"tok": {l: jpp_vectors(jac, l, gain, w_vocab).float() for l in all_band_layers(jac)}}

    def score(t: Target, logits: torch.Tensor, pair: str) -> dict:
        top = int(logits.argmax())
        return {"top": top, "success": top in t.tokens["swap_answer"], "kept": top in t.tokens["answer"]}

    meta = RESULTS / "posctl_items.json"
    if not meta.exists():
        write_once(meta, {"source": PROBE_SWAP_SOURCE, "n_items": len(items), "n_usable": len(targets),
                          "chat_template": POSCTL_CHAT, "instruction": POSCTL_INSTRUCTION if POSCTL_CHAT else None,
                          "multi_token_words": {k: v for k, v in multi.items() if v}})
    run_arms("posctl", targets, posctl_arms(), families, jac, score, batch=30)


# --- dev grid, selection, test --------------------------------------------------------------------------------------
def alphas(band: str) -> tuple[float, ...]:
    lo, hi = SPEC.bands[band]
    return SINGLE_LAYER_ALPHAS if lo == hi else BAND_ALPHAS


def dev_arms() -> list[Arm]:
    """Every band x alpha for the raw J++ key swap and the full difference-of-means key swap, each with a
    random-direction null at its own edit size; the NLA reconstructor's swap and null at its layer (Qwen); the J++
    answer swap (gold -> a') at the two smallest alphas per band; one alpha-0 J++ arm (no edit, no change)."""
    arms = [Arm("zero", "swap", "jpp", None, next(iter(SPEC.bands)), 0.0)]
    for band in SPEC.bands:
        fams = [("jpp", "jpp"), ("dm", "dm_full")] + ([("ar", "ar_full")] if band == SPEC.ar_band else [])
        for a in alphas(band):
            for fam, main in fams:
                swap = f"{fam}|{band}|{a:g}"
                arms += [Arm(swap, "swap", main, None, band, a),
                         Arm(f"null_{fam}|{band}|{a:g}", "null", None, swap, band, a)]
        for a in alphas(band)[:len(ANSWER_ALPHAS)]:
            arms.append(Arm(f"jpp_answer|{band}|{a:g}", "swap", "jpp", None, band, a, pair="answer"))
    return arms


def dev() -> None:
    targets, colours = brew_targets("dev")
    run_arms("dev", targets, dev_arms(), families_on_cuda(), jpp_jacobians(), brew_scorer(colours))


def eligible(trials: list[dict]) -> set[str]:
    """Targets whose clean pass inside the trial batch also gives the gold answer."""
    return {r["item_id"] for r in trials if r["arm"] == "clean" and r["outcome"] == "unchanged"}


def rates(trials: list[dict], keep: set[str]) -> dict[str, dict]:
    by = collections.defaultdict(list)
    for r in trials:
        if r["item_id"] in keep and r["arm"] != "clean":
            by[r["arm"]].append(r)
    out = {}
    for arm, rs in by.items():
        n = len(rs)
        c = collections.Counter(r["outcome"] for r in rs)
        out[arm] = {"n": n, **{k: round(v / n, 4) for k, v in sorted(c.items())},
                    "dose": round(sum(r["dose"] for r in rs) / n, 4),
                    "colour_mass": round(sum(r["colour_mass"] for r in rs) / n, 4)}
    return out


def select() -> None:
    """Each family's (band, alpha): the largest dev key-swap success minus its own null's success; within 0.01 of the
    best, the smaller alpha, then the band with fewer layers, then the earlier band (the manifest's rule)."""
    trials = load_trials("dev")
    keep = eligible(trials)
    r = rates(trials, keep)
    jac_layers = set(jpp_jacobians())
    chosen = {}
    for fam in ("jpp", "dm", "ar"):
        cands = []
        for arm, v in r.items():
            if not arm.startswith(f"{fam}|"):
                continue
            _, band, a = arm.split("|")
            null = r[f"null_{fam}|{band}|{a}"]
            cands.append({"band": band, "alpha": float(a), "success": v.get("success", 0.0),
                          "null": null.get("success", 0.0), "gain": round(v.get("success", 0.0)
                                                                          - null.get("success", 0.0), 4),
                          "echo": v.get("echo", 0.0), "dose": v["dose"],
                          "n_layers": len(band_layers(band, jac_layers)), "first": SPEC.bands[band][0]})
        if not cands:
            continue
        best = max(c["gain"] for c in cands)
        near = [c for c in cands if c["gain"] >= best - 0.01]
        chosen[fam] = {**min(near, key=lambda c: (c["alpha"], c["n_layers"], c["first"])),
                       "grid": sorted(cands, key=lambda c: -c["gain"])}
    write_once(RESULTS / "dev_selection.json",
               {"n_eligible": len(keep), "zero_arm": r.get("zero"), "chosen": chosen,
                "answer_swaps": {k: v for k, v in r.items() if k.startswith("jpp_answer|")}})
    print(json.dumps({f: {k: v for k, v in c.items() if k != "grid"} for f, c in chosen.items()}, indent=1))


DOSE_SCALES = (2.0, 4.0)


def test_arms(lock: dict) -> tuple[list[Arm], list[Arm]]:
    """(post-table arms, all-position arms) at the locked settings: the J-part and remainder at the full direction's
    edit size (primary) and at their own coordinate-swap size (the paper's protocol); the raw J++ colour swap at the
    full direction's edit size (jpp_matched: same band, layers, positions and norms as dm_full, J++ direction); the
    J-part and jpp_matched at 2x and 4x that size, each with a null of the same size (dose-response); and, when locked,
    the in-task control: the J++ answer swap (gold -> a') at a later band, with its null."""
    j, d = lock["jpp"], lock["dm"]
    jb, ja, db, da = j["band"], j["alpha"], d["band"], d["alpha"]
    arms = [Arm("jpp_raw", "swap", "jpp", None, jb, ja),
            Arm("null_jpp", "null", None, "jpp_raw", jb, ja),
            Arm("jpp_answer", "swap", "jpp", None, jb, ja, pair="answer"),
            Arm("dm_full", "swap", "dm_full", None, db, da),
            Arm("jpp_matched", "matched", "jpp", "dm_full", db, da),
            Arm("dm_jpart", "matched", "dm_jpart", "dm_full", db, da),
            Arm("dm_rem", "matched", "dm_rem", "dm_full", db, da),
            Arm("dm_jpart_own", "swap", "dm_jpart", None, db, da),
            Arm("dm_rem_own", "swap", "dm_rem", None, db, da),
            Arm("dm_rem_clamp", "matched", "dm_rem", "dm_full", db, da, clamp="jpp"),
            Arm("null_dm", "null", None, "dm_full", db, da),
            Arm("other_item", "swap", "dm_full", None, db, da, pair="other")]
    for s in DOSE_SCALES:
        arms += [Arm(f"dm_jpart_x{s:g}", "matched", "dm_jpart", "dm_full", db, da, scale=s),
                 Arm(f"jpp_matched_x{s:g}", "matched", "jpp", "dm_full", db, da, scale=s),
                 Arm(f"null_dm_x{s:g}", "null", None, "dm_full", db, da, scale=s)]
    if "answer_control" in lock:
        cb, ca = lock["answer_control"]["band"], lock["answer_control"]["alpha"]
        arms += [Arm("jpp_answer_ctl", "swap", "jpp", None, cb, ca, pair="answer"),
                 Arm("null_answer_ctl", "null", None, "jpp_answer_ctl", cb, ca, pair="answer")]
    if "ar" in lock:
        ab, aa = lock["ar"]["band"], lock["ar"]["alpha"]
        arms += [Arm("ar_full", "swap", "ar_full", None, ab, aa),
                 Arm("ar_jpart", "matched", "ar_jpart", "ar_full", ab, aa),
                 Arm("ar_rem", "matched", "ar_rem", "ar_full", ab, aa),
                 Arm("null_ar", "null", None, "ar_full", ab, aa)]
    everywhere = [Arm("jpp_raw_all", "swap", "jpp", None, jb, ja),
                  Arm("dm_full_all", "swap", "dm_full", None, db, da),
                  Arm("null_dm_all", "null", None, "dm_full_all", db, da)]
    return arms, everywhere


def preview() -> None:
    """The post-table test arms on dev's correct h=2 items at this model's dev selection (before the lock): what the
    decomposition arms do at the setting that would be locked. Dev only; nothing here is selected on."""
    chosen = json.loads((RESULTS / "dev_selection.json").read_text())["chosen"]
    targets, colours = brew_targets("dev")
    arms, _ = test_arms(chosen)
    run_arms("preview", targets, arms, families_on_cuda(), jpp_jacobians(), brew_scorer(colours))


def test() -> None:
    lock = locked_settings()[RUN_KEY]
    targets, colours = brew_targets("test")
    jac = jpp_jacobians()
    control = band_layers(lock["answer_control"]["band"], jac) if "answer_control" in lock else ()
    families = families_on_cuda(extra_jpp_layers=control)
    score = brew_scorer(colours)
    arms, everywhere = test_arms(lock)
    run_arms("test", targets, arms, families, jac, score)
    for t in targets:
        t.start = 0
    run_arms("test_all", targets, everywhere, families, jac, score, batch=8)


STAGES = {"check": check, "prep": prep, "posctl": posctl, "dev": dev, "select": select, "preview": preview,
          "bank": write_test_bank, "test": test}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stages", nargs="+", help=f"baseline:dev, baseline:test, {', '.join(STAGES)}")
    a = p.parse_args()
    for stage in a.stages:
        if stage not in STAGES and stage not in ("baseline:dev", "baseline:test"):
            raise SystemExit(f"unknown stage {stage}")
    for path in (MODEL_CACHE, CACHE, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    for stage in a.stages:
        print(f"exp17 {RUN_KEY}: {stage}", flush=True)
        if stage.startswith("baseline:"):
            baseline(stage.split(":")[1])
        else:
            STAGES[stage]()


if __name__ == "__main__":
    main()
