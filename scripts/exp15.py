"""exp15: reading and writing the wrong intermediate in Qwen3.6-27B's no-CoT brew answers
(results/exp15_wrong_keys/manifest.json).

A single-table brew item is a derangement T of ten colours, a start s_0 and a stir count h; gold is s_h = T^h(s_0).
When the model answers a', the key of its last lookup is w = T^-1(a') (unique: T is a bijection). On an off-cycle
error a' is not on s_0's cycle, so neither is w: the prompt alone never produces it.

  check NAME     day-0 instrument checks -> results/exp15_wrong_keys/checks_day0/NAME.json
  bank           the fresh test bank (exp11's generator and format), written once, after the manifest is locked
  requests       the C0 rows of the test bank (exp10's rendering, as exp11)
  capture        one forward per row: the 10 colour logits at the answer position and the residual stream at the
                 read cells, every layer (cache/exp15/capture_<bank>/, chunked, resumable)
  lens_reads     colour and count-word scores per token lens, cell and layer, from the captures (no model)
  trained_reads  cross-fitted key-colour probe and difference-of-means scores
  text_reads     NLA / Oracle lens / AO samples at the text-read cells and layers (resumable)
  vectors        direction vectors per method, layer and colour
  swaps          P3: key swaps on correct h=2 items (dev: selection grid; test: locked settings)
  repair         P4: repair and redirect on off-cycle errors

Reports, the manifest, the matcher audit and the external check live in exp15_report.py; run_exp15.sh chains the
stages above.

Layer l is the output of decoder block l (hidden_states[l + 1]): the lenses', NLA's and the Oracle lenses' numbering.
Run with /venv/main/bin/python from scripts/; HF_HOME must point at the model cache.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch

import cc_config as cfg
import exp11 as X
import exp11_brew as B

# The kernel configuration of every exp15 forward (checks_day0/throughput_final.json): flash-linear-attention 0.5.2
# for the DeltaNet layers, and cuDNN attention off (its per-call graph build cost 15 ms of CPU per full-attention
# layer per decode step; SDPA falls back to the flash / memory-efficient kernels).
torch.backends.cuda.enable_cudnn_sdp(False)

EXP = cfg.Exp("exp15_wrong_keys", "exp15")
MODEL = "Qwen3.6-27B"
FAMILY = "qwen3.6"
N_LAYERS = 64
DEV_BANK = "single_test"            # exp11's single-table test bank: seen in exp11, exp15's dev set
TEST_BANK = "test"
# The plan's 300 / 1000 at h = 2 / 3 was enlarged before the lock (kill rule K2: at the p >= 0.3 gate, dev projects
# 26 + 136 consistent off-cycle errors at these sizes).
TEST_SPEC = {"seed": "exp15_test_20261012", "format": "single_first", "per_h": {1: 100, 2: 600, 3: 2000}}
BANKS = (DEV_BANK, TEST_BANK)

# read cells of the final problem, in storage order: three rule-line ends (before the start colour is given: the
# key cannot be computed there), the start and count sentence, and the twelve tokens from '?' to the prefilled ':'
RULE_CELLS = ("rule3", "rule6", "rule9")
HEAD_CELLS = ("start", "start_period", "count", "count_period")
TAIL_CELLS = ("question_end", "im_end", "nl_user", "im_start", "assistant", "nl_asst", "think_open", "nl_think",
              "think_close", "nl_post_think", "answer_word", "final")
TAIL_TOKENS = ("?", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n",
               "Answer", ":")
CELLS = RULE_CELLS + HEAD_CELLS + TAIL_CELLS
CAPTURE_CHUNK = 50


def require_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; written once")
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_save(obj, path: Path) -> None:
    """torch.save to <path>.partial, then rename: a crash never leaves a truncated file under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(obj, tmp)
    tmp.rename(path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text)
    tmp.rename(path)


# --- banks and requests ---------------------------------------------------------------------------------------------
def bank_path(bank: str) -> Path:
    return X.bank_path(bank) if bank == DEV_BANK else EXP.cache / f"items_{bank}.jsonl"


def requests_path(bank: str) -> Path:
    return X.requests_path(bank) if bank == DEV_BANK else EXP.cache / f"requests_{bank}.jsonl"


def load_items(bank: str) -> dict[str, dict]:
    return {r["item_id"]: r for r in map(json.loads, bank_path(bank).open())}


def load_requests(bank: str) -> list[dict]:
    return [json.loads(l) for l in requests_path(bank).open()]


def write_test_bank() -> None:
    """exp11's write_fmt_bank with exp15's seed and sizes: one h=2 shot plus per_h eval items, single_first format,
    every gold re-derived by the text parser, no problem text shared with any exp11 or exp15 bank."""
    import cc_exp10 as E10
    if not EXP.manifest.exists() or json.loads(EXP.manifest.read_text()).get("status") != "locked":
        raise SystemExit("the test bank is generated only after the manifest is locked")
    path = bank_path(TEST_BANK)
    require_absent(path)
    brew, _ = E10.nocot_import()
    fmt = TEST_SPEC["format"]
    rng = random.Random(f"{TEST_SPEC['seed']}:{fmt}")
    cells = [("shot", 2, 0)] + [("eval", h, k) for h, n in TEST_SPEC["per_h"].items() for k in range(n)]
    rows = []
    for split, h, k in cells:
        gen = B.generate(rng, h, B.FORMAT_SPEC[fmt][0], brew.COLORS, brew.INGREDIENTS)
        problem = B.render(gen, fmt, brew.COLORS)
        assert B.parse(problem).path()[-1] == gen["answer"]
        rows.append({"item_id": f"exp15_{TEST_BANK}:{fmt}:{split}:h{h}:{k}", "bank": f"exp15_{TEST_BANK}",
                     "format": fmt, "split": split, "h": h, "problem": problem, "answer": gen["answer"],
                     "instruction": brew.INSTRUCTION})
    texts = collections.Counter(r["problem"] for r in rows)
    texts.update(r["problem"] for b in X.ALL_BANKS if X.bank_path(b).exists() for r in X.load_bank(b))
    if max(texts.values()) > 1:
        raise SystemExit("a problem text repeats across exp11's and exp15's banks")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = {"bank": TEST_BANK, **TEST_SPEC, "n_eval": sum(r["split"] == "eval" for r in rows),
              "content_key": cfg.content_key(rows)}
    atomic_write_text(json.dumps(record, indent=2) + "\n", EXP.results / "test_bank.json")
    print(f"{len(rows)} rows -> {path}")


def write_test_requests() -> None:
    import exp10_conditions as C
    path = requests_path(TEST_BANK)
    require_absent(path)
    items = list(load_items(TEST_BANK).values())
    shot = next(r for r in items if r["split"] == "shot")
    rows = []
    for it in (r for r in items if r["split"] == "eval"):
        row = {"item_id": it["item_id"], "bank": f"exp15_{TEST_BANK}", "h": it["h"], "source": "brew",
               "exp10_render": True, "seed": 0, "format": it["format"], **C.fields("C0", it, shot)}
        row["request_id"] = cfg.content_key(row)
        rows.append(row)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {path}")


# --- prompt and read cells ------------------------------------------------------------------------------------------
def prompt(tokenizer, row: dict) -> tuple[str, list[int]]:
    """exp10's rendering (as exp11): the chat text up to the assistant turn, and its ids with 'Answer:' prefilled."""
    import cc_exp10_render as R
    text = R.prompt_text(tokenizer, FAMILY, row)
    ids = R.prompt_ids(tokenizer, FAMILY, row)
    n_text = len(tokenizer.encode(text, add_special_tokens=False))
    if tokenizer.encode(text, add_special_tokens=False) != ids[:n_text]:
        raise RuntimeError(f"{row['item_id']}: the rendered text does not tokenize to the prompt's prefix")
    return text, ids


def read_cells(tokenizer, text: str, ids: list[int], brew: B.Brew) -> dict[str, int]:
    """Token index of every read cell (CELLS order) in the final problem of the prompt."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def token_ending_at(char_end: int) -> int:
        return next(i for i, (a, b) in enumerate(offsets) if a < char_end <= b)

    problem = text.rindex("Problem: ")
    rules_start = text.index("The rules:\n", problem) + len("The rules:\n")
    rule_lines = text[rules_start:].split("\n")[:len(brew.table)]
    out = {}
    cursor = rules_start
    for k, line in enumerate(rule_lines, 1):
        assert B.RULE_SINGLE.match(line), line
        if f"rule{k}" in RULE_CELLS:
            out[f"rule{k}"] = token_ending_at(cursor + len(line))
        cursor += len(line) + 1
    named = X.token_positions(tokenizer, text, brew)
    out["start"] = named["start"]
    out["start_period"] = token_ending_at(text.index(f"starts out {brew.start}.", problem)
                                          + len(f"starts out {brew.start}."))
    out["count"] = named["stirs"]
    count = B.COUNT_WORDS[len(brew.stirs)]
    out["count_period"] = token_ending_at(text.index(f"You stir it {count}.", problem) + len(f"You stir it {count}."))
    tail = list(range(len(ids) - len(TAIL_CELLS), len(ids)))
    pieces = tuple(tokenizer.decode(ids[i]) for i in tail)
    if pieces != TAIL_TOKENS:
        raise RuntimeError(f"unexpected prompt tail {pieces}")
    assert tail[0] == named["question_end"], (tail[0], named["question_end"])
    out.update(zip(TAIL_CELLS, tail))
    return {c: out[c] for c in CELLS}


# --- model ----------------------------------------------------------------------------------------------------------
def model_path() -> str:
    return str(cfg.model_dir(MODEL))


def load_base(device_map="cuda"):
    """Qwen3.6-27B as a text CausalLM (the LoRA readers' module paths: model.layers.N)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path())
    model = AutoModelForCausalLM.from_pretrained(model_path(), dtype=torch.bfloat16, device_map=device_map).eval()
    return tokenizer, model


def colour_ids(tokenizer, colours) -> list[int]:
    ids = []
    for c in colours:
        toks = tokenizer.encode(" " + c, add_special_tokens=False)
        assert len(toks) == 1, f"{c!r} is {len(toks)} tokens"
        ids.append(toks[0])
    return ids


# --- capture --------------------------------------------------------------------------------------------------------
def capture_dir(bank: str) -> Path:
    return EXP.cache / f"capture_{bank}"


@torch.no_grad()
def capture(bank: str) -> None:
    """Per chunk of CAPTURE_CHUNK rows: hidden [n, n_cells, 65, d] bf16 and one metadata row per item. Existing
    chunks are skipped, so a killed run resumes."""
    tokenizer, model = load_base()
    items = load_items(bank)
    rows = load_requests(bank)
    colours = sorted(B.parse(items[rows[0]["item_id"]]["problem"]).table)
    cids = colour_ids(tokenizer, colours)
    out_dir = capture_dir(bank)
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), CAPTURE_CHUNK):
        path = out_dir / f"chunk_{start // CAPTURE_CHUNK:03d}.pt"
        if path.exists():
            continue
        hidden, metas = [], []
        for row in rows[start:start + CAPTURE_CHUNK]:
            brew = B.parse(items[row["item_id"]]["problem"])
            text, ids = prompt(tokenizer, row)
            cells = read_cells(tokenizer, text, ids, brew)
            out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False)
            logits = out.logits[0, -1].float()
            hidden.append(torch.stack([h[0, list(cells.values())] for h in out.hidden_states], 1).cpu())
            metas.append({"item_id": row["item_id"], "h": row["h"], "request_id": row["request_id"],
                          "n_tokens": len(ids), "cells": cells, "colours": colours,
                          "colour_logits": [round(float(x), 4) for x in logits[cids]],
                          "colour_mass": round(float(torch.softmax(logits, -1)[cids].sum()), 6),
                          "top_token": tokenizer.decode(int(logits.argmax()))})
        atomic_save({"hidden": torch.stack(hidden), "metas": metas, "cells": CELLS}, path)
        print(f"{bank} capture {start + len(metas)}/{len(rows)}", flush=True)


def load_capture(bank: str, with_hidden: bool = True) -> tuple[list[dict], torch.Tensor | None]:
    """All chunks in row order: metas, and hidden [n, n_cells, 65, d] (bf16) if asked."""
    chunks = sorted(capture_dir(bank).glob("chunk_*.pt"))
    metas, hidden = [], []
    for path in chunks:
        blob = torch.load(path, weights_only=False, mmap=True)
        assert tuple(blob["cells"]) == CELLS
        metas += blob["metas"]
        if with_hidden:
            hidden.append(blob["hidden"])
    return metas, (torch.cat(hidden) if with_hidden else None)


# --- lens reads -----------------------------------------------------------------------------------------------------
def lens_reads_path(bank: str) -> Path:
    return EXP.cache / f"lens_reads_{bank}.pt"


@torch.no_grad()
def lens_reads(bank: str) -> None:
    """Per item, cell and layer 0..62: the 10 colour and 6 count-word scores of the logit lens, J-lens, R-lens, J++
    (readout logits) and the template lens (cosines). Saved fp16 [n, n_cells, 63, 16] per lens."""
    import exp15_readers as RD
    from transformers import AutoTokenizer
    path = lens_reads_path(bank)
    require_absent(path)
    tokenizer = AutoTokenizer.from_pretrained(model_path())
    metas, _ = load_capture(bank, with_hidden=False)
    colours = metas[0]["colours"]
    count_ids = [tokenizer.encode(" " + w, add_special_tokens=False) for w in RD.COUNT_WORDS]
    assert all(len(t) == 1 for t in count_ids)
    lenses = RD.TokenLenses(colour_ids(tokenizer, colours) + [t[0] for t in count_ids])
    template = RD.TemplateLens(colours + list(RD.COUNT_WORDS))
    scores = {k: [] for k in (*RD.TOKEN_LENSES, "template")}
    item_ids = []
    for chunk in sorted(capture_dir(bank).glob("chunk_*.pt")):
        blob = torch.load(chunk, weights_only=False)
        for m, hidden in zip(blob["metas"], blob["hidden"]):
            for k, v in lenses.scores(hidden).items():
                scores[k].append(v.half())
            scores["template"].append(template.scores(hidden).half())
            item_ids.append(m["item_id"])
        print(f"lens_reads {bank} {len(item_ids)}/{len(metas)}", flush=True)
    atomic_save({"item_ids": item_ids, "cells": CELLS, "layers": RD.LENS_LAYERS,
                 "labels": colours + [f"count{k}" for k in range(1, 7)],
                 "scores": {k: torch.stack(v) for k, v in scores.items()}}, path)
    print(f"-> {path}")


def load_lens_reads(bank: str) -> dict:
    return torch.load(lens_reads_path(bank), weights_only=False)


# --- trained readers: probe and difference of means -------------------------------------------------------------------
PRIMARY_CELLS = ("question_end", "nl_asst", "final")
PRIMARY_LAYERS = tuple(range(36, 49))           # token lenses, probe, diffmeans
AO_PRIMARY_LAYERS = (36, 40, 44, 48)
AO_EXPLORATORY_LAYERS = (20, 24, 28, 32, 52, 56, 60)


def probe_reads_path(bank: str) -> Path:
    return EXP.cache / f"probe_reads_{bank}.pt"


def cell_layer_hidden(bank: str, cells=PRIMARY_CELLS, layers=PRIMARY_LAYERS) -> torch.Tensor:
    """hidden at (cells, layers) for every captured row: [n, n_cells, n_layers, d] fp32 (layer l = hidden[l + 1])."""
    _, hidden = load_capture(bank)
    ci = [CELLS.index(c) for c in cells]
    li = [l + 1 for l in layers]
    return hidden[:, ci][:, :, li].float()


@torch.no_grad()
def diffmeans_directions(x: torch.Tensor, y: torch.Tensor, n_classes: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """x [L, n, d], y [n] -> (training mean [L, 1, d], unit directions [L, C, d]): mean(x | y=c) - mean(x | y!=c)."""
    mean = x.mean(1, keepdim=True)
    dirs = []
    for c in range(n_classes):
        on = y == c
        dirs.append(x[:, on].mean(1) - x[:, ~on].mean(1))
    dirs = torch.stack(dirs, 1)
    return mean, dirs / dirs.norm(dim=-1, keepdim=True)


def trained_reads(bank: str) -> None:
    """Probe (exp11_probe: PCA-64 + multinomial logistic regression) and difference of means, trained to predict the
    key colour (the true s_{h-1}) on correct h=2 and h=3 items, per primary cell and layer, 5-fold cross-fit
    stratified by (h, key). Training items get held-out scores; every other row gets the mean over the 5 fold
    models. Saved [n, n_cells, n_layers, 10]: probe log-probabilities and diffmeans projections."""
    import exp11_probe as P
    path = probe_reads_path(bank)
    require_absent(path)
    table = item_table(bank)
    metas, _ = load_capture(bank, with_hidden=False)
    colours = metas[0]["colours"]
    train = [i for i, it in enumerate(table) if it.group == "correct" and it.h in (2, 3)]
    y = torch.tensor([colours.index(table[i].key) for i in train], device="cuda")
    folds = P.stratified_folds([f"{table[i].h}:{table[i].key}" for i in train], n_folds=5, seed=0)
    folds_t = torch.tensor(folds, device="cuda")
    x_all = cell_layer_hidden(bank)
    n, n_cells, n_layers, _ = x_all.shape
    out = {k: torch.zeros(n, n_cells, n_layers, 10) for k in ("probe", "diffmeans")}
    train_t = torch.tensor(train)
    train_set = set(train)
    rest = torch.tensor([i for i in range(n) if i not in train_set])
    for ci in range(n_cells):
        x = x_all[:, ci].to("cuda").transpose(0, 1)                      # [L, n, d]
        x_train = x[:, train_t.to("cuda")]
        for f in range(5):
            fit, held = folds_t != f, folds_t == f
            z_fit, z_eval = P._features(x_train[:, fit], torch.cat([x_train[:, held], x[:, rest.to("cuda")]], 1), 64)
            w, b = P._fit(z_fit, y[fit], 10, 1e-2)
            logp = torch.log_softmax(z_eval @ w + b, -1).transpose(0, 1).cpu()      # [n_eval, L, C]
            mean, dirs = diffmeans_directions(x_train[:, fit], y[fit])
            x_eval = torch.cat([x_train[:, held], x[:, rest.to("cuda")]], 1)
            proj = torch.einsum("lnd,lcd->nlc", x_eval - mean, dirs).cpu()
            n_held = int(held.sum())
            held_rows = train_t[held.cpu()]
            out["probe"][held_rows, ci] = logp[:n_held]
            out["diffmeans"][held_rows, ci] = proj[:n_held]
            out["probe"][rest, ci] += logp[n_held:] / 5
            out["diffmeans"][rest, ci] += proj[n_held:] / 5
        print(f"trained_reads {bank} cell {PRIMARY_CELLS[ci]} done", flush=True)
    atomic_save({"item_ids": [m["item_id"] for m in metas], "cells": PRIMARY_CELLS, "layers": PRIMARY_LAYERS,
                 "labels": colours, "train_rows": train, "scores": out}, path)
    print(f"-> {path}")


def load_probe_reads(bank: str) -> dict:
    return torch.load(probe_reads_path(bank), weights_only=False)


# --- text reads -----------------------------------------------------------------------------------------------------
def text_jobs(reader: str) -> list[tuple[str, int]]:
    """(cell, layer) pairs each text reader is run at: primary cells x primary layers, plus (AO only) the
    exploratory layers at the final cell."""
    import exp15_readers as RD
    if reader == "ao":
        return [(c, l) for l in AO_PRIMARY_LAYERS for c in PRIMARY_CELLS] + [("final", l) for l in AO_EXPLORATORY_LAYERS]
    return [(c, l) for l in RD.READERS[reader]["layers"] for c in PRIMARY_CELLS]


TEXT_READ_SPEC = {"dev": "all h=2 and h=3 rows",
                  "test": {"seed": "exp15_text_rows", "correct_per_h": 300, "miscount": 300, "offcycle": "all"}}


def text_rows(bank: str) -> list[int]:
    """Captured rows the text readers run on (TEXT_READ_SPEC): dev = every h=2/h=3 row; test = every off-cycle
    error plus seeded subsamples of correct items (300 per h) and miscounts (300; secondary count target)."""
    table = item_table(bank)
    if bank == DEV_BANK:
        return [i for i, it in enumerate(table) if it.h in (2, 3)]
    spec = TEXT_READ_SPEC["test"]
    rng = random.Random(spec["seed"])
    rows = [i for i, it in enumerate(table) if it.h in (2, 3) and it.group == "offcycle"]
    for h in (2, 3):
        pool = [i for i, it in enumerate(table) if it.h == h and it.group == "correct"]
        rows += rng.sample(pool, min(spec["correct_per_h"], len(pool)))
    pool = [i for i, it in enumerate(table) if it.h in (2, 3) and it.group.startswith("miscount")]
    rows += rng.sample(pool, min(spec["miscount"], len(pool)))
    return sorted(rows)


def text_reads_path(bank: str, reader: str) -> Path:
    return EXP.cache / f"text_reads_{bank}_{reader}.jsonl"


@torch.no_grad()
def text_reads(bank: str, reader_name: str, chunk: int = 48) -> None:
    """Raw samples of one text reader at its text_jobs, for text_rows(bank); one json line per item, appended per
    chunk (resumable: items already in the file are skipped)."""
    import exp15_readers as RD
    path = text_reads_path(bank, reader_name)
    done = set()
    if path.exists():
        done = {json.loads(line)["item_id"] for line in path.read_text().splitlines()}
    metas, hidden = load_capture(bank)
    rows = [i for i in text_rows(bank) if metas[i]["item_id"] not in done]
    print(f"text_reads {bank} {reader_name}: {len(rows)} rows to read ({len(done)} done)", flush=True)
    if not rows:
        return
    reader = RD.TextReader(reader_name)
    jobs = text_jobs(reader_name)
    by_layer = collections.defaultdict(list)
    for cell, layer in jobs:
        by_layer[layer].append(cell)
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        reads = {metas[i]["item_id"]: {} for i in part}
        for layer, cells in by_layer.items():
            vectors = torch.stack([hidden[i, CELLS.index(c), layer + 1] for c in cells for i in part]).float()
            seed = int(cfg.content_key({"bank": bank, "reader": reader_name, "layer": layer, "rows": part}), 16) % 2**31
            samples = reader.read(vectors, layer, seed=seed, batch=384)
            for k, (c, i) in enumerate((c, i) for c in cells for i in part):
                reads[metas[i]["item_id"]][f"{c}|{layer}"] = samples[k]
        with path.open("a") as f:
            for item_id, r in reads.items():
                f.write(json.dumps({"item_id": item_id, "reads": r}) + "\n")
        print(f"text_reads {bank} {reader_name} {start + len(part)}/{len(rows)}", flush=True)


def load_text_reads(bank: str, reader: str) -> dict[str, dict]:
    return {r["item_id"]: r["reads"] for r in map(json.loads, text_reads_path(bank, reader).read_text().splitlines())}


# --- write directions -----------------------------------------------------------------------------------------------
def directions_path() -> Path:
    return EXP.cache / "directions.pt"


@torch.no_grad()
def build_directions(parts=("lens", "diffmeans", "nla_ar", "ptag_ar")) -> None:
    """v_layer(colour) [10, d] per method and write layer (exp15_edits.METHOD_LAYERS), saved to directions.pt.
    Parts already in the file are kept, so the GPU parts can run after the CPU ones."""
    import exp15_edits as ED
    import exp15_readers as RD
    from transformers import AutoTokenizer
    path = directions_path()
    out = torch.load(path, weights_only=False) if path.exists() else {}
    tokenizer = AutoTokenizer.from_pretrained(model_path())
    metas, _ = load_capture(DEV_BANK, with_hidden=False)
    colours = metas[0]["colours"]
    if "lens" in parts and "logit" not in out:
        lenses = RD.TokenLenses(colour_ids(tokenizer, colours), device="cpu")
        for m in ED.LENS_METHODS:
            out[m] = {l: torch.stack([lenses.direction(m, l, k) for k in range(10)]) for l in ED.METHOD_LAYERS[m]}
        template = RD.TemplateLens(colours, device="cpu")
        out["template"] = {l: template.direction(l, slice(0, 10)).clone() for l in ED.METHOD_LAYERS["template"]}
        del lenses, template
    if "diffmeans" in parts and "diffmeans" not in out:
        table = item_table(DEV_BANK)
        train = [i for i, it in enumerate(table) if it.group == "correct" and it.h in (2, 3)]
        y = torch.tensor([colours.index(table[i].key) for i in train])
        layers = ED.METHOD_LAYERS["diffmeans"]
        x = cell_layer_hidden(DEV_BANK, cells=("final",), layers=layers)[train, 0].transpose(0, 1)    # [L, n, d]
        _, dirs = diffmeans_directions(x, y)
        out["diffmeans"] = {l: dirs[k] for k, l in enumerate(layers)}
    if "nla_ar" in parts and "nla_ar" not in out:
        ar = RD.NlaAR()
        pred = ar.predict([t.format(c=c) for c in colours for t in ED.NLA_TEMPLATES]).float()
        out["nla_ar"] = {42: pred.view(10, len(ED.NLA_TEMPLATES), -1).mean(1)}
        del ar
        release()
    if "ptag_ar" in parts and "ptag_ar" not in out:
        ar = RD.PtagAR()
        whiteners = torch.load(whitener_path(), weights_only=False)
        spans = [tokenizer.encode(t.format(c=c), add_special_tokens=False) for c in colours for t in ED.PTAG_TEMPLATES]
        out["ptag_ar"] = {}
        for l in ED.METHOD_LAYERS["ptag_ar"]:
            pred = ar.predict(spans, l).view(10, len(ED.PTAG_TEMPLATES), -1).mean(1)
            w = RD.Whitener(whiteners["mean"][l], whiteners["cov"][l], WHITENER_SPEC["ridge"])
            out["ptag_ar"][l] = w.unwhiten_delta(pred)
        del ar
        release()
    out["colours"] = colours
    atomic_save(out, path.with_name(path.name + ".new"))
    path.with_name(path.name + ".new").replace(path)
    print({k: sorted(v) for k, v in out.items() if isinstance(v, dict)})


# --- swap and repair trials -----------------------------------------------------------------------------------------
def write_lock_path() -> Path:
    """Each method's write setting chosen on dev (written once, before any test-bank write trial)."""
    return EXP.results / "write_lock.json"


def trials_path(bank: str, phase: str) -> Path:
    return EXP.cache / f"trials_{bank}_{phase}.jsonl"


def trial_key(r) -> tuple:
    r = r if isinstance(r, dict) else r.__dict__
    return (r["item_id"], r["variant"], r["method"], r["layer"], r["alpha"])


@torch.no_grad()
def run_trials(bank: str, phase: str, trials: list, batch: int = 32) -> None:
    """Run edit trials (one forward each; answer = argmax over the colour tokens at the final position), appending
    one record per trial to trials_<bank>_<phase>.jsonl. Trials already recorded are skipped."""
    import exp15_edits as ED
    path = trials_path(bank, phase)
    done = set()
    if path.exists():
        done = {trial_key(json.loads(line)) for line in path.read_text().splitlines()}
    trials = [t for t in trials if trial_key(t) not in done]
    print(f"trials {bank} {phase}: {len(trials)} to run ({len(done)} done)", flush=True)
    if not trials:
        return
    tokenizer, model = load_base()
    directions = torch.load(directions_path(), weights_only=False)
    colours = directions["colours"]
    cids = torch.tensor(colour_ids(tokenizer, colours), device="cuda")
    rows = {r["item_id"]: r for r in load_requests(bank)}
    metas = {m["item_id"]: m for m in load_capture(bank, with_hidden=False)[0]}
    ids_cache = {}
    d = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
    by_layer = collections.defaultdict(list)
    for t in trials:
        by_layer[t.layer].append(t)
    n_done = 0
    for layer, group in sorted(by_layer.items()):
        edit = ED.AdditiveEdit(model, layer)
        group.sort(key=lambda t: metas[t.item_id]["n_tokens"])
        for start in range(0, len(group), batch):
            part = group[start:start + batch]
            seqs = []
            for t in part:
                if t.item_id not in ids_cache:
                    ids_cache[t.item_id] = prompt(tokenizer, rows[t.item_id])[1]
                seqs.append(ids_cache[t.item_id])
            width = max(map(len, seqs))
            ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs], device="cuda")
            mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device="cuda")
            end = torch.tensor([len(s) - 1 for s in seqs])
            first = torch.tensor([metas[t.item_id]["cells"]["start"] for t in part])
            u = torch.stack([ED.edit_vector(t, directions, colours, d) for t in part])
            alpha = torch.tensor([t.alpha for t in part])
            edit.batch(u, alpha, first, end)
            try:
                hs = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            finally:
                edit.clear()
            last = hs[torch.arange(len(part)), end.to(hs.device)]
            logits = model.lm_head(last).float()[:, cids]
            probs = torch.softmax(logits, -1)
            with path.open("a") as f:
                for t, p in zip(part, probs):
                    k = int(p.argmax())
                    f.write(json.dumps(ED.trial_record(t, colours[k], float(p[k]))) + "\n")
            n_done += len(part)
        edit.remove()
        print(f"trials {bank} {phase}: layer {layer} done ({n_done}/{len(trials)})", flush=True)


def load_trials(bank: str, phase: str) -> list[dict]:
    return [json.loads(line) for line in trials_path(bank, phase).read_text().splitlines()]


def swap_items(bank: str):
    """Correct h=2 items with their parsed brews (P3's population)."""
    items = load_items(bank)
    table = item_table(bank)
    return [(it, B.parse(items[it.item_id]["problem"])) for it in table if it.h == 2 and it.group == "correct"]


def dev_swap_grid() -> list[tuple[str, int, float]]:
    import exp15_edits as ED
    return [(m, l, a) for m in ED.METHODS for l in ED.METHOD_LAYERS[m] for a in ED.ALPHAS]


def swaps(bank: str) -> None:
    """P3. Dev: every method x layer x alpha (key_swap, answer_swap, random_null) for choosing each method's setting.
    Test: each method at its locked (layer, alpha) from the manifest (all four variants), plus the head-to-head at
    layer 42 at each method's best dev alpha there."""
    import exp15_edits as ED
    colours = torch.load(directions_path(), weights_only=False)["colours"]
    if bank == DEV_BANK:
        grid, variants, phase = dev_swap_grid(), ("key_swap", "answer_swap", "random_null"), "select"
    else:
        locked = json.loads(write_lock_path().read_text())
        grid = sorted({(m, s["layer"], s["alpha"]) for m, s in locked["primary"].items()} |
                      {(m, 42, s["alpha"]) for m, s in locked["layer42"].items()})
        variants, phase = ("key_swap", "answer_swap", "random_null", "other_item"), "locked"
    trials = [t for it, brew in swap_items(bank) for t in ED.swap_trials(it, colours, brew, grid, variants)]
    run_trials(bank, phase, trials)


def repair(bank: str) -> None:
    """P4 on consistent off-cycle h=3 errors, for methods that passed the dev key-swap threshold, at their locked
    settings (repair, redirect, random null). On dev the same trials run at the locked settings as a check."""
    import exp15_edits as ED
    import exp15_score as S
    colours = torch.load(directions_path(), weights_only=False)["colours"]
    locked = json.loads(write_lock_path().read_text())
    grid = [(m, s["layer"], s["alpha"]) for m, s in locked["primary"].items() if m in locked["repair_methods"]]
    items = load_items(bank)
    targets = [it for it in item_table(bank) if it.h == 3 and it.group == "offcycle" and it.p_pred >= S.CONSISTENT_P]
    trials = [t for it in targets for t in ED.repair_trials(it, colours, B.parse(items[it.item_id]["problem"]), grid)]
    run_trials(bank, "repair", trials)


# --- day-0 checks ---------------------------------------------------------------------------------------------------
def check_dir() -> Path:
    return EXP.results / "checks_day0"


def write_check(name: str, record: dict) -> None:
    path = check_dir() / f"{name}.json"
    require_absent(path)
    atomic_write_text(json.dumps(record, indent=2) + "\n", path)
    print(json.dumps(record, indent=2))


@torch.no_grad()
def check_throughput(name: str, batches=(16, 64)) -> None:
    """Forward time of a dev prompt and sampled-generation tokens/s per batch size, with the kernels installed now
    (the decision on flash-linear-attention is made from this, before any capture)."""
    import importlib.util
    tokenizer, model = load_base()
    row = load_requests(DEV_BANK)[0]
    _, ids = prompt(tokenizer, row)
    x = torch.tensor([ids], device="cuda")
    model(input_ids=x, use_cache=False)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(5):
        model(input_ids=x, output_hidden_states=True, use_cache=False)
    torch.cuda.synchronize()
    record = {"kernels": {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "kernels")},
              "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(), "torch": torch.__version__, "forward_s": round((time.time() - t) / 5, 4), "prompt_tokens": len(ids),
              "generate": []}
    msgs = [{"role": "user", "content": "Describe what a potion recipe usually contains, in four bullet points."}]
    p_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                          return_dict=True)["input_ids"]
    for batch in batches:
        inp = torch.tensor([p_ids] * batch, device="cuda")
        torch.cuda.synchronize()
        t = time.time()
        out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=128,
                             min_new_tokens=128, do_sample=True, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t
        n_new = (out.shape[1] - inp.shape[1]) * batch
        record["generate"].append({"batch": batch, "new_tokens": n_new, "seconds": round(dt, 2),
                                   "tokens_per_s": round(n_new / dt, 1)})
    record["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    write_check(name, record)


# exp11's dev-bank answers (results/exp11_wrong_intermediates/analysis/test_v1/test_analysis.txt, Q1 single table)
EXP11_SINGLE_TEST = {
    1: {"correct": 250},
    2: {"correct": 191, "off_cycle": 14, "+1 stirs": 30, "+2 stirs": 7, "+3 stirs": 3, "-1 stirs": 3, "-2 stirs": 2},
    3: {"correct": 88, "off_cycle": 51, "+1 stirs": 31, "+2 stirs": 20, "+3 stirs": 15, "-1 stirs": 41,
        "-2 stirs": 2, "-3 stirs": 2},
}


def exp11_label(brew: B.Brew, pred: str) -> str:
    """exp11_report.q1_single's label: stirs applied along the start's cycle, nearest to h; off_cycle if none."""
    h = len(brew.stirs)
    hops = [k for k in range(h + 4) if brew.apply(brew.start, [B.SINGLE] * k) == pred]
    if pred == brew.path()[-1]:
        return "correct"
    return f"{min(hops, key=lambda k: abs(k - h)) - h:+d} stirs" if hops else "off_cycle"


def item_table(bank: str):
    """exp15_score.Item per captured row (row order)."""
    import exp15_score as S
    items = load_items(bank)
    metas, _ = load_capture(bank, with_hidden=False)
    return [S.analyse(m["item_id"], B.parse(items[m["item_id"]]["problem"]), m["colours"], m["colour_logits"])
            for m in metas]


def check_behaviour(name: str) -> None:
    """Day-0 check 3: the dev bank's answers under exp15's loader and kernels reproduce exp11's label counts within
    3 items per (h, label); and exp15's groups (for the K2 projection)."""
    items = load_items(DEV_BANK)
    table = item_table(DEV_BANK)
    labels = collections.defaultdict(collections.Counter)
    for it in table:
        labels[it.h][exp11_label(B.parse(items[it.item_id]["problem"]), it.pred)] += 1
    diffs = {h: {lab: labels[h].get(lab, 0) - ref.get(lab, 0) for lab in set(ref) | set(labels[h])}
             for h, ref in EXP11_SINGLE_TEST.items()}
    groups = collections.Counter((it.h, it.group, it.consistent) for it in table)
    record = {"labels": {h: dict(c) for h, c in sorted(labels.items())}, "diff_vs_exp11": diffs,
              "max_abs_diff": max(abs(v) for d in diffs.values() for v in d.values()),
              "pass": all(abs(v) <= 3 for d in diffs.values() for v in d.values()),
              "groups": {f"h{h} {g} {'consistent' if c else 'unsure'}": n for (h, g, c), n in sorted(groups.items())}}
    write_check(name, record)


def check_lens_sanity(name: str) -> None:
    """Day-0 check 4: at layer 60 (and 62) of the final cell, the model's own answer colour is ranked first by each
    token lens on >= 90% of dev items; broken down by h and by whether the model's colour softmax is confident."""
    reads = load_lens_reads(DEV_BANK)
    table = item_table(DEV_BANK)
    assert [it.item_id for it in table] == reads["item_ids"]
    final = reads["cells"].index("final")
    colours = list(reads["labels"][:10])
    record = {}
    for kind, scores in reads["scores"].items():
        per = {}
        for layer in (60, 62):
            top = scores[:, final, layer, :10].float().argmax(-1)
            hit = [colours[int(t)] == it.pred for t, it in zip(top, table)]
            strata = {"all": [True] * len(table), **{f"h{h}": [it.h == h for it in table] for h in (1, 2, 3)},
                      "p_pred>=0.8": [it.p_pred >= 0.8 for it in table], "p_pred<0.8": [it.p_pred < 0.8 for it in table]}
            per[f"L{layer}"] = {k: {"top1": round(sum(x for x, m in zip(hit, mask) if m) / max(sum(mask), 1), 3),
                                    "n": sum(mask)} for k, mask in strata.items()}
        record[kind] = per
    record["pass_by_lens"] = {k: v["L60"]["all"]["top1"] >= 0.9 for k, v in record.items()
                              if k in ("logit", "jlens", "rlens", "jpp")}
    write_check(name, record)


def release() -> None:
    """Return GPU memory after the caller has `del`-ed its last reference to a model (hooks make cycles)."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def check_nla(name: str) -> None:
    """Day-0 check 5: (a) the layer convention: ceselder's 64 example activations against our forward on their
    source texts, hidden[42] vs hidden[43] (pass: cosine >= 0.99 for one); (b) the AV -> AR round trip on those
    activations (4 samples each), FVE = 1 - MSE / predict-the-mean MSE on L2-normalised vectors (pass: >= 0.6)."""
    import pyarrow.parquet as pq
    import exp15_readers as RD
    RD._easynla()
    from nla.schema import compute_predict_mean_baselines, normalize_activation
    t = pq.read_table(RD.NLA_DIR / "data" / "example_activations.parquet").to_pylist()
    texts = [r["detokenized_text_truncated"] for r in t]
    acts = torch.tensor([r["activation_vector"] for r in t], dtype=torch.float32)
    tokenizer, model = load_base()
    cos = {42: [], 43: []}
    for text, act in zip(texts, acts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        hs = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
        for k in cos:
            cos[k].append(float(torch.cosine_similarity(hs[k][0, -1].float().cpu(), act, dim=0)))
    del model, hs
    release()
    reader = RD.TextReader("nla")
    samples = reader.read(acts, 42, seed=0)
    drift = reader.drift
    del reader
    release()
    expl = [RD.explanation_text("nla", s) for row in samples for s in row]
    tagged = sum("<explanation>" in s for row in samples for s in row) / len(expl)
    ar = RD.NlaAR()
    pred = ar.predict(expl).float()
    del ar
    release()
    scale = 5120 ** 0.5
    gold = acts.repeat_interleave(4, 0)
    mse = ((normalize_activation(pred, scale) - normalize_activation(gold, scale)) ** 2).mean(-1)
    _, baseline = compute_predict_mean_baselines(acts, scale)
    fve = 1 - mse / baseline
    best = fve.view(-1, 4).max(1).values
    record = {"cos_hidden42": round(sum(cos[42]) / len(t), 4), "cos_hidden43": round(sum(cos[43]) / len(t), 4),
              "min_cos_hidden43": round(min(cos[43]), 4), "layer_index_42_is_hidden": 43 if
              sum(cos[43]) > sum(cos[42]) else 42, "av_adapter_logit_drift": round(drift, 3),
              "explanation_tag_rate": round(tagged, 3), "fve_baseline_mse": round(baseline, 4),
              "fve_mean": round(float(fve.mean()), 4), "fve_best_of_4_by_fve": round(float(best.mean()), 4),
              "pass_convention": max(sum(cos[42]), sum(cos[43])) / len(t) >= 0.99,
              "pass_fve": float(fve.mean()) >= 0.6, "examples": [expl[i][:400] for i in (0, 4, 8)]}
    write_check(name, record)


STOPWORDS = set("""that this with from have will what when where which while would could should about their there
these those them they then than been being into onto over under your yours ours also just only very more most such
some many much each other another because before after again ever never always often like into upon here were""".split())


def content_words(text: str) -> set[str]:
    import re
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


def natural_positions(tokenizer, n: int = 64, prefix: int = 20, cont: int = 20, seed: int = 0):
    """n generated passages from the template-lens corpus (entries read 'word -> passage'): ids up to position
    `prefix` and the text of the next `cont` tokens (the continuation the readers should anticipate)."""
    import exp15_readers as RD
    passages = json.loads((RD.LENS_DIR / "template-lens" / "passages" / "passages.json").read_text())
    rng = random.Random(seed)
    words = sorted(passages)
    out = []
    while len(out) < n:
        p = rng.choice(passages[rng.choice(words)]).split(" -> ", 1)[-1]
        ids = tokenizer.encode(p, add_special_tokens=False)
        if len(ids) >= prefix + cont:
            out.append((ids[:prefix], tokenizer.decode(ids[prefix:prefix + cont])))
    return out


@torch.no_grad()
def check_oracles(name: str) -> None:
    """Day-0 check 6, per reader (ceselder's Oracle lens at layer 42, agu18dec's AO at layer 44): the adapter is
    live (asserted at load), and on 64 natural-text positions the readout's content words overlap the actual next 20
    tokens more than a shuffled pairing does (paired bootstrap CI above 0). For the AO, both renderings of the
    README's prompt (one line; the README's line breaks) are scored and the better one is kept."""
    import exp15_readers as RD
    import exp15_score as S
    tokenizer, model = load_base()
    pos = natural_positions(tokenizer)
    acts = {L: [] for L in (42, 44)}
    for ids, _ in pos:
        hs = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
        for L in acts:
            acts[L].append(hs[L + 1][0, -1].float().cpu())
    del model, hs
    release()
    conts = [content_words(c) for _, c in pos]
    shuffled = conts[1:] + conts[:1]

    def overlap(samples):
        true, shuf = [], []
        for k, row in enumerate(samples):
            words = set().union(*(content_words(s) for s in row))
            true.append(len(words & conts[k]) / max(len(words), 1))
            shuf.append(len(words & shuffled[k]) / max(len(words), 1))
        diff = [a - b for a, b in zip(true, shuf)]
        return {"overlap_true": round(sum(true) / len(true), 4), "overlap_shuffled": round(sum(shuf) / len(shuf), 4),
                "paired_diff": [round(x, 4) for x in S.boot_ci(diff)[:3]], "pass": S.boot_ci(diff)[1] > 0}

    record = {}
    variants = {"olens": {"olens": None},
                "ao": {"ao_one_line": RD.AO_PROMPT_ONE_LINE, "ao_readme_breaks": RD.AO_PROMPT}}
    for reader_name, prompts in variants.items():
        reader = RD.TextReader(reader_name)
        layer = 42 if reader_name == "olens" else 44
        for label, prompt_text in prompts.items():
            if prompt_text is not None:
                reader.template = prompt_text
            samples = reader.read(torch.stack(acts[layer]), layer, seed=0)
            record[label] = {"layer": layer, "adapter_logit_drift": round(reader.drift, 3), **overlap(samples),
                             "examples": [samples[i][0][:300] for i in (0, 1, 2)]}
        del reader
        release()
    ao = {k: v for k, v in record.items() if k.startswith("ao_")}
    record["ao_prompt_kept"] = max(ao, key=lambda k: ao[k]["overlap_true"] - ao[k]["overlap_shuffled"])
    write_check(name, record)


PTAG_LAYERS = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)
WHITENER_SPEC = {"passages": 3000, "skip_first": 4, "seed": 1, "ridge": 0.1}


def whitener_path() -> Path:
    return EXP.cache / f"whiteners_natural_{cfg.content_key(WHITENER_SPEC)}.pt"


def passage_ids(tokenizer, n: int, seed: int, min_len: int = 16) -> list[list[int]]:
    import exp15_readers as RD
    passages = json.loads((RD.LENS_DIR / "template-lens" / "passages" / "passages.json").read_text())
    rng = random.Random(seed)
    words = sorted(w for w in passages if passages[w])
    out = []
    while len(out) < n:
        ids = tokenizer.encode(rng.choice(passages[rng.choice(words)]).split(" -> ", 1)[-1], add_special_tokens=False)
        if len(ids) >= min_len:
            out.append(ids)
    return out


@torch.no_grad()
def padded_hidden(model, rows: list[list[int]]):
    """Right-padded forward over token rows: (hidden_states tuple, attention mask)."""
    width = max(map(len, rows))
    ids = torch.tensor([r + [0] * (width - len(r)) for r in rows], device="cuda")
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device="cuda")
    hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
    return hs, mask


@torch.no_grad()
def estimate_whiteners(model, tokenizer) -> dict:
    """Per-layer mean and covariance of natural-text residuals at PTAG_LAYERS (positions >= skip_first, so the
    attention-sink tokens' huge norms don't dominate). Saved once; content-keyed by WHITENER_SPEC."""
    path = whitener_path()
    if path.exists():
        return torch.load(path, weights_only=False)
    spec = WHITENER_SPEC
    rows = passage_ids(tokenizer, spec["passages"], spec["seed"])
    d = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
    sums = {L: torch.zeros(d, dtype=torch.float64, device="cuda") for L in PTAG_LAYERS}
    outer = {L: torch.zeros(d, d, dtype=torch.float64, device="cuda") for L in PTAG_LAYERS}
    count = 0
    for start in range(0, len(rows), 64):
        hs, mask = padded_hidden(model, rows[start:start + 64])
        keep = mask.bool().clone()
        keep[:, :spec["skip_first"]] = False
        count += int(keep.sum())
        for L in PTAG_LAYERS:
            x = hs[L + 1][keep].double()
            sums[L] += x.sum(0)
            outer[L] += x.T @ x
    blob = {"spec": spec, "count": count, "mean": {}, "cov": {}}
    for L in PTAG_LAYERS:
        mu = sums[L] / count
        blob["mean"][L] = mu.float().cpu()
        blob["cov"][L] = (outer[L] / count - torch.outer(mu, mu)).float().cpu()
    atomic_save(blob, path)
    return blob


@torch.no_grad()
def check_ptag(name: str, n_spans: int = 200) -> None:
    """Day-0 check 7: agu18dec's ptag AR retrieves its own target among 200 held-out natural-text spans (retrieval@1
    >= 0.8 at layer 44; published 0.93). Their pooled whitener is gated, so targets are whitened with ours
    (WHITENER_SPEC); raw-space retrieval is reported alongside."""
    import exp15_readers as RD
    tokenizer, model = load_base()
    whiteners = estimate_whiteners(model, tokenizer)
    rng = random.Random(2)
    docs = passage_ids(tokenizer, n_spans, seed=2, min_len=24)
    spans, ends = [], []
    for ids in docs:
        t = rng.randrange(8, len(ids))
        length = min(rng.randint(1, 32), t + 1)
        spans.append(ids[t - length + 1:t + 1])
        ends.append(t)
    targets = {L: [] for L in PTAG_LAYERS}
    for start in range(0, n_spans, 32):
        hs, _ = padded_hidden(model, docs[start:start + 32])
        for L in PTAG_LAYERS:
            targets[L] += [hs[L + 1][i, ends[start + i]].float().cpu() for i in range(len(hs[0]))]
    del model, hs
    release()
    ar = RD.PtagAR()
    record = {"n_spans": n_spans, "lora_pairs_merged": ar.n_merged, "whitener_count": whiteners["count"],
              "whitener_spec": WHITENER_SPEC, "published_ret1": {44: 0.93}, "layers": {}}
    for L in PTAG_LAYERS:
        pred = ar.predict(spans, L)
        gold = torch.stack(targets[L])
        w = RD.Whitener(whiteners["mean"][L], whiteners["cov"][L], WHITENER_SPEC["ridge"])
        spaces = {"ours_whitened": w.white(gold), "raw_centered": gold - whiteners["mean"][L]}
        entry = {}
        for space, g in spaces.items():
            sim = torch.nn.functional.normalize(pred, dim=-1) @ torch.nn.functional.normalize(g, dim=-1).T
            entry[space] = {"ret1": round(float((sim.argmax(1) == torch.arange(n_spans)).float().mean()), 3),
                            "diag_cos": round(float(sim.diag().mean()), 3),
                            "offdiag_cos": round(float((sim.sum() - sim.diag().sum()) / (n_spans ** 2 - n_spans)), 3)}
        record["layers"][L] = entry
    record["pass"] = record["layers"][44]["ours_whitened"]["ret1"] >= 0.8
    write_check(name, record)


CHECKS = {"throughput": check_throughput,     # the first two ran before cuDNN attention was turned off
          "throughput_fla": lambda name: check_throughput(name, batches=(16, 64, 192)),
          "throughput_final": lambda name: check_throughput(name, batches=(64, 192)),
          "behaviour": check_behaviour, "lens_sanity": check_lens_sanity, "nla": check_nla, "oracles": check_oracles,
          "ptag": check_ptag}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "bank", "requests", "capture", "lens_reads", "trained_reads",
                                   "text_reads", "vectors", "swaps", "repair"])
    p.add_argument("--parts", nargs="*", default=["lens", "diffmeans", "nla_ar", "ptag_ar"])
    p.add_argument("name", nargs="?", help="check name, or the text reader (nla | olens | ao)")
    p.add_argument("--bank", choices=BANKS, default=DEV_BANK)
    a = p.parse_args()
    if a.cmd == "check":
        CHECKS[a.name](a.name)
    elif a.cmd == "bank":
        write_test_bank()
    elif a.cmd == "requests":
        write_test_requests()
    elif a.cmd == "capture":
        capture(a.bank)
    elif a.cmd == "lens_reads":
        lens_reads(a.bank)
    elif a.cmd == "trained_reads":
        trained_reads(a.bank)
    elif a.cmd == "text_reads":
        text_reads(a.bank, a.name)
    elif a.cmd == "vectors":
        build_directions(tuple(a.parts))
    elif a.cmd == "swaps":
        swaps(a.bank)
    elif a.cmd == "repair":
        repair(a.bank)


if __name__ == "__main__":
    main()
