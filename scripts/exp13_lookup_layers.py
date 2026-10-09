"""exp13: do in-context lookups need attention layers that can reach the table? (results/exp13_lookup_layers/
manifest.json)

Gemma-4-31B-it has 50 sliding-window layers (window 1024 tokens) and 10 global layers; on a brew prompt (~600
tokens) every layer can attend to the table. Qwen3.6-27B has 16 full-attention blocks and 48 linear-attention
(Gated DeltaNet) blocks. Two tests:

  passage      the neutral padding passage: WikiText-103 validation sentences with no colour, ingredient, potion,
               stir or colour-word mention, detokenised (cache/exp13/passage.txt, written once)
  requests     Gemma rows: exp11's test items (three-ingredient, h=1 and h=2) and single_test items (h=2 and h=3),
               each rendered under the padding conditions in CONDITIONS (cache/exp13/requests_gemma.jsonl)
  mask-check   Gemma harness check: block 0 (sliding) at position q must see q-1023 and not q-1024; block 5 (the
               first global block) must see q-1500
  gemma        one forward per row: the 10 colour logits at the answer position and the token spans of the target
               problem's rule lines (cache/exp13/gemma_rows.jsonl)
  qwen         Qwen3.6-27B: for each block in QWEN_BLOCKS, the token mixer's output (self-attention or linear
               attention) at the answer position replaced by another item's (resample ablation)
               (cache/exp13/qwen_ablation.pt)
  report       the manifest's numbers and reading rules (results/exp13_lookup_layers/<run>/report.txt)

Layer numbering: block b is decoder layer b (0-indexed); "the output of block b" is hidden_states[b + 1].
Run with /venv/main/bin/python from scripts/; HF_HOME must point at the model cache.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch

import cc_config as cfg
import cc_exp10_render as R
import exp10_conditions as C
import exp11 as X
import exp11_brew as B

EXP = cfg.Exp("exp13_lookup_layers", "exp13")
GEMMA = "google/gemma-4-31B-it"
GEMMA_FAMILY = "gemma4"
WINDOW = 1024              # Gemma-4-31B sliding_window: key k is visible from query q iff q - k <= WINDOW - 1
GEMMA_GLOBAL_BLOCKS = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)

# Padding conditions: (placement, target distance in tokens from the answer position back to the last token of the
# table). "between" puts the passage between the table and the start sentence; "before" puts a passage of the same
# length as the named "between" condition before the table (same total length, the table stays near the question).
SWEEP = (700, 800, 900, 950, 1000, 1050, 1100, 1200, 1500, 2500)
CONDITIONS = {"base": ("none", None),
              **{f"between_{d:04d}": ("between", d) for d in SWEEP},
              "before_1200": ("before", 1200), "before_2500": ("before", 2500)}
# Which conditions each item set is rendered under
ITEM_SETS = {"three": {"bank": "test", "h": (1, 2), "conditions": tuple(CONDITIONS)},
             "single": {"bank": "single_test", "h": (2, 3),
                        "conditions": ("base", "between_0700", "between_1200", "before_1200")}}
PASSAGE_OPEN = "[An unrelated passage follows. It has nothing to do with the problem.]"
PASSAGE_CLOSE = "[End of the unrelated passage.]"
EXCLUDED_WORDS = re.compile(
    r"\b(red\w*|blue\w*|green\w*|gold\w*|pink\w*|gr[ae]y\w*|brown\w*|black\w*|white\w*|purple\w*|salt\w*|ash\w*|"
    r"dew\w*|moss\w*|clay\w*|bark\w*|sand\w*|mint\w*|soot\w*|chalk\w*|potion\w*|stir\w*|colou?r\w*|ingredient\w*)\b",
    re.IGNORECASE)
PASSAGE_TOKENS = 3200      # enough for the longest condition

QWEN_FAMILY = "qwen3.6"
QWEN_BLOCKS = tuple(range(16, 64))
QWEN_FULL_ATTENTION = tuple(b for b in range(64) if b % 4 == 3)
N_BOOT = 2000


def passage_path() -> Path:
    return EXP.cache / "passage.txt"


def gemma_requests_path() -> Path:
    return EXP.cache / "requests_gemma.jsonl"


def gemma_rows_path() -> Path:
    return EXP.cache / "gemma_rows.jsonl"


def qwen_path() -> Path:
    return EXP.cache / "qwen_ablation.pt"


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# --- passage --------------------------------------------------------------------------------------------------------
def detokenise(text: str) -> str:
    """Undo WikiText-103's raw tokenisation (' , ', ' @-@ ', spaces inside brackets)."""
    text = re.sub(r" @(.)@ ", r"\1", text)
    text = re.sub(r" ([,.;:!?%)\]'])", r"\1", text)
    text = re.sub(r"([(\[$]) ", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def write_passage() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GEMMA)
    records = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation", streaming=True)
    sentences, n_tokens = [], 0
    for record in records:
        text = record["text"].strip()
        if not text or text.startswith("="):           # headings
            continue
        for sentence in re.split(r"(?<=[.!?]) (?=[A-Z])", detokenise(text)):
            if len(sentence) < 40 or EXCLUDED_WORDS.search(sentence):
                continue
            sentences.append(sentence)
            n_tokens += len(tokenizer.encode(" " + sentence, add_special_tokens=False))
        if n_tokens >= PASSAGE_TOKENS:
            break
    write_once(passage_path(), "\n".join(sentences) + "\n")
    print(f"{len(sentences)} sentences, ~{n_tokens} tokens -> {passage_path()}")


def passage_sentences() -> list[str]:
    return passage_path().read_text().splitlines()


# --- requests -------------------------------------------------------------------------------------------------------
def padded_problem(problem: str, placement: str, passage: str) -> str:
    """The problem text with the framed passage inserted before the table or between the table and the start
    sentence; 'none' returns the problem unchanged."""
    if placement == "none":
        return problem
    block = f"{PASSAGE_OPEN}\n{passage}\n{PASSAGE_CLOSE}\n"
    if placement == "before":
        return block + problem
    at = problem.index("The potion starts out ")
    return problem[:at] + block + problem[at:]


def target_spans(tokenizer, text: str, brew: B.Brew) -> dict:
    """Token indices in the rendered prompt text of the target (last) problem: each rule line's first and last
    token, the start sentence's first token and the last table token."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def first_token_at(char: int) -> int:
        return next(i for i, (a, b) in enumerate(offsets) if b > char)

    def last_token_before(char_end: int) -> int:
        return max(i for i, (a, b) in enumerate(offsets) if a < char_end)

    problem = text.rindex("Problem: ")
    lines = {}
    for colour in brew.rule_order:
        start = text.index(f"\nA {colour} potion turns ", problem) + 1
        end = text.index("\n", start)
        lines[colour] = (first_token_at(start), last_token_before(end))
    start_sentence = first_token_at(text.index("The potion starts out ", problem))
    return {"lines": lines, "start_sentence": start_sentence, "table_last": max(b for _, b in lines.values())}


def passage_for(tokenizer, n_tokens: int) -> str:
    """The longest prefix of the passage's sentences of at most n_tokens tokens."""
    sentences = passage_sentences()
    lo, hi = 0, len(sentences)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode(" ".join(sentences[:mid]), add_special_tokens=False)) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(sentences[:lo])


def write_gemma_requests() -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GEMMA)
    rows = []
    for set_name, spec in ITEM_SETS.items():
        items = X.load_bank(spec["bank"])
        shot = next(it for it in items if it["split"] == "shot")
        evals = [it for it in items if it["split"] == "eval" and it["h"] in spec["h"]]
        # passage lengths: chosen once per condition from the first item (the post-table tail is the same length
        # for every item of a set up to a token or two); every row's actual distances are recorded by `gemma`
        ref = evals[0]
        ref_row = C.fields("C0", {**ref, "problem": padded_problem(ref["problem"], "between", "")}, shot)
        ref_text = R.prompt_text(tokenizer, GEMMA_FAMILY, ref_row)
        ref_ids = R.prompt_ids(tokenizer, GEMMA_FAMILY, ref_row)
        overhead = (len(ref_ids) - 1) - target_spans(tokenizer, ref_text, B.parse(ref["problem"]))["table_last"]
        passages = {name: passage_for(tokenizer, d - overhead) if d else ""
                    for name, (_, d) in CONDITIONS.items()}
        for it in evals:
            for name in spec["conditions"]:
                placement, _ = CONDITIONS[name]
                problem = padded_problem(it["problem"], placement, passages[name])
                assert B.parse(problem).path() == B.parse(it["problem"]).path()
                row = {"item_id": it["item_id"], "item_set": set_name, "bank": spec["bank"], "h": it["h"],
                       "padding": name, "answer": it["answer"], "problem": problem,
                       **C.fields("C0", {**it, "problem": problem}, shot)}
                row["request_id"] = cfg.content_key(row)
                rows.append(row)
    write_once(gemma_requests_path(), "".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {gemma_requests_path()}")


# --- Gemma ----------------------------------------------------------------------------------------------------------
def load_gemma():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GEMMA)
    model = AutoModelForImageTextToText.from_pretrained(GEMMA, dtype=torch.bfloat16, device_map="cuda").eval()
    return tokenizer, model


def colour_token_ids(tokenizer, colours) -> list[int]:
    ids = []
    for colour in colours:
        toks = tokenizer.encode(" " + colour, add_special_tokens=False)
        assert len(toks) == 1, f"{colour!r} is {len(toks)} tokens"
        ids.append(toks[0])
    return ids


@torch.no_grad()
def mask_check() -> None:
    """Change one token at distance d before the answer position and compare the output of block 0 (sliding) and
    block 5 (global) at the answer position with the unchanged prompt's."""
    tokenizer, model = load_gemma()
    rows = [json.loads(line) for line in gemma_requests_path().open()]
    row = next(r for r in rows if r["padding"] == "between_1500")
    ids = R.prompt_ids(tokenizer, GEMMA_FAMILY, row)
    q = len(ids) - 1

    def block_outputs(input_ids):
        out = model(input_ids=torch.tensor([input_ids], device="cuda"), output_hidden_states=True, use_cache=False)
        return out.hidden_states[1][0, q].float(), out.hidden_states[6][0, q].float()

    clean0, clean5 = block_outputs(ids)
    lines = [f"prompt of {len(ids)} tokens (row {row['item_id']} {row['condition']}); answer position q={q}"]
    expected = {1022: (True, True), 1023: (True, True), 1024: (False, True), 1025: (False, True),
                1500: (False, True)}
    ok = True
    for d, (sees0, sees5) in expected.items():
        changed = list(ids)
        changed[q - d] = tokenizer.encode(" the", add_special_tokens=False)[0] if ids[q - d] != tokenizer.encode(
            " the", add_special_tokens=False)[0] else tokenizer.encode(" a", add_special_tokens=False)[0]
        out0, out5 = block_outputs(changed)
        diff0, diff5 = float((out0 - clean0).abs().max()), float((out5 - clean5).abs().max())
        good = (diff0 > 0) == sees0 and (diff5 > 0) == sees5
        ok &= good
        lines.append(f"  d={d}: block 0 max|diff| {diff0:.3g} (expected {'>0' if sees0 else '0'}), block 5 "
                     f"{diff5:.3g} (expected {'>0' if sees5 else '0'}) {'ok' if good else 'MISMATCH'}")
    lines.append("MASK CHECK " + ("PASSED" if ok else "FAILED"))
    text = "\n".join(lines)
    print(text)
    write_once(EXP.cache / "mask_check.txt", text + "\n")
    if not ok:
        raise SystemExit("mask check failed")


@torch.no_grad()
def run_gemma() -> None:
    out = gemma_rows_path()
    if out.exists():
        raise SystemExit(f"{out} exists; captures are written once")
    tokenizer, model = load_gemma()
    rows = [json.loads(line) for line in gemma_requests_path().open()]
    colours = sorted(B.parse(rows[0]["problem"]).table)
    cids = colour_token_ids(tokenizer, colours)
    records = []
    with out.with_suffix(".partial").open("w") as f:
        for n, row in enumerate(rows):
            text = R.prompt_text(tokenizer, GEMMA_FAMILY, row)
            ids = R.prompt_ids(tokenizer, GEMMA_FAMILY, row)
            if tokenizer.encode(text, add_special_tokens=False) != ids[:len(tokenizer.encode(
                    text, add_special_tokens=False))]:
                raise RuntimeError(f"{row['request_id']}: the rendered text does not tokenize to the prompt prefix")
            spans = target_spans(tokenizer, text, B.parse(row["problem"]))
            logits = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float()
            colour_logits = logits[cids].cpu()
            record = {"request_id": row["request_id"], "item_id": row["item_id"], "item_set": row["item_set"],
                      "h": row["h"], "padding": row["padding"], "answer": row["answer"],
                      "n_tokens": len(ids), "final": len(ids) - 1, **spans, "colours": colours,
                      "colour_logits": [round(float(x), 4) for x in colour_logits],
                      "predicted": colours[int(colour_logits.argmax())],
                      "colour_mass": round(float(torch.softmax(logits, -1)[cids].sum()), 6),
                      "top_token_is_colour": int(logits.argmax()) in cids}
            f.write(json.dumps(record) + "\n")
            if n % 250 == 0:
                print(f"{n}/{len(rows)} {row['padding']} n_tokens={len(ids)}", flush=True)
    out.with_suffix(".partial").rename(out)
    print(f"{len(rows)} rows -> {out}")


# --- Qwen -----------------------------------------------------------------------------------------------------------
def derangement(n: int, seed: str) -> list[int]:
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


class MixerPatch:
    """Forward hooks on every block's token mixer (self_attn or linear_attn): record its output at the answer
    position, or replace it there with a given vector at one block."""

    def __init__(self, layers):
        self.mixers = [layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn for layer in layers]
        self.recorded: dict[int, torch.Tensor] = {}
        self.replace: tuple[int, torch.Tensor] | None = None
        self.handles = [m.register_forward_hook(self._hook(b)) for b, m in enumerate(self.mixers)]

    def _hook(self, block: int):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.recorded[block] = out[0, -1].detach().clone()
            if self.replace is not None and self.replace[0] == block:
                out = out.clone()
                out[0, -1] = self.replace[1].to(out.dtype)
                return (out, *output[1:]) if isinstance(output, tuple) else out
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_qwen() -> None:
    out = qwen_path()
    if out.exists():
        raise SystemExit(f"{out} exists; written once")
    import exp11_model as M
    tokenizer, model = M.load()
    layers, _, _ = M.text_parts(model)
    for b, layer in enumerate(layers):
        assert hasattr(layer, "self_attn") == (b in QWEN_FULL_ATTENTION), f"block {b} type"
    items = {r["item_id"]: r for r in X.load_bank("test")}
    rows = [json.loads(line) for line in X.requests_path("test").open()]
    colours = sorted(B.parse(items[rows[0]["item_id"]]["problem"]).table)
    cids = list(M.colour_ids(tokenizer, colours).values())
    patch = MixerPatch(layers)
    results = {}
    for h in (1, 2):
        hrows = [r for r in rows if r["h"] == h]
        ids_all = [R.prompt_ids(tokenizer, QWEN_FAMILY, r) for r in hrows]
        clean_logits, clean_mix = [], []
        for ids in ids_all:
            patch.replace = None
            logits = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float()
            clean_logits.append(logits[cids].cpu())
            clean_mix.append(torch.stack([patch.recorded[b] for b in range(len(layers))]).cpu())
        donor = derangement(len(hrows), f"exp13 qwen h={h}")
        patched = torch.zeros(len(hrows), len(QWEN_BLOCKS), len(cids))
        for i, ids in enumerate(ids_all):
            for k, b in enumerate(QWEN_BLOCKS):
                patch.replace = (b, clean_mix[donor[i]][b].cuda())
                logits = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float()
                patched[i, k] = logits[cids].cpu()
            if i % 25 == 0:
                print(f"h={h} {i}/{len(hrows)}", flush=True)
        results[h] = {"item_ids": [r["item_id"] for r in hrows], "gold": [items[r["item_id"]]["answer"] for r in hrows],
                      "donor": donor, "clean_logits": torch.stack(clean_logits), "patched_logits": patched}
    patch.remove()
    torch.save({"colours": colours, "blocks": QWEN_BLOCKS, "full_attention": QWEN_FULL_ATTENTION,
                "results": results}, out)
    print(f"-> {out}")


# --- report ---------------------------------------------------------------------------------------------------------
def boot_ci(x: torch.Tensor, g: torch.Generator) -> tuple[float, float, float]:
    """Mean over items of x [n] with a bootstrap 95% CI."""
    boot = x[torch.randint(len(x), (N_BOOT, len(x)), generator=g)].mean(1)
    return float(x.mean()), float(boot.quantile(0.025)), float(boot.quantile(0.975))


def fmt(ci) -> str:
    return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"


def needed_lines(record: dict, items: dict) -> list[str]:
    """The rule lines the item's hops read: the line of each state before a stir (s_0 .. s_{h-1})."""
    return B.parse(items[record["item_id"]]["problem"]).path()[:-1]


def line_visibility(record: dict, colour: str) -> str:
    """'visible': the whole line is within the window of the answer position (so of every later query position
    too); 'hidden': no token of the line is within the window of the start sentence's first token (so of no
    post-table position); 'straddle' otherwise."""
    first, last = record["lines"][colour]
    if record["final"] - first <= WINDOW - 1:
        return "visible"
    if record["start_sentence"] - last > WINDOW - 1:
        return "hidden"
    return "straddle"


def report_gemma(lines: list[str], g: torch.Generator) -> None:
    rows = [json.loads(line) for line in gemma_rows_path().open()]
    items = {**{r["item_id"]: r for r in X.load_bank("test")}, **{r["item_id"]: r for r in X.load_bank("single_test")}}
    by = defaultdict(dict)                      # (item_set, h) -> condition -> {item_id: record}
    for r in rows:
        r["correct"] = r["predicted"] == r["answer"]
        by[(r["item_set"], r["h"])].setdefault(r["padding"], {})[r["item_id"]] = r

    def acc(cell: dict) -> torch.Tensor:
        return torch.tensor([float(r["correct"]) for _, r in sorted(cell.items())])

    def paired_diff(a: dict, b: dict) -> torch.Tensor:
        return torch.tensor([float(a[i]["correct"]) - float(b[i]["correct"]) for i in sorted(a)])

    lines.append("## Gemma-4-31B-it: accuracy by padding condition (argmax over the 10 colours; chance 0.10)")
    lines.append("distance = answer position minus the last table token (median over items); 'window' = every "
                 "needed line visible / every needed line hidden from all post-table positions")
    for key in sorted(by):
        lines.append(f"\n### {key[0]} h={key[1]}")
        for name in CONDITIONS:
            cell = by[key].get(name)
            if not cell:
                continue
            recs = list(cell.values())
            dist = sorted(r["final"] - r["table_last"] for r in recs)[len(recs) // 2]
            vis = [set(line_visibility(r, c) for c in needed_lines(r, items)) for r in recs]
            all_vis = sum(v == {"visible"} for v in vis) / len(vis)
            all_hid = sum(v == {"hidden"} for v in vis) / len(vis)
            mass = sum(r["colour_mass"] for r in recs) / len(recs)
            lines.append(f"  {name:13s} n={len(recs):3d} tokens~{recs[0]['n_tokens']:5d} distance {dist:5d}  "
                         f"accuracy {fmt(boot_ci(acc(cell), g))}  colour mass {mass:.3f}  "
                         f"needed lines all visible {all_vis:.2f} / all hidden {all_hid:.2f}")

    lines.append("\n## primary contrasts (paired over items; positive = accuracy lost)")
    drops = {}
    for key in sorted(by):
        cells = by[key]
        if "between_1200" not in cells:
            continue
        within, far, before = cells["between_0700"], cells["between_1200"], cells["before_1200"]
        drop = paired_diff(within, far)
        length = paired_diff(within, before)
        drops[key] = drop
        lines.append(f"  {key[0]} h={key[1]}: within-window (between_0700) - beyond-window (between_1200) "
                     f"{fmt(boot_ci(drop, g))}; within-window - matched-length table-near (before_1200) "
                     f"{fmt(boot_ci(length, g))}")
    if ("three", 1) in drops and ("three", 2) in drops:
        d1, d2 = drops[("three", 1)], drops[("three", 2)]
        # h=1 and h=2 are different items: an unpaired bootstrap of the difference of means
        boot = (d2[torch.randint(len(d2), (N_BOOT, len(d2)), generator=g)].mean(1)
                - d1[torch.randint(len(d1), (N_BOOT, len(d1)), generator=g)].mean(1))
        diff = (float(d2.mean() - d1.mean()), float(boot.quantile(0.025)), float(boot.quantile(0.975)))
        lines.append(f"  three: (h=2 drop) - (h=1 drop) {fmt(diff)}")
        lines.append("\n## reading rules (manifest 'reading_rules_gemma'), three-ingredient items")
        r1 = d1.mean() >= 0.25
        r2 = d2.mean() >= 0.25 and diff[0] >= 0.15 and diff[1] > 0
        r3 = boot_ci(d2, g)[2] < 0.15
        lines.append(f"  lookups_need_sliding (h=1 drop >= 0.25): {bool(r1)}")
        lines.append(f"  composition_needs_sliding (h=2 drop >= 0.25 and exceeds the h=1 drop by >= 0.15, CI "
                     f"low end > 0): {bool(r2)}")
        lines.append(f"  global_layers_suffice (h=2 drop CI high end < 0.15): {bool(r3)}")

    lines.append("\n## per-line visibility in the transition zone (between_0900 .. between_1100 pooled), "
                 "three-ingredient items: accuracy by which needed lines are hidden")
    for h in (1, 2):
        groups = defaultdict(list)
        for name in ("between_0900", "between_0950", "between_1000", "between_1050", "between_1100"):
            for r in by[("three", h)].get(name, {}).values():
                vis = [line_visibility(r, c) for c in needed_lines(r, items)]
                if "straddle" in vis:
                    continue
                label = "+".join(f"s{j}:{'vis' if v == 'visible' else 'HID'}" for j, v in enumerate(vis))
                groups[label].append(float(r["correct"]))
        for label, xs in sorted(groups.items()):
            lines.append(f"  h={h} {label:22s} n={len(xs):4d} accuracy {fmt(boot_ci(torch.tensor(xs), g))}")


def report_qwen(lines: list[str], g: torch.Generator) -> None:
    blob = torch.load(qwen_path(), weights_only=False)
    blocks, full = list(blob["blocks"]), set(blob["full_attention"])
    lines.append("\n## Qwen3.6-27B: resample ablation of one block's token mixer at the answer position")
    lines.append("keep = argmax unchanged from the clean pass; donor = argmax becomes the donor item's clean argmax "
                 "(items whose clean argmax equals the donor's excluded from 'donor'); F = full attention")
    for h, res in sorted(blob["results"].items()):
        clean = res["clean_logits"].argmax(-1)
        donor_pred = clean[torch.tensor(res["donor"])]
        patched = res["patched_logits"].argmax(-1)          # [n, blocks]
        keep = (patched == clean[:, None]).float()
        differs = donor_pred != clean
        to_donor = (patched[differs] == donor_pred[differs][:, None]).float()
        gold = torch.tensor([blob["colours"].index(c) for c in res["gold"]])
        lines.append(f"\n### h={h}: n={len(clean)}, clean accuracy {float((clean == gold).float().mean()):.3f}")
        for k, b in enumerate(blocks):
            lines.append(f"  block {b:2d}{'F' if b in full else ' '} keep {keep[:, k].mean():.3f}  "
                         f"donor {to_donor[:, k].mean():.3f}")
        is_full = torch.tensor([b in full for b in blocks])
        band = torch.tensor([20 <= b <= 56 for b in blocks])
        effect = 1 - keep                                    # [n, blocks]
        per_item = effect[:, band & is_full].mean(1) - effect[:, band & ~is_full].mean(1)
        top = blocks[int(effect.mean(0).masked_fill(~band, -1).argmax())]
        lines.append(f"  blocks 20-56: mean effect (1 - keep) at full-attention blocks minus at linear blocks "
                     f"{fmt(boot_ci(per_item, g))}; largest single-block effect at block {top} "
                     f"({'full attention' if top in full else 'linear'})")
        if h == 1:
            lines.append(f"  reading rule 'lookup_at_full_attention' (difference >= 0.10 with CI low end > 0 and "
                         f"the largest effect at a full-attention block): "
                         f"{bool(per_item.mean() >= 0.10 and boot_ci(per_item, g)[1] > 0 and top in full)}")


def report(run: str) -> None:
    out = EXP.results / run / "report.txt"
    if out.exists():
        raise SystemExit(f"{out} exists; reports are written once")
    g = torch.Generator().manual_seed(0)
    lines = [f"exp13 report ({run}); manifest results/exp13_lookup_layers/manifest.json", ""]
    if (EXP.cache / "mask_check.txt").exists():
        lines += ["## harness: sliding-window mask check", (EXP.cache / "mask_check.txt").read_text().strip(), ""]
    if gemma_rows_path().exists():
        report_gemma(lines, g)
    if qwen_path().exists():
        report_qwen(lines, g)
    text = "\n".join(lines)
    print(text)
    write_once(out, text + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["passage", "requests", "mask-check", "gemma", "qwen", "report"])
    p.add_argument("--run", help="report: the analysis run name (results/exp13_lookup_layers/<run>)")
    a = p.parse_args()
    {"passage": write_passage, "requests": write_gemma_requests, "mask-check": mask_check, "gemma": run_gemma,
     "qwen": run_qwen, "report": lambda: report(a.run)}[a.cmd]()


if __name__ == "__main__":
    main()
