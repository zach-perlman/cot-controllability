"""exp15 secondary analyses (exploratory, labelled as such in the write-up).

  count_target --bank B --tag T     on miscount items (unique implied count k != h): is the implied count read above
                                    the other counts? token lenses (number-word rows), template, text readers
  selfreport --bank B               the model's own no-CoT report of the colour before the last stir, and a
                                    thinking-on prompt-only baseline (100 test items)
  nla_iter300 --bank B              NLA AV iter_000300 vs iter_000800 on the dev reads
  best_of_16 --bank B               16 NLA samples per vector, ranked by AR reconstruction cosine: top-1 vs random-1
  ar_edit --bank B                  P3 key swaps with per-item AR-edit directions: AR(explanation with s1 -> s1')
                                    - AR(explanation), from the item's own NLA reads
  read_then_write --bank B          P4 repair with each reader's decoded key in place of w
  count_repair --bank B             repair of miscount errors by count-word edits (implied count -> h)
  edits_report --bank B --tag T     the AR-edit, read-then-write and count-repair results
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import re

import torch

import exp15 as E
import exp15_report as R
import exp15_score as S
from exp15 import DEV_BANK, EXP, PRIMARY_CELLS, PRIMARY_LAYERS


# --- count target ---------------------------------------------------------------------------------------------------
def count_target(bank: str, tag: str) -> None:
    """Miscount items with a unique implied count k in 1..6 (k != h): AUC of S(k) vs the other counts (excluding the
    true h), and of S(h) vs the others (excluding k), per reader at its primary pairs."""
    import exp15_readers as RD
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    table = E.item_table(bank)
    targets = [it for it in table if it.group == "miscount" and it.implied_count in S.COUNTS and it.h in (2, 3)]
    lr = E.load_lens_reads(bank)
    pc = [list(lr["cells"]).index(c) for c in PRIMARY_CELLS]
    count_cols = [lr["labels"].index(f"count{k}") for k in S.COUNTS]
    readers = {}
    for kind in (*R.TOKEN_READERS,):
        x = lr["scores"][kind][:, pc][:, :, list(PRIMARY_LAYERS)][..., count_cols].float()
        readers[kind] = S.ReaderScores(kind, "rank", [], {i: x[n].reshape(-1, len(S.COUNTS))
                                                          for n, i in enumerate(lr["item_ids"])})
    for name in R.TEXT_READERS:
        if not E.text_reads_path(bank, name).exists():
            continue
        tpairs = [(c, l) for c, l in E.text_jobs(name) if l in R.TEXT_PRIMARY_LAYERS[name]]
        scores = {}
        for item_id, r in E.load_text_reads(bank, name).items():
            scores[item_id] = torch.stack([
                S.naming_fraction([RD.explanation_text(name, s) for s in r[f"{c}|{l}"]], S.COUNTS, S.counts_named)
                for c, l in tpairs])
        readers[name] = S.ReaderScores(name, "fraction", tpairs, scores)
    record = {"bank": bank, "n_items": len(targets), "exploratory": True, "readers": {}}
    for name, rs in readers.items():
        implied, true = [], []
        for it in targets:
            s = rs.score(it.item_id)
            if s is None:
                continue
            k, h = S.COUNTS.index(it.implied_count), S.COUNTS.index(it.h)
            others = [j for j in range(len(S.COUNTS)) if j not in (k, h)]
            implied.append(S.item_auc(s, k, others))
            true.append(S.item_auc(s, h, others))
        record["readers"][name] = {"implied_count_auc": S.summarise([{"x": v} for v in implied], "x", 0.5),
                                   "true_count_auc": S.summarise([{"x": v} for v in true], "x", 0.5)}
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "count_target.json")
    for name, e in record["readers"].items():
        print(f"{name:9s} implied {e['implied_count_auc']}  true {e['true_count_auc']}")


# --- self-report and the thinking-on prompt-only baseline -----------------------------------------------------------
SELFREPORT_Q = "What color was the potion just before the last stir? Answer with a single color word."
FINAL_Q = "What color is the potion at the end?"
THINKING_N = 100


def selfreport_path(bank: str):
    return EXP.cache / f"selfreport_{bank}.jsonl"


@torch.no_grad()
def selfreport(bank: str) -> None:
    """(a) Self-report: after its own no-CoT answer (C0, thinking off), the model is asked SELFREPORT_Q, again with
    thinking off and 'Answer:' prefilled; its reply is the argmax over the 10 colour tokens. Every h=2/h=3 item.
    (b) Prompt-only with thinking on: the same question asked about the problem alone (no answer given), thinking
    on, up to 4096 new tokens, on THINKING_N seeded off-cycle errors (test bank only). Raw rows are cached."""
    path = selfreport_path(bank)
    E.require_absent(path)
    tokenizer, model = E.load_base()
    items = E.load_items(bank)
    rows = {r["item_id"]: r for r in E.load_requests(bank)}
    metas, _ = E.load_capture(bank, with_hidden=False)
    colours = metas[0]["colours"]
    cids = torch.tensor(E.colour_ids(tokenizer, colours), device="cuda")
    table = [it for it in E.item_table(bank) if it.h in (2, 3)]
    tail = (f"<|im_end|>\n<|im_start|>user\n{SELFREPORT_Q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "Answer:")
    out = []
    for start in range(0, len(table), 32):
        part = table[start:start + 32]
        seqs = [E.prompt(tokenizer, rows[it.item_id])[1] + tokenizer.encode(f" {it.pred}" + tail,
                                                                             add_special_tokens=False)
                for it in part]
        width = max(map(len, seqs))
        ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs], device="cuda")
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device="cuda")
        hs = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = hs[torch.arange(len(part)), torch.tensor([len(s) - 1 for s in seqs], device="cuda")]
        probs = torch.softmax(model.lm_head(last).float()[:, cids], -1)
        for it, p in zip(part, probs):
            out.append({"kind": "selfreport", "item_id": it.item_id, "report": colours[int(p.argmax())],
                        "p_report": round(float(p.max()), 4)})
    if bank != DEV_BANK:
        rng = random.Random("exp15_thinking_baseline")
        off = [it for it in table if it.group == "offcycle"]
        chosen = rng.sample(off, min(THINKING_N, len(off)))
        prompts = []
        for it in chosen:
            problem = items[it.item_id]["problem"]
            assert problem.endswith(FINAL_Q)
            msgs = [{"role": "user", "content": problem.removesuffix(FINAL_Q) + SELFREPORT_Q}]
            prompts.append(tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True,
                                                         return_dict=True)["input_ids"])
        width = max(map(len, prompts))
        ids = torch.tensor([[0] * (width - len(p)) + p for p in prompts], device="cuda")           # left padding
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device="cuda")
        gen = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=4096,
                             pad_token_id=tokenizer.eos_token_id)
        for it, g in zip(chosen, gen[:, width:]):
            text = tokenizer.decode(g, skip_special_tokens=False)
            text = text.replace("<|endoftext|>", "").replace("<|im_end|>", "")
            final = text.split("</think>")[-1]
            out.append({"kind": "thinking_prompt_only", "item_id": it.item_id, "final": final.strip()[:200],
                        "finished_thinking": "</think>" in text, "named": sorted(S.colours_named(final)),
                        "n_tokens": int((g != tokenizer.eos_token_id).sum())})
    E.atomic_write_text("".join(json.dumps(r) + "\n" for r in out), path)
    print(f"-> {path}")


def selfreport_report(bank: str, tag: str) -> None:
    """Rates at which the self-report names w = T^-1(answer), the true s_{h-1}, or another colour, per group; and
    for the thinking-on baseline the rates of naming the true s_{h-1} and w."""
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    table = {it.item_id: it for it in E.item_table(bank)}
    rows = [json.loads(line) for line in selfreport_path(bank).read_text().splitlines()]
    record = {"bank": bank, "exploratory": True, "selfreport": {}, "thinking_prompt_only": {}}
    groups = {"correct": lambda it: it.group == "correct",
              "offcycle_consistent": lambda it: it.group == "offcycle" and it.p_pred >= S.CONSISTENT_P,
              "offcycle_all": lambda it: it.group == "offcycle",
              "miscount": lambda it: it.group.startswith("miscount")}
    for g, member in groups.items():
        for h in (2, 3):
            sel = [r for r in rows if r["kind"] == "selfreport" and member(table[r["item_id"]])
                   and table[r["item_id"]].h == h]
            if not sel:
                continue
            key = [float(r["report"] == table[r["item_id"]].key) for r in sel]
            true = [float(r["report"] == table[r["item_id"]].true_prev) for r in sel]
            record["selfreport"][f"{g}_h{h}"] = {"names_w": S.summarise([{"x": v} for v in key], "x", None),
                                                 "names_true": S.summarise([{"x": v} for v in true], "x", None)}
    think = [r for r in rows if r["kind"] == "thinking_prompt_only"]
    if think:
        record["thinking_prompt_only"] = {
            "n": len(think),
            "names_true": round(sum(table[r["item_id"]].true_prev in r["named"] for r in think) / len(think), 4),
            "names_w": round(sum(table[r["item_id"]].key in r["named"] for r in think) / len(think), 4)}
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "selfreport.json")
    print(json.dumps(record, indent=1))


# --- NLA checkpoint and best-of-16 ----------------------------------------------------------------------------------
def nla_variant_path(bank: str, name: str):
    return EXP.cache / f"text_reads_{bank}_{name}.jsonl"


@torch.no_grad()
def nla_iter300(bank: str) -> None:
    """The main NLA read (primary cells, layer 42, 4 samples) with AV adapter iter_000300 instead of iter_000800."""
    import exp15_readers as RD
    path = nla_variant_path(bank, "nla_iter300")
    E.require_absent(path)
    metas, hidden = E.load_capture(bank)
    rows = E.text_rows(bank)
    reader = RD.TextReader("nla", nla_adapter="iter_000300")
    out = []
    for start in range(0, len(rows), 48):
        part = rows[start:start + 48]
        vectors = torch.stack([hidden[i, E.CELLS.index(c), 43] for c in PRIMARY_CELLS for i in part]).float()
        samples = reader.read(vectors, 42, seed=start, batch=384)
        for k, i in enumerate(part):
            out.append({"item_id": metas[i]["item_id"],
                        "reads": {f"{c}|42": samples[j * len(part) + k] for j, c in enumerate(PRIMARY_CELLS)}})
        print(f"nla_iter300 {start + len(part)}/{len(rows)}", flush=True)
    E.atomic_write_text("".join(json.dumps(r) + "\n" for r in out), path)


@torch.no_grad()
def best_of_16(bank: str) -> None:
    """16 NLA AV samples per item at the final cell, layer 42 (iter_000800); each sample's explanation is
    reconstructed by the NLA AR and scored by cosine with the true activation. Cached per item."""
    import exp15_readers as RD
    path = nla_variant_path(bank, "nla_best_of_16")
    E.require_absent(path)
    metas, hidden = E.load_capture(bank)
    rows = E.text_rows(bank)
    vectors = torch.stack([hidden[i, E.CELLS.index("final"), 43] for i in rows]).float()
    reader = RD.TextReader("nla")
    samples = reader.read(vectors, 42, seed=16, n_samples=16, batch=384)
    del reader
    E.release()
    ar = RD.NlaAR()
    flat = [RD.explanation_text("nla", s) for row in samples for s in row]
    pred = ar.predict(flat).float().view(len(rows), 16, -1)
    cos = torch.nn.functional.cosine_similarity(pred, vectors[:, None, :], dim=-1)
    out = [{"item_id": metas[i]["item_id"], "samples": samples[k], "ar_cos": [round(float(x), 4) for x in cos[k]]}
           for k, i in enumerate(rows)]
    E.atomic_write_text("".join(json.dumps(r) + "\n" for r in out), path)


def nla_variants_report(bank: str, tag: str) -> None:
    """P1/P2 key AUCs for: NLA iter_000300 vs iter_000800 (same cells and layer); best-of-16 top-1 by AR cosine vs
    a seeded random single sample vs all 16, at the final cell; plus each one's rate of naming any control colour."""
    import exp15_readers as RD
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    table = E.item_table(bank)
    metas, _ = E.load_capture(bank, with_hidden=False)
    colours = metas[0]["colours"]
    index = {c: k for k, c in enumerate(colours)}
    groups = R.read_groups(table, metas)

    def fractions(samples):
        return S.naming_fraction([RD.explanation_text("nla", s) for s in samples], colours, S.colours_named)

    variants = {}
    main = E.load_text_reads(bank, "nla")
    variants["iter800_primary_cells"] = {i: torch.stack([fractions(r[f"{c}|42"]) for c in PRIMARY_CELLS])
                                         for i, r in main.items()}
    if nla_variant_path(bank, "nla_iter300").exists():
        alt = {r["item_id"]: r["reads"] for r in map(json.loads, nla_variant_path(bank, "nla_iter300").read_text()
                                                       .splitlines())}
        variants["iter300_primary_cells"] = {i: torch.stack([fractions(r[f"{c}|42"]) for c in PRIMARY_CELLS])
                                             for i, r in alt.items()}
    if nla_variant_path(bank, "nla_best_of_16").exists():
        rng = random.Random("exp15_best_of_16")
        bo = [json.loads(line) for line in nla_variant_path(bank, "nla_best_of_16").read_text().splitlines()]
        variants["final_top1_by_ar"] = {r["item_id"]: fractions([r["samples"][max(range(16), key=lambda k:
                                        r["ar_cos"][k])]])[None] for r in bo}
        variants["final_random1"] = {r["item_id"]: fractions([r["samples"][rng.randrange(16)]])[None] for r in bo}
        variants["final_all16"] = {r["item_id"]: fractions(r["samples"])[None] for r in bo}
    record = {"bank": bank, "exploratory": True, "variants": {}}
    for name, scores in variants.items():
        rs = S.ReaderScores(name, "fraction", [], scores)
        entry = {}
        for g in ("P1_correct_h2", "P1_correct_h3", "P2_offcycle", "P2_offcycle_p>=0.0"):
            rows = [S.read_metrics(s, it, index) for it in groups[g] if (s := rs.score(it.item_id)) is not None]
            entry[g] = {m: S.summarise(rows, m, 0.5) for m in ("key_auc", "direction")}
            ctl = [float(any(s[index[c]] > 0 for c in it.controls)) for it in groups[g]
                   if (s := rs.score(it.item_id)) is not None]
            entry[g]["names_a_control"] = S.summarise([{"x": v} for v in ctl], "x", None)
        record["variants"][name] = entry
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "nla_variants.json")
    for name, e in record["variants"].items():
        print(name, {g: (v["key_auc"]["mean"], v["key_auc"]["lo"], v["names_a_control"]["mean"]) for g, v in e.items()})


# --- AR-edit recipe and read-then-write repair ----------------------------------------------------------------------
@contextlib.contextmanager
def custom_directions(vector_for):
    """While active, E.run_trials edits each trial along vector_for(trial) (normalised) when that is not None,
    instead of the difference of two rows of the colour tables in directions.pt."""
    import exp15_edits as ED
    table_vector = ED.edit_vector

    def edit_vector(trial, directions, colours, d):
        v = vector_for(trial)
        if v is None:
            return table_vector(trial, directions, colours, d)
        v = v.float()
        return v / v.norm()

    ED.edit_vector = edit_vector
    try:
        yield
    finally:
        ED.edit_vector = table_vector


AR_EDIT_LAYER = 42


def swap_colour_word(text: str, old: str, new: str) -> str:
    """Replace every mention of colour `old` (as the matcher finds it) with `new`, keeping a leading capital."""
    return re.sub(S.COLOUR_PATTERNS[old], lambda m: new.capitalize() if m.group(0)[0].isupper() else new, text,
                  flags=re.IGNORECASE)


def ar_edit_path(bank: str):
    return EXP.cache / f"ar_edit_vectors_{bank}.pt"


@torch.no_grad()
def ar_edit_vectors(bank: str) -> dict:
    """Per correct h=2 item with NLA reads: every AV explanation at the primary cells (layer 42) that names s1 is
    edited s1 -> s1' (the item's key-swap target), both versions are reconstructed by the NLA AR, and the item's
    direction is the mean of AR(edited) - AR(original). Items whose explanations never name s1 get none. Cached."""
    import collections

    import exp15_edits as ED
    import exp15_readers as RD
    path = ar_edit_path(bank)
    if path.exists():
        return torch.load(path, weights_only=False)
    colours = torch.load(E.directions_path(), weights_only=False)["colours"]
    reads = E.load_text_reads(bank, "nla")
    texts, n_items = [], 0                                   # texts: (item_id, original, edited)
    for it, brew in E.swap_items(bank):
        target = ED.swap_targets(it, colours, brew)
        if it.item_id not in reads or target is None:
            continue
        n_items += 1
        s1 = it.path[1]
        for cell in PRIMARY_CELLS:
            for sample in reads[it.item_id][f"{cell}|{AR_EDIT_LAYER}"]:
                text = RD.explanation_text("nla", sample)
                if s1 in S.colours_named(text):
                    texts.append((it.item_id, text, swap_colour_word(text, s1, target["s1_new"])))
    ar = RD.NlaAR()
    original = ar.predict([t[1] for t in texts]).float()
    edited = ar.predict([t[2] for t in texts]).float()
    del ar
    E.release()
    deltas = collections.defaultdict(list)
    for (item_id, _, _), a, b in zip(texts, original, edited):
        deltas[item_id].append(b - a)
    out = {"vectors": {i: torch.stack(v).mean(0) for i, v in deltas.items()},
           "n_explanations": {i: len(v) for i, v in deltas.items()}, "n_items": n_items, "layer": AR_EDIT_LAYER}
    E.atomic_save(out, path)
    return out


def ar_edit(bank: str) -> None:
    """Key swaps (c = s1 -> c' = s1') at layer 42 along each item's AR-edit direction, at every alpha, next to the
    template nla_ar direction and the random null on the same items (same null seeds as P3)."""
    import exp15_edits as ED
    vectors = ar_edit_vectors(bank)["vectors"]
    colours = torch.load(E.directions_path(), weights_only=False)["colours"]
    grid = [(m, AR_EDIT_LAYER, a) for m in ("nla_ar_edit", "nla_ar") for a in ED.ALPHAS]
    trials = [t for it, brew in E.swap_items(bank) if it.item_id in vectors
              for t in ED.swap_trials(it, colours, brew, grid, ("key_swap", "random_null"))]
    with custom_directions(lambda t: vectors[t.item_id] if t.method == "nla_ar_edit" else None):
        E.run_trials(bank, "ar_edit", trials)


def repair_targets(bank: str) -> list:
    """P4's population: consistent off-cycle h=3 errors."""
    return [it for it in E.item_table(bank) if it.h == 3 and it.group == "offcycle" and it.p_pred >= S.CONSISTENT_P]


def decoded_key(rs: S.ReaderScores, it: S.Item, colours: list[str]) -> str | None:
    """The reader's top colour other than the model's answer (T has no fixed points, so the key is never the
    answer); ties broken by a seeded draw. None if the reader has no read of the item or names no colour."""
    import exp15_edits as ED
    s = rs.score(it.item_id)
    if s is None:
        return None
    candidates = [k for k, c in enumerate(colours) if c != it.pred]
    best = max(float(s[k]) for k in candidates)
    if rs.kind == "fraction" and best == 0:
        return None
    top = [colours[k] for k in candidates if float(s[k]) == best]
    return ED.item_rng(it.item_id, f"decode:{rs.name}").choice(top)


def read_then_write(bank: str) -> None:
    """P4 with c = each reader's decoded key instead of w (c' = the true s2, success = gold), for each
    repair-eligible method at its locked setting; variant 'repair_read:<reader>'. Where the decoded key is s2 itself
    the edit would be the zero vector, so no trial is run (the report counts the item as unrepaired)."""
    import exp15_edits as ED
    lock = json.loads(E.write_lock_path().read_text())
    if not lock["repair_methods"]:
        print("K4: no repair-eligible method, so read-then-write is untestable")
        return
    colours = torch.load(E.directions_path(), weights_only=False)["colours"]
    readers = R.reader_scores(bank)
    trials = []
    for it in repair_targets(bank):
        for name, rs in readers.items():
            c = decoded_key(rs, it, colours)
            if c is None or c == it.true_prev:
                continue
            for m in lock["repair_methods"]:
                s = lock["primary"][m]
                trials.append(ED.Trial(it.item_id, f"repair_read:{name}", m, s["layer"], s["alpha"], c, it.true_prev,
                                       it.gold, {"success": it.gold, "unchanged": it.pred}))
    E.run_trials(bank, "read_then_write", trials)


COUNT_METHODS = ("logit", "jlens", "jpp", "template")


def count_directions() -> dict:
    """{method: {layer: [6, d]}} for the count words one..six (exp15_readers.COUNT_WORDS, the count-target read's
    words), built like the colour directions: g * W_U[w] pulled back through J_l, or the template row. Cached."""
    import exp15_edits as ED
    import exp15_readers as RD
    from transformers import AutoTokenizer
    path = EXP.cache / "count_directions.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(E.model_path())
    ids = [tokenizer.encode(" " + w, add_special_tokens=False) for w in RD.COUNT_WORDS]
    assert all(len(t) == 1 for t in ids)
    lenses = RD.TokenLenses([t[0] for t in ids], kinds=("logit", "jlens", "jpp"), device="cpu")
    out = {m: {l: torch.stack([lenses.direction(m, l, k) for k in range(len(ids))]) for l in ED.WRITE_LAYERS}
           for m in ("logit", "jlens", "jpp")}
    template = RD.TemplateLens(list(RD.COUNT_WORDS), device="cpu")
    out["template"] = {l: template.direction(l, slice(0, len(ids))).clone() for l in ED.WRITE_LAYERS}
    E.atomic_save(out, path)
    return out


def count_repair_targets(bank: str) -> list:
    """Consistent miscount errors with a unique implied count k != h (h = 2 or 3)."""
    return [it for it in E.item_table(bank) if it.group == "miscount" and it.h in (2, 3)
            and it.implied_count in S.COUNTS and it.p_pred >= S.CONSISTENT_P]


def count_minus_null(cells: dict, method: str, layer: int, alpha: float) -> list[float]:
    return R.success_minus_null(cells, f"count_{method}", layer, alpha, "count_repair", "count_null")


def mean_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def count_dev_best() -> dict[str, tuple[int, float]]:
    """Each COUNT_METHODS entry's (layer, alpha) maximising dev repair - null (ties: smaller alpha)."""
    import exp15_edits as ED
    dev = R.by_setting(E.load_trials(DEV_BANK, "count_repair"))
    out = {}
    for m in COUNT_METHODS:
        score = {(l, a): mean_or_none(count_minus_null(dev, m, l, a)) for l in ED.WRITE_LAYERS for a in ED.ALPHAS}
        out[m] = max((s for s in score if score[s] is not None), key=lambda s: (score[s], -s[1]))
    return out


def count_repair(bank: str) -> None:
    """Repair by count edits: c = count word k (the implied count) -> c' = count word h, success = gold, plus the
    random null (method 'count_<m>'; c and c' are stored as 'count<k>'). Dev: each COUNT_METHODS direction over the
    P3 layer x alpha grid. Test: each method at its dev-best setting only."""
    import exp15_edits as ED
    dirs = count_directions()
    if bank == DEV_BANK:
        grid = [(m, l, a) for m in COUNT_METHODS for l in ED.WRITE_LAYERS for a in ED.ALPHAS]
    else:
        grid = [(m, l, a) for m, (l, a) in count_dev_best().items()]
    trials = []
    for it in count_repair_targets(bank):
        outcomes = {"success": it.gold, "unchanged": it.pred}
        for m, layer, alpha in grid:
            trials.append(ED.Trial(it.item_id, "count_repair", f"count_{m}", layer, alpha,
                                   f"count{it.implied_count}", f"count{it.h}", it.gold, outcomes))
        for layer, alpha in sorted({(l, a) for _, l, a in grid}):
            trials.append(ED.Trial(it.item_id, "count_null", "random", layer, alpha, None, None, it.gold, outcomes,
                                   seed=ED.item_rng(it.item_id, f"null:{layer}").randrange(2**31)))

    def vector_for(t):
        if not t.method.startswith("count_"):
            return None
        table = dirs[t.method.removeprefix("count_")][t.layer]
        return table[int(t.c_new.removeprefix("count")) - 1] - table[int(t.c.removeprefix("count")) - 1]

    with custom_directions(vector_for):
        E.run_trials(bank, "count_repair", trials)


def outcome_shares(records: dict) -> dict:
    import collections
    counts = collections.Counter(r["outcome"] for r in records.values())
    return {k: round(v / len(records), 4) for k, v in sorted(counts.items())} | {"n": len(records)}


def edits_report(bank: str, tag: str) -> None:
    """AR-edit: per alpha, AR-edit and template nla_ar key swap - null, and AR-edit - template paired on the same items.
    Read-then-write: per repair method and reader, how often the decoded key is w / the true s2 / missing, and the
    repair rate over all targets (unrun items count as unrepaired) minus the repair rate with c = w.
    Count repair: success - null over the grid, and on the test bank at each method's dev-best setting."""
    import exp15_edits as ED
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    record = {"bank": bank, "exploratory": True}
    if E.trials_path(bank, "ar_edit").exists():
        cells = R.by_setting(E.load_trials(bank, "ar_edit"))
        vec = torch.load(ar_edit_path(bank), weights_only=False)
        n_expl = list(vec["n_explanations"].values())
        entry = {"items_with_nla_reads": vec["n_items"], "items_with_vector": len(vec["vectors"]),
                 "mean_explanations_naming_s1": round(sum(n_expl) / max(len(n_expl), 1), 3), "by_alpha": {}}
        for a in ED.ALPHAS:
            edit = cells.get(("key_swap", "nla_ar_edit", AR_EDIT_LAYER, a), {})
            template = cells.get(("key_swap", "nla_ar", AR_EDIT_LAYER, a), {})
            paired = [float(r["outcome"] == "success") - float(template[i]["outcome"] == "success")
                      for i, r in edit.items() if i in template]
            entry["by_alpha"][f"a{a}"] = {
                "ar_edit_minus_null": S.summarise([{"x": v} for v in R.success_minus_null(
                    cells, "nla_ar_edit", AR_EDIT_LAYER, a)], "x", 0.0),
                "template_minus_null": S.summarise([{"x": v} for v in R.success_minus_null(
                    cells, "nla_ar", AR_EDIT_LAYER, a)], "x", 0.0),
                "ar_edit_minus_template": S.summarise([{"x": v} for v in paired], "x", 0.0),
                "ar_edit_outcomes": outcome_shares(edit) if edit else {}}
        record["ar_edit"] = entry
    if E.trials_path(bank, "read_then_write").exists():
        lock = json.loads(E.write_lock_path().read_text())
        colours = torch.load(E.directions_path(), weights_only=False)["colours"]
        rtw = R.by_setting(E.load_trials(bank, "read_then_write"))
        main = R.by_setting(E.load_trials(bank, "repair")) if E.trials_path(bank, "repair").exists() else {}
        readers = R.reader_scores(bank)
        targets = repair_targets(bank)
        record["read_then_write"] = {}
        for m in lock["repair_methods"]:
            layer, alpha = lock["primary"][m]["layer"], lock["primary"][m]["alpha"]
            with_w = main.get(("repair", m, layer, alpha), {})
            entry = {"layer": layer, "alpha": alpha, "n_targets": len(targets),
                     "repair_with_w": S.summarise([{"x": float(r["outcome"] == "success")} for r in with_w.values()],
                                                  "x", None), "readers": {}}
            for name, rs in readers.items():
                recs = rtw.get((f"repair_read:{name}", m, layer, alpha), {})
                rows = []
                for it in targets:
                    c = decoded_key(rs, it, colours)
                    repaired = float(recs[it.item_id]["outcome"] == "success") if it.item_id in recs else 0.0
                    w_ok = float(with_w[it.item_id]["outcome"] == "success") if it.item_id in with_w else None
                    rows.append({"no_decode": float(c is None),
                                 "decoded_is_w": None if c is None else float(c == it.key),
                                 "decoded_is_true": None if c is None else float(c == it.true_prev),
                                 "repaired": repaired, "minus_with_w": None if w_ok is None else repaired - w_ok})
                entry["readers"][name] = {k: S.summarise(rows, k, None) for k in
                                          ("no_decode", "decoded_is_w", "decoded_is_true", "repaired", "minus_with_w")}
            record["read_then_write"][m] = entry
    if E.trials_path(bank, "count_repair").exists():
        cells = R.by_setting(E.load_trials(bank, "count_repair"))
        entry = {"n_targets": len(count_repair_targets(bank)), "grid": {}, "at_dev_best": {}}
        if bank == DEV_BANK:
            for m in COUNT_METHODS:
                entry["grid"][m] = {f"L{l}|a{a}": mean_or_none(count_minus_null(cells, m, l, a))
                                    for l in ED.WRITE_LAYERS for a in ED.ALPHAS}
        for m, (layer, alpha) in count_dev_best().items():
            entry["at_dev_best"][m] = {"layer": layer, "alpha": alpha, "repair_minus_null": S.summarise(
                [{"x": v} for v in count_minus_null(cells, m, layer, alpha)], "x", 0.0)}
        record["count_repair"] = entry
    E.atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "edits_secondary.json")
    print(json.dumps(record, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["count_target", "selfreport", "selfreport_report", "nla_iter300", "best_of_16",
                                   "nla_variants_report", "ar_edit", "read_then_write", "count_repair",
                                   "edits_report"])
    p.add_argument("--bank", choices=E.BANKS, default=DEV_BANK)
    p.add_argument("--tag")
    a = p.parse_args()
    if a.cmd == "count_target":
        count_target(a.bank, a.tag)
    elif a.cmd == "selfreport":
        selfreport(a.bank)
    elif a.cmd == "selfreport_report":
        selfreport_report(a.bank, a.tag)
    elif a.cmd == "nla_iter300":
        nla_iter300(a.bank)
    elif a.cmd == "best_of_16":
        best_of_16(a.bank)
    elif a.cmd == "nla_variants_report":
        nla_variants_report(a.bank, a.tag)
    elif a.cmd == "ar_edit":
        ar_edit(a.bank)
    elif a.cmd == "read_then_write":
        read_then_write(a.bank)
    elif a.cmd == "count_repair":
        count_repair(a.bank)
    elif a.cmd == "edits_report":
        edits_report(a.bank, a.tag)


if __name__ == "__main__":
    main()
