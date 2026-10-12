"""exp15 day-0 check 7 debugging: agu18dec's ptag AR failed retrieval (ret@1 0.48 at L44 against a 0.8 bar; their
val ret@1 is 0.93). Pre-registered budget: at most 3 hours, then ptag_ar is dropped from the write tests.

Hypothesis 1 (read site): their README says 'the read is the final hidden state at the span's last real token', and
the LoRA covers all 64 blocks, so the head reads the backbone's last layer for every tag. exp15_readers.PtagAR reads
the block-L output for tag L instead. Variants, scored on check 7's 200 spans and targets:
  block_L        what check 7 ran: the AR's block-L output
  final_normed   hidden_states[64], after the final RMSNorm (HF's last_hidden_state)
  final_raw      the block-63 output before the final norm (forward hook)

Later rounds (target_space: whitener mean and chat-formatted targets; pred_space: raw-space head, diagonal
whitener) are described in their docstrings.

  python exp15_ptag_debug.py read_site      -> checks_day0/ptag_debug_read_site.json
  python exp15_ptag_debug.py target_space   -> checks_day0/ptag_debug_target_space.json
  python exp15_ptag_debug.py pred_space     -> checks_day0/ptag_debug_pred_space.json
  python exp15_ptag_debug.py v2_directions  if read_site passed: 'ptag_ar_v2' write directions at the fixed read
  python exp15_ptag_debug.py v2_dev_trials  its dev selection grid (before select_writes)
"""
from __future__ import annotations

import argparse
import json
import random

import torch

import exp15 as E
import exp15_readers as RD
from exp15 import PTAG_LAYERS, WHITENER_SPEC

READS = ("block_L", "final_normed", "final_raw")


@torch.no_grad()
def check7_spans_and_targets(n_spans: int = 200):
    """The exact spans and base-model targets of exp15.check_ptag (same seeds)."""
    tokenizer, model = E.load_base()
    rng = random.Random(2)
    docs = E.passage_ids(tokenizer, n_spans, seed=2, min_len=24)
    spans, ends = [], []
    for ids in docs:
        t = rng.randrange(8, len(ids))
        length = min(rng.randint(1, 32), t + 1)
        spans.append(ids[t - length + 1:t + 1])
        ends.append(t)
    targets = {L: [] for L in PTAG_LAYERS}
    for start in range(0, n_spans, 32):
        hs, _ = E.padded_hidden(model, docs[start:start + 32])
        for L in PTAG_LAYERS:
            targets[L] += [hs[L + 1][i, ends[start + i]].float().cpu() for i in range(len(hs[0]))]
    del model, hs
    E.release()
    return spans, {L: torch.stack(v) for L, v in targets.items()}


@torch.no_grad()
def predict(ar: RD.PtagAR, spans: list[list[int]], layer: int, read: str, batch: int = 64) -> torch.Tensor:
    """PtagAR.predict with the read site as a parameter."""
    captured = {}

    def keep_block_output(module, args, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    hook = ar.text.layers[len(ar.text.layers) - 1].register_forward_hook(keep_block_output) \
        if read == "final_raw" else None
    out = []
    try:
        for start in range(0, len(spans), batch):
            rows = [ar.tag_ids[layer] + s for s in spans[start:start + batch]]
            width = max(map(len, rows))
            ids = torch.tensor([r + [0] * (width - len(r)) for r in rows], device="cuda")
            mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device="cuda")
            hs = ar.text(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
            h = {"block_L": hs[layer + 1], "final_normed": hs[-1]}.get(read, captured.get("h"))
            last = h[torch.arange(len(rows)), mask.sum(1) - 1].float()
            normed = torch.nn.functional.layer_norm(last, (last.shape[-1],))
            out.append((normed @ ar.w.T + ar.b).cpu())
    finally:
        if hook is not None:
            hook.remove()
    return torch.cat(out)


def retrieval(pred: torch.Tensor, gold: torch.Tensor) -> dict:
    sim = torch.nn.functional.normalize(pred, dim=-1) @ torch.nn.functional.normalize(gold, dim=-1).T
    n = len(sim)
    return {"ret1": round(float((sim.argmax(1) == torch.arange(n)).float().mean()), 3),
            "diag_cos": round(float(sim.diag().mean()), 3),
            "offdiag_cos": round(float((sim.sum() - sim.diag().sum()) / (n * n - n)), 3)}


def read_site() -> None:
    spans, targets = check7_spans_and_targets()
    whiteners = torch.load(E.whitener_path(), weights_only=False)
    ar = RD.PtagAR()
    record = {"hypothesis": "the AR head reads the backbone's final hidden state for every tag", "n_spans": len(spans),
              "reads": {}}
    for read in READS:
        record["reads"][read] = {}
        for L in PTAG_LAYERS:
            pred = predict(ar, spans, L, read)
            w = RD.Whitener(whiteners["mean"][L], whiteners["cov"][L], WHITENER_SPEC["ridge"])
            record["reads"][read][L] = {"ours_whitened": retrieval(pred, w.white(targets[L])),
                                        "raw_centered": retrieval(pred, targets[L] - whiteners["mean"][L])}
        print(read, {L: v["ours_whitened"]["ret1"] for L, v in record["reads"][read].items()}, flush=True)
    best = max(READS, key=lambda r: record["reads"][r][44]["ours_whitened"]["ret1"])
    record["best_read"] = best
    record["pass"] = record["reads"][best][44]["ours_whitened"]["ret1"] >= 0.8
    E.write_check("ptag_debug_read_site", record)


@torch.no_grad()
def chat_spans_and_targets(n_spans: int = 200):
    """Hypothesis 3 (training distribution): the AR was trained on assistant-turn activations of chat transcripts
    ('ar.asst.ptag', pairs 'ml_pairs_iolens_chat'). The check-7 passages as an assistant reply in the chat template;
    the target is a seeded position inside the reply, the span the up-to-32 tokens ending there."""
    tokenizer, model = E.load_base()
    rng = random.Random(3)
    docs, spans, ends = [], [], []
    for ids in E.passage_ids(tokenizer, n_spans, seed=2, min_len=24):
        reply = tokenizer.decode(ids)
        chat = tokenizer.apply_chat_template([{"role": "user", "content": "Write a short passage."},
                                              {"role": "assistant", "content": reply}], tokenize=True,
                                             return_dict=True)["input_ids"]
        start = len(chat) - len(ids) - 2                     # roughly where the reply begins
        t = rng.randrange(start + 8, len(chat) - 2)
        length = min(rng.randint(1, 32), t + 1)
        docs.append(chat)
        spans.append(chat[t - length + 1:t + 1])
        ends.append(t)
    targets = {L: [] for L in PTAG_LAYERS}
    for start in range(0, n_spans, 32):
        hs, _ = E.padded_hidden(model, docs[start:start + 32])
        for L in PTAG_LAYERS:
            targets[L] += [hs[L + 1][i, ends[start + i]].float().cpu() for i in range(len(hs[0]))]
    del model, hs
    E.release()
    return spans, {L: torch.stack(v) for L, v in targets.items()}


def target_space() -> None:
    """Hypotheses 2 and 3: (2) our natural-text whitener is not theirs (pooled chat statistics), which shifts the
    mean and per-direction scale; scored with each side centred on its own batch mean as well. (3) chat-formatted
    assistant-turn targets instead of bare passages."""
    sets = {"natural": check7_spans_and_targets(), "chat_assistant": chat_spans_and_targets()}
    whiteners = torch.load(E.whitener_path(), weights_only=False)
    ar = RD.PtagAR()
    record = {"hypotheses": target_space.__doc__, "sets": {}}
    for name, (spans, targets) in sets.items():
        record["sets"][name] = {}
        for L in PTAG_LAYERS:
            pred = predict(ar, spans, L, "block_L")
            w = RD.Whitener(whiteners["mean"][L], whiteners["cov"][L], WHITENER_SPEC["ridge"])
            white = w.white(targets[L])
            record["sets"][name][L] = {
                "ours_whitened": retrieval(pred, white),
                "ours_whitened_batch_centred": retrieval(pred - pred.mean(0), white - white.mean(0)),
                "raw_batch_centred": retrieval(pred - pred.mean(0), targets[L] - targets[L].mean(0))}
        print(name, {L: {k: v["ret1"] for k, v in e.items()} for L, e in record["sets"][name].items()
                     if L in (20, 44, 60)}, flush=True)
    best = max(((s, sp) for s in record["sets"] for sp in record["sets"][s][44]),
               key=lambda x: record["sets"][x[0]][44][x[1]]["ret1"])
    record["best_at_44"] = {"set": best[0], "space": best[1], **record["sets"][best[0]][44][best[1]]}
    record["pass"] = record["best_at_44"]["ret1"] >= 0.8
    E.write_check("ptag_debug_target_space", record)


def pool_ret1(pred: torch.Tensor, gold: torch.Tensor, pool: int, draws: int = 500, seed: int = 0) -> float:
    """Mean retrieval@1 within random candidate pools of `pool` spans (context only; the bar is among all 200)."""
    sim = torch.nn.functional.normalize(pred, dim=-1) @ torch.nn.functional.normalize(gold, dim=-1).T
    g = torch.Generator().manual_seed(seed)
    hits = []
    for _ in range(draws):
        idx = torch.randperm(len(sim), generator=g)[:pool]
        sub = sim[idx][:, idx]
        hits.append(float((sub.argmax(1) == torch.arange(pool)).float().mean()))
    return round(sum(hits) / draws, 3)


def scaled_fve(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """FVE of the best scalar multiple of pred: 1 - min_s ||s pred - g||^2 / ||g - mean(g)||^2."""
    s = float((pred * gold).sum() / (pred * pred).sum())
    return round(1 - float(((s * pred - gold) ** 2).sum() / ((gold - gold.mean(0)) ** 2).sum()), 3)


def pred_space() -> None:
    """Hypotheses 4-5 on check 7's natural spans (block-L read): (4) the head predicts raw activations and only
    the training loss was whitened, so the prediction must be whitened as well; (5) their whitener is diagonal
    (per-dimension standardisation with the same ridge rule), not ZCA. Reported for context, not for the bar:
    retrieval within pools of 24 (their eval batch size is unknown), the norm ratio of prediction to whitened target
    and the scalar-optimal FVE (published L44 FVE 0.22 implies a diagonal cosine near 0.47)."""
    spans, targets = check7_spans_and_targets()
    whiteners = torch.load(E.whitener_path(), weights_only=False)
    ar = RD.PtagAR()
    record = {"hypotheses": pred_space.__doc__, "layers": {}}
    for L in PTAG_LAYERS:
        pred = predict(ar, spans, L, "block_L")
        mean, cov = whiteners["mean"][L], whiteners["cov"][L]
        zca = RD.Whitener(mean, cov, WHITENER_SPEC["ridge"])
        var = cov.diagonal()
        diag_scale = (var + WHITENER_SPEC["ridge"] * var.mean()).rsqrt()
        gold_zca, gold_diag = zca.white(targets[L]), (targets[L] - mean) * diag_scale
        record["layers"][L] = {
            "ours_whitened": retrieval(pred, gold_zca),
            "pred_whitened_too": retrieval(zca.white(pred), gold_zca),
            "diagonal_whitener": retrieval(pred, gold_diag),
            "context": {"ret1_pool24_ours_whitened": pool_ret1(pred, gold_zca, 24),
                        "ret1_pool24_diagonal": pool_ret1(pred, gold_diag, 24),
                        "norm_ratio_pred_to_whitened_target": round(float(pred.norm(dim=-1).mean()
                                                                          / gold_zca.norm(dim=-1).mean()), 3),
                        "scaled_fve_ours_whitened": scaled_fve(pred, gold_zca),
                        "scaled_fve_diagonal": scaled_fve(pred, gold_diag)}}
        e = record["layers"][L]
        print(L, {k: e[k]["ret1"] for k in ("ours_whitened", "pred_whitened_too", "diagonal_whitener")},
              e["context"], flush=True)
    spaces = ("ours_whitened", "pred_whitened_too", "diagonal_whitener")
    best = max(spaces, key=lambda s: record["layers"][44][s]["ret1"])
    record["best_at_44"] = {"space": best, **record["layers"][44][best]}
    record["pass"] = record["best_at_44"]["ret1"] >= 0.8
    E.write_check("ptag_debug_pred_space", record)


PTAG_V2 = "ptag_ar_v2"      # ptag AR directions with the corrected read site (only if read_site passes)


def debug_result() -> dict:
    return json.loads((E.check_dir() / "ptag_debug_read_site.json").read_text())


@torch.no_grad()
def build_v2_directions() -> None:
    """directions.pt gains PTAG_V2: as ptag_ar (mean AR output over PTAG_TEMPLATES, mapped back with our S^1/2),
    but read at the debug's best read site. Other entries are kept."""
    import exp15_edits as ED
    from transformers import AutoTokenizer
    result = debug_result()
    assert result["pass"], "read_site did not pass; ptag_ar stays out of the write tests"
    path = E.directions_path()
    out = torch.load(path, weights_only=False)
    assert PTAG_V2 not in out
    tokenizer = AutoTokenizer.from_pretrained(E.model_path())
    whiteners = torch.load(E.whitener_path(), weights_only=False)
    spans = [tokenizer.encode(t.format(c=c), add_special_tokens=False) for c in out["colours"]
             for t in ED.PTAG_TEMPLATES]
    ar = RD.PtagAR()
    out[PTAG_V2] = {}
    for L in ED.METHOD_LAYERS["ptag_ar"]:
        pred = predict(ar, spans, L, result["best_read"]).view(len(out["colours"]), len(ED.PTAG_TEMPLATES), -1)
        w = RD.Whitener(whiteners["mean"][L], whiteners["cov"][L], WHITENER_SPEC["ridge"])
        out[PTAG_V2][L] = w.unwhiten_delta(pred.mean(1))
    del ar
    E.release()
    E.atomic_save(out, path.with_name(path.name + ".new"))
    path.with_name(path.name + ".new").replace(path)
    print({k: sorted(v) for k, v in out.items() if isinstance(v, dict)})


def dev_v2_trials() -> None:
    """The dev selection grid for PTAG_V2 (key and answer swaps; the random nulls at the same layer and alpha are
    already in trials_<dev>_select), appended to the same phase."""
    import exp15_edits as ED
    colours = torch.load(E.directions_path(), weights_only=False)["colours"]
    grid = [(PTAG_V2, l, a) for l in ED.METHOD_LAYERS["ptag_ar"] for a in ED.ALPHAS]
    trials = [t for it, brew in E.swap_items(E.DEV_BANK)
              for t in ED.swap_trials(it, colours, brew, grid, ("key_swap", "answer_swap"))]
    E.run_trials(E.DEV_BANK, "select", trials)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["read_site", "target_space", "pred_space", "v2_directions", "v2_dev_trials"])
    a = p.parse_args()
    if a.cmd == "read_site":
        read_site()
    elif a.cmd == "target_space":
        target_space()
    elif a.cmd == "pred_space":
        pred_space()
    elif a.cmd == "v2_directions":
        build_v2_directions()
    elif a.cmd == "v2_dev_trials":
        dev_v2_trials()


if __name__ == "__main__":
    main()
