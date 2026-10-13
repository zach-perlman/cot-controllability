"""exp17 positive-control follow-up: is the factual intermediate swap just writing the answer?

On Qwen, the J++ difference vector of the intermediate pair (v(swap_to) - v(intermediate)) has cosine ~0.4 with the
answer pair's (v(swap_answer) - v(answer)), and more on the items where the swap succeeds (0.51 vs 0.24 at layer 36).
So part of the swap's effect may be a direct answer write. Per band x alpha of the positive control, on the same
prompts and window (exp17.posctl):
  inter          the intermediate swap again (the reference for the matched arm; reproduces exp17's)
  inter_orth     the swap along the intermediate pair with the answer pair's span projected out, per layer and item
  inter_orth_m   the same at inter's recorded edit norms
  inter_clampans the intermediate swap with the answer pair's coordinates held at their clean values
-> cache/exp17/<model>/trials_posctl_orth.jsonl, and the overlap per item -> results/.../<model>/posctl_overlap.json

  EXP17_MODEL=qwen|gemma python exp17_orth.py
"""
from __future__ import annotations

import json

import torch

import exp17 as E


def targets_and_families():
    """exp17.posctl's targets, re-indexed so that pair 'pt' = rows (2k, 2k + 1) of per-target families:
    int (intermediate, swap_to), int_orth (the same with span{answer, swap_answer} removed), ans (answer, swap_answer)."""
    tokenizer = E.load_tokenizer()
    items = json.loads(E.PROBE_SWAP.read_text())["items"]
    assert E.sha256(E.PROBE_SWAP) == E.PROBE_SWAP_SOURCE["sha256"]
    targets, heads = [], []
    for it in items:
        toks = {k: tokenizer.encode(" " + it[k].strip(), add_special_tokens=False)[0]
                for k in ("intermediate", "swap_to", "answer", "swap_answer")}
        if toks["intermediate"] == toks["swap_to"] or toks["answer"] == toks["swap_answer"]:
            continue
        ids = tokenizer(it["prompt"].rstrip(" "), add_special_tokens=True)["input_ids"]
        start = 1 if ids[0] in tokenizer.all_special_ids else 0
        k = len(targets)
        targets.append(E.Target(it["name"], ids, start, pairs={"pt": (2 * k, 2 * k + 1)},
                                tokens={"answer": toks["answer"], "swap_answer": toks["swap_answer"]}))
        heads.append(toks)
    jac = E.jpp_jacobians()
    gain, w_u = E.gain_and_unembed()
    inter_ids = torch.tensor([t for h in heads for t in (h["intermediate"], h["swap_to"])])
    ans_ids = torch.tensor([t for h in heads for t in (h["answer"], h["swap_answer"])])
    fams = {"int": {}, "int_orth": {}, "ans": {}}
    overlap = {}
    for l in E.all_band_layers(jac):
        v_int = E.jpp_vectors(jac, l, gain, w_u[inter_ids].to("cuda")).float()      # [2n, d]
        v_ans = E.jpp_vectors(jac, l, gain, w_u[ans_ids].to("cuda")).float()
        n = len(targets)
        q, _ = torch.linalg.qr(v_ans.view(n, 2, -1).transpose(1, 2))                    # [n, d, 2] orthonormal
        vi = v_int.view(n, 2, -1)
        orth = vi - (vi @ q) @ q.transpose(1, 2)
        fams["int"][l], fams["ans"][l], fams["int_orth"][l] = v_int, v_ans, orth.reshape(2 * n, -1)
        di, da = vi[:, 1] - vi[:, 0], v_ans.view(n, 2, -1)[:, 1] - v_ans.view(n, 2, -1)[:, 0]
        overlap[l] = {"cos_diff": [round(float(c), 4) for c in torch.nn.functional.cosine_similarity(di, da, -1)],
                      "kept_norm_fraction": [round(float(x), 4) for x in
                                             (orth[:, 1] - orth[:, 0]).norm(dim=-1) / di.norm(dim=-1)]}
    return targets, fams, jac, overlap


def arms() -> list[E.Arm]:
    out = []
    for band, (lo, hi) in E.SPEC.bands.items():
        if lo == hi:
            continue
        for a in E.POSCTL_ALPHAS:
            ref = f"inter|{band}|{a:g}"
            out += [E.Arm(ref, "swap", "int", None, band, a, pair="pt"),
                    E.Arm(f"inter_orth|{band}|{a:g}", "swap", "int_orth", None, band, a, pair="pt"),
                    E.Arm(f"inter_orth_m|{band}|{a:g}", "matched", "int_orth", ref, band, a, pair="pt"),
                    E.Arm(f"inter_clampans|{band}|{a:g}", "swap", "int", None, band, a, pair="pt", clamp="ans")]
    return out


def main() -> None:
    targets, fams, jac, overlap = targets_and_families()
    path = E.RESULTS / "posctl_overlap.json"
    if not path.exists():
        E.write_once(path, {"items": [t.item_id for t in targets], "by_layer": overlap})

    def score(t, logits, pair):
        top = int(logits.argmax())
        return {"top": top, "success": top == t.tokens["swap_answer"], "kept": top == t.tokens["answer"]}

    E.run_arms("posctl_orth", targets, arms(), fams, jac, score, batch=30)


if __name__ == "__main__":
    main()
