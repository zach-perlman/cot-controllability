"""exp11: what is in Qwen3.6-27B's residual stream when it gets a no-CoT brew item wrong (manifest:
results/exp11_wrong_intermediates/manifest.json).

  banks             generate the dev and test banks (nocot-bench's brew generator, exp10's configuration, h in 1..3)
  requests --bank   the C0 rows of a bank (exp10's rendering: one h=2 shot, thinking off, 'Answer:' prefilled)
  capture  --bank   one bf16 forward pass per row: the colour answer distribution, and the residual stream at the
                    start colour, each stir, the end of the question and the final position (cache/exp11, gitignored)
"""
from __future__ import annotations

import argparse
import collections
import json

import cc_config as cfg
import cc_exp10 as E10
import exp10_conditions as C
import exp11_brew as B

EXP = cfg.Exp("exp11_wrong_intermediates", "exp11")
H_VALUES = (1, 2, 3)
BANKS = {"dev": {"seed": "exp11_dev_20261008", "per_h": 60}, "test": {"seed": "exp11_test_20261008", "per_h": 250}}
FMT_BANKS = {   # manifest deviations 1 and 2
    "fmt_dev": {"seed": "exp11_fmt_dev_20261008", "per_h": 60, "formats": B.FORMATS, "h_values": H_VALUES},
    "fmt_same_dev": {"seed": "exp11_fmt_same_dev_20261008", "per_h": 60, "formats": ("three_same",),
                     "h_values": (2, 3)},
}
ALL_BANKS = [*BANKS, *FMT_BANKS]


def bank_path(bank: str):
    return EXP.cache / f"items_{bank}.jsonl"


def requests_path(bank: str):
    return EXP.cache / f"requests_{bank}.jsonl"


def capture_paths(bank: str):
    return EXP.cache / f"capture_{bank}.pt", EXP.cache / f"capture_{bank}.jsonl"


def load_bank(bank: str) -> list[dict]:
    return [json.loads(l) for l in bank_path(bank).open()]


# --- Banks ----------------------------------------------------------------------------------------------------------
def bank_config(per_h: int):
    """exp10's brew configuration (cc_exp10.bank_config) with exp11's depths."""
    brew, _ = E10.nocot_import()
    n = per_h * len(H_VALUES)
    return brew.Config(rungs=tuple((f"brew:h{h}", ((h, per_h),)) for h in H_VALUES), n_shots=1, shot_h=2,
                       saturating=False, bank_gold_cap=max(7, round(n * 1.4 / len(brew.COLORS))),
                       rung_gold_cap=max(2, round(per_h * 1.5 / len(brew.COLORS))), chance=None)


def write_banks() -> None:
    brew, _ = E10.nocot_import()
    if any(bank_path(b).exists() for b in BANKS):
        raise SystemExit("exp11 banks exist; banks are generated once")
    rows = {b: [E10.to_row(it, f"exp11_{b}") for it in brew.generate(bank_config(spec["per_h"]), spec["seed"])]
            for b, spec in BANKS.items()}
    texts = collections.Counter(r["problem"] for bank_rows in rows.values() for r in bank_rows)
    texts.update(r["problem"] for b in ("dev", "test", "shipped") for r in E10.load_bank(b))
    if max(texts.values()) > 1:
        raise SystemExit("a problem text occurs twice across exp11's and exp10's banks")
    record = {"nocot_commit": E10.NOCOT_COMMIT, "h_values": H_VALUES, "banks": {}}
    for b, bank_rows in rows.items():
        failed = [r["item_id"] for r in bank_rows if not all(E10.check_item(r).values())]
        if failed:
            raise SystemExit(f"{b}: {len(failed)} items fail exp10's checks: {failed[:5]}")
        assert all(B.parse(r["problem"]).path()[-1] == r["answer"] for r in bank_rows)
        evals = [r for r in bank_rows if r["split"] == "eval"]
        record["banks"][b] = {**BANKS[b], "n_eval": len(evals), "n_shot": len(bank_rows) - len(evals),
                              "per_h": dict(collections.Counter(r["h"] for r in evals)),
                              "gold_counts": dict(collections.Counter(r["answer"] for r in evals).most_common()),
                              "content_key": cfg.content_key(bank_rows)}
        EXP.cache.mkdir(parents=True, exist_ok=True)
        bank_path(b).write_text("".join(json.dumps(r) + "\n" for r in bank_rows))
        print(f"{b}: {len(evals)} eval items + {len(bank_rows) - len(evals)} shot -> {bank_path(b)}")
    EXP.results.mkdir(parents=True, exist_ok=True)
    (EXP.results / "banks.json").write_text(json.dumps(record, indent=2) + "\n")


def write_fmt_bank(name: str) -> None:
    """A format-pilot bank: formats x h_values x per_h eval items plus one h=2 shot per format, nocot-bench's
    colours, ingredients and instruction; every gold re-derived by the independent text parser."""
    import random
    brew, _ = E10.nocot_import()
    spec = FMT_BANKS[name]
    path = bank_path(name)
    if path.exists():
        raise SystemExit(f"{path} exists; banks are generated once")
    rows = []
    for fmt in spec["formats"]:
        rng = random.Random(f"{spec['seed']}:{fmt}")
        cells = [("shot", 2, 0)] + [("eval", h, k) for h in spec["h_values"] for k in range(spec["per_h"])]
        for split, h, k in cells:
            gen = B.generate(rng, h, B.FORMAT_SPEC[fmt][0], brew.COLORS, brew.INGREDIENTS)
            problem = B.render(gen, fmt, brew.COLORS)
            assert B.parse(problem).path()[-1] == gen["answer"]
            rows.append({"item_id": f"{name}:{fmt}:{split}:h{h}:{k}", "bank": name, "format": fmt,
                         "split": split, "h": h, "problem": problem, "answer": gen["answer"],
                         "instruction": brew.INSTRUCTION})
    texts = collections.Counter(r["problem"] for r in rows)
    texts.update(r["problem"] for b in ALL_BANKS if b != name and bank_path(b).exists() for r in load_bank(b))
    if max(texts.values()) > 1:
        raise SystemExit("a problem text repeats")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = json.loads((EXP.results / "banks.json").read_text())
    record["banks"][name] = {**spec, "n_eval": sum(r["split"] == "eval" for r in rows),
                             "content_key": cfg.content_key(rows)}
    (EXP.results / "banks.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(rows)} rows -> {path}")


# --- Requests -------------------------------------------------------------------------------------------------------
def write_requests(bank: str) -> None:
    path = requests_path(bank)
    if path.exists():
        raise SystemExit(f"{path} exists; request files are fixed once written")
    items = load_bank(bank)
    shots = {r.get("format"): r for r in items if r["split"] == "shot"}
    rows = []
    for it in (r for r in items if r["split"] == "eval"):
        shot = shots[it.get("format")]
        row = {"item_id": it["item_id"], "bank": bank, "h": it["h"], "source": "brew", "exp10_render": True,
               "seed": 0, **({"format": it["format"]} if "format" in it else {}), **C.fields("C0", it, shot)}
        row["request_id"] = cfg.content_key(row)
        rows.append(row)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with (EXP.results / "requests.jsonl").open("a") as f:
        f.write(json.dumps({"bank": bank, "n_rows": len(rows), "file": str(path.relative_to(cfg.REPO_ROOT)),
                            "content_key": cfg.content_key([r["request_id"] for r in rows])}) + "\n")
    print(f"{len(rows)} rows -> {path}")


# --- Capture --------------------------------------------------------------------------------------------------------
def token_positions(tokenizer, text: str, brew: B.Brew) -> dict[str, int]:
    """Token index of the last token of each named word in the FINAL problem of the rendered prompt (the shot's
    problem comes earlier and is skipped)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def token_ending_at(char_end: int) -> int:
        return next(i for i, (a, b) in enumerate(offsets) if a < char_end <= b)

    def word_token(word: str, at: int) -> int:
        i = token_ending_at(at + len(word))
        piece = text[offsets[i][0]:offsets[i][1]].strip()
        assert piece and word.endswith(piece), f"token {piece!r} does not end {word!r}"
        return i

    problem = text.rindex("Problem: ")
    sentence = text.index("The potion starts out ", problem)
    out = {"start": word_token(brew.start, sentence + len("The potion starts out "))}
    if B.SINGLE in brew.stirs:
        count = B.COUNT_WORDS[len(brew.stirs)]
        out["stirs"] = word_token(count, text.index(f"You stir it {count}.", problem) + len("You stir it "))
    else:
        cursor = text.index("one at a time: ", problem) + len("one at a time: ")
        for k, ingredient in enumerate(brew.stirs, 1):
            at = text.index(ingredient, cursor)
            out[f"stir{k}"] = word_token(ingredient, at)
            cursor = at + len(ingredient)
    out["question_end"] = token_ending_at(text.index(B.QUESTION, problem) + len(B.QUESTION))
    return out


def capture(bank: str) -> None:
    import torch
    import cc_exp10_render as R
    import exp11_model as M
    pt_path, meta_path = capture_paths(bank)
    if pt_path.exists() or meta_path.exists():
        raise SystemExit(f"{pt_path} exists; captures are written once")
    tokenizer, model = M.load()
    items = {r["item_id"]: r for r in load_bank(bank)}
    rows = [json.loads(l) for l in requests_path(bank).open()]
    colours = sorted(B.parse(items[rows[0]["item_id"]]["problem"]).table)
    cids = M.colour_ids(tokenizer, colours)
    hidden_all, metas = [], []
    for n, row in enumerate(rows):
        brew = B.parse(items[row["item_id"]]["problem"])
        text = R.prompt_text(tokenizer, M.FAMILY, row)
        ids = R.prompt_ids(tokenizer, M.FAMILY, row)
        if tokenizer.encode(text, add_special_tokens=False) != ids[:len(ids) - len(
                tokenizer.encode(row["response_prefix"], add_special_tokens=False))]:
            raise RuntimeError(f"{row['item_id']}: the rendered text does not tokenize to the prompt's prefix")
        named = {**token_positions(tokenizer, text, brew), "final": len(ids) - 1}
        hidden, logits = M.forward(model, ids, list(named.values()))
        probs_all = torch.softmax(logits, -1)
        colour_logits = logits[list(cids.values())]
        hidden_all.append(hidden)
        metas.append({"item_id": row["item_id"], "h": row["h"], "request_id": row["request_id"],
                      "n_tokens": len(ids), "positions": named, "colours": colours,
                      "colour_logits": [round(float(x), 4) for x in colour_logits],
                      "colour_mass": round(float(probs_all[list(cids.values())].sum()), 6),
                      "top_token": tokenizer.decode(int(logits.argmax()))})
        if n % 50 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    torch.save({"item_ids": [m["item_id"] for m in metas], "hidden": hidden_all,
                "position_names": [list(m["positions"]) for m in metas]}, pt_path)
    meta_path.write_text("".join(json.dumps(m) + "\n" for m in metas))
    print(f"{len(metas)} items -> {pt_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["banks", "fmt_bank", "requests", "capture"])
    p.add_argument("--bank", choices=ALL_BANKS)
    a = p.parse_args()
    if a.cmd == "banks":
        write_banks()
    elif a.cmd == "fmt_bank":
        write_fmt_bank(a.bank)
    elif a.cmd == "requests":
        write_requests(a.bank)
    else:
        capture(a.bank)


if __name__ == "__main__":
    main()
