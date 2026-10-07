"""exp10 depth stage: the no-CoT answer log-probability, a continuous reading of the depth a model reaches without
reasoning. Accuracy at chance (or at ceiling) cannot show a gain of less than one step; log P(gold) can, so this is
the sensitive test of the loop dials (Ouro's passes, Huginn's recurrent steps) and of looped vs non-looped pairs.

Rows: the depth stage's direct rows, C0 and F100..F8000 (exp10_task_conditions.fields, as cc_exp10_tasks.build_rows
writes them), on every eval item of the bank: no example turns are involved, so every model scores the same items.
Huginn has no reasoning tokens to hold the dots, so it scores C0 only.
Score: log P(gold | prompt ending in "Answer:"), summed over the gold's tokens, for both spellings " <gold>" and
"<gold>" (their first tokens differ, so P(either) is the sum). No end token: a longer number that starts with the
gold also counts, the same way in every condition and at every loop count.
Engines: vLLM prompt logprobs (cc_generate.load_llm with the spec's vllm_args, which carry the loop overrides);
Huginn: a transformers forward pass with the spec's num_steps.

  /venv/main/bin/python scripts/exp10_answer_logprob.py requests --model M --bank dev
  VLLM_VENV=... scripts/vllm_python.sh scripts/exp10_answer_logprob.py score --model M --bank dev   (Huginn: /venv/main)
Output: cache/exp10/<task>/logprob/<model>__<bank>__<key>.jsonl (content-addressed; an existing output is kept).
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
from pathlib import Path

import cc_config as cfg
import cc_exp10_render as render

TASKS = ("chain", "arithmetic")
NO_FILLER_FAMILIES = {"huginn"}


def logprob_dir(task: str) -> Path:
    return cfg.REPO_ROOT / "cache" / "exp10" / task / "logprob"


def requests_path(task: str, bank: str, model: str) -> Path:
    return logprob_dir(task) / f"requests_{bank}_{model}.jsonl"


# --- Requests (main venv) -------------------------------------------------------------------------------------------
def build_rows(task: str, bank: str, model: str) -> list[dict]:
    import cc_exp10_tasks as X
    import exp10_conditions as C
    import exp10_task_conditions as T
    tokenizer = cfg.load_tokenizer(model)
    family = cfg.ALL_MODELS[model]["family"]
    conditions = ["C0"] if family in NO_FILLER_FAMILIES else ["C0", *T.FILLER]
    per_dot = len(tokenizer.encode(C.dots(1000), add_special_tokens=False)) / 1000
    dose_dots = {f: max(1, round(n / per_dot)) for f, n in T.FILLER.items()}
    demo = X.shots(task, bank)
    rows = []
    for condition in conditions:
        for it in X.evals(task, bank):
            row = {"item_id": it["item_id"], "task": task, "bank": bank, "h": it["h"], "gold": str(it["answer"]),
                   "exp10_render": True, "seed": cfg.rollout_seed(it["item_id"], "exp10", 0),
                   **T.fields(task, condition, it, demo, (), dose_dots.get(condition))}
            row["request_id"] = cfg.content_key(row)
            rows.append(row)
    return rows


def write_requests(task: str, bank: str, model: str) -> None:
    path = requests_path(task, bank, model)
    if path.exists():
        print(f"{path} exists; request files are fixed once written")
        return
    rows = build_rows(task, bank, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{path}: {len(rows)} rows")


# --- Scoring (the model's engine venv) ------------------------------------------------------------------------------
def sequences(tokenizer, family: str, row: dict) -> list[tuple[list[int], int]]:
    """(prompt + target ids, number of target tokens) for each spelling of the gold."""
    prompt = render.prompt_ids(tokenizer, family, row)
    out = []
    for spelling in (" " + row["gold"], row["gold"]):
        target = tokenizer.encode(spelling, add_special_tokens=False)
        out.append((prompt + target, len(target)))
    return out


def score_vllm(model: str, seqs: list[tuple[list[int], int]]) -> list[float]:
    import cc_generate as gen
    from vllm import SamplingParams
    spec = cfg.ALL_MODELS[model]
    llm = gen.load_llm(cfg.model_dir(model), spec.get("gpu_memory_utilization", 0.9), cfg.VLLM_MAX_MODEL_LEN,
                       spec.get("vllm_args", {}))
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids, _ in seqs], params)
    scores = []
    for (ids, n_target), out in zip(seqs, outputs):
        plp = out.prompt_logprobs
        scores.append(sum(plp[i][ids[i]].logprob for i in range(len(ids) - n_target, len(ids))))
    return scores


def score_huginn(model: str, seqs: list[tuple[list[int], int]]) -> list[float]:
    import torch
    from transformers import AutoModelForCausalLM
    spec = cfg.ALL_MODELS[model]
    net = AutoModelForCausalLM.from_pretrained(cfg.model_dir(model), torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).to("cuda").eval()
    scores = []
    with torch.no_grad():
        for ids, n_target in seqs:
            x = torch.tensor([ids], device="cuda")
            logits = net(input_ids=x, num_steps=spec["num_steps"]).logits[0].float()
            logp = torch.log_softmax(logits[:-1], dim=-1)  # position i predicts token i + 1
            scores.append(sum(logp[i - 1, ids[i]].item() for i in range(len(ids) - n_target, len(ids))))
    return scores


def output_path(task: str, bank: str, model: str, rows: list[dict]) -> Path:
    material = {"rows": [r["request_id"] for r in rows], "spec": cfg.ALL_MODELS[model],
                "code": [inspect.getsource(f) for f in (sequences, score_vllm, score_huginn, render.prompt_text,
                                                        render.suffix_pieces, render.prompt_ids)]}
    return logprob_dir(task) / f"{model}__{bank}__{cfg.content_key(material)}.jsonl"


def score(task: str, bank: str, model: str) -> None:
    rows = [json.loads(l) for l in requests_path(task, bank, model).open()]
    out = output_path(task, bank, model, rows)
    if out.exists():
        print(f"{out} exists")
        return
    family = cfg.ALL_MODELS[model]["family"]
    tokenizer = cfg.load_tokenizer(model)
    seqs = [s for r in rows for s in sequences(tokenizer, family, r)]
    flat = (score_huginn if family == "huginn" else score_vllm)(model, seqs)
    with out.open("w") as f:
        for k, r in enumerate(rows):
            space, bare = flat[2 * k], flat[2 * k + 1]
            f.write(json.dumps({"request_id": r["request_id"], "item_id": r["item_id"], "model": model,
                                "task": task, "bank": bank, "h": r["h"], "condition": r["condition"],
                                "logp_space": space, "logp_bare": bare,
                                "logp": max(space, bare) + math.log1p(math.exp(-abs(space - bare)))}) + "\n")
    print(f"wrote {out}")


def load(task: str, bank: str, model: str) -> list[dict]:
    paths = glob.glob(str(logprob_dir(task) / f"{model}__{bank}__*.jsonl"))
    if len(paths) != 1:
        raise SystemExit(f"{task} {bank} {model}: expected one logprob file, found {len(paths)}")
    return [json.loads(l) for l in open(paths[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["requests", "score"])
    parser.add_argument("--model", required=True, choices=list(cfg.ALL_MODELS))
    parser.add_argument("--bank", required=True, choices=["dev", "test"])
    parser.add_argument("--task", choices=TASKS, nargs="+", default=list(TASKS))
    args = parser.parse_args()
    for task in args.task:
        (write_requests if args.command == "requests" else score)(task, args.bank, args.model)


if __name__ == "__main__":
    main()
