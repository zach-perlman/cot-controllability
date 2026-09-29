"""Two-phase vLLM generation for exp01 / exp02 (and the judge, which imports generate_two_phase).

Phase 1 samples the reasoning until </think> (or im_end), capped per request at
min(REASONING_CAP_TOKENS, VLLM_MAX_MODEL_LEN - prompt - answer cap - forced close). If the cap is hit, "\\n</think>\\n\\n"
is appended by us (status "truncated"). Phase 2 continues from prompt + phase-1 tokens for at most ANSWER_CAP_TOKENS.
Everything is kept as token ids, so the reasoning the grader sees is exactly what the model wrote between <think> and
</think>.

Output: cache/<exp>/generations/<model>__<key>.jsonl, key = content hash of (model spec, sampling, caps, requests),
written in chunks under <...>.parts/ (resumable) and joined when complete. An existing output is never rewritten.

Run (vLLM venv):
  scripts/vllm_python.sh scripts/cc_generate.py --exp exp02 --model Qwen3-8B [--requests PATH]
      [--prompts baseline --modes word_suppression lowercase_thinking] [--sampling card|greedy]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cc_config as cfg

CHUNK_SIZE = 500  # requests per chunk; a crash loses at most one chunk


def sampling_for(exp_key: str, family: str, override: str | None) -> tuple[str, dict]:
    design = cfg.EXP01_DESIGN if exp_key == "exp01" else cfg.EXP02_DESIGN
    name = override or design["sampling"]
    return name, (cfg.GREEDY_SAMPLING if name == "greedy" else cfg.FAMILIES[family]["sampling"])


def check_family_tokens(tokenizer, family: str) -> None:
    fam = cfg.FAMILIES[family]
    expected = {"think_start": "<think>", "think_end": "</think>", "im_end": "<|im_end|>"}
    for key, text in expected.items():
        got = tokenizer.convert_ids_to_tokens(fam[key])
        if got != text:
            raise RuntimeError(f"{family}: token {fam[key]} is {got!r}, expected {text!r}")


def render_prompt_ids(tokenizer, family: str, system: str, user: str) -> list[int]:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **cfg.FAMILIES[family]["chat_template_kwargs"])
    return tokenizer.encode(text, add_special_tokens=False)


def load_llm(model_path: Path, gpu_memory_utilization: float, max_model_len: int):
    from vllm import LLM
    config = json.loads((model_path / "config.json").read_text())
    multimodal = any("ConditionalGeneration" in a for a in config.get("architectures", []))
    kwargs = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if multimodal else {}
    return LLM(model=str(model_path), max_model_len=max_model_len, max_num_seqs=cfg.VLLM_MAX_NUM_SEQS,
               gpu_memory_utilization=gpu_memory_utilization, seed=cfg.VLLM_ENGINE_SEED,
               enable_prefix_caching=True, **kwargs)


def split_reasoning(completion, fam: dict) -> tuple[list[int], list[int], str]:
    """(reasoning ids, phase-1 ids ending in </think> for the answer phase, status) of one phase-1 completion.

    Status: "closed" (</think> written), "truncated" (cap hit; we add the close), "no_think_close" (the model ended
    its turn without </think>; no answer phase).
    """
    tokens = list(completion.token_ids)
    if tokens and tokens[-1] in (fam["think_end"], fam["im_end"]):
        tokens = tokens[:-1]  # the stop token, if vLLM kept it
    body = tokens[1:] if (not fam["template_opens_think"] and tokens[:1] == [fam["think_start"]]) else tokens
    if completion.finish_reason == "length":
        return body, tokens, "truncated"
    if completion.stop_reason == fam["think_end"]:
        return body, tokens + [fam["think_end"]], "closed"
    return body, tokens, "no_think_close"


def generate_two_phase(llm, tokenizer, family: str, jobs: list[dict], sampling: dict, reasoning_cap: int,
                       answer_cap: int, max_model_len: int) -> list[dict]:
    """jobs: dicts with prompt_ids and seed. Returns one result dict per job, same order."""
    from vllm import SamplingParams
    fam = cfg.FAMILIES[family]
    close_ids = tokenizer.encode(cfg.FORCED_THINK_CLOSE, add_special_tokens=False)

    phase1_params, caps = [], []
    for job in jobs:
        cap = min(reasoning_cap, max_model_len - len(job["prompt_ids"]) - answer_cap - len(close_ids) - 1)
        if cap < 1:
            raise RuntimeError(f"prompt of {len(job['prompt_ids'])} tokens leaves no room for reasoning")
        caps.append(cap)
        phase1_params.append(SamplingParams(**sampling, max_tokens=cap, seed=job["seed"],
                                            stop_token_ids=[fam["think_end"], fam["im_end"]],
                                            skip_special_tokens=False))
    t0 = time.time()
    phase1 = llm.generate([{"prompt_token_ids": j["prompt_ids"]} for j in jobs], phase1_params, use_tqdm=True)
    t1 = time.time()

    results, phase2_prompts, phase2_params, phase2_index = [], [], [], []
    for i, (job, out) in enumerate(zip(jobs, phase1)):
        reasoning_ids, phase1_ids, status = split_reasoning(out.outputs[0], fam)
        result = {"reasoning": tokenizer.decode(reasoning_ids, skip_special_tokens=False).strip(),
                  "reasoning_tokens": len(reasoning_ids), "think_status": status, "reasoning_cap": caps[i],
                  "prompt_tokens": len(job["prompt_ids"]), "answer": "", "answer_tokens": 0,
                  "answer_finish": None}
        results.append(result)
        if status == "no_think_close":
            continue  # the model ended its turn without closing <think>; there is no answer phase
        continuation = phase1_ids + (close_ids if status == "truncated" else [])
        phase2_prompts.append({"prompt_token_ids": job["prompt_ids"] + continuation})
        phase2_params.append(SamplingParams(**sampling, max_tokens=answer_cap, seed=cfg.phase2_seed(job["seed"]),
                                            stop_token_ids=[fam["im_end"]], skip_special_tokens=True))
        phase2_index.append(i)
    phase2 = llm.generate(phase2_prompts, phase2_params, use_tqdm=True) if phase2_prompts else []
    for i, out in zip(phase2_index, phase2):
        completion = out.outputs[0]
        results[i]["answer"] = completion.text.strip()
        results[i]["answer_tokens"] = len(completion.token_ids)
        results[i]["answer_finish"] = completion.finish_reason
    print(f"phase 1 {t1 - t0:.0f}s, phase 2 {time.time() - t1:.0f}s for {len(jobs)} jobs", flush=True)
    return results


def select_requests(path: Path, prompts: list[str] | None, modes: list[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    return [r for r in rows if (not prompts or r["prompt"] in prompts) and (not modes or r["mode"] in modes)]


def output_path(requests_path: Path, model: str, sampling_name: str, sampling: dict, requests: list[dict]) -> Path:
    """<requests dir>/generations/..., so smoke outputs (cache/<exp>/smoke/) stay apart from the experiment's."""
    spec = cfg.ALL_MODELS[model]
    key = cfg.content_key({"model": spec, "sampling": sampling, "reasoning_cap": cfg.REASONING_CAP_TOKENS,
                           "answer_cap": cfg.ANSWER_CAP_TOKENS, "max_model_len": cfg.VLLM_MAX_MODEL_LEN,
                           "forced_close": cfg.FORCED_THINK_CLOSE, "engine_seed": cfg.VLLM_ENGINE_SEED,
                           "requests": [r["request_id"] for r in requests]})
    return requests_path.parent / "generations" / f"{model}__{sampling_name}__{key}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=list(cfg.EXPERIMENTS), required=True)
    parser.add_argument("--model", choices=list(cfg.ALL_MODELS), required=True)
    parser.add_argument("--requests", type=Path, default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--sampling", choices=["card", "greedy"], default=None)
    parser.add_argument("--max-model-len", type=int, default=cfg.VLLM_MAX_MODEL_LEN,
                        help="lower only when the KV cache cannot hold one full-length sequence (bf16 32B check)")
    args = parser.parse_args()

    spec = cfg.ALL_MODELS[args.model]
    family = spec["family"]
    requests_path = args.requests or cfg.EXPERIMENTS[args.exp].requests
    requests = select_requests(requests_path, args.prompts, args.modes)
    sampling_name, sampling = sampling_for(args.exp, family, args.sampling)
    out = output_path(requests_path, args.model, sampling_name, sampling, requests)
    if out.exists():
        print(f"{out} exists; nothing to do")
        return
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    (parts / "meta.json").write_text(json.dumps(
        {"exp": args.exp, "model": args.model, "spec": spec, "sampling_name": sampling_name, "sampling": sampling,
         "n_requests": len(requests), "max_model_len": args.max_model_len,
         "requests_file": str(requests_path)}, indent=2))
    print(f"{args.model}: {len(requests)} requests, sampling {sampling_name} -> {out.name}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(args.model))
    check_family_tokens(tokenizer, family)
    llm = load_llm(cfg.model_dir(args.model), spec["gpu_memory_utilization"], args.max_model_len)

    chunks = [requests[i:i + CHUNK_SIZE] for i in range(0, len(requests), CHUNK_SIZE)]
    for index, chunk in enumerate(chunks):
        chunk_path = parts / f"chunk_{index:03d}.jsonl"
        if chunk_path.exists():
            continue
        jobs = [{"prompt_ids": render_prompt_ids(tokenizer, family, r["system"], r["user"]), "seed": r["seed"]}
                for r in chunk]
        results = generate_two_phase(llm, tokenizer, family, jobs, sampling, cfg.REASONING_CAP_TOKENS,
                                     cfg.ANSWER_CAP_TOKENS, args.max_model_len)
        tmp = chunk_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for request, result in zip(chunk, results):
                row = {k: request[k] for k in ("request_id", "item_id", "source", "mode", "prompt", "rollout",
                                               "seed")}
                f.write(json.dumps({**row, "model": args.model, "sampling": sampling_name, **result}) + "\n")
        tmp.rename(chunk_path)
        print(f"chunk {index + 1}/{len(chunks)} written", flush=True)

    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for index in range(len(chunks)):
            f.write((parts / f"chunk_{index:03d}.jsonl").read_text())
    tmp.rename(out)
    print(f"wrote {out}", flush=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from exp03_exit import exit_without_teardown
    exit_without_teardown()


if __name__ == "__main__":
    main()
