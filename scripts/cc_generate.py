"""Two-phase vLLM generation for exp01 / exp02 (and the judge, which imports generate_two_phase).

Phase 1 samples the reasoning until </think> (or im_end), capped per request at
min(REASONING_CAP_TOKENS, VLLM_MAX_MODEL_LEN - prompt - answer cap - forced close). If the cap is hit, "\\n</think>\\n\\n"
is appended by us (status "truncated"). Phase 2 continues from prompt + phase-1 tokens for at most ANSWER_CAP_TOKENS.
Everything is kept as token ids, so the reasoning the grader sees is exactly what the model wrote between <think> and
</think>.

Output: cache/<exp>/generations/<model>__<key>.jsonl, key = content hash of (model spec, sampling, caps, requests),
written in chunks under <...>.parts/ (resumable) and joined when complete. An existing output is never rewritten.

Engine profiles (cc_config.ENGINE_PROFILES) change only GPU scheduling. "v1" (default) is the batch path above.
"stream" issues the same requests (prompts, SamplingParams, seeds) through one queue (generate_streaming): each
trace's answer phase starts when its reasoning ends, and rows are appended to <...>.parts/stream.jsonl as they
finish (resumable). Non-default profiles are part of the output key and file name (<model>__<sampling>__<profile>__).

Run (vLLM venv):
  scripts/vllm_python.sh scripts/cc_generate.py --exp exp02 --model Qwen3-8B [--requests PATH]
      [--prompts baseline --modes word_suppression lowercase_thinking] [--sampling card|greedy] [--engine stream]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cc_config as cfg

# Requests per chunk; a crash loses at most one chunk. Each chunk ends waiting on its longest (up to 25k-token)
# trace, so fewer, larger chunks waste less GPU time.
CHUNK_SIZE = 1000


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


def load_llm(model_path: Path, gpu_memory_utilization: float, max_model_len: int, extra: dict | None = None):
    """extra: further vLLM engine arguments of a non-default engine profile (engine_args)."""
    from vllm import LLM
    config = json.loads((model_path / "config.json").read_text())
    multimodal = any("ConditionalGeneration" in a for a in config.get("architectures", []))
    kwargs = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if multimodal else {}
    return LLM(model=str(model_path), max_model_len=max_model_len, max_num_seqs=cfg.VLLM_MAX_NUM_SEQS,
               gpu_memory_utilization=gpu_memory_utilization, seed=cfg.VLLM_ENGINE_SEED,
               enable_prefix_caching=True, **kwargs, **(extra or {}))


# Under vLLM's priority scheduling a lower value is scheduled first: answer phases go ahead of waiting reasoning
# requests, so they run while their prompt + reasoning is still in the prefix cache.
REASONING_PRIORITY, ANSWER_PRIORITY = 1, 0


def engine_args(profile_name: str, family: str) -> dict:
    """vLLM engine arguments an engine profile adds to load_llm's."""
    profile = cfg.ENGINE_PROFILES[profile_name]
    extra = {}
    if profile["scheduler"] == "streaming":
        extra["scheduling_policy"] = "priority"
    if profile["speculative"]:
        if family not in cfg.SPECULATIVE_CONFIGS:
            raise ValueError(f"engine profile {profile_name}: no draft head configured for family {family}")
        extra["speculative_config"] = cfg.SPECULATIVE_CONFIGS[family]
    return extra


def engine_key_material(profile_name: str, family: str) -> dict:
    """What a non-default engine profile adds to the generation output key."""
    return {"name": profile_name, **cfg.ENGINE_PROFILES[profile_name], "engine_args": engine_args(profile_name, family)}


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
                       answer_cap: int, max_model_len: int, answer_after_no_close: bool = False) -> list[dict]:
    """jobs: dicts with prompt_ids and seed. Returns one result dict per job, same order.

    answer_after_no_close (judge only): if the model ends its turn without </think>, close it and ask for the
    answer anyway, as for a truncated trace. Subject models keep "no_think_close" with no answer phase.
    """
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
        if status == "no_think_close" and not answer_after_no_close:
            continue  # the model ended its turn without closing <think>; there is no answer phase
        continuation = phase1_ids + (close_ids if status != "closed" else [])
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


def generate_streaming(llm, tokenizer, family: str, jobs: list[dict], sampling: dict, reasoning_cap: int,
                       answer_cap: int, max_model_len: int, on_result) -> None:
    """The requests of generate_two_phase (same prompts, SamplingParams and seeds; subject models, so no
    answer_after_no_close) through one engine queue. The LLM must use scheduling_policy="priority" (engine_args).

    A job's answer phase is enqueued as soon as its reasoning ends. on_result(job index, result) is called once per
    job in completion order; result has generate_two_phase's fields. check_streaming.py compares the two paths'
    requests on a fake engine.
    """
    from tqdm import tqdm
    from vllm import SamplingParams
    fam = cfg.FAMILIES[family]
    close_ids = tokenizer.encode(cfg.FORCED_THINK_CLOSE, add_special_tokens=False)

    caps, reasoning_params = [], []
    for job in jobs:
        cap = min(reasoning_cap, max_model_len - len(job["prompt_ids"]) - answer_cap - len(close_ids) - 1)
        if cap < 1:
            raise RuntimeError(f"prompt of {len(job['prompt_ids'])} tokens leaves no room for reasoning")
        caps.append(cap)
        reasoning_params.append(SamplingParams(**sampling, max_tokens=cap, seed=job["seed"],
                                               stop_token_ids=[fam["think_end"], fam["im_end"]],
                                               skip_special_tokens=False))
    request_ids = llm.enqueue([{"prompt_token_ids": j["prompt_ids"]} for j in jobs], reasoning_params,
                              priority=[REASONING_PRIORITY] * len(jobs), use_tqdm=False)
    reasoning_of = dict(zip(request_ids, range(len(jobs))))  # engine request id -> job index
    answer_of = {}  # engine request id -> (job index, result without the answer yet)

    bar = tqdm(total=len(jobs), desc="Finished traces", dynamic_ncols=True)
    engine = llm.llm_engine
    while engine.has_unfinished_requests():
        for out in engine.step():
            if not out.finished:
                continue
            if out.request_id in reasoning_of:
                i = reasoning_of.pop(out.request_id)
                reasoning_ids, phase1_ids, status = split_reasoning(out.outputs[0], fam)
                result = {"reasoning": tokenizer.decode(reasoning_ids, skip_special_tokens=False).strip(),
                          "reasoning_tokens": len(reasoning_ids), "think_status": status, "reasoning_cap": caps[i],
                          "prompt_tokens": len(jobs[i]["prompt_ids"]), "answer": "", "answer_tokens": 0,
                          "answer_finish": None}
                if status == "no_think_close":
                    on_result(i, result)  # the model ended its turn without closing <think>; no answer phase
                    bar.update()
                    continue
                continuation = phase1_ids + (close_ids if status != "closed" else [])
                answer_params = SamplingParams(**sampling, max_tokens=answer_cap,
                                               seed=cfg.phase2_seed(jobs[i]["seed"]),
                                               stop_token_ids=[fam["im_end"]], skip_special_tokens=True)
                [answer_id] = llm.enqueue([{"prompt_token_ids": jobs[i]["prompt_ids"] + continuation}],
                                          [answer_params], priority=[ANSWER_PRIORITY], use_tqdm=False)
                answer_of[answer_id] = (i, result)
            else:
                i, result = answer_of.pop(out.request_id)
                completion = out.outputs[0]
                result.update(answer=completion.text.strip(), answer_tokens=len(completion.token_ids),
                              answer_finish=completion.finish_reason)
                on_result(i, result)
                bar.update()
    bar.close()
    if reasoning_of or answer_of:
        raise RuntimeError(f"engine stopped with {len(reasoning_of)} reasoning and {len(answer_of)} answer "
                           f"requests unfinished")


def select_requests(path: Path, prompts: list[str] | None, modes: list[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    return [r for r in rows if (not prompts or r["prompt"] in prompts) and (not modes or r["mode"] in modes)]


def output_path(requests_path: Path, model: str, sampling_name: str, sampling: dict, requests: list[dict],
                engine: str = cfg.DEFAULT_ENGINE_PROFILE) -> Path:
    """<requests dir>/generations/..., so smoke outputs (cache/<exp>/smoke/) stay apart from the experiment's.
    The default engine profile adds nothing to the key, so files generated before profiles existed keep theirs."""
    spec = cfg.ALL_MODELS[model]
    material = {"model": spec, "sampling": sampling, "reasoning_cap": cfg.REASONING_CAP_TOKENS,
                "answer_cap": cfg.ANSWER_CAP_TOKENS, "max_model_len": cfg.VLLM_MAX_MODEL_LEN,
                "forced_close": cfg.FORCED_THINK_CLOSE, "engine_seed": cfg.VLLM_ENGINE_SEED,
                "requests": [r["request_id"] for r in requests]}
    label = sampling_name
    if engine != cfg.DEFAULT_ENGINE_PROFILE:
        material["engine"] = engine_key_material(engine, spec["family"])
        label = f"{sampling_name}__{engine}"
    return requests_path.parent / "generations" / f"{model}__{label}__{cfg.content_key(material)}.jsonl"


def output_row(request: dict, model: str, sampling_name: str, engine: str, result: dict) -> dict:
    row = {k: request[k] for k in ("request_id", "item_id", "source", "mode", "prompt", "rollout", "seed")}
    row.update(model=model, sampling=sampling_name)
    if engine != cfg.DEFAULT_ENGINE_PROFILE:
        row["engine"] = engine
    return {**row, **result}


def read_stream(path: Path) -> dict[str, dict]:
    """Rows an interrupted streaming run already finished, by request_id. The file is rewritten without the line
    that was being written when the run stopped, so new rows are not appended to a partial line."""
    if not path.exists():
        return {}
    rows, complete = {}, []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[row["request_id"]] = row
        complete.append(line)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(line + "\n" for line in complete))
    tmp.rename(path)
    return rows


def run_batch(llm, tokenizer, family: str, requests: list[dict], sampling: dict, max_model_len: int, parts: Path,
              row_of) -> list[str]:
    """Engine profile v1: CHUNK_SIZE requests at a time through generate_two_phase, one file per chunk.
    Returns the output's JSON lines, in request order."""
    chunks = [requests[i:i + CHUNK_SIZE] for i in range(0, len(requests), CHUNK_SIZE)]
    for index, chunk in enumerate(chunks):
        chunk_path = parts / f"chunk_{index:03d}.jsonl"
        if chunk_path.exists():
            continue
        jobs = [{"prompt_ids": render_prompt_ids(tokenizer, family, r["system"], r["user"]), "seed": r["seed"]}
                for r in chunk]
        results = generate_two_phase(llm, tokenizer, family, jobs, sampling, cfg.REASONING_CAP_TOKENS,
                                     cfg.ANSWER_CAP_TOKENS, max_model_len)
        tmp = chunk_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for request, result in zip(chunk, results):
                f.write(json.dumps(row_of(request, result)) + "\n")
        tmp.rename(chunk_path)
        print(f"chunk {index + 1}/{len(chunks)} written", flush=True)
    return [line for index in range(len(chunks))
            for line in (parts / f"chunk_{index:03d}.jsonl").read_text().splitlines()]


def run_streaming(llm, tokenizer, family: str, requests: list[dict], sampling: dict, max_model_len: int,
                  parts: Path, row_of) -> list[str]:
    """Streaming profiles: every unfinished request in one queue; each row is appended to stream.jsonl when done.
    Returns the output's JSON lines, in request order."""
    stream_path = parts / "stream.jsonl"
    done = read_stream(stream_path)
    todo = [r for r in requests if r["request_id"] not in done]
    print(f"{len(done)} rows from an earlier run, {len(todo)} to generate", flush=True)
    if todo:
        jobs = [{"prompt_ids": render_prompt_ids(tokenizer, family, r["system"], r["user"]), "seed": r["seed"]}
                for r in todo]
        with stream_path.open("a") as f:
            def write(index: int, result: dict) -> None:
                row = row_of(todo[index], result)
                done[row["request_id"]] = row
                f.write(json.dumps(row) + "\n")
                f.flush()
            generate_streaming(llm, tokenizer, family, jobs, sampling, cfg.REASONING_CAP_TOKENS,
                               cfg.ANSWER_CAP_TOKENS, max_model_len, write)
    return [json.dumps(done[r["request_id"]]) for r in requests]


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
    parser.add_argument("--engine", choices=list(cfg.ENGINE_PROFILES), default=cfg.DEFAULT_ENGINE_PROFILE,
                        help="GPU scheduling only (cc_config.ENGINE_PROFILES); non-default profiles change the key")
    args = parser.parse_args()

    spec = cfg.ALL_MODELS[args.model]
    family = spec["family"]
    profile = cfg.ENGINE_PROFILES[args.engine]
    requests_path = args.requests or cfg.EXPERIMENTS[args.exp].requests
    requests = select_requests(requests_path, args.prompts, args.modes)
    sampling_name, sampling = sampling_for(args.exp, family, args.sampling)
    out = output_path(requests_path, args.model, sampling_name, sampling, requests, args.engine)
    if out.exists():
        print(f"{out} exists; nothing to do")
        return
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    gpu_memory_utilization = profile["gpu_memory_utilization"] or spec["gpu_memory_utilization"]
    extra = engine_args(args.engine, family)
    (parts / "meta.json").write_text(json.dumps(
        {"exp": args.exp, "model": args.model, "spec": spec, "sampling_name": sampling_name, "sampling": sampling,
         "n_requests": len(requests), "max_model_len": args.max_model_len, "requests_file": str(requests_path),
         "engine": args.engine, "gpu_memory_utilization": gpu_memory_utilization, "engine_args": extra}, indent=2))
    print(f"{args.model}: {len(requests)} requests, sampling {sampling_name}, engine {args.engine} -> {out.name}",
          flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(args.model))
    check_family_tokens(tokenizer, family)
    llm = load_llm(cfg.model_dir(args.model), gpu_memory_utilization, args.max_model_len, extra)

    def row_of(request: dict, result: dict) -> dict:
        return output_row(request, args.model, sampling_name, args.engine, result)

    run = run_streaming if profile["scheduler"] == "streaming" else run_batch
    started = time.time()
    lines = run(llm, tokenizer, family, requests, sampling, args.max_model_len, parts, row_of)
    (parts / f"timing_{int(started)}.json").write_text(json.dumps(
        {"engine": args.engine, "seconds": round(time.time() - started), "n_requests": len(requests)}))

    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for line in lines:
            f.write(line + "\n")
    tmp.rename(out)
    print(f"wrote {out}", flush=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from exp03_exit import exit_without_teardown
    exit_without_teardown()


if __name__ == "__main__":
    main()
