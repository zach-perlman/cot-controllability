"""exp03 generation: one streaming engine queue per GPU group, stopping a trace at its first confirmed violation.

Same requests as exp02's two-phase generation (prompt ids from cc_generate.render_prompt_ids, card sampling, the
request's seed for the reasoning and cfg.phase2_seed for the answer, 25k reasoning cap, forced close, 1024-token
answer), with two differences:
- Scheduling (as cc_generate's "stream" profile): a trace's answer phase is enqueued as soon as its reasoning ends,
  ahead of waiting reasoning requests (vLLM priority scheduling), so it reuses the reasoning's prefix cache.
- Abort: a request with abort_on_violation streams its reasoning tokens back (DELTA output); every
  cc_abort.check_interval(n) new tokens the reasoning so far is decoded and checked with
  cc_abort.confirmed_violation. On a confirmed violation the request is aborted: think_status "aborted", the
  reasoning is what was generated up to that step, there is no answer phase, abort_char is the violation's offset.

Output: cache/exp03/generations/<model>__card__stream_abort__<key>.jsonl, key = hash of (model spec, sampling,
caps, engine seed, the requests and their abort flags, the abort rule's code and settings). Shards (--shard K/N,
one per GPU group) append rows to <...>.parts/stream.shard<K>of<N>.jsonl as they finish (resumable: finished rows
are kept, unfinished requests rerun); --merge-shards N writes the output in request order.

Run (vLLM venv): CC_TENSOR_PARALLEL=2 scripts/vllm_python.sh scripts/cc_generate_abort.py --exp exp03 \
    --model Qwen3-32B --shard 0/4           (scripts/run_sharded_abort.sh launches all shards and the merge)
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cc_abort
import cc_config as cfg
import cc_generate as gen
import cc_grade

ENGINE = "stream_abort"


def abort_key_material() -> dict:
    return {"engine": ENGINE, "margin_chars": cfg.ABORT_MARGIN_CHARS, "check_min_tokens": cfg.ABORT_CHECK_MIN_TOKENS,
            "rule_code": [inspect.getsource(f) for f in (cc_abort.confirmed_violation, cc_abort.check_interval)],
            "locator_code_key": cc_grade.code_key()}


def output_path(requests_path: Path, model: str, sampling: dict, requests: list[dict]) -> Path:
    material = {"model": cfg.ALL_MODELS[model], "sampling": sampling, "reasoning_cap": cfg.REASONING_CAP_TOKENS,
                "answer_cap": cfg.ANSWER_CAP_TOKENS, "max_model_len": cfg.VLLM_MAX_MODEL_LEN,
                "forced_close": cfg.FORCED_THINK_CLOSE, "engine_seed": cfg.VLLM_ENGINE_SEED,
                "requests": [[r["request_id"], r["abort_on_violation"]] for r in requests],
                "abort": abort_key_material()}
    return requests_path.parent / "generations" / f"{model}__card__{ENGINE}__{cfg.content_key(material)}.jsonl"


def reasoning_caps(jobs: list[dict], close_len: int, max_model_len: int) -> list[int]:
    caps = []
    for job in jobs:
        cap = min(cfg.REASONING_CAP_TOKENS,
                  max_model_len - len(job["prompt_ids"]) - cfg.ANSWER_CAP_TOKENS - close_len - 1)
        if cap < 1:
            raise RuntimeError(f"prompt of {len(job['prompt_ids'])} tokens leaves no room for reasoning")
        caps.append(cap)
    return caps


def generate_streaming_abort(llm, tokenizer, family: str, jobs: list[dict], sampling: dict, max_model_len: int,
                             on_result) -> None:
    """jobs: dicts with prompt_ids, seed, abort (bool), mode, item. on_result(job index, result) once per job, in
    completion order; result has generate_two_phase's fields plus abort_char and abort_checks."""
    from tqdm import tqdm
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    fam = cfg.FAMILIES[family]
    close_ids = tokenizer.encode(cfg.FORCED_THINK_CLOSE, add_special_tokens=False)
    caps = reasoning_caps(jobs, len(close_ids), max_model_len)
    params = [SamplingParams(**sampling, max_tokens=cap, seed=job["seed"],
                             stop_token_ids=[fam["think_end"], fam["im_end"]], skip_special_tokens=False,
                             output_kind=RequestOutputKind.DELTA if job["abort"] else RequestOutputKind.FINAL_ONLY)
              for job, cap in zip(jobs, caps)]
    request_ids = llm.enqueue([{"prompt_token_ids": j["prompt_ids"]} for j in jobs], params,
                              priority=[gen.REASONING_PRIORITY] * len(jobs), use_tqdm=False)
    reasoning_of = dict(zip(request_ids, range(len(jobs))))
    streamed = {rid: [] for rid, i in reasoning_of.items() if jobs[i]["abort"]}  # token ids so far
    next_check = {rid: cfg.ABORT_CHECK_MIN_TOKENS for rid in streamed}
    n_checks = {rid: 0 for rid in streamed}
    answer_of = {}
    engine = llm.llm_engine
    bar = tqdm(total=len(jobs), desc="Finished traces", dynamic_ncols=True)

    def base_result(i: int, reasoning_ids: list[int], status: str) -> dict:
        return {"reasoning": tokenizer.decode(reasoning_ids, skip_special_tokens=False).strip(),
                "reasoning_tokens": len(reasoning_ids), "think_status": status, "reasoning_cap": caps[i],
                "prompt_tokens": len(jobs[i]["prompt_ids"]), "answer": "", "answer_tokens": 0,
                "answer_finish": None, "abort_char": None, "abort_checks": None}

    def body_of(ids: list[int]) -> list[int]:
        opens = not fam["template_opens_think"] and ids[:1] == [fam["think_start"]]
        return ids[1:] if opens else ids

    while engine.has_unfinished_requests():
        to_abort = []
        for out in engine.step():
            rid = out.request_id
            if rid in reasoning_of:
                i = reasoning_of[rid]
                completion = out.outputs[0]
                if rid in streamed:
                    streamed[rid].extend(completion.token_ids)
                    if not out.finished and len(streamed[rid]) >= next_check[rid]:
                        n_checks[rid] += 1
                        body = body_of(streamed[rid])
                        text = tokenizer.decode(body, skip_special_tokens=False).lstrip()
                        char = cc_abort.confirmed_violation(jobs[i]["mode"], text, jobs[i]["item"])
                        if char is not None:
                            to_abort.append(rid)
                            reasoning_of.pop(rid)
                            result = base_result(i, body, "aborted")
                            # The last token may end inside a multi-byte character, which decodes as U+FFFD.
                            result.update(reasoning=result["reasoning"].rstrip("\ufffd").rstrip(), abort_char=char,
                                          abort_checks=n_checks.pop(rid))
                            streamed.pop(rid)
                            on_result(i, result)
                            bar.update()
                            continue
                        next_check[rid] = len(streamed[rid]) + cc_abort.check_interval(len(streamed[rid]))
                    if not out.finished:
                        continue
                    completion = SimpleNamespace(token_ids=streamed.pop(rid), finish_reason=completion.finish_reason,
                                                 stop_reason=completion.stop_reason)
                elif not out.finished:
                    continue
                reasoning_of.pop(rid)
                reasoning_ids, phase1_ids, status = gen.split_reasoning(completion, fam)
                result = base_result(i, reasoning_ids, status)
                if rid in n_checks:
                    result["abort_checks"] = n_checks.pop(rid)
                if status == "no_think_close":
                    on_result(i, result)
                    bar.update()
                    continue
                continuation = phase1_ids + (close_ids if status != "closed" else [])
                answer_params = SamplingParams(**sampling, max_tokens=cfg.ANSWER_CAP_TOKENS,
                                               seed=cfg.phase2_seed(jobs[i]["seed"]), stop_token_ids=[fam["im_end"]],
                                               skip_special_tokens=True, output_kind=RequestOutputKind.FINAL_ONLY)
                [answer_id] = llm.enqueue([{"prompt_token_ids": jobs[i]["prompt_ids"] + continuation}],
                                          [answer_params], priority=[gen.ANSWER_PRIORITY], use_tqdm=False)
                answer_of[answer_id] = (i, result)
            elif rid in answer_of and out.finished:
                i, result = answer_of.pop(rid)
                completion = out.outputs[0]
                result.update(answer=completion.text.strip(), answer_tokens=len(completion.token_ids),
                              answer_finish=completion.finish_reason)
                on_result(i, result)
                bar.update()
        if to_abort:
            engine.abort_request(to_abort)
    bar.close()
    if reasoning_of or answer_of:
        raise RuntimeError(f"engine stopped with {len(reasoning_of)} reasoning and {len(answer_of)} answer requests "
                           f"unfinished")


def shard_stream_path(parts: Path, shard: tuple[int, int]) -> Path:
    return parts / f"stream.shard{shard[0]}of{shard[1]}.jsonl"


def output_row(request: dict, model: str, result: dict) -> dict:
    row = {k: request[k] for k in ("request_id", "item_id", "source", "mode", "prompt", "rollout", "seed",
                                   "abort_on_violation", "full_trace_cell")}
    row.update(model=model, sampling="card", engine=ENGINE)
    return {**row, **result}


def run_shard(llm, tokenizer, family: str, requests: list[dict], items: dict, sampling: dict, max_model_len: int,
              parts: Path, shard: tuple[int, int], model: str) -> None:
    mine = requests[shard[0]::shard[1]]
    path = shard_stream_path(parts, shard)
    done = gen.read_stream(path)
    todo = [r for r in mine if r["request_id"] not in done]
    print(f"shard {shard[0]}/{shard[1]}: {len(mine)} requests, {len(done)} done earlier, {len(todo)} to run",
          flush=True)
    if not todo:
        return
    jobs = [{"prompt_ids": gen.render_prompt_ids(tokenizer, family, r["system"], r["user"]), "seed": r["seed"],
             "abort": r["abort_on_violation"], "mode": r["mode"], "item": items[r["item_id"]]} for r in todo]
    with path.open("a") as f:
        def write(index: int, result: dict) -> None:
            f.write(json.dumps(output_row(todo[index], model, result)) + "\n")
            f.flush()
        generate_streaming_abort(llm, tokenizer, family, jobs, sampling, max_model_len, write)


def merge(requests: list[dict], parts: Path, n_shards: int) -> list[str]:
    rows = {}
    for k in range(n_shards):
        path = shard_stream_path(parts, (k, n_shards))
        if not path.exists():
            raise SystemExit(f"missing {path.name}; run its shard first")
        shard_rows = gen.read_stream(path)
        expected = {r["request_id"] for r in requests[k::n_shards]}
        if set(shard_rows) != expected:
            raise SystemExit(f"{path.name}: {len(expected - set(shard_rows))} of its requests unfinished, "
                             f"{len(set(shard_rows) - expected)} rows of other shards")
        rows.update(shard_rows)
    return [json.dumps(rows[r["request_id"]]) for r in requests]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["exp03"], required=True)
    parser.add_argument("--model", choices=list(cfg.ALL_MODELS), required=True)
    parser.add_argument("--requests", type=Path, default=None, help="default: cache/<exp>/requests.jsonl")
    parser.add_argument("--items", type=Path, default=None, help="default: cfg.EXP03_ITEMS_PATH")
    parser.add_argument("--sampling", choices=["card", "greedy"], default="card",
                        help="greedy only for the engine check against the batch engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard", help="K/N")
    group.add_argument("--merge-shards", type=int, metavar="N")
    args = parser.parse_args()

    spec = cfg.ALL_MODELS[args.model]
    family = spec["family"]
    requests_path = args.requests or cfg.EXPERIMENTS[args.exp].requests
    requests = [json.loads(line) for line in requests_path.open()]
    items = {it["item_id"]: it for it in map(json.loads, (args.items or cfg.EXP03_ITEMS_PATH).open())}
    sampling = cfg.GREEDY_SAMPLING if args.sampling == "greedy" else cfg.FAMILIES[family]["sampling"]
    out = output_path(requests_path, args.model, sampling, requests)
    if args.sampling == "greedy":
        out = out.with_name(out.name.replace("__card__", "__greedy__"))
    if out.exists():
        print(f"{out} exists; nothing to do")
        return
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        gen.write_jsonl(out, merge(requests, parts, args.merge_shards))
        print(f"wrote {out}", flush=True)
        return

    shard = tuple(int(x) for x in args.shard.split("/"))
    meta = {"exp": args.exp, "model": args.model, "spec": spec, "sampling_name": args.sampling, "sampling": sampling,
            "n_requests": len(requests), "requests_file": str(requests_path), "engine": ENGINE,
            "abort": {k: v for k, v in abort_key_material().items() if k != "rule_code"},
            "gpus": gen.visible_gpus(), "tensor_parallel": gen.TENSOR_PARALLEL, "shard": args.shard,
            "gpu_memory_utilization": spec["gpu_memory_utilization"],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (parts / f"meta_shard{shard[0]}of{shard[1]}_{int(time.time())}.json").write_text(json.dumps(meta, indent=2))
    print(f"{args.model}: shard {args.shard} of {len(requests)} requests, TP {gen.TENSOR_PARALLEL} -> {out.name}",
          flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(args.model))
    gen.check_family_tokens(tokenizer, family)
    llm = gen.load_llm(cfg.model_dir(args.model), spec["gpu_memory_utilization"], cfg.VLLM_MAX_MODEL_LEN,
                       {"scheduling_policy": "priority"})
    started = time.time()
    run_shard(llm, tokenizer, family, requests, items, sampling, cfg.VLLM_MAX_MODEL_LEN, parts, shard, args.model)
    (parts / f"timing_shard{shard[0]}of{shard[1]}_{int(started)}.json").write_text(json.dumps(
        {"seconds": round(time.time() - started), "gpus": meta["gpus"], "tensor_parallel": gen.TENSOR_PARALLEL}))

    sys.path.insert(0, str(Path(__file__).parent))
    from exp03_exit import exit_without_teardown
    exit_without_teardown()


if __name__ == "__main__":
    main()
