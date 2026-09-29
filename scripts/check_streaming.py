"""CPU check: the streaming engine profile issues the same requests and returns the same results as the v1 batch
path, and resumes correctly after a crash.

A fake vLLM engine answers every request with a pure function of (prompt ids, sampling parameters), so the two paths
return identical results exactly when they issue identical requests. The fake finishes requests in a shuffled order
and covers the three reasoning outcomes (closed, cap hit, turn ended without </think>). SamplingParams is replaced
by a stub that records its arguments, so no GPU or vLLM engine is needed.

Checks:
1. generate_two_phase and generate_streaming issue the same multiset of (prompt, sampling arguments) requests.
2. Their per-job results are identical.
3. Answer phases are enqueued at ANSWER_PRIORITY, reasoning at REASONING_PRIORITY.
4. run_streaming, crashed after some rows with a partial last line, then resumed, writes the same lines as an
   uninterrupted run.

Run: /venv/main/bin/python scripts/check_streaming.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import types
from pathlib import Path

import cc_config as cfg

FAMILY = "qwen3"
N_JOBS = 60


class StubSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


sys.modules["vllm"] = types.SimpleNamespace(SamplingParams=StubSamplingParams)
import cc_generate  # noqa: E402  (after the vllm stub)


def request_signature(prompt_ids: list[int], params: StubSamplingParams) -> str:
    return json.dumps([prompt_ids, params.kwargs], sort_keys=True)


def fake_completion(prompt_ids: list[int], params: StubSamplingParams, tokenizer):
    """Deterministic output for one request, covering every reasoning outcome."""
    fam = cfg.FAMILIES[FAMILY]
    rng = random.Random(hashlib.sha256(request_signature(prompt_ids, params).encode()).hexdigest())
    kw = params.kwargs
    body = [rng.randrange(1000, 30000) for _ in range(rng.randint(0, min(40, kw["max_tokens"])))]
    if fam["think_end"] in kw["stop_token_ids"]:  # reasoning request
        if rng.random() < 0.5:
            body = [fam["think_start"]] + body  # the model writes <think> itself under this template
        outcome = rng.choice(["closed", "closed", "truncated", "no_close"])
        if outcome == "truncated":
            tokens = (body + [rng.randrange(1000, 30000) for _ in range(kw["max_tokens"])])[:kw["max_tokens"]]
            return types.SimpleNamespace(token_ids=tokens, finish_reason="length", stop_reason=None, text="")
        stop = fam["think_end"] if outcome == "closed" else fam["im_end"]
        tokens = body + ([stop] if rng.random() < 0.5 else [])  # vLLM may or may not keep the stop token
        return types.SimpleNamespace(token_ids=tokens, finish_reason="stop", stop_reason=stop, text="")
    text = tokenizer.decode(body, skip_special_tokens=True)
    return types.SimpleNamespace(token_ids=body, finish_reason="stop", stop_reason=fam["im_end"], text=text)


class FakeEngine:
    """Enough of vllm.LLM (generate, enqueue, llm_engine.step) for both generation paths."""

    def __init__(self, tokenizer, crash_after: int | None = None):
        self.tokenizer = tokenizer
        self.issued = []  # (signature, priority) of every request, in issue order
        self.waiting = {}  # request id -> (prompt ids, params)
        self.counter = 0
        self.rng = random.Random(0)
        self.finished_count = 0
        self.crash_after = crash_after
        self.llm_engine = self

    def generate(self, prompts, params, use_tqdm=True):
        for p, sp in zip(prompts, params):
            self.issued.append((request_signature(p["prompt_token_ids"], sp), None))
        return [types.SimpleNamespace(outputs=[fake_completion(p["prompt_token_ids"], sp, self.tokenizer)])
                for p, sp in zip(prompts, params)]

    def enqueue(self, prompts, params, priority, use_tqdm=True):
        ids = []
        for p, sp, pr in zip(prompts, params, priority):
            request_id = str(self.counter)
            self.counter += 1
            self.waiting[request_id] = (p["prompt_token_ids"], sp)
            self.issued.append((request_signature(p["prompt_token_ids"], sp), pr))
            ids.append(request_id)
        return ids

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting)

    def step(self):
        """Finish a random subset of the waiting requests, in shuffled order."""
        ids = list(self.waiting)
        self.rng.shuffle(ids)
        outputs = []
        for request_id in ids[:self.rng.randint(1, 5)]:
            if self.crash_after is not None and self.finished_count >= self.crash_after:
                raise KeyboardInterrupt("simulated crash")
            prompt_ids, sp = self.waiting.pop(request_id)
            self.finished_count += 1
            outputs.append(types.SimpleNamespace(request_id=request_id, finished=True,
                                                 outputs=[fake_completion(prompt_ids, sp, self.tokenizer)]))
        return outputs


def main() -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir("Qwen3-8B"))
    sampling = cfg.FAMILIES[FAMILY]["sampling"]
    rng = random.Random(1)
    reasoning_cap, answer_cap, max_model_len = 30, 20, 400
    jobs = [{"prompt_ids": [rng.randrange(1000, 30000) for _ in range(rng.randint(5, 50))],
             "seed": rng.randrange(2 ** 31)} for _ in range(N_JOBS)]

    batch = FakeEngine(tokenizer)
    batch_results = cc_generate.generate_two_phase(batch, tokenizer, FAMILY, jobs, sampling, reasoning_cap,
                                                   answer_cap, max_model_len)
    stream = FakeEngine(tokenizer)
    stream_results = [None] * len(jobs)

    def collect(index: int, result: dict) -> None:
        assert stream_results[index] is None, f"job {index} reported twice"
        stream_results[index] = result

    cc_generate.generate_streaming(stream, tokenizer, FAMILY, jobs, sampling, reasoning_cap, answer_cap,
                                   max_model_len, collect)

    statuses = sorted({r["think_status"] for r in batch_results})
    assert statuses == ["closed", "no_think_close", "truncated"], f"fake engine missed an outcome: {statuses}"
    assert sorted(s for s, _ in batch.issued) == sorted(s for s, _ in stream.issued), "requests differ"
    assert batch_results == stream_results, "results differ"
    n_reasoning = len(jobs)
    assert all(p == cc_generate.REASONING_PRIORITY for _, p in stream.issued[:n_reasoning])
    assert all(p == cc_generate.ANSWER_PRIORITY for _, p in stream.issued[n_reasoning:])
    print(f"1-3 ok: {len(batch.issued)} identical requests and {len(jobs)} identical results "
          f"({', '.join(f'{s} {sum(r['think_status'] == s for r in batch_results)}' for s in statuses)})")

    requests = [{"request_id": f"r{i:03d}", "item_id": f"i{i}", "source": "gpqa", "mode": "baseline",
                 "prompt": "baseline", "rollout": 0, "seed": rng.randrange(2 ** 31), "system": "",
                 "user": f"Question {i}: what is {i} + {i}?"} for i in range(N_JOBS)]

    def row_of(request: dict, result: dict) -> dict:
        return cc_generate.output_row(request, "Qwen3-8B", "card", "stream", result)

    def run(parts: Path, crash_after: int | None = None) -> list[str]:
        parts.mkdir(exist_ok=True)  # main() creates the .parts directory
        return cc_generate.run_streaming(FakeEngine(tokenizer, crash_after), tokenizer, FAMILY, requests, sampling,
                                         cfg.VLLM_MAX_MODEL_LEN, parts, row_of)

    with tempfile.TemporaryDirectory() as tmp:
        uninterrupted = run(Path(tmp) / "a")
        resumed_dir = Path(tmp) / "b"
        try:
            run(resumed_dir, crash_after=37)
        except KeyboardInterrupt:
            pass
        stream_file = resumed_dir / "stream.jsonl"
        n_before = len(stream_file.read_text().splitlines())
        with stream_file.open("a") as f:
            f.write('{"request_id": "r0')  # the line being written when the run died
        resumed = run(resumed_dir)
        n_after = len(stream_file.read_text().splitlines())
    assert 0 < n_before < N_JOBS, f"crash left {n_before} rows"
    assert resumed == uninterrupted, "resumed run differs from an uninterrupted one"
    assert n_after == N_JOBS, f"stream file has {n_after} rows, expected {N_JOBS}"
    print(f"4 ok: crashed with {n_before} of {N_JOBS} rows plus a partial line; resumed output identical")


if __name__ == "__main__":
    main()
