"""CPU check of exp03's streaming-with-abort generator (cc_generate_abort) against the v1 batch path.

A fake engine plays real exp02 reasoning traces (Qwen3-8B, abortable modes, re-tokenized with its tokenizer): each
prompt maps to one trace, streamed back in random-sized chunks to DELTA requests and whole to FINAL_ONLY requests;
answers are a pure function of the answer prompt. No GPU or vLLM engine is needed (SamplingParams is a stub).

Checks:
1. Requests that may not abort get exactly generate_two_phase's results.
2. Requests that may abort: a trace whose rule the abort rule can confirm is aborted, with abort_char equal to the
   full trace's first violation and reasoning a prefix of the full reasoning; no answer phase is issued for it and
   it is aborted in the engine. Every other trace gets generate_two_phase's result.
3. Four shards + merge == one shard, and a shard that crashed mid-run with a partial line resumes to the same rows.

Run: /venv/main/bin/python scripts/check_abort_streaming.py
"""

from __future__ import annotations

import enum
import hashlib
import json
import random
import sys
import tempfile
import types
from pathlib import Path

import cc_config as cfg


class StubSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubOutputKind(enum.Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


sys.modules["vllm"] = types.SimpleNamespace(SamplingParams=StubSamplingParams)
sys.modules["vllm.sampling_params"] = types.SimpleNamespace(RequestOutputKind=StubOutputKind)
import cc_abort  # noqa: E402
import cc_generate  # noqa: E402
import cc_generate_abort as ga  # noqa: E402
import cc_grade  # noqa: E402

FAMILY = "qwen3"
N_TRACES = 80


class FakeEngine:
    """generate / enqueue / llm_engine.step / abort_request, replaying fixed traces."""

    def __init__(self, tokenizer, traces: dict, crash_after: int | None = None):
        self.tokenizer, self.traces, self.crash_after = tokenizer, traces, crash_after
        self.running = {}  # request id -> state
        self.counter, self.finished, self.aborted, self.answer_prompts = 0, 0, [], []
        self.rng = random.Random(0)
        self.llm_engine = self

    def completion(self, prompt_ids, kw):
        fam = cfg.FAMILIES[FAMILY]
        if fam["think_end"] in kw["stop_token_ids"]:
            body = self.traces[tuple(prompt_ids)]
            if len(body) >= kw["max_tokens"]:
                return body[:kw["max_tokens"]], "length", None
            return body + [fam["think_end"]], "stop", fam["think_end"]
        self.answer_prompts.append(tuple(prompt_ids))
        rng = random.Random(hashlib.sha256(json.dumps([prompt_ids, kw["seed"]]).encode()).hexdigest())
        return [rng.randrange(1000, 30000) for _ in range(rng.randint(1, 12))], "stop", fam["im_end"]

    def output(self, rid, tokens, finished, finish_reason=None, stop_reason=None):
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return types.SimpleNamespace(request_id=rid, finished=finished, outputs=[types.SimpleNamespace(
            token_ids=tokens, finish_reason=finish_reason, stop_reason=stop_reason, text=text)])

    def generate(self, prompts, params, use_tqdm=True):
        outs = []
        for p, sp in zip(prompts, params):
            tokens, fr, sr = self.completion(p["prompt_token_ids"], sp.kwargs)
            outs.append(self.output(None, tokens, True, fr, sr))
        return outs

    def enqueue(self, prompts, params, priority, use_tqdm=True):
        ids = []
        for p, sp in zip(prompts, params):
            rid = f"req{self.counter}"
            self.counter += 1
            tokens, fr, sr = self.completion(p["prompt_token_ids"], sp.kwargs)
            # Chunk sizes depend only on the request (vLLM without speculative decoding returns one token per step,
            # so where a trace is checked does not depend on which requests share the engine).
            chunks = random.Random(hashlib.sha256(json.dumps(p["prompt_token_ids"]).encode()).hexdigest())
            self.running[rid] = {"tokens": tokens, "sent": 0, "fr": fr, "sr": sr, "kind": sp.kwargs["output_kind"],
                                 "chunks": chunks}
            ids.append(rid)
        return ids

    def has_unfinished_requests(self):
        return bool(self.running)

    def step(self):
        outs = []
        ids = list(self.running)
        self.rng.shuffle(ids)
        for rid in ids[:self.rng.randint(1, 8)]:
            if self.crash_after is not None and self.finished + len(self.aborted) >= self.crash_after:
                raise KeyboardInterrupt("simulated crash")
            st = self.running[rid]
            new = min(len(st["tokens"]), st["sent"] + st["chunks"].randint(1, 400))
            done = new == len(st["tokens"])
            if st["kind"] == StubOutputKind.DELTA:
                outs.append(self.output(rid, st["tokens"][st["sent"]:new], done, st["fr"] if done else None,
                                        st["sr"] if done else None))
            elif done:
                outs.append(self.output(rid, st["tokens"], True, st["fr"], st["sr"]))
            st["sent"] = new
            if done:
                self.running.pop(rid)
                self.finished += 1
        return outs

    def abort_request(self, ids):
        for rid in ids:
            assert rid in self.running, f"aborted {rid}, which is not running"
            self.running.pop(rid)
        self.aborted.extend(ids)


def load_traces(tokenizer) -> tuple[list[dict], dict]:
    """(requests, prompt ids -> reasoning token ids) for N_TRACES real Qwen3-8B exp02 traces of abortable modes."""
    items = {it["item_id"]: it for it in map(json.loads, cfg.ITEMS_PATH.open())}
    gens = [g for g in map(json.loads, (cfg.EXP02.generations / "Qwen3-8B__card__7af8921892df.jsonl").open())
            if g["mode"] in cfg.ABORT_MODES]
    rng = random.Random(3)
    rng.shuffle(gens)
    fam = cfg.FAMILIES[FAMILY]
    requests, traces = [], {}
    for i, g in enumerate(gens[:N_TRACES]):
        body = tokenizer.encode(g["reasoning"], add_special_tokens=False)[:3000]
        user = f"trace {i}"
        prompt_ids = cc_generate.render_prompt_ids(tokenizer, FAMILY, "", user)
        traces[tuple(prompt_ids)] = [fam["think_start"]] + body
        requests.append({"request_id": f"r{i:03d}", "item_id": g["item_id"], "source": g["source"], "mode": g["mode"],
                         "prompt": g["prompt"], "rollout": 0, "seed": g["seed"], "system": "", "user": user,
                         "abort_on_violation": i % 4 != 0, "full_trace_cell": i % 4 == 0})
    return requests, traces, items


def main() -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir("Qwen3-8B"))
    requests, traces, items = load_traces(tokenizer)
    sampling = cfg.FAMILIES[FAMILY]["sampling"]
    max_len = cfg.VLLM_MAX_MODEL_LEN

    jobs = [{"prompt_ids": cc_generate.render_prompt_ids(tokenizer, FAMILY, "", r["user"]), "seed": r["seed"],
             "abort": r["abort_on_violation"], "mode": r["mode"], "item": items[r["item_id"]]} for r in requests]
    reference = cc_generate.generate_two_phase(FakeEngine(tokenizer, traces), tokenizer, FAMILY, jobs, sampling,
                                               cfg.REASONING_CAP_TOKENS, cfg.ANSWER_CAP_TOKENS, max_len)
    engine = FakeEngine(tokenizer, traces)
    results = [None] * len(jobs)

    def collect(i, result):
        assert results[i] is None, f"job {i} reported twice"
        results[i] = result

    ga.generate_streaming_abort(engine, tokenizer, FAMILY, jobs, sampling, max_len, collect)
    fields = list(reference[0])
    n_abort, n_confirmable = 0, 0
    for job, ref, res in zip(jobs, reference, results):
        n_confirmable += job["abort"] and cc_abort.confirmed_violation(job["mode"], ref["reasoning"], job["item"]) \
            is not None
        if res["think_status"] == "aborted":
            # A trace can also end before its next check and run to the end; only aborted ones are checked here.
            n_abort += 1
            assert job["abort"], "a request without abort_on_violation was aborted"
            assert res["abort_char"] == cc_grade.first_violation_char(job["mode"], ref["reasoning"], job["item"])
            assert ref["reasoning"].startswith(res["reasoning"]), "aborted reasoning is not a prefix"
            assert res["answer"] == "" and res["answer_tokens"] == 0
        else:
            assert {k: res[k] for k in fields} == ref, f"{job['mode']} abort={job['abort']}: result differs"
    assert len(engine.aborted) == n_abort, f"{len(engine.aborted)} engine aborts for {n_abort} aborted traces"
    assert len(engine.answer_prompts) == len(jobs) - n_abort, "an aborted trace got an answer phase"
    assert n_abort >= 0.8 * n_confirmable, f"only {n_abort} of {n_confirmable} confirmable violations aborted"
    n_plain = sum(not j["abort"] for j in jobs)
    print(f"1-2 ok: {n_plain} non-abortable results identical to the batch path; of {len(jobs) - n_plain} "
          f"abortable ({n_confirmable} with a confirmable violation), {n_abort} aborted at the full trace's first "
          f"violation, the rest identical to the batch path")

    crashes = []

    def run_all(parts: Path, n: int, crash: dict | None = None) -> list[str]:
        for k in range(n):
            eng = FakeEngine(tokenizer, traces, crash_after=(crash or {}).get(k))
            try:
                ga.run_shard(eng, tokenizer, FAMILY, requests, items, sampling, max_len, parts, (k, n), "Qwen3-8B")
            except KeyboardInterrupt:
                crashes.append(k)
                with ga.shard_stream_path(parts, (k, n)).open("a") as f:
                    f.write('{"request_id": "r0')  # the line being written when the run died
                ga.run_shard(FakeEngine(tokenizer, traces), tokenizer, FAMILY, requests, items, sampling, max_len,
                             parts, (k, n), "Qwen3-8B")
        return ga.merge(requests, parts, n)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "one").mkdir()
        (tmp / "four").mkdir()
        one = [json.loads(line) for line in run_all(tmp / "one", 1)]
        four = [json.loads(line) for line in run_all(tmp / "four", 4, crash={1: 7, 3: 2})]
    assert [r["request_id"] for r in one] == [r["request_id"] for r in requests]
    assert crashes == [1, 3], f"simulated crashes happened in shards {crashes}, expected [1, 3]"
    diff = [(a["request_id"], k) for a, b in zip(one, four) for k in a if a[k] != b.get(k)]
    assert not diff, f"4 shards (two crashed and resumed) differ from 1 shard: {diff[:5]}"
    print(f"3 ok: 4 shards, two crashed with a partial line and resumed, merge to the 1-shard rows ({len(one)})")
    print("all abort-streaming checks passed")


if __name__ == "__main__":
    main()
