"""CPU check of cc_generate's --shard / --merge-shards path (no GPU, no vLLM).

Generation is replaced by a deterministic function of each request's seed, so a correct shard + merge must give
byte-identical output to one unsharded run_batch. Checks:
 1. shard (N = 1, 2, 3, 4) + merge == unsharded output, byte for byte;
 2. an already-written chunk is kept as is, and only unwritten chunks are sharded (the exp02 Qwen3.6 resume case);
 3. the merge refuses a missing shard file and a shard file whose rows are not that shard's requests.
Run: /venv/main/bin/python scripts/check_sharding.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cc_generate as gen

N_REQUESTS = 2 * gen.CHUNK_SIZE + 37  # three chunks, the last one partial
calls = []  # (number of jobs) per fake generate_two_phase call


def fake_generate_two_phase(llm, tokenizer, family, jobs, sampling, reasoning_cap, answer_cap, max_model_len):
    calls.append(len(jobs))
    return [{"reasoning": f"r{j['seed']}", "reasoning_tokens": j["seed"] % 997, "think_status": "closed",
             "answer": f"ANSWER: {j['seed'] % 4}"} for j in jobs]


def fake_render(tokenizer, family, system, user):
    return [len(user)]


def requests() -> list[dict]:
    return [{"request_id": f"q{i:05d}", "system": "", "user": "x" * (i % 7), "seed": 1000 + 7919 * i}
            for i in range(N_REQUESTS)]


def row_of(request: dict, result: dict) -> dict:
    return {"request_id": request["request_id"], "seed": request["seed"], **result}


def run(parts: Path, shard=None):
    return gen.run_batch(None, None, "qwen3", requests(), {}, 0, parts, row_of, shard)


def sharded(parts: Path, n: int) -> list[str]:
    for k in range(n):
        assert run(parts, (k, n)) is None
    return gen.merge_shards(requests(), parts, n)


def main() -> None:
    gen.generate_two_phase = fake_generate_two_phase
    gen.render_prompt_ids = fake_render
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ref").mkdir()
        reference = run(tmp / "ref")
        ref_bytes = {p.name: p.read_bytes() for p in (tmp / "ref").glob("chunk_???.jsonl")}
        assert len(reference) == N_REQUESTS and len(ref_bytes) == 3

        for n in (1, 2, 3, 4):
            parts = tmp / f"n{n}"
            parts.mkdir()
            assert sharded(parts, n) == reference, f"N={n}: merged lines differ"
            for name, data in ref_bytes.items():
                assert (parts / name).read_bytes() == data, f"N={n}: {name} differs"
        print("1. shard + merge == unsharded, byte for byte (N = 1, 2, 3, 4)")

        parts = tmp / "resume"
        parts.mkdir()
        kept = b'{"request_id": "sentinel"}\n'  # a chunk 0 written earlier (other hardware) must not be touched
        (parts / "chunk_000.jsonl").write_bytes(kept)
        calls.clear()
        for k in range(4):
            run(parts, (k, 4))
        assert sum(calls) == N_REQUESTS - gen.CHUNK_SIZE, f"sharded {sum(calls)} requests, expected only chunks 1-2"
        assert not list(parts.glob("chunk_000.shard*")), "chunk 0 was sharded although it was written"
        lines = gen.merge_shards(requests(), parts, 4)
        assert (parts / "chunk_000.jsonl").read_bytes() == kept and lines[0] == kept.decode().strip()
        assert lines[1:] == reference[gen.CHUNK_SIZE:]  # the kept chunk 0 is one line
        print("2. a written chunk is kept; only unwritten chunks are sharded and merged")

        parts = tmp / "missing"
        parts.mkdir()
        run(parts, (0, 2))
        try:
            gen.merge_shards(requests(), parts, 2)
            raise AssertionError("merge accepted a missing shard")
        except SystemExit as e:
            assert "missing" in str(e)
        wrong = gen.shard_path(parts, 0, (1, 2))
        wrong.write_text(gen.shard_path(parts, 0, (0, 2)).read_text())  # shard 0's rows under shard 1's name
        try:
            gen.merge_shards(requests(), parts, 2)
            raise AssertionError("merge accepted a shard file with the wrong rows")
        except SystemExit as e:
            assert "not shard 1" in str(e)
        print("3. merge refuses a missing shard and a shard with the wrong rows")
    print("all sharding checks passed")


if __name__ == "__main__":
    sys.exit(main())
