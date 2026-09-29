"""GPU check before a pre-registered run uses a non-default engine profile: does it produce the same traces as v1?

Engine profiles are meant to change only scheduling (cc_config.ENGINE_PROFILES), and check_streaming.py shows on a
fake engine that "stream" issues exactly v1's requests. What remains is the real engine: batch composition changes
floating-point results, so long traces can diverge at some token even when both paths are correct. The strongest
test is greedy decoding, where a correct profile should reproduce most traces exactly and diverge late when it
does not. With card sampling, only the distributions (status rates, lengths, compliance, accuracy) should agree.

  prepare: copy a request subset (default: the 120 smoke requests) into cache/<exp>/engine_check/<key>/, so check
           generations never land in cache/<exp>/generations/ (which grading and analysis read).
  compare: two generation files over the same requests -> per-file summary, paired agreement, wall time.

Run:
  /venv/main/bin/python scripts/check_engine_equivalence.py prepare --exp exp02 [--source experiment
      --prompts baseline --modes word_suppression]
  scripts/vllm_python.sh scripts/cc_generate.py --exp exp02 --model Qwen3-8B --requests <printed path> --sampling greedy
  scripts/vllm_python.sh scripts/cc_generate.py ... same ... --engine stream
  /venv/main/bin/python scripts/check_engine_equivalence.py compare --exp exp02 <v1 file> <stream file>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cc_config as cfg
import cc_grade


def prepare(exp_key: str, source: str, prompts: list[str] | None, modes: list[str] | None) -> Path:
    exp = cfg.EXPERIMENTS[exp_key]
    path = exp.cache / "smoke" / "requests.jsonl" if source == "smoke" else exp.requests
    rows = [r for r in map(json.loads, path.open())
            if (not prompts or r["prompt"] in prompts) and (not modes or r["mode"] in modes)]
    if not rows:
        raise SystemExit("no requests match")
    out = exp.cache / "engine_check" / cfg.content_key([r["request_id"] for r in rows]) / "requests.jsonl"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} requests from {path.relative_to(cfg.REPO_ROOT)} -> {out}")
    return out


def first_difference(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def wall_seconds(gen_path: Path) -> int | None:
    timings = sorted(gen_path.with_suffix(".parts").glob("timing_*.json"))
    return json.loads(timings[-1].read_text())["seconds"] if timings else None


def summary(gens: list[dict], grades: list[dict]) -> dict:
    tokens = [g["reasoning_tokens"] for g in gens]
    graded = [r for r in grades if r["mode"] not in cfg.JUDGED_MODES and r["mode"] != cfg.NO_CONSTRAINT]
    return {"n": len(gens),
            **{f"{s} %": 100 * sum(g["think_status"] == s for g in gens) / len(gens)
               for s in ("closed", "truncated", "no_think_close")},
            "median reasoning tokens": statistics.median(tokens), "mean reasoning tokens": statistics.mean(tokens),
            "accuracy %": 100 * sum(r["correct"] for r in grades) / len(grades),
            "programmatic compliance %": 100 * sum(bool(r["compliant"]) for r in graded) / max(1, len(graded))}


def compare(exp_key: str, path_a: Path, path_b: Path) -> None:
    items = {it["item_id"]: it for it in map(json.loads, cfg.ITEMS_PATH.open())}
    requests = cc_grade.load_requests(exp_key)
    gens, grades = {}, {}
    for path in (path_a, path_b):
        requests.update({r["request_id"]: r for r in map(json.loads, (path.parent.parent / "requests.jsonl").open())})
        gens[path] = {g["request_id"]: g for g in map(json.loads, path.open())}
        grades[path] = [json.loads(line) for line in cc_grade.grade_file(path, requests, items).open()]
    if set(gens[path_a]) != set(gens[path_b]):
        raise SystemExit("the two files cover different requests")

    print(f"{'':28s}{path_a.name[:40]:>42s}{path_b.name[:40]:>42s}")
    sa, sb = summary(list(gens[path_a].values()), grades[path_a]), summary(list(gens[path_b].values()), grades[path_b])
    for key in sa:
        print(f"{key:28s}{sa[key]:>42.1f}{sb[key]:>42.1f}")
    print(f"{'wall seconds':28s}{str(wall_seconds(path_a)):>42s}{str(wall_seconds(path_b)):>42s}")

    ids = sorted(gens[path_a])
    same_reasoning = [i for i in ids if gens[path_a][i]["reasoning"] == gens[path_b][i]["reasoning"]]
    letters = {p: {r["request_id"]: r["answer_letter"] for r in grades[p]} for p in (path_a, path_b)}
    same_letter = sum(letters[path_a][i] == letters[path_b][i] for i in ids)
    diverged = [i for i in ids if i not in set(same_reasoning)]
    print(f"\nidentical reasoning: {len(same_reasoning)}/{len(ids)}; same answer letter: {same_letter}/{len(ids)}")
    if diverged:
        rel = [first_difference(gens[path_a][i]["reasoning"], gens[path_b][i]["reasoning"])
               / max(1, len(gens[path_a][i]["reasoning"])) for i in diverged]
        print(f"diverged traces: first difference at median {100 * statistics.median(rel):.0f}% of the v1 trace "
              f"(early divergence, near 0%, points to different requests rather than floating-point drift)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--exp", choices=list(cfg.EXPERIMENTS), required=True)
    p.add_argument("--source", choices=["smoke", "experiment"], default="smoke")
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--modes", nargs="*", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--exp", choices=list(cfg.EXPERIMENTS), required=True)
    c.add_argument("reference", type=Path, help="v1 generation file")
    c.add_argument("candidate", type=Path, help="generation file of the profile under test")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.exp, args.source, args.prompts, args.modes)
    else:
        compare(args.exp, args.reference, args.candidate)


if __name__ == "__main__":
    main()
