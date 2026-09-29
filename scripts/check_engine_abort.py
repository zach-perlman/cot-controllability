"""GPU check before exp03: does the streaming-with-abort generator (cc_generate_abort) behave like the v1 batch
engine on real traces?

Both run the 120 exp02 smoke requests (Qwen3-8B, greedy), with abort_on_violation set for the abortable modes.
Greedy decoding makes the two paths comparable trace by trace, up to floating-point drift from different batching.
  - Traces the candidate did not abort: share with identical reasoning, and where diverged ones first differ.
  - Aborted traces whose reasoning is a prefix of the v1 trace (no drift before the abort): the v1 trace must be
    non-compliant with its first violation at the abort character. This is the pass rule (exit 1 otherwise).
  - Aborted traces that drifted before the abort: counted, and checked against their own prefix only.
  - Compliance (aborted = non-compliant), reasoning tokens generated, wall seconds.

Run:
  /venv/main/bin/python scripts/check_engine_abort.py prepare
  scripts/vllm_python.sh scripts/cc_generate.py --exp exp02 --model Qwen3-8B --requests <path> --sampling greedy
  scripts/vllm_python.sh scripts/cc_generate_abort.py --exp exp03 --model Qwen3-8B --requests <path> \
      --items cache/exp02/items.jsonl --sampling greedy --shard 0/1   (then --merge-shards 1)
  /venv/main/bin/python scripts/check_engine_abort.py compare <v1 file> <abort file>
Output of compare: results/exp03_abort_survival/engine_check.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cc_config as cfg
import cc_grade


def prepare() -> Path:
    rows = [json.loads(line) for line in (cfg.EXP02.cache / "smoke" / "requests.jsonl").open()]
    for r in rows:
        r.update(abort_on_violation=r["mode"] in cfg.ABORT_MODES, full_trace_cell=False)
    out = cfg.EXP03.cache / "engine_check" / cfg.content_key([r["request_id"] for r in rows]) / "requests.jsonl"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(out)
    return out


def wall_seconds(gen_path: Path) -> int:
    return sum(json.loads(p.read_text())["seconds"] for p in gen_path.with_suffix(".parts").glob("timing_*.json"))


def compare(path_v1: Path, path_abort: Path) -> None:
    items = {it["item_id"]: it for it in map(json.loads, cfg.ITEMS_PATH.open())}
    requests = {r["request_id"]: r for r in map(json.loads, (path_v1.parent.parent / "requests.jsonl").open())}
    v1 = {g["request_id"]: g for g in map(json.loads, path_v1.open())}
    ab = {g["request_id"]: g for g in map(json.loads, path_abort.open())}
    if set(v1) != set(ab):
        raise SystemExit("the two files cover different requests")
    grades_v1 = {r["request_id"]: r for r in map(json.loads, cc_grade.grade_file(path_v1, requests, items).open())}
    grades_ab = {r["request_id"]: r for r in map(json.loads, cc_grade.grade_file(path_abort, requests, items).open())}

    aborted = [i for i in ab if ab[i]["think_status"] == "aborted"]
    kept = [i for i in ab if i not in set(aborted)]
    same = [i for i in kept if ab[i]["reasoning"] == v1[i]["reasoning"]]
    same_prefix = [i for i in aborted if v1[i]["reasoning"].startswith(ab[i]["reasoning"])]
    failures = [i for i in same_prefix
                if grades_v1[i]["compliant"] or grades_v1[i]["fv_char"] != ab[i]["abort_char"]]
    drifted = [i for i in aborted if i not in set(same_prefix)]
    drift_ok = [i for i in drifted if not grades_ab[i]["compliant"] and grades_ab[i]["fv_char"] == ab[i]["abort_char"]]

    def first_diff(a: str, b: str) -> float:
        n = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return n / max(1, len(a))

    graded = [i for i in ab if requests[i]["mode"] in cfg.ABORT_MODES]
    result = {
        "requests": len(ab), "aborted": len(aborted),
        "not_aborted_identical_reasoning": f"{len(same)}/{len(kept)}",
        "not_aborted_diverged_first_difference_median_rel": (
            statistics.median(first_diff(v1[i]["reasoning"], ab[i]["reasoning"]) for i in kept if i not in same)
            if len(same) < len(kept) else None),
        "aborted_prefix_of_v1": len(same_prefix),
        "aborted_prefix_of_v1_but_v1_disagrees (must be 0)": len(failures),
        "aborted_after_drift": len(drifted), "aborted_after_drift_consistent_with_own_prefix": len(drift_ok),
        "abortable_mode_compliance_v1": sum(bool(grades_v1[i]["compliant"]) for i in graded) / len(graded),
        "abortable_mode_compliance_abort": sum(bool(grades_ab[i]["compliant"]) for i in graded) / len(graded),
        "reasoning_tokens_v1": sum(g["reasoning_tokens"] for g in v1.values()),
        "reasoning_tokens_abort": sum(g["reasoning_tokens"] for g in ab.values()),
        "wall_seconds_v1": wall_seconds(path_v1), "wall_seconds_abort": wall_seconds(path_abort),
        "failures": failures[:10], "pass": not failures and len(drift_ok) == len(drifted),
        "files": [path_v1.name, path_abort.name],
    }
    out = cfg.EXP03.results / "engine_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    c = sub.add_parser("compare")
    c.add_argument("reference", type=Path)
    c.add_argument("candidate", type=Path)
    args = parser.parse_args()
    prepare() if args.command == "prepare" else compare(args.reference, args.candidate)
