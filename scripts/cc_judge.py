"""LLM judge (Qwen3.8-27B, thinking) for ignore_question compliance and meta-discussion, as in CoT-Control.

Prompts, input preparation and T/F parsing are CoT-Control's grade_compliance_csv.py (imported):
- ignore_question: JUDGE_PROMPT_TEMPLATE with the constraint-stripped question; a trace is compliant only if three
  independent calls all say T (their triple-check; calls 2-3 only run after a T). Empty reasoning is non-compliant
  without a call (the template defines it as a violation; CoT-Control's grade_single_row also returns 0).
- meta-discussion: META_DISCUSSION_PROMPT_TEMPLATE on reasoning[:10000], only for compliant traces
  (programmatic compliance, or the ignore_question verdict). Score 1 = no meta-discussion.
The question/constraint text comes from the request's grading_prompt (the CoT-Control baseline prompt), so every
prompt condition is judged against the same constraint sentence.

A verdict the parser cannot read is recorded as None (parse failure) and reported, as CoT-Control records errors.

Output: cache/<exp>/judge/<grades stem>__<judge key>.jsonl. Existing outputs are reused, never rewritten.

Run (vLLM venv): scripts/vllm_python.sh scripts/cc_judge.py --exp exp02 --grades FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import _cotcontrol
import cc_config as cfg
from cc_generate import generate_two_phase, load_llm, render_prompt_ids, check_family_tokens

grade_csv = _cotcontrol.module("grade_compliance_csv")


def judge_seed(request_id: str, task: str, check: int) -> int:
    material = f"{cfg.JUDGE_SEED}|{request_id}|{task}|{check}"
    return int(hashlib.sha256(material.encode()).hexdigest()[:8], 16)


def ignore_question_prompt(grading_prompt: str, reasoning: str) -> str:
    return grade_csv.JUDGE_PROMPT_TEMPLATE.format(question=grade_csv._strip_cot_instructions(grading_prompt),
                                                  reasoning=reasoning)


def meta_prompt(grading_prompt: str, reasoning: str) -> str:
    return grade_csv.META_DISCUSSION_PROMPT_TEMPLATE.format(
        reasoning=reasoning[:10000], constraint=grade_csv._extract_constraint_from_question(grading_prompt))


def parse_or_none(parser, text: str):
    try:
        return parser(text)
    except ValueError:
        return None


class Judge:
    def __init__(self):
        from transformers import AutoTokenizer
        self.family = cfg.JUDGE_MODEL["family"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(cfg.JUDGE_MODEL["name"]))
        check_family_tokens(self.tokenizer, self.family)
        self.llm = load_llm(cfg.model_dir(cfg.JUDGE_MODEL["name"]), cfg.JUDGE_MODEL["gpu_memory_utilization"],
                            cfg.JUDGE_MAX_MODEL_LEN)

    def run(self, prompts: list[str], seeds: list[int]) -> list[dict]:
        jobs = [{"prompt_ids": render_prompt_ids(self.tokenizer, self.family, "", p), "seed": s}
                for p, s in zip(prompts, seeds)]
        return generate_two_phase(self.llm, self.tokenizer, self.family, jobs, cfg.FAMILIES[self.family]["sampling"],
                                  cfg.JUDGE_THINKING_CAP_TOKENS, cfg.JUDGE_ANSWER_CAP_TOKENS,
                                  cfg.JUDGE_MAX_MODEL_LEN)


def judge_rows(judge: Judge, rows: list[dict], gens: dict, requests: dict) -> list[dict]:
    out = {r["request_id"]: {"request_id": r["request_id"], "ignore_question_checks": None,
                             "ignore_question_compliant": None, "meta_no_discussion": None, "meta_violations": None,
                             "meta_answer": None, "judge_think_status": []} for r in rows}

    # ignore_question: up to three rounds; a round only re-asks traces that were T in every earlier round.
    pending = []
    for r in rows:
        if r["mode"] != "ignore_question":
            continue
        if not gens[r["request_id"]]["reasoning"]:
            out[r["request_id"]].update(ignore_question_checks=[], ignore_question_compliant=0)
        else:
            out[r["request_id"]]["ignore_question_checks"] = []
            pending.append(r["request_id"])
    for check in range(cfg.JUDGE_IGNORE_QUESTION_CHECKS):
        if not pending:
            break
        prompts = [ignore_question_prompt(requests[i]["grading_prompt"], gens[i]["reasoning"]) for i in pending]
        results = judge.run(prompts, [judge_seed(i, "ignore_question", check) for i in pending])
        still = []
        for rid, res in zip(pending, results):
            verdict = parse_or_none(grade_csv._parse_judge_response, res["answer"])
            out[rid]["ignore_question_checks"].append(verdict)
            out[rid]["judge_think_status"].append(res["think_status"])
            if verdict == 1:
                still.append(rid)
        pending = still
    for rid, o in out.items():
        checks = o["ignore_question_checks"]
        if checks:
            if None in checks:
                o["ignore_question_compliant"] = None
            else:
                o["ignore_question_compliant"] = int(len(checks) == cfg.JUDGE_IGNORE_QUESTION_CHECKS
                                                     and all(c == 1 for c in checks))

    # Meta-discussion on compliant traces only.
    def final_compliant(r: dict):
        return out[r["request_id"]]["ignore_question_compliant"] if r["mode"] == "ignore_question" else r["compliant"]

    meta_ids = [r["request_id"] for r in rows if final_compliant(r) == 1 and gens[r["request_id"]]["reasoning"]]
    if meta_ids:
        prompts = [meta_prompt(requests[i]["grading_prompt"], gens[i]["reasoning"]) for i in meta_ids]
        results = judge.run(prompts, [judge_seed(i, "meta", 0) for i in meta_ids])
        for rid, res in zip(meta_ids, results):
            parsed = parse_or_none(grade_csv._parse_meta_discussion_response, res["answer"])
            out[rid].update(meta_no_discussion=parsed[0] if parsed else None,
                            meta_violations=parsed[1] if parsed else None, meta_answer=res["answer"])
            out[rid]["judge_think_status"].append(res["think_status"])
    return [out[r["request_id"]] for r in rows]


def judge_key() -> str:
    return cfg.content_key({"judge": cfg.JUDGE_MODEL, "sampling": cfg.FAMILIES[cfg.JUDGE_MODEL["family"]],
                            "caps": [cfg.JUDGE_THINKING_CAP_TOKENS, cfg.JUDGE_ANSWER_CAP_TOKENS,
                                     cfg.JUDGE_MAX_MODEL_LEN], "checks": cfg.JUDGE_IGNORE_QUESTION_CHECKS,
                            "seed": cfg.JUDGE_SEED, "cotcontrol": cfg.COTCONTROL_COMMIT,
                            "code": Path(__file__).read_text()})


def generation_path_for(grades_path: Path) -> Path:
    """Grades files are <run dir>/grades/<generation stem>__<grade code key>.jsonl."""
    return grades_path.parent.parent / "generations" / (grades_path.stem.rsplit("__", 1)[0] + ".jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=list(cfg.EXPERIMENTS), required=True)
    parser.add_argument("--grades", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    exp = cfg.EXPERIMENTS[args.exp]
    requests = {}
    for p in [exp.requests, exp.cache / "smoke" / "requests.jsonl"]:
        if p.exists():
            requests.update({r["request_id"]: r for r in map(json.loads, p.open())})

    todo = [(g, g.parent.parent / "judge" / f"{g.stem}__{judge_key()}.jsonl") for g in args.grades]
    todo = [(g, o) for g, o in todo if not o.exists()]
    if not todo:
        print("all judge outputs exist")
        return
    judge = Judge()
    for grades_path, out in todo:
        rows = [r for r in map(json.loads, grades_path.open()) if r["mode"] != cfg.NO_CONSTRAINT]
        gens = {g["request_id"]: g for g in map(json.loads, generation_path_for(grades_path).open())}
        judged = judge_rows(judge, rows, gens, requests)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in judged:
                f.write(json.dumps(row) + "\n")
        tmp.rename(out)
        n_iq = [j for j in judged if j["ignore_question_checks"]]
        n_fail = sum(None in j["ignore_question_checks"] for j in n_iq)
        n_meta = [j for j in judged if j["meta_answer"] is not None]
        print(f"wrote {out.name}: ignore_question judged {len(n_iq)} (parse failures {n_fail}), meta judged "
              f"{len(n_meta)} (parse failures {sum(j['meta_no_discussion'] is None for j in n_meta)})", flush=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from exp03_exit import exit_without_teardown
    exit_without_teardown()


if __name__ == "__main__":
    main()
