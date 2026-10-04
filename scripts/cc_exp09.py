"""exp09_final_test: one locked, pre-registered test of the project's prompt (A: the stacked rule, a failure guide and
up to 12 example turns of the model's own rewritten traces) against CoT-Control's own prompt (baseline), on
questions no earlier experiment has used (results/exp09_final_test/manifest.json, written before any exp09
generation).

Design:
  questions  120 never-used questions, 40 per source (GPQA, HLE, MMLU-Pro), drawn once with DRAW_SEED
  rules      exp07's 9 + 3 new ones (exp09_rules), one formatting, one insertion and one content rule
  models     DESIGN_MODELS (A was chosen on them in exp07b) and FRESH_MODELS (never used to choose anything)
  short rows every model x rule x arm x question: thinking on, stop at the first violation (CoT-Control's rules) or
             at 1200 reasoning tokens, no answer (S(200), S(1000))
  full rows  FULL_MODELS x question x (no rule; FULL_RULES x {baseline, A}): the whole trace (up to 16000 reasoning
             tokens) and the answer (whole-trace compliance; accuracy)
Rows carry "exp07_cell", so cc_generate_abort renders them with cc_exp07's prompt_ids and job, unchanged.

Request order: the engine starts waiting requests in file order, and a run ends when its slowest requests do, so
each request file is sorted longest first (full rows, then A, then baseline), rule by rule within each group
(rows of one rule share their long example prefix in the prefix cache).

Steps (/venv/main/bin/python scripts/cc_exp09.py ...):
  items                 draw the questions (once)
  requests --model M    the model's request file (once) and its record (prompt tokens, A's example counts)
  plan --gpus N         the gpu_lanes.py plan for N H200s (one engine per GPU)
  grade --model M       grade the model's generation (cache/exp09/grades/)
Generation: scripts/gpu_lanes.py results/exp09_final_test/plan_<N>xH200.json
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import random
from functools import lru_cache
from pathlib import Path

import _cotcontrol
import cc_config as cfg
import cc_exp07 as e7
import cc_exp07b as e7b
import cc_prompts
import exp07_candidates as cands
import exp07b_candidates as b
import exp09_rules as R

EXP = cfg.Exp("exp09_final_test", "exp09")
ITEMS_PATH = EXP.cache / "items.jsonl"
DESIGN_MODELS = ["Qwen3.8-27B-FP8", "Gemma-4-31B-FP8", "Qwen3-32B"]
FRESH_MODELS = ["Qwen3.6-27B-FP8", "Qwen3.6-35B-A3B-FP8", "GLM-4.7-Flash-FP8", "Gemma-4-12B-FP8"]
MODELS = DESIGN_MODELS + FRESH_MODELS
# Whose own unconstrained traces (exp07's or exp07b's pre-pass) give a model's few-shot examples: the FP8-quantized
# GLM and Gemma 12B use the traces of the same checkpoints in bf16.
EXAMPLE_SOURCE = {"GLM-4.7-Flash-FP8": "GLM-4.7-Flash", "Gemma-4-12B-FP8": "Gemma-4-12B"}
DRAW_SEED = 20261005
N_PER_SOURCE = 40
ARMS = ["A", "baseline"]  # longest prompt first
FULL_MODELS = MODELS
FULL_RULES = ["lowercase_thinking", "no_commas"]
FULL_ARMS = ["A", "baseline"]
FULL_REASONING_CAP = 16000
CONTEXT_MARGIN = 256  # tokens kept free beyond the prompt and the reasoning stop (short rows; exp07b's and exp08's)
# Full rows: beyond the prompt, the 16000 reasoning tokens and the answer, only room for the forced close, so every
# full row keeps the whole reasoning cap (cc_generate_abort.reasoning_caps would otherwise shorten it).
FULL_CONTEXT_MARGIN = 64
PROMPT_LIMIT = {"short": cfg.VLLM_MAX_MODEL_LEN - e7.STOP_TOKENS - CONTEXT_MARGIN,
                "full": cfg.VLLM_MAX_MODEL_LEN - FULL_REASONING_CAP - cfg.ANSWER_CAP_TOKENS - FULL_CONTEXT_MARGIN}


# --- Items ----------------------------------------------------------------------------------------------------------
def used_item_ids() -> set[str]:
    """Every question in any earlier experiment's item or request file (exp01-exp08)."""
    paths = [Path(p) for pattern in ("cache/exp0[1-8]*/**/requests*.jsonl", "cache/exp0[1-8]*/**/items*.jsonl")
             for p in glob.glob(str(cfg.REPO_ROOT / pattern), recursive=True)]
    return e7.used_item_ids() | {json.loads(line)["item_id"] for p in paths for line in p.open() if line.strip()}


def draw_items() -> None:
    if ITEMS_PATH.exists():
        raise SystemExit(f"{ITEMS_PATH} exists; the questions are fixed once drawn")
    import cc_items
    used = used_item_ids()
    run_cceval = _cotcontrol.module("run_cceval")
    rng = random.Random(DRAW_SEED)
    out, record = [], {"seed": DRAW_SEED, "excluded_used_items": len(used), "unused_left": {}}
    for source in e7.SOURCES:
        pool = [r for r in cc_items.load_source(run_cceval, source) if r["item_id"] not in used and r["options"]]
        rng.shuffle(pool)
        if len(pool) < N_PER_SOURCE:
            raise SystemExit(f"{source}: only {len(pool)} unused items")
        out += [{**r, "split": "final_test"} for r in pool[:N_PER_SOURCE]]
        record["unused_left"][source] = len(pool) - N_PER_SOURCE
    EXP.cache.mkdir(parents=True, exist_ok=True)
    EXP.results.mkdir(parents=True, exist_ok=True)
    ITEMS_PATH.write_text("".join(json.dumps(it) + "\n" for it in out))
    record["item_ids"] = [it["item_id"] for it in out]
    record["items_file_key"] = cfg.content_key({"items": out})
    (EXP.results / "items.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(out)} items; unused left: {record['unused_left']}")


@lru_cache(maxsize=1)
def items() -> list[dict]:
    return [json.loads(line) for line in ITEMS_PATH.open()]


# --- Examples and prompts -------------------------------------------------------------------------------------------
def bare_context(item: dict, mode: str) -> cands.Context:
    return cands.Context(rule=R.rule_of(item, mode), question_block=cc_prompts.question_block(item),
                         cotcontrol_user=R.cotcontrol_user(item, mode))


def context(item: dict, mode: str, examples: list[cands.Example]) -> cands.Context:
    return cands.Context(rule=R.rule_of(item, mode), question_block=cc_prompts.question_block(item),
                         cotcontrol_user=R.cotcontrol_user(item, mode), examples=examples)


@lru_cache(maxsize=None)
def usable_examples(model: str, mode: str) -> list[cands.Example]:
    """Every own trace (example-pool order) whose rewritten 1000-character cut passes the grader: exp07b's (and
    exp08's) examples for exp07's rules, the same construction with exp09_rules for the new ones."""
    import cc_exp06a
    source = EXAMPLE_SOURCE.get(model, model)
    if mode not in R.NEW_RULES:
        return e7b.usable_examples(source, mode)
    out = []
    for trace in e7b.own_traces(source):
        item = trace["item"]

        def reasoning(max_chars: int, text=trace["reasoning"], item=item) -> str:
            cut = cc_exp06a.cut_example(text) if max_chars == cfg.EXP06A_FEWSHOT_MAX_CHARS else e7.cut_at(text,
                                                                                                         max_chars)
            rewritten = R.rewrite(item, mode, cut)
            if not R.compliant(mode, rewritten, item):
                raise ValueError(f"{item['item_id']} {mode}: rewritten example fails the grader")
            return rewritten
        try:
            reasoning(cfg.EXP06A_FEWSHOT_MAX_CHARS)
        except ValueError:
            continue
        out.append(cands.Example(context=bare_context(item, mode), reasoning=reasoning,
                                 answer=trace["answer"].strip()))
    if len(out) < 3:
        raise SystemExit(f"{model} {mode}: only {len(out)} usable examples")
    return out


def prompt_tokens(tokenizer, family: str, prompt: cands.Prompt) -> int:
    return len(e7.prompt_ids(tokenizer, family, {"system": prompt.system, "user": prompt.user,
                                                 "history": prompt.history, "thinking": True, "request_id": "fit"}))


def fit_examples(tokenizer, family: str, mode: str, examples: list[cands.Example], limit: int) -> int:
    """A's n: the largest n <= 12 (and <= the usable examples) whose prompt fits under limit for every question."""
    for n in range(min(b.MAX_EXAMPLES, len(examples)), 0, -1):
        if max(prompt_tokens(tokenizer, family, b.many_examples(context(it, mode, examples), mode, n))
               for it in items()) <= limit:
            return n
    return 0


# --- Rows -----------------------------------------------------------------------------------------------------------
REQUEST_ORDER = [("full", "A"), ("full", "baseline"), ("full", "no_rule"), ("short", "A"), ("short", "baseline")]


def row(item: dict, mode: str, arm: str, prompt: cands.Prompt, channel: str) -> dict:
    full = channel == "full"
    graded_by_cotcontrol = mode not in R.LOCATOR_GRADED and mode != cfg.NO_CONSTRAINT
    return e7.with_id({
        "item_id": item["item_id"], "source": item["source"], "mode": mode, "prompt": arm, "channel": channel,
        "system": prompt.system, "user": prompt.user, "history": prompt.history, "thinking": True,
        "answer_phase": full, "reasoning_stop_tokens": FULL_REASONING_CAP if full else e7.STOP_TOKENS,
        "exp07_cell": f"exp09|{arm}|{channel}", "full_trace_cell": full, "n_examples": len(prompt.history),
        "grading_prompt": R.cotcontrol_user(item, mode) if graded_by_cotcontrol else None,
        "abort_on_violation": not full and mode in R.ABORTABLE,
        # full rows are separate samples from the short rows of the same prompt
        "seed": cfg.rollout_seed(item["item_id"], f"exp09_full_{mode}" if full else mode, 0)})


def model_rows(model: str) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(model))
    family = cfg.ALL_MODELS[model]["family"]
    e7.check_history_template(tokenizer, family)
    out, record = [], {"model": model, "example_source": EXAMPLE_SOURCE.get(model, model), "rules": {}}
    for mode in R.ALL_RULES:
        examples = usable_examples(model, mode)
        n_short = fit_examples(tokenizer, family, mode, examples, PROMPT_LIMIT["short"])
        n_full = fit_examples(tokenizer, family, mode, examples[:n_short], PROMPT_LIMIT["full"])
        record["rules"][mode] = {"usable_examples": len(examples), "A_examples": n_short}
        if n_short <= 3:
            raise SystemExit(f"{model} {mode}: A fits only {n_short} examples, no more than exp06's 3")
        for it in items():
            ctx = context(it, mode, examples)
            prompts = {"A": b.many_examples(ctx, mode, n_short), "baseline": cands.baseline(ctx)}
            out += [row(it, mode, arm, prompts[arm], "short") for arm in ARMS]
            if model in FULL_MODELS and mode in FULL_RULES:
                if n_full != n_short:
                    raise SystemExit(f"{model} {mode}: A's {n_short} examples do not fit the full-trace limit")
                out += [row(it, mode, arm, prompts[arm], "full") for arm in FULL_ARMS]
    if model in FULL_MODELS:
        for it in items():
            no_rule = cands.Prompt(*cc_prompts.cotcontrol_prompt(it, "baseline"), [])
            out.append(row(it, cfg.NO_CONSTRAINT, "no_rule", no_rule, "full"))
    if len({r["request_id"] for r in out}) != len(out):
        raise RuntimeError(f"{model}: duplicate request ids")
    rule_order = {m: k for k, m in enumerate(R.ALL_RULES + [cfg.NO_CONSTRAINT])}
    item_order = {it["item_id"]: k for k, it in enumerate(items())}
    out.sort(key=lambda r: (REQUEST_ORDER.index((r["channel"], r["prompt"])), rule_order[r["mode"]],
                            item_order[r["item_id"]]))
    tokens = {}
    for r in out:
        key = f"{r['channel']}|{r['prompt']}"
        n = len(e7.prompt_ids(tokenizer, family, r))
        tokens.setdefault(key, {"rows": 0, "prompt_tokens": 0, "longest_prompt": 0})
        tokens[key]["rows"] += 1
        tokens[key]["prompt_tokens"] += n
        tokens[key]["longest_prompt"] = max(tokens[key]["longest_prompt"], n)
    record["cells"] = tokens
    return out, record


# --- Paths and grading ----------------------------------------------------------------------------------------------
def requests_path(model: str) -> Path:
    return EXP.cache / f"requests_{model}.jsonl"


def generation_path(model: str) -> Path:
    paths = glob.glob(str(EXP.generations / f"{model}__card__stream_abort_{model}__*.jsonl"))
    if len(paths) != 1:
        raise SystemExit(f"{model}: expected one generation file, found {len(paths)}")
    return Path(paths[0])


def grades_path(model: str) -> Path:
    return EXP.grades / generation_path(model).name


def grade(model: str) -> Path:
    """exp07's grading (cc_grade.grade_row; exp07_rules' locator for no_digits and no_first_person); for exp09's new
    rules the same flags, with compliance and the first violation from exp09_rules."""
    import cc_grade
    from transformers import AutoTokenizer
    out = grades_path(model)
    if out.exists():
        return out
    requests = {r["request_id"]: r for r in map(json.loads, requests_path(model).open())}
    by_id = {it["item_id"]: it for it in items()}
    gens = [json.loads(line) for line in generation_path(model).open()]
    graded = {r["request_id"]: r for r in e7.grade_generations(
        [g for g in gens if g["mode"] not in R.NEW_RULES], requests, by_id, model)}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(model))

    def offsets_of(text: str) -> list[int]:
        return [s for s, _ in tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]

    for g in gens:
        if g["mode"] not in R.NEW_RULES:
            continue
        request, item = requests[g["request_id"]], by_id[g["item_id"]]
        r = cc_grade.grade_row({**g, "mode": cfg.NO_CONSTRAINT}, request, item, offsets_of)
        char = R.first_violation_new(g["mode"], g["reasoning"], item) if g["reasoning"] else None
        r.update(mode=g["mode"], compliant=char is None and bool(g["reasoning"]), locator_agrees=True,
                 prompt=request["prompt"])
        if char is not None:
            r.update(fv_char=char, fv_token=max(0, bisect.bisect_right(offsets_of(g["reasoning"]), char) - 1),
                     fv_rel=char / max(1, len(g["reasoning"])))
        graded[g["request_id"]] = r
    rows = []
    for g in gens:
        r, request = graded[g["request_id"]], requests[g["request_id"]]
        r.update(channel=request["channel"], cell=f"{request['prompt']}|{request['channel']}",
                 n_examples=request["n_examples"], kind=R.RULE_KIND.get(g["mode"], "none"),
                 model_group="design" if model in DESIGN_MODELS else "fresh")
        rows.append(r)
    EXP.grades.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


# --- Plans (scripts/gpu_lanes.py) -----------------------------------------------------------------------------------
# Per GPU count: per GPU, its lanes (models run one after another). One engine per GPU with 0.90 of its memory: in a
# benchmark on exp08 requests (log/exp09/lanes_bench_*), Qwen3.8 and Qwen3.6-27B sharing a GPU under MPS (0.45
# each) generated about a quarter of the tokens one of them generates alone in the same time (a third of the KV cache
# each, for 12-example prompts and 16000-token traces). Lanes are balanced by each model's estimated minutes.
LANES = {
    1: {"0": [["Qwen3.6-27B-FP8", "Gemma-4-31B-FP8", "Qwen3.8-27B-FP8", "Qwen3-32B", "Gemma-4-12B-FP8",
               "Qwen3.6-35B-A3B-FP8", "GLM-4.7-Flash-FP8"]]},
    2: {"0": [["Qwen3.6-27B-FP8", "Gemma-4-31B-FP8"]],
        "1": [["Qwen3.8-27B-FP8", "Qwen3-32B", "Gemma-4-12B-FP8", "Qwen3.6-35B-A3B-FP8", "GLM-4.7-Flash-FP8"]]},
}
MEMORY = {model: 0.90 for model in MODELS}
USE_MPS = False


def write_plan(n_gpus: int) -> Path:
    def job(model: str) -> dict:
        return {"exp": "exp09", "model": model, "requests": str(requests_path(model).relative_to(cfg.REPO_ROOT)),
                "items": str(ITEMS_PATH.relative_to(cfg.REPO_ROOT)), "memory": MEMORY[model],
                "after": ["/venv/main/bin/python", "cc_exp09.py", "grade", "--model", model]}
    lanes = LANES[n_gpus]
    planned = sorted(m for gpu in lanes.values() for lane in gpu for m in lane)
    if planned != sorted(MODELS):
        raise SystemExit(f"plan for {n_gpus} GPUs does not run every model once: {planned}")
    plan = {"name": f"exp09_{n_gpus}xH200", "mps": USE_MPS,
            "gpus": {gpu: [[job(m) for m in lane] for lane in gpu_lanes] for gpu, gpu_lanes in lanes.items()}}
    path = EXP.results / f"plan_{n_gpus}xH200.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


# --- CLI ------------------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["items", "check", "requests", "plan", "grade"],
                        help="check: build a model's rows and print its record, writing nothing")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--gpus", type=int, choices=sorted(LANES))
    args = parser.parse_args()
    if args.command == "items":
        draw_items()
    elif args.command == "check":
        rows, record = model_rows(args.model)
        check_dir = Path("/tmp/exp09_check")
        check_dir.mkdir(exist_ok=True)
        (check_dir / f"record_{args.model}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=1))
        print(f"{len(rows)} rows; first: {rows[0]['channel']} {rows[0]['prompt']} {rows[0]['mode']}; "
              f"last: {rows[-1]['channel']} {rows[-1]['prompt']} {rows[-1]['mode']}")
    elif args.command == "requests":
        rows, record = model_rows(args.model)
        e7.write_requests(requests_path(args.model), rows)
        (EXP.results / f"requests_record_{args.model}.json").write_text(json.dumps(record, indent=2) + "\n")
    elif args.command == "plan":
        print(write_plan(args.gpus))
    else:
        print(grade(args.model))


if __name__ == "__main__":
    main()
