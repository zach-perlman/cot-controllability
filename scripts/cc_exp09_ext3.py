"""exp09 extension 3 (results/exp09_final_test/manifest_extension3.json, written before its generation): three parts
on exp09's 120 questions and 7 models, each in its own request file per model:

  necessity      (stem "nec") exp04's CoT-necessity rows: thinking off, CoT-Control's no-rule prompt, the response
                 starts with "ANSWER:" and is one option letter; 5 samples per question. A question is "necessary" for
                 a model if at most 1 of 5 is right, "unnecessary" if at least 4 are (cfg.EXP04_NECESSITY_*)
  examples only  (stem "exonly") A's example turns (exp09's examples, as many as exp09's A row) with no rule anywhere:
                 every user message is CoT-Control's no-rule prompt, the example reasoning is still the rewritten,
                 rule-following one. Thinking on, stop at the first violation or 1200 tokens, graded under the rule
                 (separates following the instruction from copying the examples)
  openings off   (stem "offopen") extension 2's openings with thinking off: A thinking off (extension 2's prompt and
                 example count) and the no-rule thinking-off prompt, with the response prefilled
                 "<output_reasoning>\\n" + the opening (exp06's thinking-off prefill); the continuation is graded

Rows of the last two parts are extension 2's row format (cc_exp09_ext2.row: "exp06_cell", cc_exp06's renderer;
GLM-4.7-Flash's through cc_exp09_render); necessity rows are exp04's (cc_generate_abort's exp04 rendering and its
one-letter restriction), unchanged.

Steps (/venv/main/bin/python scripts/cc_exp09_ext3.py ...):
  check --model M      build the model's three files' rows and run the checks (writes nothing)
  requests --model M   write cache/exp09/requests_{nec,exonly,offopen}_<M>.jsonl (once each) and the record
  plan                 the gpu_lanes.py plan (results/exp09_final_test/plan_ext3_1xB200.json): extension 2, examples
                       only and necessity per model, then openings off per model (the longest part, last)
  grade --model M --part P
"""

from __future__ import annotations

import argparse
import json

import cc_config as cfg
import cc_exp04
import cc_exp07 as e7
import cc_exp07b as e7b
import cc_exp08 as e8
import cc_exp09 as e9
import cc_exp09_ext as x1
import cc_exp09_ext2 as x2
import cc_prompts
import exp07_candidates as cands
import exp07b_candidates as b
import exp09_rules as R

PARTS = ["nec", "exonly", "offopen"]
# Openings off is not run for the two models with the longest thinking-off responses (about 3000 tokens; human
# decision, to save about 4.7 GPU hours); their request files are written but not generated.
OFFOPEN_SKIPPED = ["Qwen3.6-27B-FP8", "Qwen3.8-27B-FP8"]
OFFOPEN_ORDER = [("A", "compliant"), ("A", "noncompliant"), ("A", "commitment"), ("no_rule", "compliant")]


# --- Necessity (exp04's rows, unchanged but for the questions) ------------------------------------------------------
def necessity_rows() -> list[dict]:
    rows = []
    for item in e9.items():
        system, user = cc_prompts.cotcontrol_prompt(item, "baseline")
        letters = [chr(ord("A") + i) for i in range(len(item["options"]))]
        for r in range(cfg.EXP04_NECESSITY_ROLLOUTS):
            rows.append(cc_exp04.with_ids({
                "item_id": item["item_id"], "source": item["source"], "mode": cfg.NO_CONSTRAINT,
                "prompt": cfg.NO_CONSTRAINT, "rollout": r, "condition": "necessity", "prefill": None,
                "thinking": False, "system": system, "user": user, "response_prefix": cfg.EXP04_NECESSITY_PREFIX,
                "response_letters": letters, "grading_prompt": None,
                "seed": cfg.rollout_seed(item["item_id"], "necessity", r), "full_trace_cell": False,
                "abort_on_violation": False}))
    return rows


# --- Examples only --------------------------------------------------------------------------------------------------
def examples_only(item: dict, examples: list[cands.Example], example_items: list[dict], n: int) -> cands.Prompt:
    """A's example turns with CoT-Control's no-rule prompt as every user message (A's turns: the example question's
    stacked message, the rewritten reasoning cut as A cuts it, the answer)."""
    def no_rule_user(it: dict) -> str:
        system, user = cc_prompts.cotcontrol_prompt(it, "baseline")
        if system:
            raise ValueError("CoT-Control's no-rule prompt has a system message")
        return user

    history = [{"user": no_rule_user(ex_item), "reasoning": ex.reasoning(cfg.EXP06A_FEWSHOT_MAX_CHARS),
                "answer": ex.answer} for ex, ex_item in list(zip(examples, example_items))[:n]]
    return cands.Prompt("", no_rule_user(item), history)


# --- Rows -----------------------------------------------------------------------------------------------------------
def model_rows(model: str) -> tuple[dict[str, list[dict]], dict]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(model))
    source = e9.EXAMPLE_SOURCE.get(model, model)
    by_block = {cc_prompts.question_block(t["item"]): t["item"] for t in e7b.own_traces(source)}
    ext2 = json.loads((e9.EXP.results / f"requests_record_{x2.STEM}_{model}.json").read_text())["rules"]
    exp09_a = x1.exp09_short_rows(model, "A")
    parts = {"nec": necessity_rows(), "exonly": [], "offopen": []}
    for mode in R.ALL_RULES:
        examples = e9.usable_examples(model, mode)
        example_items = [by_block[ex.context.question_block] for ex in examples]
        n_on, n_off = ext2[mode]["A_examples_on"], ext2[mode]["A_examples_off"]
        for it in e9.items():
            a_on = b.many_examples(e9.context(it, mode, examples), mode, n_on)
            short = exp09_a[it["item_id"], mode]
            if [t["reasoning"] for t in a_on.history] != [t["reasoning"] for t in short["history"]]:
                raise RuntimeError(f"{model} {mode} {it['item_id']}: A's examples differ from exp09's")
            parts["exonly"].append(x2.row(it, mode, "examples_only", "none", "on",
                                          examples_only(it, examples, example_items, n_on), "examples_only"))
            if mode in x2.OPENING_RULES:
                texts = x2.openings(it, mode)
                prompts = {"A": x2.a_off(it, mode, examples, example_items, n_off), "no_rule": x2.no_rule_off(it)}
                for arm, kind in OFFOPEN_ORDER:
                    if kind in texts:
                        parts["offopen"].append(x2.row(it, mode, arm, kind, "off", prompts[arm], "openings_off",
                                                       prefill=texts[kind]))
    rule_order = {m: k for k, m in enumerate(R.ALL_RULES)}
    item_order = {it["item_id"]: k for k, it in enumerate(e9.items())}
    parts["exonly"].sort(key=lambda r: (rule_order[r["mode"]], item_order[r["item_id"]]))
    parts["offopen"].sort(key=lambda r: (OFFOPEN_ORDER.index((r["prompt"], r["condition"])), rule_order[r["mode"]],
                                         item_order[r["item_id"]]))
    if model in x2.RENDERED_BY_EXP09:
        for r in parts["exonly"] + parts["offopen"]:
            r["exp09_render"] = True
    for name, rows in parts.items():
        if len({r["request_id"] for r in rows}) != len(rows):
            raise RuntimeError(f"{model} {name}: duplicate request ids")
    record = {"model": model, "rows": {p: len(rows) for p, rows in parts.items()},
              "cells": {p: {c: sum(r.get("exp06_cell") == c for r in rows)
                            for c in sorted({r.get("exp06_cell") for r in rows if r.get("exp06_cell")})}
                        for p, rows in parts.items()},
              "checks": check_rows(tokenizer, model, parts)}
    return parts, record


def check_rows(tokenizer, model: str, parts: dict[str, list[dict]]) -> dict:
    """Every prompt fits: thinking on beside the 1200-token stop; thinking off, the prompt without its opening as
    extension 2's (the opening starts the response), and with it, beside the response cap within the context. Necessity
    rows render with exp04's rendering and every option letter is one token."""
    import cc_generate_abort
    family = cfg.ALL_MODELS[model]["family"]
    on_limit = cfg.VLLM_MAX_MODEL_LEN - e7.STOP_TOKENS - e8.CONTEXT_MARGIN
    longest = {}
    for name in ("exonly", "offopen"):
        for r in parts[name]:
            n = len(x2.rendered(tokenizer, model, r))
            if r["thinking"]:
                fits = n <= on_limit
            else:
                fits = (len(x2.rendered(tokenizer, model, {**r, "prefill": None})) <= x2.OFF_PROMPT_LIMIT
                        and n + cfg.EXP04_EXTERNAL_CAP_TOKENS <= cfg.VLLM_MAX_MODEL_LEN)
            if not fits:
                raise RuntimeError(f"{model} {name} {r['mode']} {r['item_id']}: prompt of {n} tokens does not fit")
            longest[name] = max(longest.get(name, 0), n)
    for r in parts["nec"]:
        cc_generate_abort.letter_token_ids(tokenizer, r)
        longest["nec"] = max(longest.get("nec", 0), len(cc_generate_abort.request_prompt_ids(tokenizer, family, r)))
    return {"longest_prompt": longest}


# --- Grading and plan -----------------------------------------------------------------------------------------------
def grade(model: str, part: str):
    if part != "nec":
        return x2.grade(model, part)
    out = x2.grades_path(model, part)
    if out.exists():
        return out
    requests = {r["request_id"]: r for r in map(json.loads, x2.requests_path(model, part).open())}
    gens = [json.loads(line) for line in x2.generation_path(model, part).open()]
    rows = e7.grade_generations(gens, requests, {it["item_id"]: it for it in e9.items()}, model)
    for r in rows:
        r.update(condition="necessity", rollout=requests[r["request_id"]]["rollout"])
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def write_plan():
    def job(model: str, stem: str, after: list[str]) -> dict:
        # The jobs do not depend on each other, so a failed one does not stop the lane (rerunning the plan resumes it)
        return {"exp": "exp09", "model": model,
                "requests": str(x2.requests_path(model, stem).relative_to(cfg.REPO_ROOT)),
                "items": str(e9.ITEMS_PATH.relative_to(cfg.REPO_ROOT)), "memory": e9.MEMORY[model],
                "after": ["/venv/main/bin/python"] + after + ["--model", model], "optional": True}
    lane = []
    for m in x2.MODEL_ORDER:
        lane.append(job(m, x2.STEM, ["cc_exp09_ext2.py", "grade"]))
        lane.append(job(m, "exonly", ["cc_exp09_ext3.py", "grade", "--part", "exonly"]))
        lane.append(job(m, "nec", ["cc_exp09_ext3.py", "grade", "--part", "nec"]))
    lane += [job(m, "offopen", ["cc_exp09_ext3.py", "grade", "--part", "offopen"])
             for m in x2.MODEL_ORDER if m not in OFFOPEN_SKIPPED]
    plan = {"name": "exp09_ext3_1xB200", "mps": False, "gpus": {"0": [lane]}}
    path = e9.EXP.results / "plan_ext3_1xB200.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "requests", "plan", "grade"])
    parser.add_argument("--model", choices=x2.MODEL_ORDER)
    parser.add_argument("--part", choices=PARTS)
    args = parser.parse_args()
    if args.command == "plan":
        print(write_plan())
        return
    if args.command == "grade":
        print(grade(args.model, args.part))
        return
    if args.command == "requests" and any(x2.requests_path(args.model, p).exists() for p in PARTS):
        raise SystemExit(f"{args.model}: an extension-3 request file exists; request files are fixed once written")
    parts, record = model_rows(args.model)
    print(json.dumps(record, indent=1))
    if args.command == "requests":
        for p, rows in parts.items():
            e7.write_requests(x2.requests_path(args.model, p), rows)
        (e9.EXP.results / f"requests_record_ext3_{args.model}.json").write_text(json.dumps(record, indent=2) + "\n")
        print("wrote", [str(x2.requests_path(args.model, p)) for p in PARTS])


if __name__ == "__main__":
    main()
