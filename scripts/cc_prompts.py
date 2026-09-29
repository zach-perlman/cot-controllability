"""Prompt builders and the request files of exp01 / exp02.

Four constrained prompt conditions per (item, mode):
- baseline: CoT-Control's create_controlled_prompt (through its create_prompt_for_sample), verbatim.
- repetition, self_monitor, stacked: Arun Jose's zero-shot templates (cc_config holds the texts).
Plus the no_constraint reference: CoT-Control's unconstrained "baseline" mode prompt.

Every request also carries `grading_prompt`, the baseline user prompt of its (item, mode). CoT-Control's grader reads
the rule (keyword, synonyms, target string) back out of the prompt text, so grading every condition against the
same baseline prompt keeps the metric code identical across conditions.

Run:
  /venv/main/bin/python scripts/cc_prompts.py requests --exp exp01|exp02 [--smoke-items 3]
  /venv/main/bin/python scripts/cc_prompts.py examples   # results/exp02_prompt_grid/prompt_examples.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _cotcontrol
import cc_config as cfg

MODEL_NAME_FOR_TEMPLATES = "qwen"  # CoT-Control only switches wording for "gpt-oss"


def load_items() -> list[dict]:
    return [json.loads(line) for line in cfg.ITEMS_PATH.open()]


def cotcontrol_sample(item: dict, mode: str) -> dict:
    """The sample dict CoT-Control's prompt functions expect, with the mode's keyword data."""
    sample = {"question": item["question"], "options": item["options"]}
    if mode == "word_suppression":
        sample.update(keyword=item["keyword"], synonyms=item["synonyms"])
    elif mode == "multiple_word_suppression":
        sample.update(keyword=item["multi_keyword"], synonyms=item["multi_synonyms"])
    return sample


def constraint_sentence(item: dict, mode: str) -> str:
    """CoT-Control's requirement sentence for the mode, without its trailing "Additionally, ..." clause."""
    prompts = _cotcontrol.module("prompts")
    run_cceval = _cotcontrol.module("run_cceval")
    control_value = run_cceval.get_control_value(mode, cotcontrol_sample(item, mode))
    text = prompts.CONTROL_TYPE_REQUIREMENTS[mode].format(control_value=control_value,
                                                          reasoning_term=cfg.REASONING_TERM)
    return prompts._adapt_text_for_model(text, MODEL_NAME_FOR_TEMPLATES)


def question_block(item: dict) -> str:
    choices_text, format_instruction = _cotcontrol.module("prompts")._format_choices(item["options"])
    return f"Question: {item['question']}{choices_text}{format_instruction}"


def cotcontrol_prompt(item: dict, mode: str) -> tuple[str, str]:
    """CoT-Control's (system, user) prompt; mode "baseline" in its terms is our no_constraint."""
    run_cceval = _cotcontrol.module("run_cceval")
    return run_cceval.create_prompt_for_sample(
        cotcontrol_sample(item, mode), mode, MODEL_NAME_FOR_TEMPLATES, non_reasoning=False,
        adversarial_pressure="none", output_no_letter_no_grading=False, monitor_aware=False, eval_aware=False)


# --- Stacked's inline example -----------------------------------------------------------------------------
def _alternate_word(match: re.Match) -> str:
    word = match.group(0)
    return "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(word))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def stacked_example(mode: str) -> str:
    """Arun's lowercase example passage with the mode's rule applied (cc_config documents the choices)."""
    base = cfg.STACKED_EXAMPLE_SENTENCE_CASE
    if mode == "lowercase_thinking":
        return cfg.STACKED_EXAMPLE_LOWERCASE
    if mode == "uppercase_thinking":
        return cfg.STACKED_EXAMPLE_LOWERCASE.upper()
    if mode == "alternating_case":
        return re.sub(r"[a-zA-Z]+", _alternate_word, cfg.STACKED_EXAMPLE_LOWERCASE)
    if mode in ("word_suppression", "multiple_word_suppression"):
        return base
    if mode == "ignore_question":
        return cfg.STACKED_EXAMPLE_OFF_TOPIC
    if mode == "repeat_sentences":
        target = _cotcontrol.module("run_cceval").get_control_value(mode, {})
        return f"{target} {base} {target}"
    if mode == "end_of_sentence":
        target = _cotcontrol.module("run_cceval").get_control_value(mode, {})
        out = []
        for sentence in _sentences(base):
            body, punct = re.match(r"(.*?)([.!?]+)$", sentence).groups()
            out.append(f"{body} {target}{punct}")
        return " ".join(out)
    if mode == "meow_between_words":
        target = _cotcontrol.module("run_cceval").get_control_value(mode, {})
        return " ".join(f" {target} ".join(s.split()) for s in _sentences(base))
    raise ValueError(mode)


# --- The four conditions --------------------------------------------------------------------------------------
def build_prompt(item: dict, mode: str, prompt: str) -> tuple[str, str]:
    """(system, user) for one constrained condition."""
    if prompt == "baseline":
        return cotcontrol_prompt(item, mode)
    constraint = constraint_sentence(item, mode)
    block = question_block(item)
    if prompt == "repetition":
        rules = "\n".join([cfg.RULE_LINE.format(constraint=constraint)] * cfg.REPETITION_COUNT)
        return "", cfg.REPETITION_TEMPLATE.format(rules=rules, question_block=block)
    if prompt == "self_monitor":
        return "", cfg.SELF_MONITOR_TEMPLATE.format(constraint=constraint, question_block=block)
    if prompt == "stacked":
        rules = "\n".join([cfg.RULE_LINE.format(constraint=constraint)] * cfg.STACKED_REPETITION_COUNT)
        system = cfg.STACKED_SYSTEM_TEMPLATE.format(constraint=constraint)
        user = cfg.STACKED_USER_TEMPLATE.format(rules=rules, example=stacked_example(mode),
                                                example_note=cfg.STACKED_EXAMPLE_NOTES[mode], question_block=block)
        return system, user
    raise ValueError(prompt)


def request_rows(exp_key: str, items: list[dict]) -> list[dict]:
    design = cfg.EXP01_DESIGN if exp_key == "exp01" else cfg.EXP02_DESIGN
    rows = []
    for item in items:
        for mode in cfg.MODES:
            grading_prompt = cotcontrol_prompt(item, mode)[1]
            for prompt in design["prompts"]:
                for r in range(design["rollouts"]):
                    system, user = build_prompt(item, mode, prompt)
                    rows.append({"item_id": item["item_id"], "source": item["source"], "mode": mode,
                                 "prompt": prompt, "rollout": r, "system": system, "user": user,
                                 "grading_prompt": grading_prompt})
        if design["no_constraint"]:
            system, user = cotcontrol_prompt(item, "baseline")
            for r in range(cfg.NO_CONSTRAINT_ROLLOUTS):
                rows.append({"item_id": item["item_id"], "source": item["source"], "mode": cfg.NO_CONSTRAINT,
                             "prompt": cfg.NO_CONSTRAINT, "rollout": r, "system": system, "user": user,
                             "grading_prompt": None})
    for row in rows:
        row["seed"] = cfg.rollout_seed(row["item_id"], row["mode"], row["rollout"])
        row["request_id"] = cfg.content_key({k: row[k] for k in ("item_id", "mode", "prompt", "rollout", "system",
                                                                   "user")})
    return rows


def write_requests(exp_key: str, smoke_items: int | None) -> None:
    exp = cfg.EXPERIMENTS[exp_key]
    items = load_items()
    path = exp.requests
    if smoke_items:
        # The first n items of each source, so every dataset appears.
        by_source = {}
        for item in items:
            by_source.setdefault(item["source"], []).append(item)
        items = [it for group in by_source.values() for it in group[:1]][:smoke_items]
        path = exp.cache / "smoke" / "requests.jsonl"
    rows = request_rows(exp_key, items)
    if path.exists():
        old = [json.loads(line)["request_id"] for line in path.open()]
        if old != [r["request_id"] for r in rows]:
            raise RuntimeError(f"{path} exists with different requests; the request set is fixed once written")
        print(f"{path} unchanged ({len(rows)} requests)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} requests to {path}")


def write_examples() -> None:
    """One rendered prompt per (condition, mode), on the first item, for a human read before the manifest."""
    item = load_items()[0]
    grading = _cotcontrol.module("grading")
    lines = [f"# Rendered prompts (item {item['item_id']})", "",
             "One prompt per (condition, mode), as the model receives it (system then user). Stacked's inline "
             "example is graded with CoT-Control's grader for its mode at the end of each stacked block.", ""]
    for mode in cfg.MODES:
        lines += [f"## {mode}", ""]
        for prompt in cfg.PROMPTS:
            system, user = build_prompt(item, mode, prompt)
            lines += [f"### {prompt}", ""]
            if system:
                lines += ["System:", "", "```", system, "```", ""]
            lines += ["User:", "", "```", user, "```", ""]
            if prompt == "stacked" and mode != "ignore_question":
                score = grading.grade_cot_compliance(cotcontrol_prompt(item, mode)[1], stacked_example(mode))
                lines += [f"Stacked example passes CoT-Control's grader for {mode}: {bool(score)}", ""]
    system, user = cotcontrol_prompt(item, "baseline")
    lines += ["## no_constraint (reference)", "", "```", user, "```", ""]
    out = cfg.EXP02.results / "prompt_examples.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    req = sub.add_parser("requests")
    req.add_argument("--exp", choices=list(cfg.EXPERIMENTS), required=True)
    req.add_argument("--smoke-items", type=int, default=None)
    sub.add_parser("examples")
    args = parser.parse_args()
    if args.cmd == "requests":
        write_requests(args.exp, args.smoke_items)
    else:
        write_examples()
