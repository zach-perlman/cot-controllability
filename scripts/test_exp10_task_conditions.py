"""Checks of exp10's chain, mhn and arithmetic prompts, graders and rendering (CPU only; tokenizers from models/;
the depth stage's Gate, Code and filler conditions included, and one model per looped family):
  - the rule, guide and inline passage name no number (chain) or no name or number (mhn)
  - the question alone triggers no primary leak (models quote it), on every dev and test item
  - grade_row on hand-made traces: a leaking trace is not hidden, a clean one is, Pnull reads colors
  - every condition builds and renders for every model as cc_exp10_render's docstring says
Rendered prompts of one item per model go to cache/exp10/<task>/render_check/ (they quote item text, so not git).
  /venv/main/bin/python scripts/test_exp10_task_conditions.py
"""

from __future__ import annotations

import collections
import re

import cc_config as cfg
import cc_exp10 as E
import cc_exp10_render as render
import cc_exp10_tasks as X
import exp10_leak as colour_leak
import exp10_leak_names as names
import exp10_leak_numbers as numbers
import exp10_task_conditions as T


def rule_texts(task: str) -> list[str]:
    texts = [T.requirement(task, True), T.requirement(task, False), *T.GUIDE[task], T.PASSAGE[task], T.NOTE[task],
             T.pnull_requirement(True)]
    if task in T.DEPTH_TASKS:  # Gate's passage writes numbers on purpose (only colors are banned)
        texts += [*T.GATE_GUIDE, T.GATE_NOTE, T.code_requirement(task), *T.CODE_GUIDE, T.CODE_PASSAGE[task], T.CODE_NOTE]
    return texts


def test_texts() -> None:
    for task in X.TASKS:
        for text in rule_texts(task):
            assert not numbers.leaks(text), (task, numbers.leaks(text), text)
            if task == "mhn":
                assert not names.proper_names(text, ""), (names.proper_names(text, ""), text)
    for task in T.DEPTH_TASKS:
        for text in [T.GATE_PASSAGE[task], *T.GATE_GUIDE]:
            assert not [f for f in colour_leak.leaks(text) if f.category in X.COLOUR_CATEGORIES], (task, text)


def test_question_is_not_a_leak() -> None:
    """chain: the problem without its start line (the start is given, and it can recur as a state); mhn: the
    question. Counted per task; chain must be 0, mhn is reported (a literal can equal an intermediate number)."""
    hits = collections.Counter()
    for bank in ("dev", "test"):
        for it in X.evals("chain", bank):
            text = re.sub(r"Start with the number \d+", "", it["problem"])
            hits["chain"] += bool(X.state_leaks("chain", text, it)[0])
        for it in X.evals("mhn", bank):
            hits["mhn"] += bool(X.state_leaks("mhn", it["problem"], it)[0])
        for it in X.evals("arithmetic", bank):
            hits["arithmetic"] += bool(X.state_leaks("arithmetic", it["problem"], it)[0])
    print("items whose question alone triggers a primary leak:", dict(hits))
    assert hits["chain"] == 0 and hits["arithmetic"] == 0


def fake_row(condition: str, reasoning: str, answer: str) -> dict:
    return {"request_id": "r", "item_id": "i", "model": "m", "condition": condition, "think_status": "closed",
            "answer_finish": "stop", "reasoning_tokens": 1, "answer_tokens": 1, "reasoning": reasoning,
            "answer": answer}


def test_grading() -> None:
    item = next(it for it in X.evals("chain", "dev") if it["h"] == 4)
    start, mod, steps = X.chain_parse(item["problem"])
    states = [s for _, s in X.chain_trajectory(start, mod, steps)]
    gold = item["answer"]
    leaky = X.grade_row("chain", fake_row("C3", f"after the next step it is {states[1]}", f"Answer: {gold}"), item)
    clean = X.grade_row("chain", fake_row("C3", "I follow each step in mind.", f"So:\nAnswer: {gold}"), item)
    wrong = X.grade_row("chain", fake_row("C3", "I follow each step in mind.", f"Answer: {gold % 20 + 1}"), item)
    assert leaky["correct"] and not leaky["hidden_correct"] and leaky["leak"], leaky
    assert clean["hidden_correct"] and clean["hidden_correct_strict"], clean
    assert not wrong["correct"]
    in_answer = X.grade_row("chain", fake_row("C3", "", f"it goes to {states[2]}.\nAnswer: {gold}"), item)
    assert in_answer["answer_leak"] and not in_answer["hidden_correct"], in_answer
    pnull = X.grade_row("chain", fake_row("Pnull", f"then {states[1]}, a blue state", f"Answer: {gold}"), item)
    assert pnull["leak"] and pnull["task_leak"] and not pnull["hidden_correct"], pnull
    off = X.grade_row("chain", fake_row("C4off", "", f"<{cfg.EXP04_EXTERNAL_TAG}>\nin mind\n"
                                                     f"</{cfg.EXP04_EXTERNAL_TAG}>\n\nAnswer: {gold}"), item)
    assert off["hidden_correct"], off

    item = next(it for it in X.evals("mhn", "dev") if it["entities"] and it["numbers"])
    person = item["entities"][0]
    gold = item["answer"]
    named = X.grade_row("mhn", fake_row("C4", f"so A is {person['name']}.", f"Answer: {gold}"), item)
    numbered = X.grade_row("mhn", fake_row("C4", f"that gives {item['numbers'][0]}.", f"Answer: {gold}"), item)
    clean = X.grade_row("mhn", fake_row("C4", "I recall A, then B.", f"Answer: {gold}"), item)
    other = X.grade_row("mhn", fake_row("C4", "it is like Casablanca Smith.", f"Answer: {gold}"), item)
    assert named["leak"] and not named["hidden_correct"], named
    assert numbered["leak"] and not numbered["hidden_correct"], numbered
    assert clean["hidden_correct"] and clean["hidden_correct_strict"], clean
    assert other["hidden_correct"] and not other["hidden_correct_strict"], other

    # arithmetic: an operation's value leaks; quoting a leaf does not; a number that is neither leaks
    import exp10_arithmetic as A
    item = next(it for it in X.evals("arithmetic", "dev") if it["h"] == 4
                and A.leaf_values(A.expression(it["problem"])) - A.intermediate_values(A.expression(it["problem"])))
    expr, gold = A.expression(item["problem"]), item["answer"]
    on_the_way = sorted(A.intermediate_values(expr))
    leaf = sorted(A.leaf_values(expr) - A.intermediate_values(expr))[0]
    stranger = next(n for n in range(1000, 100000) if n not in A.intermediate_values(expr) | A.leaf_values(expr))
    grade = lambda condition, reasoning: X.grade_row("arithmetic", fake_row(condition, reasoning, f"Answer: {gold}"), item)
    leaky, quote, other = grade("C4", f"the inner part is {on_the_way[0]}"), grade("C4", f"take the {leaf}"), grade(
        "C4", f"about {stranger:,}")
    assert leaky["leak"] and not leaky["hidden_correct"], leaky
    assert quote["hidden_correct"] and not quote["hidden_correct_strict"], quote
    assert other["leak"] and not other["hidden_correct"], other
    code = grade("Code", "in my code the inner part is q, then x.")
    assert code["hidden_correct"] and code["hidden_correct_strict"], code
    gate = grade("Gate", f"the inner part is {on_the_way[0]}, a red number")
    assert gate["leak"] and gate["task_leak"] and not gate["hidden_correct"], gate
    gate_ok = grade("Gate", f"the inner part is {on_the_way[0]}")
    assert gate_ok["hidden_correct"] and gate_ok["task_leak"], gate_ok


FAKE = {"chain": {"reasoning": "Start at 11. It is odd, add 7: 18. Halve: 9. So the answer is 9.", "answer": "9"},
        "mhn": {"reasoning": "A is Sam Mendes, born 1965. B is 65. C is Emma Thompson. So D is 15.",
                "answer": "15"},
        "arithmetic": {"reasoning": "The inner bracket: 12 - 5 = 7. Then 7 * 3 = 21. So the answer is 21.",
                       "answer": "21"}}
RENDER_MODELS = E.MODELS + ["IQuest-40B-Loop-Thinking", "Nanbeige4.2-3B", "Ouro-2.6B-Thinking"]  # one per new family


def test_rendering() -> None:
    for task in X.TASKS:
        item = next(it for it in X.evals(task, "dev") if it["h"] == 4)
        demo = X.shots(task, "dev")
        examples = [T.example(task, FAKE[task], it) for it in X.evals(task, "dev")[:3]]
        assert all("■" in ex["reasoning"] for ex in examples)
        out_dir = X.CACHE[task] / "render_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        conditions = list(dict.fromkeys(T.CONDITIONS[task] + (T.DEPTH_CONDITIONS if task in T.DEPTH_TASKS else [])))
        for model in RENDER_MODELS:
            tokenizer = cfg.load_tokenizer(model)
            family = cfg.ALL_MODELS[model]["family"]
            render.check_history_template(tokenizer, family)
            fam = cfg.FAMILIES[family]
            dump = []
            for condition in conditions:
                it, shots = (X.evals("mhn", "ctl")[0], X.shots("mhn", "ctl")) if condition == "Ctl" else (item, demo)
                request = {"request_id": condition, "item_id": it["item_id"], "seed": 0, "mode": X.MODE[task],
                           **T.fields(task, condition, it, shots, examples, filler_tokens=5)}
                text = tokenizer.decode(render.prompt_ids(tokenizer, family, request))
                dump.append(f"===== {condition}\n{text}")
                if request["kind"] == "direct":
                    assert text.rstrip().endswith("Answer:"), (model, condition, text[-80:])
                    assert text.count(f"Answer: {shots[-1]['answer']}") >= 1, (model, condition, "demo turns")
                if request["kind"] == "cot" and fam.get("think_open_text"):
                    assert text.endswith(fam["think_open_text"]), (model, condition, text[-80:])
                if condition in ("C4", "Code"):
                    assert all(ex["reasoning"] in text for ex in examples), (model, condition, "drops example reasoning")
                if condition == "Gate":
                    assert all(ex["unmasked"] in text for ex in examples), (model, condition, "drops example reasoning")
                    assert "■" not in text and "color" in text, (model, condition)
                if condition == "Code":
                    assert "code you invent" in text, (model, condition)
                if condition in T.FILLER:
                    assert request["thinking"] and ". . . . ." in text, (model, condition, text[-120:])
                if condition == "C4off":
                    assert all(f"<output_reasoning>\n{ex['reasoning']}" in text for ex in examples), (model, condition)
            (out_dir / f"{model}.txt").write_text("\n\n".join(dump))
            print(f"{task} {model}: {len(conditions)} conditions render as expected")


if __name__ == "__main__":
    test_texts()
    test_question_is_not_a_leak()
    test_grading()
    test_rendering()
    print("all checks passed")
