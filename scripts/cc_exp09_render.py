"""Rendering of the exp09 extension's GLM-4.7-Flash rows (rows with "exp09_render"; cc_generate_abort routes them here).

cc_exp06's renderer, which renders the extension's other rows, has no history format for GLM
(cfg.EXP06_HISTORY_REASONING), and adding one would change the cache key of every exp06/exp08 generation. This is
cc_exp07.prompt_ids (exp09's GLM short rows were rendered by it, so a thinking-on row without an opening gets exactly
their ids) with cc_exp06's two additions:
  - earlier turns without reasoning (A thinking off: the example reasoning is inside the answer's tags) are rendered
    as the template renders a plain assistant answer
  - exp04's prefill: thinking on, the opening follows the reasoning opening; thinking off, it follows
    "<output_reasoning>\\n" in the response
"""

from __future__ import annotations

import cc_config as cfg
import cc_exp07


def prompt_ids(tokenizer, family: str, request: dict) -> list[int]:
    fam = cfg.FAMILIES[family]
    with_reasoning = [turn["reasoning"] is not None for turn in request["history"]]
    if all(with_reasoning):
        ids = cc_exp07.prompt_ids(tokenizer, family, request)
    elif any(with_reasoning):
        raise RuntimeError(f"{request['request_id']}: earlier turns with and without reasoning")
    else:
        messages = [{"role": "system", "content": request["system"]}] if request["system"] else []
        for turn in request["history"]:
            messages += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["answer"]}]
        messages.append({"role": "user", "content": request["user"]})
        kwargs = {**fam["chat_template_kwargs"], "enable_thinking": request["thinking"]}
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if request["thinking"] and fam.get("think_open_text"):
            ids = ids + tokenizer.encode(fam["think_open_text"], add_special_tokens=False)
    if request["prefill"]:
        if not request["thinking"]:
            ids = ids + tokenizer.encode(f"<{cfg.EXP04_EXTERNAL_TAG}>\n", add_special_tokens=False)
        elif not fam["template_opens_think"]:
            ids = ids + tokenizer.encode(tokenizer.convert_ids_to_tokens(fam["think_start"]) + "\n",
                                         add_special_tokens=False)
        ids = ids + tokenizer.encode(request["prefill"], add_special_tokens=False)
    return ids


def check_history_template(tokenizer, family: str) -> None:
    cc_exp07.check_history_template(tokenizer, family)


def job(tokenizer, family: str, request: dict, items: dict) -> dict:
    """The engine job of one row (fields as cc_exp06.job)."""
    import cc_generate_abort
    return {"prompt_ids": prompt_ids(tokenizer, family, request), "seed": request["seed"],
            "abort": request["abort_on_violation"], "mode": request["mode"], "item": items[request["item_id"]],
            "response_cap": cc_generate_abort.response_cap(request), "allowed_token_ids": None,
            "reasoning_cap": request["reasoning_stop_tokens"], "answer_phase": request["answer_phase"]}
