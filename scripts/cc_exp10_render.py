"""Rendering of exp10's rows (rows with "exp10_render"; cc_generate_abort routes them here). Kinds: exp10_conditions.

Prompt: system, earlier turns (a turn with reasoning is written back as cc_exp07.HISTORY_REASONING says, as exp09's
prompt A was; a turn without reasoning is a plain assistant message), final user message, rendered with the row's
thinking switch and template_kwargs. Then, tokenized on their own:
  cot, thinking on    the family's reasoning opening (Gemma 4: "<|channel>thought\\n")
  direct, thinking on the reasoning opening ("<think>\\n" first if the template does not open it), the filler, the
                      family's close, then the response prefix
  direct, thinking off the response prefix (the template has already closed an empty reasoning block)
"""

from __future__ import annotations

import cc_config as cfg


# The looped models' history formats (cc_exp07.HISTORY_REASONING's convention: None where the template keeps
# reasoning_content). Kept here, not in cc_exp07's dict, which enters exp07/exp09/exp10 output keys;
# cc_generate_abort adds these to the key of rows of these families only.
LOOPED_HISTORY_REASONING = {"iquest": None, "nanbeige4.2": None, "ouro": "<think>\n{reasoning}\n</think>\n\n{answer}"}


def history_formats(family: str) -> tuple[str | None, str]:
    import cc_exp07
    base = family.split("-")[0]
    if base in LOOPED_HISTORY_REASONING:
        return LOOPED_HISTORY_REASONING[base], ""
    return cc_exp07.HISTORY_REASONING[base], cc_exp07.HISTORY_EMPTY_THINK.get(family, "")


def prompt_text(tokenizer, family: str, request: dict) -> str:
    fam = cfg.FAMILIES[family]
    history_format, empty_think = history_formats(family)
    messages = [{"role": "system", "content": request["system"]}] if request["system"] else []
    inserted = {}
    for k, turn in enumerate(request["history"]):
        messages.append({"role": "user", "content": turn["user"]})
        if turn["reasoning"] is None:
            messages.append({"role": "assistant", "content": turn["answer"]})
        elif history_format is None:
            messages.append({"role": "assistant", "content": turn["answer"], "reasoning_content": turn["reasoning"]})
        else:
            placeholder = f"@@exp10-turn-{k}@@"
            inserted[empty_think + placeholder] = history_format.format(reasoning=turn["reasoning"],
                                                                        answer=turn["answer"])
            messages.append({"role": "assistant", "content": placeholder})
    messages.append({"role": "user", "content": request["user"]})
    kwargs = {**fam["chat_template_kwargs"], "enable_thinking": request["thinking"], **request["template_kwargs"]}
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    for span, block in inserted.items():
        if text.count(span) != 1:
            raise RuntimeError(f"{request['request_id']}: history span rendered {text.count(span)} times")
        text = text.replace(span, block)
    return text


def suffix_pieces(tokenizer, family: str, request: dict) -> list[str]:
    """What follows the rendered template, piece by piece (each tokenized on its own)."""
    fam = cfg.FAMILIES[family]
    pieces = []
    if request["thinking"]:
        if fam.get("think_open_text"):
            pieces.append(fam["think_open_text"])
        if request["kind"] == "direct":
            if not fam["template_opens_think"]:
                pieces.append(tokenizer.convert_ids_to_tokens(fam["think_start"]) + "\n")
            pieces += [request["filler"], fam.get("forced_close", cfg.FORCED_THINK_CLOSE)]
    if request["kind"] == "direct":
        pieces.append(request["response_prefix"])
    return [p for p in pieces if p]


def prompt_ids(tokenizer, family: str, request: dict) -> list[int]:
    ids = tokenizer.encode(prompt_text(tokenizer, family, request), add_special_tokens=False)
    for piece in suffix_pieces(tokenizer, family, request):
        ids = ids + tokenizer.encode(piece, add_special_tokens=False)
    return ids


def check_history_template(tokenizer, family: str) -> None:
    import cc_exp07
    if family.split("-")[0] not in LOOPED_HISTORY_REASONING:
        cc_exp07.check_history_template(tokenizer, family)
        return
    probe = "exp10 probe reasoning"  # cc_exp07.check_history_template's probe, against LOOPED_HISTORY_REASONING
    messages = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1", "reasoning_content": probe},
                {"role": "user", "content": "u2"}]
    kwargs = {**cfg.FAMILIES[family]["chat_template_kwargs"], "enable_thinking": True}
    kept = probe in tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    if kept != (history_formats(family)[0] is None):
        raise RuntimeError(f"{family}: template {'keeps' if kept else 'drops'} history reasoning, unlike "
                           f"LOOPED_HISTORY_REASONING")


def job(tokenizer, family: str, request: dict, items: dict) -> dict:
    """The engine job (fields as cc_generate_abort.job_of). direct and off rows are one call of at most
    response_cap_tokens; cot rows reason (at most reasoning_stop_tokens) and then answer."""
    single = request["kind"] in ("direct", "off")
    return {"prompt_ids": prompt_ids(tokenizer, family, request), "seed": request["seed"], "abort": False,
            "mode": request["mode"], "item": items[request["item_id"]],
            "response_cap": request["response_cap_tokens"] if single else None, "allowed_token_ids": None,
            "reasoning_cap": None if single else request["reasoning_stop_tokens"],
            "answer_phase": request["answer_phase"]}


def generation_result(request: dict, result: dict, tokenizer) -> dict:
    """off rows: "reasoning" is the <output_reasoning> tag content of the response (no tags -> empty). Others as
    generated (a direct row's "answer" is the continuation after the response prefix)."""
    if request["kind"] != "off":
        return result
    import cc_exp04
    content, blocks = cc_exp04.external_reasoning(result["answer"])
    return {**result, "reasoning": content, "external_blocks": blocks,
            "reasoning_tokens": len(tokenizer.encode(content, add_special_tokens=False)) if content else 0}
