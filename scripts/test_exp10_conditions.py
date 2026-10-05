"""Checks of exp10's prompts and rendering (CPU only; tokenizers from models/):
  - the tag instruction is CoT-Control's (cc_exp08's), and C4's inline example passage follows the rule
  - every condition builds for a dev item, and every model's renders end as cc_exp10_render's docstring says
  - nocot-bench's Qwen3.8 trap: thinking off with preserve_thinking=False leaves its demo turn clean
The rendered prompts of one item per model are written to cache/exp10/render_check/ (they quote item text, so not to
git) for a read by eye.
  /venv/main/bin/python scripts/test_exp10_conditions.py
"""

from __future__ import annotations

import cc_config as cfg
import cc_exp10 as E
import cc_exp10_render as render
import exp10_conditions as C
import exp10_leak as leak

FAKE_TRACE = {"reasoning": "The potion starts out red. Salt turns red into blue. Then ash turns blue into gold. "
                           "So the answer is gold.", "answer": "Answer: gold"}


def test_texts() -> None:
    import cc_exp08
    assert C.TAG_INSTRUCTION == cc_exp08.tag_instruction()
    assert not leak.leaks(C.PASSAGE), leak.leaks(C.PASSAGE)
    assert not leak.leaks(C.CODE_NOTE.split(":", 1)[0])
    assert all(not leak.leaks(word) for word in C.CODE.values())
    masked = C.example(FAKE_TRACE, E.evals("dev")[0])["reasoning"]
    assert "■" in masked and not leak.leaks(masked), masked


def test_rendering() -> None:
    from transformers import AutoTokenizer
    item, shot = E.evals("dev")[20], E.shots("dev")[0]
    examples = [C.example(FAKE_TRACE, it) for it in E.evals("dev")[:3]]
    out_dir = E.EXP.cache / "render_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in E.MODELS:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(model))
        family = cfg.ALL_MODELS[model]["family"]
        fam = cfg.FAMILIES[family]
        dump = []
        for condition in C.CONDITIONS:
            request = {"request_id": condition, "item_id": item["item_id"], "seed": 0, "mode": E.MODE,
                       **C.fields(condition, item, shot, examples, filler_tokens=5)}
            text = tokenizer.decode(render.prompt_ids(tokenizer, family, request))
            dump.append(f"===== {condition}\n{text}")
            kind = request["kind"]
            if kind == "direct":
                assert text.rstrip().endswith("Answer:"), (model, condition, text[-80:])
                if request["thinking"]:
                    close = fam.get("forced_close", cfg.FORCED_THINK_CLOSE)
                    assert text.endswith(request["filler"] + close + "Answer:"), (model, condition, text[-120:])
            if kind == "cot" and fam.get("think_open_text"):
                assert text.endswith(fam["think_open_text"]), (model, condition, text[-80:])
            if condition == "C4":
                assert all(ex["reasoning"] in text for ex in examples), (model, "C4 drops example reasoning")
            if condition == "C4off":
                assert all(f"<output_reasoning>\n{ex['reasoning']}" in text for ex in examples), (model, condition)
        if family == "qwen3.8":
            c0 = "\n".join(d for d in dump if d.startswith("===== C0\n"))
            assert "<think>\n\n</think>\n\nAnswer: " + shot["answer"] not in c0, "Qwen3.8 demo turn carries think"
        (out_dir / f"{model}.txt").write_text("\n\n".join(dump))
        print(f"{model}: {len(C.CONDITIONS)} conditions render as expected")


if __name__ == "__main__":
    test_texts()
    test_rendering()
    print("all checks passed")
