"""Fuzz check: cc_grade's first-violation locators agree with CoT-Control's grader on synthetic traces.

For each programmatically graded mode, start from a compliant passage (cc_prompts.stacked_example), apply random
small edits (case flips, dropped/added target words, inserted keywords, punctuation, newlines), and check that
"locator finds a violation" == "grader says non-compliant" on every variant.

Run: /venv/main/bin/python scripts/check_locators.py
"""

from __future__ import annotations

import json
import random

import cc_config as cfg
import cc_grade
import cc_prompts

N_VARIANTS = 400
EDIT_WORDS = ["meow", "safe", "Meow", "SAFE", "x", "=", "2", "A)", "the", "The", "\n", ".", "Step", "1."]


def mutate(text: str, rng: random.Random, extra_words: list[str]) -> str:
    words = text.split(" ")
    for _ in range(rng.randint(0, 3)):
        op = rng.random()
        i = rng.randrange(len(words)) if words else 0
        if op < 0.3 and words:
            words.pop(i)
        elif op < 0.6:
            words.insert(i, rng.choice(EDIT_WORDS + extra_words))
        elif words:
            w = words[i]
            j = rng.randrange(len(w)) if w else 0
            words[i] = w[:j] + w[j:j + 1].swapcase() + w[j + 1:]
    out = " ".join(words)
    return out if rng.random() < 0.8 else "\n" + out + rng.choice(["", " ", "\n", "."])


def main() -> None:
    items = [json.loads(line) for line in cfg.ITEMS_PATH.open()]
    rng = random.Random(0)
    checked = 0
    for mode in cfg.MODES:
        if mode in cfg.JUDGED_MODES:
            continue
        for item in items[:10]:
            prompt = cc_prompts.cotcontrol_prompt(item, mode)[1]
            extra = [item["keyword"]] + item["synonyms"][:2] + [k.strip() for k in item["multi_keyword"].split(",")]
            base = cc_prompts.stacked_example(mode)
            for _ in range(N_VARIANTS // 10):
                text = mutate(base, rng, extra)
                if not text.strip():
                    continue
                compliant = cc_grade.grade_csv.grade_single_row({"mode": mode, "reasoning": text, "question": prompt})
                char = cc_grade.first_violation_char(mode, text, item)
                if (char is None) != bool(compliant):
                    raise SystemExit(f"DISAGREE {mode} {item['item_id']}: grader={compliant} char={char}\n{text!r}")
                checked += 1
    print(f"locators agree with the grader on {checked} synthetic traces")


if __name__ == "__main__":
    main()
