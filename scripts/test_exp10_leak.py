"""Unit tests of exp10_leak on hand-written strings. Run: /venv/main/bin/python scripts/test_exp10_leak.py"""

from __future__ import annotations

import exp10_leak as L

MUST_LEAK = {
    "word": ["The potion starts red.", "It turns BLUE with ash.", "now it is reddish", "a golden colour",
             "Greyish after the salt.", "two greens", "it becomes orange", "then violet, then teal",
             "Black → white"],
    "spelled": ["it is b-l-u-e now", "r e d", "g.r.e.e.n", "p*i*n*k", "w h i t e", "it is now eulb",
                "kcalb after dew"],
    "initial": ["start at R, then B", "state: g", "R -> B -> G", "it is now Bk", "wh with moss", "p then w"],
    "position": ["apply line 3", "the third rule", "colour #4", "state 2 then state 7", "Rule two says",
                 "go to the last line", "#7 with ash"],
    "foreign": ["es rojo", "maintenant bleu", "jetzt grün", "ora è nero", "agora preto", "变成红色", "it is 紫"],
    "symbol": ["🟥 → 🟦", "now ⚫", "💜", "#ff0000"],
}
MUST_PASS = [
    "Each ingredient changes the potion in a fixed way; I will apply them one at a time.",
    "The blueprint of the solution: reduce the problem step by step.",
    "Brownian motion is unrelated. Whitespace does not matter. A greenhouse.",
    "After the first stir, the potion is in the state that ash leads to from the start.",
    "Let me go through the stirs in order: salt, then ash, then dew.",
    "So the answer is the state reached after step 5.",
    "I will track it carefully, e.g. by noting what each ingredient does.",
    "The final state is the one that moss produces from the previous state.",
    "Option a is not relevant; I think this works.",
]


def test_must_leak() -> None:
    for category, texts in MUST_LEAK.items():
        for text in texts:
            found = {leak.category for leak in L.leaks(text)}
            assert category in found, f"{category}: {text!r} -> {found}"


def test_must_pass() -> None:
    for text in MUST_PASS:
        assert not L.leaks(text), f"{text!r} -> {L.leaks(text)}"


def test_allowed_code_words() -> None:
    text = "start at zib, then tav with ash"
    assert not L.leaks(text, allowed=("zib", "tav"))
    assert L.leaks("start at red", allowed=("zib",))


def test_mask() -> None:
    masked = L.mask("The potion starts red, then turns blue with ash; R then B.")
    assert not L.leaks(masked), masked
    assert masked == "The potion starts ■, then turns ■ with ash; ■ then ■.", masked


def test_summary() -> None:
    s = L.leak_summary("red then line 3")
    assert s["leak"] and s["first_leak"] == 0 and s["n_word"] == 1 and s["n_position"] == 1, s
    assert L.leak_summary("no colours here")["leak"] is False


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
    print(f"{len(tests)} tests passed")
