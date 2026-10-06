"""Unit tests of exp10_leak_numbers on hand-written strings (no model calls).
  /venv/main/bin/python scripts/test_exp10_leak_numbers.py
"""

from __future__ import annotations

import exp10_leak_numbers as N


def found(text: str) -> list[tuple[str, int | None]]:
    return [(f.category, f.value) for f in N.leaks(text)]


def test_forms() -> None:
    cases = {
        "it becomes 13": [("digit", 13)],
        "the 65th ceremony": [("digit", 65)],
        "now seven": [("word", 7)],
        "twenty-one and sixty five": [("word", 21), ("word", 65)],
        "the twenty-first one": [("ordinal", 21)],
        "chapter XII": [("roman", 12)],
        "born MCMLXV": [("roman", 1965)],
        "s-e-v-e-n": [("spelled", 7)],
        "neves": [("spelled", 7)],
        "siete y ocho": [("foreign", 7), ("foreign", 8)],
        "十三": [("foreign", 13)],
        "⑦ then 3\ufe0f\u20e3": [("symbol", 7), ("symbol", 3)],
        "１２": [("digit", 12)],
        "x2": [("digit", 2)],
    }
    for text, want in cases.items():
        assert found(text) == want, (text, found(text))


def test_must_pass() -> None:
    for text in ("this one is odd", "one at a time", "the next one", "Once more, then halve it", "an MD said",
                 "I halve it", "the due date", "Answer: keep going", "it is even, so halve it",
                 "the fifth one"):
        assert not N.leaks(text) or text == "the fifth one", (text, found(text))
    assert found("the fifth one") == [("ordinal", 5)]


def test_exempt_contexts() -> None:
    text = "Step 3: 13 is odd, add 7 to get 20. Then 20 - 9 = 11, the third step."
    hits = N.intermediate_leaks(text, {3, 7, 9, 13, 20, 11}, operands=True)
    assert [(f.text, f.value) for f in hits] == [("13", 13), ("20", 20), ("20", 20), ("11", 11)], hits
    assert [(f.text, f.value) for f in N.intermediate_leaks("6 x 2 = 12, halve: 6", {2, 6, 12}, True)] \
        == [("6", 6), ("12", 12), ("6", 6)]
    # mhn: no operand exemption, step positions only
    assert [f.text for f in N.intermediate_leaks("Step 2: she was born on the 2nd", {2}, operands=False)] == ["2nd"]


def test_values_only() -> None:
    assert not N.intermediate_leaks("it is 5 now", {4, 6}, operands=True)
    assert N.intermediate_leaks("it is five now", {5}, operands=True)


def test_mask_and_partial() -> None:
    assert N.mask("13 is odd, add 7: twenty") == "■ is odd, add ■: ■"
    assert N.partial_info("it is even; bigger than 10? no, odd") == 3


if __name__ == "__main__":
    for test in (test_forms, test_must_pass, test_exempt_contexts, test_values_only, test_mask_and_partial):
        test()
        print(f"{test.__name__} ok")
