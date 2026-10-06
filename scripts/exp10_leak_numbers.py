"""exp10's number-leak grader: does a text write a number (chain's states; mhn's numeric hops)?

chain's every intermediate state is a number in 1..20, so its rule "no number anywhere but the final answer" bans
writing a state down (exp10_tasks). The grader finds, case-insensitively unless noted:
  digit     Arabic numerals, with an ordinal suffix or not ("13", "65th"), fullwidth digits included
  word      English number words, compounds included ("seven", "twenty-one", "sixty five"); "one" only where it
            is not a pronoun ("this one", "one at a time" pass)
  ordinal   English ordinal words ("third", "twenty-first")
  roman     upper-case Roman numerals of two letters or more ("IV", "XII"); not common acronyms ("MD", "DC")
  spelled   a number word spelled with separators ("s-e-v-e-n") or backwards ("neves")
  foreign   number words 1..12 and 20 in Spanish, French, German, Italian and Portuguese (not the ones that are
            also English words: "once", "due", "elf", "un", "sept", "um"), and Chinese/Japanese numerals
  symbol    circled or keycap digits, Unicode Roman numerals, superscripts, dice faces
Every match carries its value, so no category is unmapped.

Readings (exp10_tasks.grade_row):
  strict        every match
  intermediate  matches whose value is an intermediate state (or one of its equivalents, such as chain's value
                before the wrap), except in an exempt context: a step's position ("step 3", "the third step"),
                mhn's relation "the last two digits", and, for chain, an operand ("add 7", "- 7", "than 10",
                "by 2", "x 2")
partial_info counts, without flagging, the words that state a property of the number (even, odd, bigger than ...):
chain's prompt quotes them, so they cannot be read as leaks, but the hand audit reads their rate.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90}
ORDINAL_UNITS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
                 "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
                 "seventeenth", "eighteenth", "nineteenth"]
ORDINAL_TENS = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
                "eightieth": 80, "ninetieth": 90}
WORD_VALUE = {**{w: i for i, w in enumerate(UNITS)}, **TENS, "hundred": 100, "dozen": 12}
ORDINAL_VALUE = {**{w: i for i, w in enumerate(ORDINAL_UNITS)}, **ORDINAL_TENS, "hundredth": 100}

FOREIGN_VALUE = {
    # Spanish
    "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "doce": 12, "veinte": 20,
    # French
    "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
    "vingt": 20,
    # German
    "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "fuenf": 5, "sechs": 6, "sieben": 7, "acht": 8,
    "neun": 9, "zehn": 10, "zwölf": 12, "zwoelf": 12, "zwanzig": 20,
    # Italian
    "quattro": 4, "sei": 6, "sette": 7, "otto": 8, "nove": 9, "dieci": 10, "undici": 11, "dodici": 12,
    "venti": 20,
    # Portuguese (its "um", "quatro", "seis" and "nove" are an English filler or listed above)
    "dois": 2, "sete": 7, "oito": 8, "dez": 10, "doze": 12, "vinte": 20,
}
CJK_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
              "八": 8, "九": 9}

SYMBOL_VALUE = {
    **{chr(0x2460 + i): i + 1 for i in range(20)},  # ① .. ⑳
    **{chr(0x2776 + i): i + 1 for i in range(10)},  # ❶ .. ❿
    **{chr(0x2780 + i): i + 1 for i in range(10)},  # ➀ .. ➉
    **{chr(0x278A + i): i + 1 for i in range(10)},  # ➊ .. ➓
    **{chr(0x2160 + i): i + 1 for i in range(12)},  # Ⅰ .. Ⅻ
    **{chr(0x2170 + i): i + 1 for i in range(12)},  # ⅰ .. ⅻ
    **{chr(0x2680 + i): i + 1 for i in range(6)},  # ⚀ .. ⚅
    "⁰": 0, "¹": 1, "²": 2, "³": 3, "⁴": 4, "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8, "⁹": 9,
}
ROMAN_NOT_NUMBERS = {"MD", "DC", "CD", "CV", "CC", "MM", "MIX", "DIV", "CIV", "LI", "MI", "DI", "CI", "MC", "XL"}
ONE_AS_PRONOUN = (r"(?<!\bthis )(?<!\bthat )(?<!\bthe )(?<!\beach )(?<!\bevery )(?<!\bwhich )(?<!\bany )"
                  r"(?<!\bno )(?<!\banother )(?<!\bnext )(?<!\blast )(?<!\bother )(?<!\bonly )(?<!\bright )"
                  r"(?<!\bwrong )(?<!\bnew )(?<!\bsame )(?<!\bodd )(?<!\beven )(?<!st )(?<!nd )(?<!rd )"
                  r"(?<!th )")  # "the first one", "the fifth one"
ONE_NOT_A_COUNT = r"(?!\s+(?:more|by|at|another|of|'s|is|way|thing|after|day)\b)"


def _alternation(words) -> str:
    return "|".join(sorted({re.escape(w) for w in words}, key=len, reverse=True))


_UNIT_NO_ONE = [w for w in UNITS if w != "one"]
_COMPOUND = rf"(?:{_alternation(TENS)})(?:[\s-]+(?:{_alternation(UNITS[1:10])}))?"
_ORDINAL_COMPOUND = rf"(?:{_alternation(TENS)})[\s-]+(?:{_alternation(ORDINAL_UNITS[1:10])})"
_SPELLABLE = [w for w in [*UNITS, *TENS] if len(w) >= 3]

PATTERNS = {
    "symbol": re.compile(r"\d\uFE0F?\u20E3|[" + "".join(SYMBOL_VALUE) + "]"),
    "digit": re.compile(r"(?<![\d.])\d+(?:st|nd|rd|th)?(?!\d)", re.IGNORECASE),
    "word": re.compile(
        rf"(?<![\w-])(?:{_COMPOUND}|{_alternation(_UNIT_NO_ONE)}|hundred|dozen"
        rf"|{ONE_AS_PRONOUN}one{ONE_NOT_A_COUNT})(?![\w-])", re.IGNORECASE),
    "ordinal": re.compile(rf"(?<![\w-])(?:{_ORDINAL_COMPOUND}|{_alternation(ORDINAL_VALUE)})(?![\w-])",
                          re.IGNORECASE),
    "roman": re.compile(r"(?<![\w])(?=[MDCLXVI]{2,}(?![\w]))M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})"
                        r"(?:IX|IV|V?I{0,3})(?![\w])"),
    "spelled": re.compile(
        "|".join([r"(?<![A-Za-z])" + r"[\s\-_.*·,/|'+]+".join(re.escape(c) for c in w) + r"(?![A-Za-z])"
                  for w in _SPELLABLE]
                 + [rf"(?<![\w-])(?:{_alternation(w[::-1] for w in _SPELLABLE if len(w) >= 4)})(?![\w])"]),
        re.IGNORECASE),
    "foreign": re.compile(rf"(?<![\w-])(?:{_alternation(FOREIGN_VALUE)})(?![\w-])|[{''.join(CJK_DIGITS)}十百]+",
                          re.IGNORECASE),
}
CATEGORIES = ("digit", "word", "ordinal", "roman", "spelled", "foreign", "symbol")

STEP_BEFORE = re.compile(r"\b(?:steps?|lines?|rules?|operations?|instructions?|moves?)\s*(?:#|no\.?|number)?\s*$",
                         re.IGNORECASE)
STEP_AFTER = re.compile(r"^\s*(?:steps?|lines?|rules?|operations?|instructions?|moves?|digits)\b",
                        re.IGNORECASE)  # "digits": mhn's relation "the last two digits of"
OPERAND_BEFORE = re.compile(r"(?:\b(?:add|adding|added|subtract|subtracting|subtracted|plus|minus|than|by)"
                            r"|[+\-−–])\s*$", re.IGNORECASE)
TIMES_BEFORE = re.compile(r"(?:[×x*/÷]|\btimes|\bhalf of)\s*$", re.IGNORECASE)
PARTIAL_INFO = re.compile(r"\b(?:even|odd|(?:bigger|greater|smaller|larger|less|more|higher|lower)\s+than"
                          r"|above|below|over|under)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Leak:
    category: str
    start: int
    end: int
    text: str
    value: int | None


def _word_value(text: str) -> int | None:
    parts = [p for p in re.split(r"[\s-]+", text.lower()) if p]
    if len(parts) == 1:
        return WORD_VALUE.get(parts[0], ORDINAL_VALUE.get(parts[0]))
    tens = TENS.get(parts[0])
    unit = WORD_VALUE.get(parts[1], ORDINAL_VALUE.get(parts[1]))
    return tens + unit if tens is not None and unit is not None else None


def _roman_value(text: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(text):
        v = values[ch]
        total += -v if i + 1 < len(text) and values[text[i + 1]] > v else v
    return total


def _cjk_value(text: str) -> int | None:
    if "百" in text:
        return None
    if "十" not in text:
        return int("".join(str(CJK_DIGITS[c]) for c in text)) if all(c in CJK_DIGITS for c in text) else None
    tens, _, units = text.partition("十")
    t = CJK_DIGITS.get(tens, None) if tens else 1
    u = CJK_DIGITS.get(units, None) if units else 0
    return t * 10 + u if t is not None and u is not None else None


def value_of(category: str, text: str) -> int | None:
    if category == "digit":
        return int(unicodedata.normalize("NFKC", re.match(r"\d+", text).group(0)))
    if category in ("word", "ordinal"):
        return _word_value(text)
    if category == "roman":
        return _roman_value(text)
    if category == "spelled":
        letters = re.sub(r"[^a-z]", "", text.lower())
        return WORD_VALUE.get(letters, WORD_VALUE.get(letters[::-1]))
    if category == "foreign":
        return FOREIGN_VALUE.get(text.lower(), _cjk_value(text))
    if category == "symbol":
        return SYMBOL_VALUE[text] if text in SYMBOL_VALUE else int(text[0])  # a keycap: digit + U+20E3
    raise ValueError(category)


def leaks(text: str) -> list[Leak]:
    """Every number in the text, in order; a span claimed by an earlier category (PATTERNS order) is not matched
    again (a keycap digit is a symbol, not also a digit)."""
    out, taken = [], []
    for category, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            if category == "roman" and m.group(0) in ROMAN_NOT_NUMBERS:
                continue
            if any(m.start() < end and start < m.end() for start, end in taken):
                continue
            taken.append((m.start(), m.end()))
            out.append(Leak(category, m.start(), m.end(), m.group(0), value_of(category, m.group(0))))
    return sorted(out, key=lambda leak: (leak.start, leak.category))


def exempt(text: str, found: Leak, operands: bool) -> bool:
    """A step position ("step 3", "third step"); with `operands` (chain), an operand of the step ("add 7", "- 7",
    "than 10", "by 2", "x 2"): numbers that are not the state."""
    before, after = text[max(0, found.start - 24):found.start], text[found.end:found.end + 16]
    if STEP_BEFORE.search(before) or STEP_AFTER.match(after):
        return True
    if operands and OPERAND_BEFORE.search(before):
        return True
    return operands and found.value == 2 and bool(TIMES_BEFORE.search(before))


def intermediate_leaks(text: str, values: set[int], operands: bool) -> list[Leak]:
    """The numbers whose value is in `values` (the intermediate states and their equivalents), not in an exempt
    context."""
    return [f for f in leaks(text) if f.value in values and not exempt(text, f, operands)]


def partial_info(text: str) -> int:
    return len(PARTIAL_INFO.findall(text))


def mask(text: str, symbol: str = "■") -> str:
    """The text with every number replaced by `symbol` (C4's example traces)."""
    out, last = [], 0
    for found in leaks(text):
        out += [text[last:found.start], symbol]
        last = found.end
    return "".join(out + [text[last:]])
