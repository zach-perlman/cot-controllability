"""exp11: brew items as explicit permutation automata, and behaviour-level hypotheses for wrong answers.

A brew item is a rule table (colour x ingredient -> colour, each ingredient a derangement of the colours), a start
colour and a sequence of stirs. Because every ingredient is a bijection, a wrong final colour y pins, for each stir j,
exactly one colour that the potion "must have been" after stir j for the remaining stirs (applied correctly) to give y.
That back-propagated path is what the activation-level analysis tests; this module only parses items and enumerates
the shortcut hypotheses that a wrong answer can be checked against without activations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RULE_LINE = re.compile(r"^A (\w+) potion turns (\w+) with (\w+), (\w+) with (\w+), and (\w+) with (\w+)\.$")
RULE_SINGLE = re.compile(r"^A (\w+) potion turns (\w+) when stirred\.$")
START = re.compile(r"The potion starts out (\w+)\.")
STIRS = re.compile(r"You stir in, one at a time: ([^.\n]+)")
STIR_COUNT = re.compile(r"You stir it (once|twice|three times)\.")
COUNT_WORDS = {1: "once", 2: "twice", 3: "three times"}
SINGLE = "stir"   # the one ingredient of a single-table item


@dataclass(frozen=True)
class Brew:
    table: dict[str, dict[str, str]]   # table[colour][ingredient] -> colour
    rule_order: tuple[str, ...]        # colours in the order their rule lines are printed
    start: str
    stirs: tuple[str, ...]

    @property
    def ingredients(self) -> tuple[str, ...]:
        return tuple(next(iter(self.table.values())).keys())

    def apply(self, colour: str, stirs) -> str:
        for ingredient in stirs:
            colour = self.table[colour][ingredient]
        return colour

    def path(self) -> list[str]:
        """s_0 (start), s_1, ..., s_h: the true colour after each stir."""
        states = [self.start]
        for ingredient in self.stirs:
            states.append(self.table[states[-1]][ingredient])
        return states

    def inverse(self, colour: str, ingredient: str) -> str:
        """The colour that turns into `colour` with `ingredient` (unique: each ingredient is a bijection)."""
        return next(c for c, row in self.table.items() if row[ingredient] == colour)

    def back_path(self, final: str) -> list[str]:
        """w_0..w_h with w_h = final and w_{j-1} = inverse(w_j, stir j): the states consistent with answering `final`
        if every stir from j on were applied correctly. Where w_j differs from the true s_j, an error at or before
        stir j is implied."""
        states = [final]
        for ingredient in reversed(self.stirs):
            states.append(self.inverse(states[-1], ingredient))
        return states[::-1]


def parse(problem: str) -> Brew:
    """Both wordings: nocot-bench's three-ingredient table and the single 'stir' table, start sentence anywhere."""
    table, order = {}, []
    for line in problem.splitlines():
        if m := RULE_LINE.match(line.strip()):
            colour, *pairs = m.groups()
            table[colour] = {pairs[i + 1]: pairs[i] for i in range(0, 6, 2)}
            order.append(colour)
        elif m := RULE_SINGLE.match(line.strip()):
            table[m.group(1)] = {SINGLE: m.group(2)}
            order.append(m.group(1))
    start = START.search(problem).group(1)
    if m := STIRS.search(problem):
        stirs = tuple(s.strip() for s in re.split(r",\s*(?:then\s+)?|\s+then\s+", m.group(1).strip()) if s.strip())
    else:
        h = {w: n for n, w in COUNT_WORDS.items()}[STIR_COUNT.search(problem).group(1)]
        stirs = (SINGLE,) * h
    brew = Brew(table, tuple(order), start, stirs)
    assert all(len(set(row[i] for row in table.values())) == len(table) for i in brew.ingredients), "not bijective"
    return brew


# --- The format pilot's generator (manifest deviation 1) ---------------------------------------------------------
# Table type x start position. three_first is nocot-bench's own wording; single_last is WorkspaceBench's.
FORMATS = ("three_first", "three_last", "single_first", "single_last")
# format -> (table type for generate, start position for render); three_same is deviation 2's control: the
# three-ingredient table and wording with one ingredient stirred every time
FORMAT_SPEC = {"three_first": ("three", "first"), "three_last": ("three", "last"), "single_first": ("single", "first"),
               "single_last": ("single", "last"), "three_same": ("three_same", "first")}
THREE_HEADER = "A potion changes color each time an ingredient is stirred in. The rules:"
SINGLE_HEADER = "A potion changes color each time it is stirred. The rules:"
QUESTION = "What color is the potion at the end?"


def _derangement(rng, n: int) -> list[int]:
    idx = list(range(n))
    while True:
        rng.shuffle(idx)
        if all(idx[i] != i for i in range(n)):
            return idx[:]


def generate(rng, h: int, table_type: str, colours, ingredients, max_tries: int = 4000) -> dict:
    """One item under nocot-bench's brew rules (datagen/banks/brew.py, _gen_brew): each ingredient a derangement,
    all states distinct, the last-stir-only answer never gold, and (three ingredients) >= 2 distinct ingredients and
    8 sampled reorderings of the stirs all changing the gold. A single table has one ingredient, SINGLE;
    'three_same' is a three-ingredient table with one of its ingredients stirred every time."""
    n = len(colours)
    for _ in range(max_tries):
        ings = [SINGLE] if table_type == "single" else rng.sample(list(ingredients), 3)
        perms = {ing: _derangement(rng, n) for ing in ings}
        start = rng.randrange(n)
        seq = [rng.choice(ings)] * h if table_type == "three_same" else [rng.choice(ings) for _ in range(h)]
        if table_type == "three" and h > 1 and len(set(seq)) < 2:
            continue
        states = [start]
        for ing in seq:
            states.append(perms[ing][states[-1]])
        if len(set(states)) != h + 1 or (h > 1 and perms[seq[-1]][start] == states[-1]):
            continue
        if table_type == "three" and h > 1:
            collide = False
            for _p in range(8):
                other = seq[:]
                rng.shuffle(other)
                if other == seq:
                    continue
                w = start
                for ing in other:
                    w = perms[ing][w]
                collide |= w == states[-1]
            if collide:
                continue
        order = list(colours)
        rng.shuffle(order)
        return {"ings": ings, "perms": perms, "start": start, "seq": seq, "line_order": order, "h": h,
                "answer": colours[states[-1]]}
    raise RuntimeError(f"rejection loop exhausted (h={h}, {table_type})")


def render(item: dict, fmt: str, colours) -> str:
    table_type, start_pos = FORMAT_SPEC[fmt]
    idx = {c: i for i, c in enumerate(colours)}
    start = f"The potion starts out {colours[item['start']]}."
    if table_type != "single":
        lines = [THREE_HEADER]
        for c in item["line_order"]:
            parts = [f"{colours[item['perms'][ing][idx[c]]]} with {ing}" for ing in item["ings"]]
            lines.append(f"A {c} potion turns {parts[0]}, {parts[1]}, and {parts[2]}.")
        stirs = f"You stir in, one at a time: {', then '.join(item['seq'])}."
    else:
        lines = [SINGLE_HEADER] + [f"A {c} potion turns {colours[item['perms'][SINGLE][idx[c]]]} when stirred."
                                   for c in item["line_order"]]
        stirs = f"You stir it {COUNT_WORDS[item['h']]}."
    lines.append(f"{start} {stirs}" if start_pos == "first" else f"{stirs} {start}")
    lines.append(QUESTION)
    return "\n".join(lines)


def hypotheses(brew: Brew) -> dict[str, set[str]]:
    """Shortcut families, each the set of final colours it predicts. Families can overlap; gold is removed by the
    caller (a wrong answer is never gold)."""
    s, h = brew.path(), len(brew.stirs)
    out: dict[str, set[str]] = {
        "stop_early": set(s[:h]),                               # an intermediate (or the start) reported as final
        "skip_one_stir": {brew.apply(brew.start, brew.stirs[:j] + brew.stirs[j + 1:]) for j in range(h)},
        "reverse_order": {brew.apply(brew.start, brew.stirs[::-1])},
        "last_stir_only": {brew.apply(brew.start, brew.stirs[-1:])},
        # right row, wrong ingredient column, at exactly one stir
        "wrong_column": {brew.apply(brew.table[s[j]][alt], brew.stirs[j + 1:])
                         for j, ing in enumerate(brew.stirs) for alt in brew.ingredients if alt != ing},
        # the rule read backwards at exactly one stir ("which colour turns into this one")
        "inverse_lookup": {brew.apply(brew.inverse(s[j], ing), brew.stirs[j + 1:])
                           for j, ing in enumerate(brew.stirs)},
    }
    return out
