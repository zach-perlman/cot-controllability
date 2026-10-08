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
START = re.compile(r"The potion starts out (\w+)\.")
STIRS = re.compile(r"You stir in, one at a time: ([^.\n]+)")


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
    table, order = {}, []
    for line in problem.splitlines():
        m = RULE_LINE.match(line.strip())
        if m:
            colour, *pairs = m.groups()
            table[colour] = {pairs[i + 1]: pairs[i] for i in range(0, 6, 2)}
            order.append(colour)
    start = START.search(problem).group(1)
    stirs = tuple(s.strip() for s in re.split(r",\s*(?:then\s+)?|\s+then\s+", STIRS.search(problem).group(1).strip())
                  if s.strip())
    brew = Brew(table, tuple(order), start, stirs)
    assert all(len(set(row[i] for row in table.values())) == len(table) for i in brew.ingredients), "not bijective"
    return brew


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
