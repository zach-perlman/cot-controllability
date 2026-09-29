"""exp03's abort rule: when may generation stop because a trace has already broken its rule for good?

In the ABORT_MODES every rule is "no unit may break it" (no forbidden word, every letter lowercase, every sentence
ends with the word, ...), and a unit, once complete, never changes. So once a violation sits in text that is
complete, the trace is non-compliant whatever follows. The text seen so far is cut at its last whitespace (no
partial word) and cc_grade's first-violation locator is run on it; the violation counts only if it lies at least
ABORT_MARGIN_CHARS before the cut, so the violating unit and its neighbours (the meow locator's math-token and
line-end rules look at the next word; a multi-word synonym may start before a single-word hit) are complete.

scripts/check_abort_rule.py replays this rule on every exp02 trace, at every 16th character: whenever it fires,
the full trace must be non-compliant with the same first violation.
"""

from __future__ import annotations

import re

import cc_config as cfg
import cc_grade

LAST_WHITESPACE = re.compile(r"\s(?=\S*$)")


def confirmed_violation(mode: str, text: str, item: dict) -> int | None:
    """Character offset of the first violation in text if no continuation can undo it, else None."""
    if mode not in cfg.ABORT_MODES:
        return None
    cut = LAST_WHITESPACE.search(text)
    if cut is None:
        return None
    seen = text[:cut.start()]
    char = cc_grade.first_violation_char(mode, seen, item)
    if char is None or char > len(seen) - cfg.ABORT_MARGIN_CHARS:
        return None
    return char


def check_interval(n_tokens: int) -> int:
    """New tokens to wait before the next check of a trace of n_tokens: at most 1/8 more text is generated after
    a violation becomes confirmable, and a long trace is checked O(log n) times."""
    return max(cfg.ABORT_CHECK_MIN_TOKENS, n_tokens // 8)
