"""exp10's name-leak grader for mhn: does a text name a person or film of the chain?

An mhn item chains facts through people, films and numbers (nocot-bench's mhn_chains.json gives every hop's
output). The rule bans every name, title and number but the final answer; numbers are read by exp10_leak_numbers.
For one entity (exp10_tasks stores its forms with the item: a person's name and family names from nocot-bench's
mhn_library.json, a film's title) the grader finds, accents and case folded unless noted:
  name      the full name or title (a title without a leading "The"); a one-word title and a one-word family
            name are matched only capitalised, so the words "crash" or "ford" in prose pass
  family    a family name of a person ("Mendes", "de Gaulle"); the name's last word if the library gives none
  initials  the person's initials ("S.M.", "S. M.", "SM")
Readings (exp10_tasks.grade_row):
  intermediate  any form of an entity the chain passes through (outputs of hops 1..N-1)
  strict        any capitalised word sequence that is not in the question, not sentence-initial and not a stop
                word (proper_names); this reading also catches a person named through a description of another
                ("the director of American Beauty"), which no list of the chain's entities can
A description that names nothing ("the British director of a war film") passes every reading; the hand audit
reads for it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Leak:
    category: str
    start: int
    end: int
    text: str
    entity: str | None


def fold(text: str) -> str:
    """Accents removed, length kept (one output character per input character), so spans index the original."""
    return "".join(unicodedata.normalize("NFKD", c)[0] for c in text)


def _phrase(text: str) -> str:
    return r"\s+".join(re.escape(w) for w in text.split())


def entity_patterns(entity: dict) -> list[tuple[str, re.Pattern]]:
    """(category, pattern) of one entity: {"kind": "person" | "film", "name": ..., "family": [...]}."""
    name = fold(entity["name"])
    if entity["kind"] == "film":
        name = re.sub(r"^The\s+", "", name)
    family = entity.get("family") or ([name.split()[-1]] if entity["kind"] == "person" else [])
    out = []
    for category, form in [("name", name)] + [("family", fold(f)) for f in family]:
        if len(form) < 2:
            continue
        flags = re.IGNORECASE if " " in form else 0
        out.append((category, re.compile(rf"(?<![\w-]){_phrase(form)}(?![\w-])", flags)))
    if entity["kind"] == "person":
        initials = [w[0] for w in name.split() if w[:1].isupper()]
        if len(initials) >= 2:
            dotted = r"\.?\s?".join(initials)
            out.append(("initials", re.compile(rf"(?<![\w.]){dotted}\.?(?![\w])")))
    return out


def entity_leaks(text: str, entities: list[dict]) -> list[Leak]:
    """Every match of a form of one of the entities, in order (a span is reported once)."""
    folded, out, taken = fold(text), [], []
    for entity in entities:
        for category, pattern in entity_patterns(entity):
            for m in pattern.finditer(folded):
                if any(m.start() < end and start < m.end() for start, end in taken):
                    continue
                taken.append((m.start(), m.end()))
                out.append(Leak(category, m.start(), m.end(), text[m.start():m.end()], entity["name"]))
    return sorted(out, key=lambda leak: leak.start)


STOP_WORDS = {
    "I", "I'm", "I'll", "I've", "I'd", "OK", "Okay", "Answer", "Let", "Step", "Steps", "Now", "So", "Then", "Next",
    "The", "This", "That", "These", "Those", "It", "Its", "He", "She", "His", "Her", "They", "We", "You", "My",
    "Wait", "Hmm", "Yes", "No", "First", "Second", "Third", "Finally", "Therefore", "Thus", "But", "And", "Also",
    "Since", "Because", "If", "In", "On", "At", "For", "From", "To", "Of", "By", "With", "As", "Is", "Was",
    "Roman", "Arabic",  # the rule's own "Roman numerals"
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
    "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}
_CAPITALISED = re.compile(r"[A-Z][\w'’.-]*[A-Za-z]|[A-Z]{2,}")
_SEQUENCE = re.compile(rf"(?:{_CAPITALISED.pattern})(?:\s+(?:(?:de|van|von|der|da|di|du|la|le)\s+)?"
                       rf"(?:{_CAPITALISED.pattern}))*")
_SENTENCE_START = re.compile(r"(?:^|[.!?:;\n\"“(*#-]\s*|\d\.\s+)$")


def proper_names(text: str, question: str) -> list[Leak]:
    """Capitalised word sequences of the text that the question does not contain: not single letters (mhn's
    variables), not STOP_WORDS, not a single sentence-initial word (one capitalised word after a sentence break
    is read as an ordinary word)."""
    folded, question_words = fold(text), set(re.findall(r"[\w'’.-]+", fold(question)))
    out = []
    for m in _SEQUENCE.finditer(folded):
        words = [w for w in m.group(0).split() if w not in STOP_WORDS and w not in question_words
                 and w[0].isupper()]
        if not words:
            continue
        sentence_initial = bool(_SENTENCE_START.search(folded[:m.start()]))
        if sentence_initial and len(m.group(0).split()) == 1:
            continue
        out.append(Leak("proper_name", m.start(), m.end(), text[m.start():m.end()], None))
    return out


def mask(text: str, entities: list[dict], question: str, symbol: str = "■") -> str:
    """The text with the entities' forms and every other proper name replaced by `symbol` (C4's example traces;
    numbers are masked by exp10_leak_numbers.mask)."""
    spans = sorted([(f.start, f.end) for f in entity_leaks(text, entities)]
                   + [(f.start, f.end) for f in proper_names(text, question)])
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    out, last = [], 0
    for start, end in merged:
        out += [text[last:start], symbol]
        last = end
    return "".join(out + [text[last:]])
