"""Unit tests of exp10_leak_names on hand-written strings (no model calls).
  /venv/main/bin/python scripts/test_exp10_leak_names.py
"""

from __future__ import annotations

import exp10_leak_names as L

MENDES = {"kind": "person", "name": "Sam Mendes", "family": ["Mendes"]}
GAULLE = {"kind": "person", "name": "Charles de Gaulle", "family": ["de Gaulle"]}
CRASH = {"kind": "film", "name": "Crash", "family": []}
GODFATHER = {"kind": "film", "name": "The Godfather Part II", "family": []}
BRONTE = {"kind": "person", "name": "Émile Brontë", "family": ["Brontë"]}


def cats(text: str, entities: list[dict]) -> list[tuple[str, str]]:
    return [(f.category, f.text) for f in L.entity_leaks(text, entities)]


def test_entities() -> None:
    assert cats("so it is Sam Mendes, then Mendes's birth", [MENDES]) == [("name", "Sam Mendes"),
                                                                           ("family", "Mendes")]
    assert cats("that was S. M. who", [MENDES]) == [("initials", "S. M.")]
    assert cats("then de Gaulle", [GAULLE]) == [("family", "de Gaulle")]
    assert cats("the film Crash won", [CRASH]) == [("name", "Crash")]
    assert cats("a car crash", [CRASH]) == []
    assert cats("the godfather part ii won", [GODFATHER]) == [("name", "godfather part ii")]
    assert cats("Emile Bronte wrote", [BRONTE]) == [("name", "Emile Bronte")]
    assert cats("Brontë", [BRONTE]) == [("family", "Brontë")]
    no_family = {"kind": "person", "name": "Frances McDormand", "family": []}
    assert cats("then McDormand", [no_family]) == [("family", "McDormand")]
    assert cats("Li was", [{"kind": "person", "name": "Li Po", "family": ["Li"]}]) == [("family", "Li")]


def test_proper_names() -> None:
    question = "Let A be the winner of the Nobel Prize in Peace awarded in 1945. Let B be the year of A."
    names = [f.text for f in L.proper_names("Now A is known. The winner was Cordell Hull, so B is that. "
                                            "I recall the Nobel Prize list. Okay.", question)]
    assert names == ["Cordell Hull"], names
    assert [f.text for f in L.proper_names("the director of American Beauty", "Let A be")] == ["American Beauty"]
    assert not L.proper_names("Then B. Next C. Wait, May is a month.", question)


def test_mask() -> None:
    masked = L.mask("A is Sam Mendes. Mendes made American Beauty.", [MENDES], "Let A be")
    assert masked == "A is ■. ■ made ■.", masked


if __name__ == "__main__":
    for test in (test_entities, test_proper_names, test_mask):
        test()
        print(f"{test.__name__} ok")
