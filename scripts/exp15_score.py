"""exp15 scoring: per-item targets and controls, colour / count matching of text readouts, rank aggregation, AUC,
bootstrap and Holm (no model, no GPU).

Notation for a single-table item: T is the table (a derangement), s_0 the start, s_k = T^k(s_0), h the stir count,
a' the model's answer (argmax over the 10 colour tokens). key = T^-1(a') is the colour the model's last lookup must
have started from; on a correct item it is s_{h-1}.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

import exp11_brew as B

# p(a') over the colour softmax for an error to count as consistent. The plan's 0.5 left 3 of 52 dev h=3 off-cycle
# errors, so 0.3 is primary (a pre-lock deviation, in the manifest); SENSITIVITY_P are reported alongside.
CONSISTENT_P = 0.3
SENSITIVITY_P = (0.0, 0.5, 0.8)
COUNTS = (1, 2, 3, 4, 5, 6)


# --- per-item targets -----------------------------------------------------------------------------------------------
def step(brew: B.Brew, colour: str, k: int = 1) -> str:
    """T^k(colour); negative k applies the inverse."""
    for _ in range(abs(k)):
        colour = brew.apply(colour, [B.SINGLE]) if k > 0 else brew.inverse(colour, B.SINGLE)
    return colour


@dataclass
class Item:
    item_id: str
    h: int
    gold: str
    pred: str
    p_pred: float                     # softmax over the 10 colour logits
    group: str                        # correct | offcycle | miscount | miscount_ambiguous
    path: list[str]                   # s_0..s_h
    implied: list[str]                # T^-h(a')..a': the trajectory consistent with a' and correct lookups
    key: str                          # T^-1(a')
    true_prev: str                    # s_{h-1}
    forward: str                      # T(a')
    implied_count: int | None         # the k != h with T^k(s_0) = a' (miscount items with a unique k)
    controls: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return self.p_pred >= CONSISTENT_P


def analyse(item_id: str, brew: B.Brew, colours: list[str], colour_logits) -> Item:
    probs = torch.softmax(torch.tensor(colour_logits, dtype=torch.float32), -1)
    pred = colours[int(probs.argmax())]
    h = len(brew.stirs)
    path = brew.path()
    implied = [step(brew, pred, -j) for j in range(h, -1, -1)]
    hops = [k for k in range(h + 4) if step(brew, path[0], k) == pred]
    if pred == path[-1]:
        group, implied_count = "correct", None
    elif not hops:
        group, implied_count = "offcycle", None
    elif len(hops) == 1:
        group, implied_count = "miscount", hops[0]
    else:
        group, implied_count = "miscount_ambiguous", None
    key, forward = step(brew, pred, -1), step(brew, pred, 1)
    excluded = set(path) | set(implied) | {forward}
    return Item(item_id=item_id, h=h, gold=path[-1], pred=pred, p_pred=float(probs.max()), group=group, path=path,
                implied=implied, key=key, true_prev=path[-2] if h >= 1 else path[0], forward=forward,
                implied_count=implied_count, controls=[c for c in colours if c not in excluded])


# --- matching text readouts -----------------------------------------------------------------------------------------
_CJK = r"\u4e00-\u9fff"


def _zh(char: str, not_after: str = "", compounds: tuple[str, ...] = ()) -> str:
    """A Chinese colour word: '<char>色' (not right after a character in `not_after`), '<char>' standing alone (no
    Chinese character on either side, as in '绿 (' or '蓝 →'; this skips words like 黑暗 or 明白), or a compound."""
    with_suffix = (f"(?<![{not_after}])" if not_after else "") + f"{char}色"
    alone = f"(?<![{_CJK}]){char}(?![{_CJK}])"
    return "|".join((with_suffix, alone, *compounds))


# English, plus Chinese: the AO answers in Chinese in about a quarter of dev samples, often with the bare character
COLOUR_PATTERNS = {
    "red": r"\bred\b|" + _zh("红", not_after="粉"),            # 粉红色 is pink
    "blue": r"\bblue\b|" + _zh("蓝"),
    "green": r"\bgreen\b|" + _zh("绿"),
    "gold": r"\bgold(?:en)?\b|" + _zh("金", compounds=("金黄色",)),
    "pink": r"\bpink\b|" + _zh("粉", compounds=("粉红色?",)),
    "gray": r"\bgr[ae]y\b|" + _zh("灰"),
    "brown": r"\bbrown\b|" + _zh("[棕褐]"),
    "black": r"\bblack\b|" + _zh("黑"),
    "white": r"\bwhite\b|" + _zh("白"),
    "purple": r"\bpurple\b|" + _zh("紫"),
}
# counts are matched only as stir counts ("twice", "three times", "3 stirs"): bare "one" / "two" are too common
_NUM = {1: ("one", "1"), 2: ("two", "2"), 3: ("three", "3"), 4: ("four", "4"), 5: ("five", "5"), 6: ("six", "6")}
COUNT_PATTERNS = {
    k: r"\b(?:" + "|".join([*(f"{w}\\s+(?:times?|stirs?|stirrings?)" for w in words),
                            *(["once"] if k == 1 else ["twice"] if k == 2 else ["thrice"] if k == 3 else [])])
       + r")\b"
    for k, words in _NUM.items()
}
_COLOUR_RE = {c: re.compile(p, re.IGNORECASE) for c, p in COLOUR_PATTERNS.items()}
_COUNT_RE = {k: re.compile(p, re.IGNORECASE) for k, p in COUNT_PATTERNS.items()}


def colours_named(text: str) -> set[str]:
    return {c for c, rx in _COLOUR_RE.items() if rx.search(text)}


def counts_named(text: str) -> set[int]:
    return {k for k, rx in _COUNT_RE.items() if rx.search(text)}


def naming_fraction(samples: list[str], labels, matcher) -> torch.Tensor:
    """Fraction of samples naming each label: [len(labels)]."""
    hits = [matcher(s) for s in samples]
    return torch.tensor([sum(lab in h for h in hits) / max(len(samples), 1) for lab in labels])


# --- aggregation and statistics -------------------------------------------------------------------------------------
def ranks(scores: torch.Tensor) -> torch.Tensor:
    """Average ranks (0 = lowest) along the last dim, ties averaged: [..., C] -> [..., C] float."""
    s = scores.float()
    less = (s[..., None, :] < s[..., :, None]).float().sum(-1)
    equal = (s[..., None, :] == s[..., :, None]).float().sum(-1)
    return less + 0.5 * (equal - 1)


def aggregate(scores: torch.Tensor) -> torch.Tensor:
    """Per-item reader score S(c): the within-(cell, layer) rank of c among the 10 colours, averaged over the given
    (cell, layer) pairs. scores [..., C] with any leading dims -> [C]."""
    return ranks(scores).reshape(-1, scores.shape[-1]).mean(0)


def item_auc(s: torch.Tensor, target: int, controls: list[int]) -> float:
    """P(S(target) > S(control)) over the item's controls, ties counted half."""
    t, c = s[target], s[controls]
    return float((t > c).float().mean() + 0.5 * (t == c).float().mean())


def item_top(s: torch.Tensor, target: int, others: list[int]) -> float:
    """1 if the target outscores every other candidate, split evenly on ties at the top."""
    best = s[[target, *others]].max()
    if s[target] < best:
        return 0.0
    return 1.0 / int((s[[target, *others]] == best).sum())


def boot_ci(values, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float, int]:
    """Mean and 95% percentile bootstrap CI over items: (mean, lo, hi, n)."""
    x = torch.as_tensor(values, dtype=torch.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    g = torch.Generator().manual_seed(seed)
    b = x[torch.randint(len(x), (n_boot, len(x)), generator=g)].mean(1)
    return float(x.mean()), float(b.quantile(0.025)), float(b.quantile(0.975)), len(x)


def boot_p_greater(values, null: float, n_boot: int = 10_000, seed: int = 0) -> float:
    """One-sided bootstrap p-value for mean(values) > null: the fraction of resampled means at or below null."""
    x = torch.as_tensor(values, dtype=torch.float64)
    if len(x) == 0:
        return float("nan")
    g = torch.Generator().manual_seed(seed)
    b = x[torch.randint(len(x), (n_boot, len(x)), generator=g)].mean(1)
    return float(((b <= null).double().sum() + 1) / (n_boot + 1))


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values (family = the dict)."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


# --- per-item read metrics ------------------------------------------------------------------------------------------
@dataclass
class ReaderScores:
    """One reader's colour scores per item at its primary (cell, layer) pairs.
    kind = 'rank': S(c) = within-pair rank, averaged over pairs (lenses, probe, diffmeans);
    kind = 'fraction': S(c) = fraction of samples naming c, averaged over pairs (text readers)."""
    name: str
    kind: str
    pairs: list[tuple[str, int]]
    scores: dict[str, torch.Tensor]          # item_id -> [n_pairs, 10]

    def score(self, item_id: str, keep: torch.Tensor | None = None) -> torch.Tensor | None:
        x = self.scores.get(item_id)
        if x is None:
            return None
        if keep is not None:
            x = x[keep]
        if len(x) == 0:
            return None
        return aggregate(x) if self.kind == "rank" else x.float().mean(0)


def direction_eligible(it: Item) -> bool:
    """The direction test (key vs T(a')) needs T(a') to be a third colour, off the true path."""
    return it.forward != it.key and it.forward not in it.path


def read_metrics(s: torch.Tensor, it: Item, index: dict[str, int]) -> dict:
    """Per-item metrics from a reader score S [10]:
    key_auc: S(key) vs controls; direction: S(key) vs S(T(a')) (None if ineligible);
    on errors also true_auc: S(true s_{h-1}) vs controls, and top shares among {key, true s_{h-1}, controls}."""
    ctl = [index[c] for c in it.controls]
    out = {"key_auc": item_auc(s, index[it.key], ctl),
           "direction": item_auc(s, index[it.key], [index[it.forward]]) if direction_eligible(it) else None}
    if it.group != "correct":
        k, t = index[it.key], index[it.true_prev]
        out["true_auc"] = item_auc(s, t, ctl)
        out["top_key"] = item_top(s, k, [t, *ctl])
        out["top_true"] = item_top(s, t, [k, *ctl])
        out["top_neither"] = 1.0 - out["top_key"] - out["top_true"]
    return out


def summarise(rows: list[dict], metric: str, null: float | None, n_boot: int = 10_000) -> dict:
    """Bootstrap mean and CI of one metric over items (skipping None), with a one-sided p against `null`."""
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    mean, lo, hi, n = boot_ci(vals, n_boot)
    out = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n}
    if null is not None:
        out["p"] = boot_p_greater(vals, null, n_boot) if n else float("nan")
    return out
