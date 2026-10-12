"""exp15 writes: the additive edit, the per-method write directions, and the swap / repair trials.

Edit (every method, matched in size): at the output of decoder block `layer`, for every position p from the start
colour's token to the final token,
    h_p <- h_p + alpha * ||h_p|| * unit(v_layer(c') - v_layer(c)),
where v_layer(colour) is the method's direction for that colour. The model's answer is the argmax over the 10 colour
tokens at the final position ('Answer:' is prefilled and thinking is off, so one forward pass decides it).
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch

import exp15_score as S

WRITE_LAYERS = (28, 32, 36, 40, 42, 44)
ALPHAS = (0.1, 0.25, 0.5, 1.0)
LENS_METHODS = ("logit", "jlens", "rlens", "jpp")
METHODS = (*LENS_METHODS, "template", "diffmeans", "nla_ar", "ptag_ar")
METHOD_LAYERS = {m: WRITE_LAYERS for m in METHODS} | {"nla_ar": (42,), "ptag_ar": (28, 32, 36, 40, 44)}

# NLA AR: explanation-style descriptions of a state whose current colour is {c} (its reconstructor reads
# explanations, not puzzle text). The direction for c is the mean AR output over the four.
NLA_TEMPLATES = (
    "Colour-change puzzle about a potion: tracking the current colour through the stir rules. The potion is {c} "
    "right now, and the model is about to apply the rule for a {c} potion.",
    "Rule lookup in progress: the relevant line is 'A {c} potion turns ...'; the answer is the colour that {c} "
    "turns into.",
    "The intermediate colour before the last stir is {c}; the final answer follows from the {c} rule.",
    "Answer slot after 'Answer:' in a short puzzle; the colour held in mind is {c}, about to be mapped through the "
    "table once more.",
)
# ptag AR: short spans whose last token would carry 'the current colour is {c}'.
PTAG_TEMPLATES = (
    "The potion is now {c}",
    "A {c} potion turns",
    "After that stir the potion is {c}",
    "Before the last stir it was {c}",
)


# --- the edit -------------------------------------------------------------------------------------------------------
class AdditiveEdit:
    """Forward hook on decoder block `layer`: adds alpha_b * ||h|| * u_b at positions [start_b, end_b] of row b.
    Set .batch(u [b, d], alpha [b], start [b], end [b]) before a forward and .clear() after."""

    def __init__(self, model, layer: int):
        self.state = None
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def batch(self, u: torch.Tensor, alpha: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> None:
        self.state = (u, alpha, start, end)

    def clear(self) -> None:
        self.state = None

    def remove(self) -> None:
        self.handle.remove()

    def _hook(self, module, args, output):
        if self.state is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        u, alpha, start, end = (t.to(h.device) for t in self.state)
        pos = torch.arange(h.shape[1], device=h.device)
        inside = (pos[None] >= start[:, None]) & (pos[None] <= end[:, None])                  # [b, T]
        delta = alpha[:, None, None] * h.float().norm(dim=-1, keepdim=True) * u[:, None, :].float()
        h = h + (delta * inside[..., None]).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h


# --- trials ---------------------------------------------------------------------------------------------------------
@dataclass
class Trial:
    item_id: str
    variant: str          # key_swap | answer_swap | random_null | other_item | repair | redirect | repair_null
    method: str           # a METHODS entry, or 'random' for the null
    layer: int
    alpha: float
    c: str | None         # colour written away from (None for the random direction)
    c_new: str | None     # colour written towards
    success: str          # the answer that counts as success
    outcomes: dict        # outcome label -> colour (success, start_swap, echo, unchanged)
    seed: int = 0


def item_rng(item_id: str, purpose: str) -> random.Random:
    return random.Random(f"exp15:{item_id}:{purpose}")


def swap_targets(it: S.Item, colours: list[str], brew) -> dict | None:
    """Per correct h=2 item: s1' and a' for the key and answer swaps, and an unrelated pair for other_item.
    s1' is off the true path with T(s1') and T(T(s1')) also off it and all three distinct, so success, start-swap,
    echo and unchanged are four different colours."""
    rng = item_rng(it.item_id, "swap")
    path = set(it.path)
    keys = [x for x in colours if x not in path and S.step(brew, x) not in path and S.step(brew, x, 2) not in path
            and len({x, S.step(brew, x), S.step(brew, x, 2)}) == 3]
    if not keys:
        return None
    s1_new = rng.choice(keys)
    taken = path | {s1_new, S.step(brew, s1_new), S.step(brew, s1_new, 2)}
    answers = [x for x in colours if x not in path]
    others = [x for x in colours if x not in taken]
    if len(others) < 2:
        return None
    other_c, other_new = rng.sample(others, 2)
    return {"s1_new": s1_new, "a_new": rng.choice(answers), "other": (other_c, other_new)}


def swap_trials(it: S.Item, colours: list[str], brew, grid: list[tuple[str, int, float]], variants) -> list[Trial]:
    """P3 trials on one correct h=2 item, for each (method, layer, alpha) in grid. c = s1, c' = s1'."""
    t = swap_targets(it, colours, brew)
    if t is None:
        return []
    s1, s1_new = it.path[1], t["s1_new"]
    key_out = {"success": S.step(brew, s1_new), "start_swap": S.step(brew, s1_new, 2), "echo": s1_new,
               "unchanged": it.gold}
    trials = []
    for method, layer, alpha in grid:
        if "key_swap" in variants:
            trials.append(Trial(it.item_id, "key_swap", method, layer, alpha, s1, s1_new, key_out["success"], key_out))
        if "answer_swap" in variants:
            trials.append(Trial(it.item_id, "answer_swap", method, layer, alpha, it.gold, t["a_new"], t["a_new"],
                                {"success": t["a_new"], "unchanged": it.gold}))
        if "other_item" in variants:
            c, c_new = t["other"]
            trials.append(Trial(it.item_id, "other_item", method, layer, alpha, c, c_new, key_out["success"], key_out))
    if "random_null" in variants:
        for layer, alpha in sorted({(l, a) for _, l, a in grid}):
            trials.append(Trial(it.item_id, "random_null", "random", layer, alpha, None, None, key_out["success"],
                                key_out, seed=item_rng(it.item_id, f"null:{layer}").randrange(2**31)))
    return trials


def repair_trials(it: S.Item, colours: list[str], brew, grid: list[tuple[str, int, float]]) -> list[Trial]:
    """P4 trials on one off-cycle error: c = w (the model's key) -> c' = the true s_{h-1} (success = gold), the
    redirect control c' = r, a random colour off both paths (success = T(r)), and a random-direction null."""
    rng = item_rng(it.item_id, "repair")
    excluded = set(it.path) | set(it.implied) | {it.forward}
    pool = [x for x in colours if x not in excluded and S.step(brew, x) not in excluded]
    trials = []
    redirect = rng.choice(pool) if pool else None
    for method, layer, alpha in grid:
        trials.append(Trial(it.item_id, "repair", method, layer, alpha, it.key, it.true_prev, it.gold,
                            {"success": it.gold, "unchanged": it.pred}))
        if redirect is not None:
            trials.append(Trial(it.item_id, "redirect", method, layer, alpha, it.key, redirect,
                                S.step(brew, redirect), {"success": S.step(brew, redirect), "unchanged": it.pred}))
    for layer, alpha in sorted({(l, a) for _, l, a in grid}):
        trials.append(Trial(it.item_id, "repair_null", "random", layer, alpha, None, None, it.gold,
                            {"success": it.gold, "unchanged": it.pred},
                            seed=item_rng(it.item_id, f"null:{layer}").randrange(2**31)))
    return trials


def classify(trial: Trial, answer: str) -> str:
    """The first outcome label whose colour is the answer, else 'other'."""
    for label, colour in trial.outcomes.items():
        if colour == answer:
            return label
    return "other"


def edit_vector(trial: Trial, directions: dict, colours: list[str], d: int) -> torch.Tensor:
    """unit(v(c') - v(c)) for the trial's method and layer, or a seeded random unit vector for the null."""
    if trial.method == "random":
        g = torch.Generator().manual_seed(trial.seed)
        v = torch.randn(d, generator=g)
    else:
        table = directions[trial.method][trial.layer]
        v = table[colours.index(trial.c_new)] - table[colours.index(trial.c)]
    return v / v.norm()


def trial_record(trial: Trial, answer: str, p_answer: float) -> dict:
    return {**asdict(trial), "answer": answer, "p_answer": round(p_answer, 4), "outcome": classify(trial, answer)}
