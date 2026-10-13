"""exp17 interchange patching: which subspace carries the information the brew lookup uses?

The swap arms write a chosen direction at a chosen dose. Patching needs neither. Per correct h=2 item x (start s0,
key s1 = T(s0), answer T(s1)), the counterfactual x' is the same prompt with the start colour replaced by
s0' = T^-1(s1'), where s1' is the item's key-swap target (exp15_edits.swap_targets): x' looks up s1' and answers
T(s1'), the swap arms' 'success' colour. One token differs; every other token, and the length, are the same. At
every J++ layer of a band and every window position, the coordinates of x's residual in a subspace are set to x''s:
  patch_full|band      the whole residual (from the band on, on the window, x' itself: the upper bound)
  patch_dm|band        span of the 10 difference-of-means key directions (dm_full)
  patch_dm_jpart|band  span of their J++ parts       patch_dm_rem|band  span of their remainders
  patch_jpp|band       span of the 10 J++ colour vectors
  patch_random|band    a random 10-dimensional subspace (inside the occupied subspace under EXP17_LENS=occupied)
On the in-task control band (later layers, where the J++ answer swap works), only full, jpp and random: the dm
families exist at dev-band layers only. Two windows:
  start   start colour .. last token (the swap arms' window)
  after   the token after the start colour .. last token: the start colour's own position, whose residual names s0'
          rather than s1', is left alone
Success: the answer is T(s1'); analysed on items whose counterfactual is answered correctly (arm cf_clean).
  dev    the dev bands and the first control band, on dev's correct h=2 items -> trials_patch_<window>.jsonl
  test   the bands and windows of the run's locked 'patch' setting (a locked manifest addendum), on the test bank
         -> trials_test_patch_<window>.jsonl

  EXP17_MODEL=qwen|gemma [EXP17_LENS=occupied] python exp17_patch.py dev|test
"""
from __future__ import annotations

import argparse
import dataclasses

import torch

import exp11_brew as B
import exp15 as E15
import exp15_score as S
import exp17 as E

WINDOWS = ("start", "after")
SPANS = {"full": None, "dm": "dm_full", "dm_jpart": "dm_jpart", "dm_rem": "dm_rem", "jpp": "jpp",
         "random": "random"}
CONTROL_SPANS = ("full", "jpp", "random")


def counterfactual_ids(tokenizer, text: str, ids: list[int], start: int, brew, s0_new: str, cid: dict) -> list[int]:
    """ids with the start colour's token replaced by s0_new's, checked against the tokenization of the prompt text
    with the final problem's start sentence rewritten."""
    assert ids[start] == cid[brew.start], (ids[start], brew.start)
    new = list(ids)
    new[start] = cid[s0_new]
    old_sentence = f"starts out {brew.start}."
    at = text.index(old_sentence, text.rindex("Problem: "))
    new_text = text[:at] + f"starts out {s0_new}." + text[at + len(old_sentence):]
    enc = tokenizer.encode(new_text, add_special_tokens=False)
    if enc != new[:len(enc)]:
        raise RuntimeError(f"start colour {brew.start} -> {s0_new}: the substituted ids are not the rewritten text's")
    return new


def counterfactual_targets(bank: str) -> tuple[list[E.Target], list[str]]:
    """exp17.brew_targets with cf_ids: the prompt whose start colour is T^-1(s1')."""
    tokenizer = E.load_tokenizer()
    targets, colours = E.brew_targets(bank)
    items = E.bank_items(bank)
    _, prepared = E.prepared_rows(bank)
    row_of = {row["item_id"]: row for row, _, _ in prepared}
    cid = dict(zip(colours, E15.colour_ids(tokenizer, colours)))
    for t in targets:
        brew = B.parse(items[t.item_id]["problem"])
        inverse = {S.step(brew, c): c for c in colours}
        assert len(inverse) == len(colours), f"{t.item_id}: the table is not a permutation"
        s1_new = t.outcomes["key"]["echo"]
        text, ids = E15.prompt(tokenizer, row_of[t.item_id])
        assert ids == t.ids
        t.cf_ids = counterfactual_ids(tokenizer, text, ids, t.start, brew, inverse[s1_new], cid)
    return targets, colours


def arms(bands: list[str]) -> list[E.Arm]:
    out = []
    for band in bands:
        spans = CONTROL_SPANS if band in E.SPEC.control_bands else tuple(SPANS)
        out += [E.Arm(f"patch_{span}|{band}", "patch", SPANS[span], None, band, 1.0) for span in spans]
    return out


@torch.no_grad()
def run(bank: str, bands: list[str], windows) -> None:
    targets, colours = counterfactual_targets(bank)
    jac = E.jpp_jacobians()
    control = [l for b in bands if b in E.SPEC.control_bands for l in E.band_layers(b, jac)]
    families = E.families_on_cuda(extra_jpp_layers=control)
    score = E.brew_scorer(colours)
    prefix = "patch" if bank == "dev" else "test_patch"
    print(f"{prefix}: bands {[(b, E.band_layers(b, jac)) for b in bands]}, windows {list(windows)}", flush=True)
    for window in windows:
        shift = {"start": 0, "after": 1}[window]
        shifted = [dataclasses.replace(t, start=t.start + shift) for t in targets]
        E.run_arms(f"{prefix}_{window}", shifted, arms(bands), families, jac, score)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bank", choices=["dev", "test"])
    bank = p.parse_args().bank
    if bank == "dev":
        multi_layer = [b for b, (lo, hi) in E.SPEC.bands.items() if lo < hi]
        run("dev", multi_layer + [next(iter(E.SPEC.control_bands))], WINDOWS)
    else:
        lock = E.locked_settings()[E.RUN_KEY]["patch"]
        run("test", lock["bands"], lock["windows"])


if __name__ == "__main__":
    main()
