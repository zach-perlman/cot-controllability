"""Scratch: how precise are exp06a's contrasts at 100 vs 50 items? Uses only reused (already graded) rows.

Thinking on : stacked - baseline, S(1000) and S(200), paired question-level bootstrap.
Thinking off: exp04's external_ceiling baseline, S(200) and P1 (no second thinking-off arm exists yet, so this is the
              single-arm SE; a paired contrast's SE is about sqrt(2 * (1 - rho)) times it).
"""
import sys

sys.path.insert(0, "/workspace/cot-controllability/scripts")
import numpy as np

import cc_analysis as ca
import cc_config as cfg
import cc_exp04_analysis as a4
import cc_exp06a
import cc_exp06a_analysis as a6

items = [it for it in cc_exp06a.load_items()[0]]
half = []
for source in sorted({it["source"] for it in items}):
    group = [it for it in items if it["source"] == source]
    half += group[: len(group) // 2]

df = a6.score(a6.reference_rows(a6.MODELS), empty_is_violation=True)
for label, subset in (("100 items", items), ("50 items", half)):
    ids = {it["item_id"] for it in subset}
    boot = ca.Bootstrap(subset, cfg.BOOTSTRAP_ITERS, cfg.BOOTSTRAP_SEED)
    d_all = df[df["item_id"].isin(ids)]
    print(f"\n== {label} ==")
    for model in a6.MODELS:
        out = []
        for t, name in ((a6.T_STAR, "S1000"), (a6.T_SHORT, "S200")):
            s = a4.macro_km(a6.cell(d_all, model, "stacked", True), a6.RULES, t, boot)
            b = a4.macro_km(a6.cell(d_all, model, "baseline", True), a6.RULES, t, boot)
            diff = s[1] - b[1]
            out.append(f"on {name} stacked-baseline {100 * (s[0] - b[0]):5.1f} (SE {100 * diff.std():.1f})")
        off = a6.cell(d_all, model, "baseline", False)
        k = a4.macro_km(off, a6.RULES, a6.T_SHORT, boot)
        p = ca.pooled(off, "P1", boot)
        out.append(f"off S200 {100 * k[0]:5.1f} (SE {100 * k[1].std():.1f})")
        out.append(f"off P1 {100 * p[0]:5.1f} (SE {100 * p[1].std():.1f})")
        print(f"{model:18s} " + " | ".join(out))
