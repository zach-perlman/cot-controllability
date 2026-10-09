"""exp11 test figures (no item text): the twin write-in curves per layer, from cache/exp11/q23_twinpatch_*.pt.

  fig_twinpatch.png  for each forward twin patch, the fraction of recipients answering
                       - the donor's answer under the treatment (the recipient's own twin as donor),
                       - the recipient's own twin-answer colour under the control (another item's twin as donor),
                       - the control donor's answer under the control,
                     with the unpatched rate of the recipient's own twin-answer colour as a dashed line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import exp11 as X

PANELS = [("test", "three_first", "final", "three columns, final position\n(h=2 <- its h=1 twin)", "s1"),
          ("single_test", "single_first", "final", "single table, final position\n(h=3 <- its h=2 twin)", "s2"),
          ("single_test", "single_first", "count", "single table, count word\n(h=3 <- its h=2 twin)", "s2")]


def rates(blob: dict) -> dict[str, list[float]]:
    colours, res = blob["colours"], blob["results"]
    idx = lambda key: torch.tensor([colours.index(r[key]) for r in res])
    twin_answer, ctrl_answer = idx("donor_gold"), idx("control_donor_gold")
    treat = torch.stack([r["treatment"] for r in res]).argmax(-1)       # [items, layers]
    ctrl = torch.stack([r["control"] for r in res]).argmax(-1)
    base = torch.stack([r["unpatched"] for r in res]).argmax(-1)
    frac = lambda a, b: (a == b[:, None]).float().mean(0).tolist()
    return {"treatment": frac(treat, twin_answer), "control_own": frac(ctrl, twin_answer),
            "control_donor": frac(ctrl, ctrl_answer), "unpatched": float((base == twin_answer).float().mean())}


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 3.6), sharey=True)
    for ax, (bank, fmt, site, title, state) in zip(axes, PANELS):
        blob = torch.load(X.EXP.cache / f"q23_twinpatch_{bank}_{fmt}_{site}.pt", weights_only=False)
        r, layers = rates(blob), blob["layers"]
        ax.plot(layers, r["treatment"], "o-", c="C0", label=f"own twin donor: answers twin's answer ({state})")
        ax.plot(layers, r["control_own"], "s-", c="C1", label=f"other item's twin: answers OWN {state}")
        ax.plot(layers, r["control_donor"], "^-", c="C2", label="other item's twin: answers that donor's answer")
        ax.axhline(r["unpatched"], ls="--", c="grey", lw=1, label=f"unpatched: answers own {state}")
        ax.set_title(f"{title}, n={len(blob['results'])}", fontsize=9)
        ax.set_xlabel("patched layer (block output)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of recipients (argmax)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Qwen3.6-27B, no-CoT brew: a twin's state transplanted into its parent", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_twinpatch.png", dpi=150)
    print("wrote", out / "fig_twinpatch.png")


if __name__ == "__main__":
    main(sys.argv[1])
