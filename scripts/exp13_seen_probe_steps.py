"""exp13 'already seen' (exploratory, run before the exp13 manifest): does exp11's brew colour probe on Qwen3.6-27B
gain accuracy at full-attention blocks?

Reads exp11's saved per-layer probe results (held-out correctness per item and layer; layer l = output of block l).
Qwen3.6-27B blocks 3, 7, 11, ... (l % 4 == 3) are full softmax attention; the rest are Gated DeltaNet.
For each (h, position, label) it prints the per-layer accuracy and the gain from layer l-1 to l (the gain block l
contributes), and compares the mean gain at full-attention blocks with the mean gain at linear blocks in blocks 20-56.
Run from scripts/.
"""
import torch

import exp11 as X

blob = torch.load(X.EXP.cache / "q23_probe_test_k64_wd0.001.pt", weights_only=False)


def is_full_attention(block: int) -> bool:
    return block % 4 == 3


for (h, pos, label), r in sorted(blob["results"].items()):
    acc = r["correct"].float().mean(0)                          # [63]: accuracy per layer (output of block l)
    gains = acc[1:] - acc[:-1]                                  # gain contributed by block l (l = 1..62)
    blocks = torch.arange(1, len(acc))
    full = torch.tensor([is_full_attention(int(b)) for b in blocks])
    band = (blocks >= 20) & (blocks <= 56)
    print(f"\n## h={h} position={pos} label={label} n={r['correct'].shape[0]}")
    print("  layer: acc (gain)   [F = full-attention block]")
    for b in range(24, 57):
        print(f"  {b:>2}{'F' if is_full_attention(b) else ' '} {acc[b]:.2f} ({acc[b] - acc[b - 1]:+.2f})")
    g_full, g_lin = gains[band & full], gains[band & ~full]
    share = g_full.clamp(min=0).sum() / gains[band].clamp(min=0).sum()
    print(f"  blocks 20..56: mean gain at full-attention blocks {g_full.mean():+.3f} (n={len(g_full)}), at linear "
          f"blocks {g_lin.mean():+.3f} (n={len(g_lin)}); share of total positive gain at full blocks {share:.2f} "
          f"(full blocks are {len(g_full) / band.sum():.2f} of the band)")
