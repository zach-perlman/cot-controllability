"""exp11 colour probes: per layer, a 10-way L2-regularised logistic regression from the residual at one position to
a colour label, cross-validated; all layers are fitted at once (independent problems summed into one loss).

Features per fold: centred on the training mean; optionally projected onto the top `k_pca` principal components of
the training fold; then each feature divided by its training standard deviation.
"""
from __future__ import annotations

import random

import torch


def stratified_folds(groups: list[str], n_folds: int = 5, seed: int = 0) -> list[int]:
    """A fold index per item; within each group, items are shuffled and dealt round-robin."""
    rng = random.Random(seed)
    fold = [0] * len(groups)
    for g in sorted(set(groups)):
        members = [i for i, x in enumerate(groups) if x == g]
        rng.shuffle(members)
        for k, i in enumerate(members):
            fold[i] = k % n_folds
    return fold


def _features(x_train: torch.Tensor, x_test: torch.Tensor, k_pca: int | None):
    """x [L, n, d] -> standardised features [L, n, k]."""
    mean = x_train.mean(1, keepdim=True)
    x_train, x_test = x_train - mean, x_test - mean
    if k_pca is not None:
        _, _, vh = torch.linalg.svd(x_train, full_matrices=False)      # vh [L, min(n, d), d]
        basis = vh[:, :k_pca].transpose(1, 2)                          # [L, d, k]
        x_train, x_test = x_train @ basis, x_test @ basis
    std = x_train.std(1, keepdim=True).clamp_min(1e-6)
    return x_train / std, x_test / std


def _fit(z: torch.Tensor, y: torch.Tensor, n_classes: int, weight_decay: float, max_iter: int = 200):
    """Multinomial logistic regression per layer: z [L, n, k], y [n] -> (W [L, k, C], b [L, C])."""
    n_layers, _, k = z.shape
    w = torch.zeros(n_layers, k, n_classes, device=z.device, requires_grad=True)
    b = torch.zeros(n_layers, 1, n_classes, device=z.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")
    target = y.expand(n_layers, -1)

    def closure():
        opt.zero_grad()
        logits = z @ w + b
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, reduction="none").mean(1).sum()
        loss = loss + weight_decay * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def cv_correct(x: torch.Tensor, y: list[int], folds: list[int], n_classes: int = 10, k_pca: int | None = 64,
               weight_decay: float = 1e-2) -> torch.Tensor:
    """x [n, L, d] (any float dtype), y [n] class indices -> held-out correctness [n, L] (bool, CPU)."""
    x = x.to("cuda", torch.float32).transpose(0, 1)                    # [L, n, d]
    y_t = torch.tensor(y, device="cuda")
    folds_t = torch.tensor(folds, device="cuda")
    correct = torch.zeros(x.shape[1], x.shape[0], dtype=torch.bool)
    for f in sorted(set(folds)):
        train, test = folds_t != f, folds_t == f
        z_train, z_test = _features(x[:, train], x[:, test], k_pca)
        w, b = _fit(z_train, y_t[train], n_classes, weight_decay)
        pred = (z_test @ w + b).argmax(-1)                             # [L, n_test]
        correct[test.cpu()] = (pred == y_t[test]).T.cpu()
    return correct
