"""exp17 core (no model loading): the J++ dictionary, non-negative gradient pursuit, and the clamped band
coordinate swap.

Coordinate swap (the global-workspace paper, section 2.5, clamped as in its experiments). For a pair of vectors
V = [v_a; v_b] at layer l, the pair coordinates of a residual h are c(h) = V^+ h (the pseudoinverse, so a correlated
pair is handled). With c0 = c(h_clean) read from the unedited forward of the same batch, the edit at every layer of
a band and every position of a window sets the coordinates to
    target = c0 + alpha * (sigma(c0) - c0)          sigma swaps the two entries; alpha = 1 exchanges them
    h <- h + (target - c(h)) V
Clamping to targets taken from the clean pass, rather than swapping the current coordinates again, keeps a swap made
at layer l from being swapped back at layer l + 1.

Edit kinds (every row of a batch has the same kind):
  swap      the main pair's edit
  matched   the main pair's edit rescaled, per layer and position, to the norm the reference arm's edit had there in
            its own forward (the J-part and remainder arms are matched to the full direction's swap)
  null      a fixed random unit direction per row and layer, at the reference arm's edit norms
and, for any kind, an optional clamp pair whose coordinates are then held at their clean values (the paper's control:
does a remainder swap act through the J++ coordinates of the two colours?).
The reference norms are recorded from the reference arm's forward, not recomputed: a clamped swap edits fully at the
band's first layer and only corrects drift after it, while a swap recomputed from an unswapped state is full-size at
every layer.

  python exp17_swaps.py selftest
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial

import torch

EPS = 1e-6


# --- the J++ dictionary ---------------------------------------------------------------------------------------------
def readable_token_ids(tokenizer) -> list[int]:
    """Vocabulary rows that can enter the dictionary: tokens whose text has a letter or a digit (J++'s readout
    filter), special tokens excluded."""
    special = set(tokenizer.all_special_ids)
    ids = []
    for i in range(len(tokenizer)):
        if i in special:
            continue
        text = tokenizer.decode([i])
        if any(ch.isalnum() for ch in text):
            ids.append(i)
    return ids


@torch.no_grad()
def lens_atoms(jacobian: torch.Tensor, gain: torch.Tensor, w_u: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Unit J++ vectors for the rows of w_u: v_t = J^T (gain * W_U[t]), returned as rows [n_tokens, d] (bf16).
    Computed in fp32 on the device of `w_u`."""
    J = jacobian.to(w_u.device, torch.float32)
    g = gain.to(w_u.device, torch.float32)
    out = torch.empty(w_u.shape[0], J.shape[1], dtype=torch.bfloat16, device=w_u.device)
    for s in range(0, w_u.shape[0], chunk):
        rows = (w_u[s:s + chunk].float() * g) @ J
        out[s:s + chunk] = torch.nn.functional.normalize(rows, dim=-1).to(torch.bfloat16)
    return out


def project_unit(rows: torch.Tensor, basis: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Unit rows of the projections of rows [n, d] onto span(basis) (orthonormal rows [r, d]), in rows' dtype and
    device."""
    B = basis.to(rows.device, torch.float32)
    out = torch.empty_like(rows)
    for s in range(0, rows.shape[0], chunk):
        x = rows[s:s + chunk].float()
        out[s:s + chunk] = torch.nn.functional.normalize((x @ B.T) @ B, dim=-1).to(rows.dtype)
    return out


@dataclass
class Decomposition:
    support: list[int]          # dictionary row indices, in the order they were selected
    coef: torch.Tensor          # non-negative coefficients [len(support)]
    j_part: torch.Tensor        # coef @ atoms[support]  [d]
    explained: float            # 1 - |x - j_part|^2 / |x|^2


@torch.no_grad()
def nn_gradient_pursuit(x: torch.Tensor, atoms: torch.Tensor, k: int, inner: int = 25) -> Decomposition:
    """Non-negative sparse approximation of x by at most k unit rows of `atoms`: at each step the atom most correlated
    with the residual joins the support, then projected gradient steps with exact line search refit the
    coefficients on the support (coefficients kept >= 0)."""
    x = x.to(atoms.device, torch.float32)
    r = x.clone()
    support: list[int] = []
    coef = x.new_zeros(0)
    for _ in range(k):
        corr = (atoms @ r.to(atoms.dtype)).float()
        if support:
            corr[support] = -float("inf")
        j = int(corr.argmax())
        if corr[j] <= 0:
            break
        support.append(j)
        coef = torch.cat([coef, coef.new_zeros(1)])
        A = atoms[support].float()                                   # [s, d]
        for _ in range(inner):
            g = A @ r                                                # minus the gradient of |r|^2 / 2 in coef
            g = torch.where((coef <= 0) & (g < 0), torch.zeros_like(g), g)
            step = g @ A
            denom = step @ step
            if denom <= 1e-20:
                break
            coef = (coef + (g @ g) / denom * g).clamp_min(0)
            r = x - coef @ A
    j_part = coef @ atoms[support].float() if support else torch.zeros_like(x)
    return Decomposition(support, coef.cpu(), j_part.cpu(), float(1 - (x - j_part).pow(2).sum() / x.pow(2).sum()))


# --- the coordinate swap --------------------------------------------------------------------------------------------
def pair_coords(V: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """V [b, 2, d], h [b, W, d] -> c [b, W, 2] with h ~ c @ V (least squares: c = h V^T (V V^T)^-1)."""
    gram = V @ V.transpose(1, 2)                                     # [b, 2, 2]
    return torch.linalg.solve(gram, V @ h.transpose(1, 2)).transpose(1, 2)


def swap_edit(V: torch.Tensor, h_clean: torch.Tensor, h_now: torch.Tensor, alpha: float) -> torch.Tensor:
    """The edit setting the pair coordinates of h_now to c0 + alpha (sigma(c0) - c0), c0 those of h_clean."""
    c0 = pair_coords(V, h_clean)
    target = c0 + alpha * (c0.flip(-1) - c0)
    return (target - pair_coords(V, h_now)) @ V


def clamp_edit(V: torch.Tensor, h_clean: torch.Tensor, h_now: torch.Tensor) -> torch.Tensor:
    """The edit holding the pair coordinates of h_now at those of h_clean."""
    return (pair_coords(V, h_clean) - pair_coords(V, h_now)) @ V


def patch_edit(h_source: torch.Tensor, h_now: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    """The edit setting h_now's coordinates in span(basis) ([b, r, d] orthonormal rows; None: all of h) to
    h_source's (interchange patching from a counterfactual forward)."""
    diff = h_source - h_now
    return diff if basis is None else (diff @ basis.transpose(1, 2)) @ basis


def orthonormal_rows(rows: torch.Tensor) -> torch.Tensor:
    """An orthonormal basis [r, d] of the span of rows [r, d] (full rank)."""
    return torch.linalg.qr(rows.float().T)[0].T


@dataclass
class EditSpec:
    """One arm's edit for a batch. Pairs are per layer, [b, 2, d] (row 0 the colour written away from, row 1 the
    colour written towards); noise is per layer [b, d] (unit rows); ref_norms per layer [b, W] (the reference arm's
    edit norms, BandEdit.norms after its forward); basis per layer [b, r, d] (patch: the patched subspace; None: the
    whole residual)."""
    kind: str                                   # swap | matched | null | patch
    alpha: float
    layers: tuple[int, ...]                     # the band
    main: dict[int, torch.Tensor] | None = None
    ref_norms: dict[int, torch.Tensor] | None = None
    noise: dict[int, torch.Tensor] | None = None
    clamp: dict[int, torch.Tensor] | None = None
    basis: dict[int, torch.Tensor] | None = None


def edit_delta(spec: EditSpec, layer: int, h_clean: torch.Tensor, h_now: torch.Tensor) -> torch.Tensor:
    """The fp32 edit [b, W, d] at one layer of the band. h_clean: the recorded pass the edit starts from (for a
    patch, the counterfactual forward)."""
    if spec.kind == "patch":
        return patch_edit(h_clean, h_now, None if spec.basis is None else spec.basis[layer])
    if spec.kind == "swap":
        delta = swap_edit(spec.main[layer], h_clean, h_now, spec.alpha)
    else:
        size = spec.ref_norms[layer][..., None]
        if spec.kind == "matched":
            own = swap_edit(spec.main[layer], h_clean, h_now, spec.alpha)
            delta = own * (size / own.norm(dim=-1, keepdim=True).clamp_min(EPS))
        elif spec.kind == "null":
            delta = spec.noise[layer][:, None, :] * size
        else:
            raise ValueError(spec.kind)
    if spec.clamp is not None:
        delta = delta + clamp_edit(spec.clamp[layer], h_clean, h_now + delta)
    return delta


class BandEdit:
    """Forward hooks on blocks[l] for every l in `layers` (the union of all bands used). Per batch:
      .window(idx, mask)  the edited positions of each row: idx [b, W] token positions, mask [b, W] (padding False)
      .record()           the next forward stores each hooked block's output at the window (the clean pass)
      .record("counterfactual")  the same for a counterfactual prompt of the same length (the patch arms' source)
      .apply(spec)        the next forward edits the window at spec.layers, from the stored clean outputs (patch:
                          the stored counterfactual outputs)
      .clear()            the next forward runs untouched
    After an edited forward, .dose holds per row the mean over band layers and window positions of |delta| / |h|, and
    .norms per band layer the edit norms [b, W] (zero at padding): a later arm's ref_norms."""

    def __init__(self, blocks, layers):
        self.layers = tuple(sorted(layers))
        self.handles = [blocks[l].register_forward_hook(partial(self._hook, l)) for l in self.layers]
        self.mode, self.spec, self.dose, self.norms = None, None, None, {}
        self.recorded = {"clean": {}, "counterfactual": {}}
        self.recording = "clean"

    @property
    def clean(self) -> dict:
        return self.recorded["clean"]

    def window(self, idx: torch.Tensor, mask: torch.Tensor) -> None:
        self.idx, self.mask = idx, mask
        self.recorded = {"clean": {}, "counterfactual": {}}

    def record(self, source: str = "clean") -> None:
        self.mode, self.recording = "record", source

    def source(self, spec: EditSpec) -> dict:
        return self.recorded["counterfactual" if spec.kind == "patch" else "clean"]

    def apply(self, spec: EditSpec) -> None:
        missing = [l for l in spec.layers if l not in self.source(spec)]
        if missing:
            raise RuntimeError(f"no {'counterfactual' if spec.kind == 'patch' else 'clean'} pass stored at layers "
                               f"{missing}")
        self.mode, self.spec = "edit", spec
        self.dose = torch.zeros(self.idx.shape[0], device=self.idx.device)
        self.norms = {}

    def clear(self) -> None:
        self.mode, self.spec = None, None

    def remove(self) -> None:
        for h in self.handles:
            h.remove()

    def _gather(self, h: torch.Tensor) -> torch.Tensor:
        return h.gather(1, self.idx.to(h.device)[..., None].expand(-1, -1, h.shape[-1]))

    def _hook(self, layer, module, args, output):
        if self.mode is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if self.mode == "record":
            self.recorded[self.recording][layer] = self._gather(h).detach()
            return output
        if layer not in self.spec.layers:
            return output
        mask = self.mask.to(h.device)
        h_now = self._gather(h).float()
        delta = edit_delta(self.spec, layer, self.source(self.spec)[layer].float(), h_now) * mask[..., None]
        self.norms[layer] = delta.norm(dim=-1)
        ratio = self.norms[layer] / h_now.norm(dim=-1).clamp_min(EPS)
        self.dose += (ratio * mask).sum(1) / mask.sum(1).clamp_min(1) / len(self.spec.layers)
        idx = self.idx.to(h.device)[..., None].expand(-1, -1, h.shape[-1])
        h = h.scatter_add(1, idx, delta.to(h.dtype))
        return (h, *output[1:]) if isinstance(output, tuple) else h


def window_index(starts: list[int], ends: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions start..end (inclusive) per row, right-padded with the row's end (masked out): idx, mask [b, W]."""
    width = max(e - s + 1 for s, e in zip(starts, ends))
    idx = torch.tensor([[min(s + k, e) for k in range(width)] for s, e in zip(starts, ends)])
    mask = torch.tensor([[s + k <= e for k in range(width)] for s, e in zip(starts, ends)])
    return idx, mask


def pair(vectors: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """Rows a and b of vectors [n, d] as one pair [2, d]."""
    return torch.stack([vectors[a], vectors[b]])


# --- self test ------------------------------------------------------------------------------------------------------
class _Block(torch.nn.Module):
    def __init__(self, d: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(d, d, generator=g) * 0.05)

    def forward(self, x):
        return (x + torch.tanh(x @ self.w),)


def selftest() -> None:
    torch.manual_seed(0)
    d, b, W = 64, 3, 5
    # (1) a swap exchanges the pair coordinates exactly and leaves the part outside the pair's span unchanged
    va, vb = torch.randn(d), torch.randn(d)
    vb = 0.6 * va + vb                                              # a correlated pair
    V = torch.stack([va, vb])[None].expand(b, -1, -1)
    h = torch.randn(b, W, d)
    c = pair_coords(V, h)
    edited = h + swap_edit(V, h, h, 1.0)
    assert torch.allclose(pair_coords(V, edited), c.flip(-1), atol=1e-4)
    Q, _ = torch.linalg.qr(torch.stack([va, vb]).T)
    outside = lambda x: x - (x @ Q) @ Q.T                           # noqa: E731
    assert torch.allclose(outside(edited), outside(h), atol=1e-4)
    # (2) clamping: once the coordinates are at their target, a later layer adds nothing (no swap back)
    assert swap_edit(V, h, edited, 1.0).abs().max() < 1e-4
    assert clamp_edit(V, h, h).abs().max() < 1e-5
    # (3) matched and null edits have the reference norms at every position
    U = torch.randn(b, 2, d)
    ref = swap_edit(V, h, h, 1.0).norm(dim=-1)
    for kind in ("matched", "null"):
        spec = EditSpec(kind, 1.0, (0,), main={0: U}, ref_norms={0: ref},
                        noise={0: torch.nn.functional.normalize(torch.randn(b, d), dim=-1)})
        assert torch.allclose(edit_delta(spec, 0, h, h).norm(dim=-1), ref, atol=1e-3), kind
    # (4) pursuit recovers a planted non-negative combination
    atoms = torch.nn.functional.normalize(torch.randn(500, d), dim=-1)
    x = 0.8 * atoms[3] + 0.4 * atoms[77] + 0.2 * atoms[401]
    dec = nn_gradient_pursuit(x, atoms, k=5)
    assert {3, 77, 401} <= set(dec.support[:3]) and dec.explained > 0.999 and (dec.coef >= 0).all(), dec
    # (5) the hook: alpha 0 changes nothing; an edit changes only window positions from its first band layer on
    blocks = torch.nn.ModuleList([_Block(d, s) for s in range(4)])

    def forward(x):
        for blk in blocks:
            x = blk(x)[0]
        return x

    T = 9
    x = torch.randn(b, T, d)
    edit = BandEdit(blocks, layers=(1, 2))
    idx, mask = window_index([4, 5, 6], [8, 8, 7])
    edit.window(idx, mask)
    edit.record()
    clean = forward(x)
    shared = torch.randn(b, 2, d)
    pairs = {l: shared for l in (1, 2)}                              # one pair at both layers, as a slow-varying lens
    edit.apply(EditSpec("swap", 0.0, (1, 2), main=pairs))
    assert torch.allclose(forward(x), clean, atol=1e-5)
    edit.apply(EditSpec("swap", 1.0, (1, 2), main=pairs))
    out = forward(x)
    edit.clear()
    changed = (out - clean).abs().amax(-1) > 1e-6                    # [b, T]
    assert not changed[0, :4].any() and not changed[1, :5].any() and not changed[2, :6].any()
    assert changed[0, 4:].all() and not changed[2, 8], changed
    assert (edit.dose > 0).all()
    # (6) a null and a matched arm given the swap arm's recorded norms edit with the same norms at every band layer
    # and position, in their own forwards (where a recomputed swap size would be full-size at every layer)
    swap_norms, swap_dose = dict(edit.norms), edit.dose.clone()
    assert swap_norms[2].sum() < swap_norms[1].sum()                 # the clamped swap only corrects at layer 2
    noise = {l: torch.nn.functional.normalize(torch.randn(b, d), dim=-1) for l in (1, 2)}
    for kind in ("null", "matched"):
        other = {l: torch.randn(b, 2, d) for l in (1, 2)}
        edit.apply(EditSpec(kind, 1.0, (1, 2), main=other, ref_norms=swap_norms, noise=noise))
        forward(x)
        edit.clear()
        for l in (1, 2):
            assert torch.allclose(edit.norms[l], swap_norms[l], atol=1e-4), (kind, l)
        assert torch.allclose(edit.dose, swap_dose, rtol=0.05), (kind, edit.dose, swap_dose)
    edit.remove()

    # (7) projecting onto a subspace keeps a vector already inside it and drops the part outside it
    basis = torch.linalg.qr(torch.randn(d, 8))[0].T                   # [8, d] orthonormal rows
    inside = torch.nn.functional.normalize(torch.randn(2, 8) @ basis, dim=-1)
    outside = torch.randn(2, d)
    outside = outside - outside @ basis.T @ basis                     # orthogonal to the subspace
    mixed = inside + 5 * outside
    assert torch.allclose(project_unit(inside, basis), inside, atol=1e-5)
    assert torch.allclose(project_unit(mixed, basis), inside, atol=1e-5)

    # (8) interchange patching (the toy blocks act per position): patching all of h at every band layer gives the
    # counterfactual forward's outputs inside the window and the clean ones outside it; patching a subspace at one
    # layer moves exactly the coordinates in that subspace to the counterfactual's
    x_cf = x + torch.randn_like(x)
    edit = BandEdit(blocks, layers=(1, 2))
    edit.window(idx, mask)
    edit.record()
    clean = forward(x)
    edit.record("counterfactual")
    cf = forward(x_cf)
    edit.apply(EditSpec("patch", 1.0, (1, 2)))
    patched = forward(x)
    inside_w = torch.zeros(b, T, dtype=torch.bool)
    for r in range(b):
        inside_w[r, idx[r][mask[r]]] = True
    assert torch.allclose(patched[inside_w], cf[inside_w], atol=1e-5)
    assert torch.allclose(patched[~inside_w], clean[~inside_w], atol=1e-5)
    edit.remove()

    def through_block1(z):
        return blocks[1](blocks[0](z)[0])[0]

    clean1, cf1 = through_block1(x), through_block1(x_cf)
    Q = orthonormal_rows(torch.randn(6, d))
    edit = BandEdit(blocks, layers=(1,))
    edit.window(idx, mask)
    edit.record()
    through_block1(x)
    edit.record("counterfactual")
    through_block1(x_cf)
    edit.apply(EditSpec("patch", 1.0, (1,), basis={1: Q.expand(b, -1, -1)}))
    patched1 = through_block1(x)
    expected = clean1 + ((cf1 - clean1) @ Q.T @ Q) * inside_w[..., None]
    assert torch.allclose(patched1, expected, atol=1e-5)
    edit.remove()
    print("exp17_swaps selftest: ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest"])
    p.parse_args()
    selftest()


if __name__ == "__main__":
    main()
