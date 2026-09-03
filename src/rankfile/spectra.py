"""Pure spectral measurements: effective rank, stable rank, top-r energy, LoRA subspace overlap."""
from __future__ import annotations

import re

import torch

_NAME = re.compile(r"blocks\.(\d+)\.(attn|mlp)\.(\w+)(?:\.weight)?$")


def singular_values(W: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(W.detach().float().cpu())


def effective_rank(s: torch.Tensor) -> float:
    """Entropy of normalized spectrum. Returns 0.0 if no positive singular value exists."""
    s = s[s > 0]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def stable_rank(s: torch.Tensor) -> float:
    """Frobenius/spectral norm squared ratio. Returns 0.0 if s[0] == 0."""
    if len(s) == 0 or s[0] == 0:
        return 0.0
    return float((s * s).sum() / (s[0] * s[0]))


def top_r_energy(s: torch.Tensor, r: int) -> float:
    e = s * s
    return float(e[:r].sum() / e.sum())


def subspace_overlap(delta: torch.Tensor, B: torch.Tensor) -> float:
    """Fraction of ‖delta‖_F² captured by projection onto colspace(B) via SVD basis.
    Returns 0.0 if delta is zero, B is zero, or B is rank-deficient beyond tolerance."""
    delta, B = delta.detach().float().cpu(), B.detach().float().cpu()
    delta_norm_sq = (delta * delta).sum()
    if delta_norm_sq == 0:
        return 0.0
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    if S.max() == 0:
        return 0.0
    mask = S > 1e-6 * S.max()
    if not mask.any():
        return 0.0
    U_basis = U[:, mask]
    proj = U_basis @ (U_basis.T @ delta)
    return float((proj * proj).sum() / delta_norm_sq)


def matrix_report(name: str, W: torch.Tensor, ranks: tuple[int, ...] = (4, 16, 64)) -> dict:
    m = _NAME.search(name)
    layer, module = (int(m.group(1)), f"{m.group(2)}.{m.group(3)}") if m else (-1, name)
    s = singular_values(W)
    out = {"name": name, "layer": layer, "module": module, "rows": W.shape[0], "cols": W.shape[1],
           "erank": effective_rank(s), "srank": stable_rank(s), "fro": float(s.norm())}
    for r in ranks:
        out[f"top{r}"] = top_r_energy(s, r)
    return out
