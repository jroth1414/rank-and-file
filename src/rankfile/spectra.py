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


def _svd_basis(M: torch.Tensor) -> torch.Tensor | None:
    """SVD left-singular-vector basis of M, columns above the rank tolerance.
    Returns None if M is zero or entirely below tolerance."""
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    if S.max() == 0:
        return None
    mask = S > 1e-6 * S.max()
    if not mask.any():
        return None
    return U[:, mask]


def subspace_overlap(delta: torch.Tensor, B: torch.Tensor) -> float:
    """Fraction of ‖delta‖_F² captured by projection onto colspace(B) via SVD basis.
    Returns 0.0 if delta is zero, B is zero, or B is rank-deficient beyond tolerance."""
    delta, B = delta.detach().float().cpu(), B.detach().float().cpu()
    delta_norm_sq = (delta * delta).sum()
    if delta_norm_sq == 0:
        return 0.0
    U_basis = _svd_basis(B)
    if U_basis is None:
        return 0.0
    proj = U_basis @ (U_basis.T @ delta)
    return float((proj * proj).sum() / delta_norm_sq)


def subspace_overlap_two_sided(delta: torch.Tensor, B: torch.Tensor, A: torch.Tensor) -> float:
    """Fraction of ‖delta‖_F² captured by projecting onto colspace(B) on the left and
    rowspace(A) (i.e. colspace(Aᵀ)) on the right, each via its own SVD basis.
    Returns 0.0 if delta is zero, or either B or A is zero/rank-deficient beyond tolerance."""
    delta, B, A = delta.detach().float().cpu(), B.detach().float().cpu(), A.detach().float().cpu()
    delta_norm_sq = (delta * delta).sum()
    if delta_norm_sq == 0:
        return 0.0
    U_basis = _svd_basis(B)
    if U_basis is None:
        return 0.0
    V_basis = _svd_basis(A.T)
    if V_basis is None:
        return 0.0
    proj = U_basis @ (U_basis.T @ delta @ V_basis) @ V_basis.T
    return float((proj * proj).sum() / delta_norm_sq)


def matrix_report(
    name: str, W: torch.Tensor, ranks: tuple[int, ...] = (4, 16, 64)
) -> dict[str, float | int | str]:
    m = _NAME.search(name)
    layer, module = (int(m.group(1)), f"{m.group(2)}.{m.group(3)}") if m else (-1, name)
    s = singular_values(W)
    erank = effective_rank(s)
    rows, cols = W.shape[0], W.shape[1]
    out: dict[str, float | int | str] = {
        "name": name, "layer": layer, "module": module, "rows": rows, "cols": cols,
        "erank": erank, "erank_norm": erank / min(rows, cols), "srank": stable_rank(s),
        "fro": float(s.norm()),
    }
    for r in ranks:
        out[f"top{r}"] = top_r_energy(s, r)
    return out
