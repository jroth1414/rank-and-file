"""Muon (Jordan et al. 2024) with the scaling and weight-decay fixes of Liu et al. 2025.

Muon replaces each 2-D weight's momentum-smoothed gradient with an approximate
orthogonalization computed by five quintic Newton–Schulz iterations, then scales
the update so its RMS matches what AdamW would produce (0.2 * sqrt(max(m, n))).
"""
from __future__ import annotations

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = _NS_COEFFS
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)  # spectral norm <= 1 so the iteration converges
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)
