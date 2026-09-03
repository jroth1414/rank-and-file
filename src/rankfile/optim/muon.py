"""Muon (Jordan et al. 2024) with the scaling and weight-decay fixes of Liu et al. 2025.

Muon replaces each 2-D weight's momentum-smoothed gradient with an approximate
orthogonalization computed by five quintic Newton–Schulz iterations, then scales
the update so its RMS matches what AdamW would produce (0.2 * sqrt(max(m, n))).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Return an approximation of U @ Vh from the SVD of G (i.e. an orthogonalized G),
    computed by `steps` quintic Newton-Schulz iterations instead of an explicit SVD.

    Dividing by the Frobenius norm before iterating guarantees every singular value of
    the input is <= 1, which is what makes the iteration converge. The quintic
    coefficients trade exactness for speed: singular values land in roughly [0.7, 1.3]
    rather than converging exactly to 1.
    """
    if G.ndim != 2:
        raise ValueError(
            f"zeropower_via_newtonschulz5 expects a 2-D tensor, got shape {tuple(G.shape)}"
        )
    a, b, c = _NS_COEFFS
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    # Out-of-place division: X may alias the caller's momentum buffer (nesterov=False),
    # so normalizing in place would corrupt it.
    X = X / (X.norm() + 1e-7)  # spectral norm <= 1 so the iteration converges
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
    ):
        params = list(params)
        for p in params:
            if isinstance(p, dict):
                raise ValueError(
                    "Muon does not support param groups (dicts); "
                    "pass a flat list of parameters"
                )
            if not isinstance(p, torch.Tensor) or p.ndim != 2:
                raise ValueError(f"Muon only handles 2-D params, got shape {tuple(p.shape)}")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
            ),
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(mu).add_(g)
                upd = g.add(buf, alpha=mu) if group["nesterov"] else buf
                orth_update = zeropower_via_newtonschulz5(upd, group["ns_steps"])
                scale = 0.2 * max(p.shape[0], p.shape[1]) ** 0.5
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(orth_update, alpha=-lr * scale)
        return loss
