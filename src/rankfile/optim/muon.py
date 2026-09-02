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


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
    ):
        params = list(params)
        for p in params:
            if p.ndim != 2:
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
    def step(self, closure=None):
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
                orth_update = orth_update * (0.2 * max(p.shape[0], p.shape[1]) ** 0.5)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(orth_update, alpha=-lr)
        return loss
