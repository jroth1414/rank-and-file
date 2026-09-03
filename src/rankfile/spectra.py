"""Pure spectral measurements: effective rank, stable rank, top-r energy, LoRA subspace overlap."""
from __future__ import annotations

import re

import torch

_NAME = re.compile(r"blocks\.(\d+)\.(attn|mlp)\.(\w+)(?:\.weight)?$")


def singular_values(W: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(W.detach().float().cpu())


def effective_rank(s: torch.Tensor) -> float:
    s = s[s > 0]
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def stable_rank(s: torch.Tensor) -> float:
    return float((s * s).sum() / (s[0] * s[0]))


def top_r_energy(s: torch.Tensor, r: int) -> float:
    e = s * s
    return float(e[:r].sum() / e.sum())


def subspace_overlap(delta: torch.Tensor, B: torch.Tensor) -> float:
    delta, B = delta.detach().float().cpu(), B.detach().float().cpu()
    Q, _ = torch.linalg.qr(B)                       # orthonormal basis of colspace(B)
    proj = Q @ (Q.T @ delta)
    return float((proj * proj).sum() / (delta * delta).sum())


def matrix_report(name: str, W: torch.Tensor, ranks: tuple[int, ...] = (4, 16, 64)) -> dict:
    m = _NAME.search(name)
    layer, module = (int(m.group(1)), f"{m.group(2)}.{m.group(3)}") if m else (-1, name)
    s = singular_values(W)
    out = {"name": name, "layer": layer, "module": module, "rows": W.shape[0], "cols": W.shape[1],
           "erank": effective_rank(s), "srank": stable_rank(s), "fro": float(s.norm())}
    for r in ranks:
        out[f"top{r}"] = top_r_energy(s, r)
    return out
