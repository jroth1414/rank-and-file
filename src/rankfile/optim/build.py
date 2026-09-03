"""Route parameters to AdamW or Muon per arm. The only difference between arms is
which optimizer updates the 2-D hidden matrices; everything else is identical."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rankfile.optim.muon import Muon

if TYPE_CHECKING:
    from rankfile.model import Transformer


def _adamw(
    groups: list[dict], lr: float, betas: tuple[float, float], adamw_lr_scale: float
) -> torch.optim.AdamW:
    params = [p for g in groups for p in g["params"]]
    on_cuda = bool(params) and all(p.is_cuda for p in params)
    for g in groups:
        g["lr_scale"] = adamw_lr_scale
    return torch.optim.AdamW(
        groups, lr=lr, betas=betas, fused=on_cuda
    )  # fused only for CUDA params; CPU stays deterministic


def build_optimizers(
    model: Transformer,
    name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    adamw_lr_scale: float = 1.0,
) -> list[torch.optim.Optimizer]:
    hidden = model.hidden_matrix_params()
    other = model.other_params()
    other_decay = [p for p in other if p.ndim >= 2]  # embeddings
    other_nodecay = [p for p in other if p.ndim < 2]  # RMSNorm gains
    if name == "adamw":
        groups = [
            dict(params=hidden + other_decay, weight_decay=weight_decay),
            dict(params=other_nodecay, weight_decay=0.0),
        ]
        return [_adamw(groups, lr, betas, adamw_lr_scale)]
    if name == "muon":
        groups = [
            dict(params=other_decay, weight_decay=weight_decay),
            dict(params=other_nodecay, weight_decay=0.0),
        ]
        muon = Muon(hidden, lr=lr, weight_decay=weight_decay)
        for g in muon.param_groups:
            g["lr_scale"] = 1.0
        return [muon, _adamw(groups, lr, betas, adamw_lr_scale)]
    raise ValueError(f"unknown optimizer {name!r}; use 'adamw' or 'muon'")


def set_lr(opts: list[torch.optim.Optimizer], lr: float) -> None:
    for o in opts:
        for g in o.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)


def optimizer_state_dicts(opts: list[torch.optim.Optimizer]) -> list[dict]:
    return [o.state_dict() for o in opts]


def load_optimizer_state_dicts(opts: list[torch.optim.Optimizer], sds: list[dict]) -> None:
    if len(opts) != len(sds):
        raise ValueError(
            f"optimizer/state-dict count mismatch: {len(opts)} optimizers vs {len(sds)} state dicts"
        )
    for o, sd in zip(opts, sds, strict=True):
        o.load_state_dict(sd)
