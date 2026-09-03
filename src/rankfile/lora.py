"""LoRA (Hu et al. 2021) implemented by hand: W x + (B A x) * alpha/r, B zero-initialized."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

TARGETS = ("q", "k", "v", "o", "gate", "up", "down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def delta(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale


def apply_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    targets: tuple[str, ...] = TARGETS,
) -> list[nn.Parameter]:
    # Stashed so merge_lora can restore each parameter's own prior requires_grad
    # (e.g. an embedding already frozen for other reasons) instead of a blanket True.
    model._lora_prev_requires_grad = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    for blk in model.blocks:
        for parent in (blk.attn, blk.mlp):
            for name in targets:
                if hasattr(parent, name) and isinstance(getattr(parent, name), nn.Linear):
                    wrapped = LoRALinear(getattr(parent, name), r, alpha)
                    setattr(parent, name, wrapped)
                    trainable += [wrapped.A, wrapped.B]
    return trainable


def _lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_state_dict(model: nn.Module) -> dict[str, dict]:
    return {n: {"A": m.A.detach().cpu().clone(), "B": m.B.detach().cpu().clone(), "scale": m.scale}
            for n, m in _lora_modules(model)}


def lora_deltas(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: m.delta().detach() for n, m in _lora_modules(model)}


@torch.no_grad()
def merge_lora(model: nn.Module) -> None:
    for name, mod in _lora_modules(model):
        mod.base.weight.add_(mod.delta().to(mod.base.weight.dtype))
        parent_name, attr = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, mod.base)
    prev = getattr(model, "_lora_prev_requires_grad", None)
    for p in model.parameters():
        p.requires_grad_(prev.get(id(p), True) if prev is not None else True)
