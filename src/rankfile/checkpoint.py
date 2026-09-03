"""Checkpoint format: model, optimizers, step, data position, RNG. Exact resume."""
from __future__ import annotations

from pathlib import Path

import torch

from rankfile.optim.build import (
    load_optimizer_state_dicts,
    optimizer_state_dicts,
)


def unwrap(model):
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    path: str | Path,
    model,
    opts,
    step: int,
    position: int,
    tokens_seen: int,
    extra: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap(model).state_dict(),
        "optimizers": optimizer_state_dicts(opts),
        "step": step,
        "position": position,
        "tokens_seen": tokens_seen,
        "extra": extra,
        "rng": {
            "cpu": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic on the same volume


def load_checkpoint(path: str | Path, model, opts=None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(state["model"])
    if opts is not None:
        load_optimizer_state_dicts(opts, state["optimizers"])
    torch.set_rng_state(state["rng"]["cpu"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return {
        k: state[k]
        for k in ("step", "position", "tokens_seen", "extra")
    }


def load_model_state(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def write_latest(run_dir: str | Path, ckpt_name: str) -> None:
    (Path(run_dir) / "latest.txt").write_text(ckpt_name, encoding="utf-8")


def read_latest(run_dir: str | Path) -> Path | None:
    p = Path(run_dir) / "latest.txt"
    if not p.exists():
        return None
    return Path(run_dir) / p.read_text(encoding="utf-8").strip()
