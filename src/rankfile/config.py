"""YAML config loading with key=value overrides into dataclasses."""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return {} if d is None else dict(d)


def to_yaml(obj: Any, path: str | Path) -> None:
    d = asdict(obj) if is_dataclass(obj) else dict(obj)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    out = dict(d)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        if k not in out:
            raise KeyError(f"unknown config key {k!r}; known: {sorted(out)}")
        parsed = yaml.safe_load(v)
        # Fallback to float parsing if YAML returned a string that looks numeric
        if isinstance(parsed, str):
            try:
                parsed = float(parsed)
            except ValueError:
                pass  # Keep as string if conversion fails
        out[k] = parsed
    return out


def from_dict(cls: type[T], d: dict) -> T:
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**d)
