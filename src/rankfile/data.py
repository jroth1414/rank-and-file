"""Token shards and a fixed-order, resumable micro-batch loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rankfile.tokenizer import EOT_ID  # noqa: F401  # Task 2 uses this


def write_shard(tokens: np.ndarray, path: str | Path) -> None:
    assert tokens.dtype == np.uint16
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(str(path))


def list_shards(dir: str | Path, split: str) -> list[Path]:
    return sorted(Path(dir).glob(f"{split}_*.bin"))


class TokenStream:
    """All shards as one logical uint16 array, memory-mapped."""

    def __init__(self, shard_paths: list[Path]):
        if not shard_paths:
            raise ValueError("no shards")
        self.mm = [np.memmap(p, dtype=np.uint16, mode="r") for p in shard_paths]
        self.sizes = np.array([len(m) for m in self.mm], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total_tokens = int(self.offsets[-1])

    def window(self, start: int, length: int) -> np.ndarray:
        if start < 0 or start + length > self.total_tokens:
            raise ValueError(
                f"window [{start},{start + length}) outside [0,{self.total_tokens})"
            )
        out = np.empty(length, dtype=np.uint16)
        filled = 0
        i = int(np.searchsorted(self.offsets, start, side="right") - 1)
        pos = start - self.offsets[i]
        while filled < length:
            take = min(length - filled, int(self.sizes[i]) - pos)
            out[filled : filled + take] = self.mm[i][pos : pos + take]
            filled += take
            i += 1
            pos = 0
        return out
