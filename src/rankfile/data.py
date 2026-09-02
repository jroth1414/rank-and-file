"""Token shards and a fixed-order, resumable micro-batch loader."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from rankfile.tokenizer import EOT_ID


def write_shard(tokens: np.ndarray, path: str | Path) -> None:
    if tokens.dtype != np.uint16:
        raise TypeError(f"tokens must be uint16, got {tokens.dtype}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tokens.tofile(str(tmp))
    os.replace(tmp, path)


def list_shards(dir: str | Path, split: str) -> list[Path]:
    return sorted(Path(dir).glob(f"{split}_*.bin"))


class TokenStream:
    """All shards as one logical uint16 array, memory-mapped."""

    def __init__(self, shard_paths: list[Path]):
        if not shard_paths:
            raise ValueError("no shards")
        shard_paths = [Path(p) for p in shard_paths]
        self.mm = [np.memmap(p, dtype=np.uint16, mode="r") for p in shard_paths]
        self.sizes = np.array([len(m) for m in self.mm], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total_tokens = int(self.offsets[-1])

        manifest_path = shard_paths[0].parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            shard_counts = manifest.get("shards", {})
            for path, size in zip(shard_paths, self.sizes, strict=True):
                expected = shard_counts.get(path.name)
                if expected is not None and int(size) != expected:
                    raise ValueError(
                        f"shard {path.name} has {int(size)} tokens, "
                        f"manifest expects {expected}"
                    )

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


class FixedOrderSampler:
    """Seeded permutation of non-overlapping windows of seq_len+1 tokens."""

    def __init__(self, total_tokens: int, seq_len: int, seed: int):
        self.stride = seq_len + 1
        self.n_windows = total_tokens // self.stride
        self.perm = np.random.default_rng(seed).permutation(self.n_windows)

    def start(self, i: int) -> int:
        if not 0 <= i < self.n_windows:
            raise IndexError(f"window {i} out of range [0, {self.n_windows})")
        return int(self.perm[i]) * self.stride


def doc_ids_from_tokens(x: torch.Tensor) -> torch.Tensor:
    """Document index per token; an EOT token belongs to the document it ends."""
    eot = (x == EOT_ID).long()
    return torch.cumsum(eot, dim=1) - eot


def make_batch(stream: TokenStream, sampler: FixedOrderSampler, position: int,
               micro_batch: int, seq_len: int,
               device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    buf = np.stack([stream.window(sampler.start(position + j), seq_len + 1)
                    for j in range(micro_batch)])
    t = torch.from_numpy(buf.astype(np.int64))
    x, y = t[:, :-1], t[:, 1:]
    d = doc_ids_from_tokens(x)
    return x.to(device, non_blocking=True), y.to(device,
                                                  non_blocking=True), d.to(
                                                      device, non_blocking=True)
