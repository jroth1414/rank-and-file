r"""Stream FineWeb-Edu (sample-10BT), tokenize, write uint16 shards.

Usage:
  set HF_HOME=D:\hf_cache
  python scripts/prepare_data.py --out data/fineweb_edu --max-tokens 4200000000
Writes val_0000.bin first (20M tokens), then train_XXXX.bin shards of 100M tokens.
4.2B tokens covers P3 (3.75B) with margin. ~1M tok/s single process, so ~70 min.
"""
from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from rankfile.data import write_shard
from rankfile.tokenizer import encode_docs, load_tokenizer


def _batches(it: Iterator[str], n: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def shard_documents(
    docs: Iterable[str], tok, out_dir: str | Path, shard_tokens: int, max_tokens: int, val_tokens: int
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buf: list[np.ndarray] = []
    buf_n = 0
    split, shard_idx = "val", 0
    target = val_tokens
    stats = {"val_tokens": 0, "train_tokens": 0, "docs": 0}

    def flush() -> None:
        nonlocal buf, buf_n, split, shard_idx, target
        if buf_n == 0:
            return
        arr = np.concatenate(buf)
        write_shard(arr, out_dir / f"{split}_{shard_idx:04d}.bin")
        stats[f"{split}_tokens"] += len(arr)
        buf, buf_n = [], 0
        if split == "val":
            split, shard_idx, target = "train", 0, shard_tokens
        else:
            shard_idx += 1

    done = False
    for batch in _batches(iter(docs), 1000):
        for ids in encode_docs(tok, batch):
            buf.append(ids)
            buf_n += len(ids)
            stats["docs"] += 1
            if buf_n >= target:
                flush()
                if stats["train_tokens"] >= max_tokens:
                    done = True
                    break
        if done:
            break
    if not done and split == "train":
        flush()  # partial last shard when the source ran out first
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fineweb_edu")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--max-tokens", type=int, default=4_200_000_000)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--val-tokens", type=int, default=20_000_000)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    t0 = time.time()
    stats = shard_documents(
        (r["text"] for r in ds), load_tokenizer(a.tokenizer), a.out, a.shard_tokens, a.max_tokens, a.val_tokens
    )
    print(stats, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
