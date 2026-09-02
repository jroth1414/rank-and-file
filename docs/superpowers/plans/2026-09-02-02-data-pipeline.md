# Plan 2: Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tokenized FineWeb-Edu shards on disk, and a loader that yields `(x, y, doc_ids)` micro-batches in an order fixed by seed, identical across optimizers, and resumable from an integer position.

**Architecture:** Shards are flat `uint16` `.bin` files of concatenated documents each ending in `EOT_ID`. `TokenStream` memory-maps all shards as one logical token array. `FixedOrderSampler` is a seeded permutation of non-overlapping windows of `seq_len + 1` tokens. `make_batch` slices windows, derives `doc_ids` from EOT positions, and returns tensors. The training loop (Plan 4) stores only an integer `position` in checkpoints.

**Tech Stack:** numpy memmap, `datasets` streaming, `tokenizers`, torch.

**Spec:** `CLAUDE.md` §3 rule 2 (same data, same order), §6 layout. `proposal.md` §4.1 (FineWeb-Edu, 2048 context, intra-document masking), §4.2 (2.5B and 3.75B tokens).

## Global Constraints

- Python 3.11 in `.venv`; run as `.venv\Scripts\python.exe ...`.
- `EOT_ID = 0` from `rankfile.tokenizer`; every document ends with it.
- Data lives under `data/` on D: (gitignored). Set `HF_HOME=D:\hf_cache` before downloads.
- Shard order and window permutation depend only on `seed`; P1 and P2 with the same seed see identical batches. P3 (3.75B) continues through the same permutation past P1's 2.5B.
- Commit prefix `data:`; no AI attribution trailers.

---

### Task 1: Shard writer and TokenStream reader

**Files:**
- Create: `src/rankfile/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `write_shard(tokens: np.ndarray, path: Path) -> None` (uint16 raw); `TokenStream(shard_paths: list[Path])` with `.total_tokens: int` and `.window(start: int, length: int) -> np.ndarray` (uint16, may span shards); `list_shards(dir: Path, split: str) -> list[Path]` returning sorted `dir/{split}_*.bin`.
- Consumes: `rankfile.tokenizer.EOT_ID`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
import numpy as np
import pytest
from pathlib import Path
from rankfile.data import write_shard, TokenStream, list_shards

def _mk(tmp_path, sizes):
    paths = []
    off = 0
    for i, n in enumerate(sizes):
        p = tmp_path / f"train_{i:04d}.bin"
        write_shard(np.arange(off, off + n, dtype=np.uint16), p)
        paths.append(p); off += n
    return paths

def test_stream_spans_shards(tmp_path):
    paths = _mk(tmp_path, [10, 5, 7])
    s = TokenStream(paths)
    assert s.total_tokens == 22
    assert s.window(8, 6).tolist() == [8, 9, 10, 11, 12, 13]
    assert s.window(14, 8).tolist() == list(range(14, 22))

def test_window_out_of_range(tmp_path):
    s = TokenStream(_mk(tmp_path, [10]))
    with pytest.raises(ValueError):
        s.window(5, 6)

def test_list_shards_sorted_by_split(tmp_path):
    _mk(tmp_path, [3, 3])
    write_shard(np.zeros(3, dtype=np.uint16), tmp_path / "val_0000.bin")
    assert [p.name for p in list_shards(tmp_path, "train")] == ["train_0000.bin", "train_0001.bin"]
    assert [p.name for p in list_shards(tmp_path, "val")] == ["val_0000.bin"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/data.py
"""Token shards and a fixed-order, resumable micro-batch loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rankfile.tokenizer import EOT_ID


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
            raise ValueError(f"window [{start},{start + length}) outside [0,{self.total_tokens})")
        out = np.empty(length, dtype=np.uint16)
        filled = 0
        i = int(np.searchsorted(self.offsets, start, side="right") - 1)
        pos = start - self.offsets[i]
        while filled < length:
            take = min(length - filled, int(self.sizes[i]) - pos)
            out[filled:filled + take] = self.mm[i][pos:pos + take]
            filled += take
            i += 1
            pos = 0
        return out
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/data.py tests/test_data.py
git commit -m "data: uint16 shard writer and multi-shard memory-mapped TokenStream"
```

---

### Task 2: Fixed-order sampler, doc_ids, make_batch

**Files:**
- Modify: `src/rankfile/data.py`
- Test: `tests/test_data.py` (append)

**Interfaces:**
- Produces: `FixedOrderSampler(total_tokens: int, seq_len: int, seed: int)` with `.n_windows: int` and `.start(i: int) -> int` (token offset of the i-th window in seeded order); `doc_ids_from_tokens(x: LongTensor[B,T]) -> LongTensor[B,T]`; `make_batch(stream, sampler, position: int, micro_batch: int, seq_len: int, device) -> tuple[x, y, doc_ids]` where `x,y` are `LongTensor[micro_batch, seq_len]`, `y` is `x` shifted by one, and `position` counts windows consumed so far (the loop advances it by `micro_batch`).
- Plan 4 consumes all of these and stores `position` in checkpoints.

- [ ] **Step 1: Write the failing tests**

```python
def test_sampler_is_permutation_and_deterministic():
    from rankfile.data import FixedOrderSampler
    a = FixedOrderSampler(total_tokens=1000, seq_len=9, seed=0)
    b = FixedOrderSampler(total_tokens=1000, seq_len=9, seed=0)
    c = FixedOrderSampler(total_tokens=1000, seq_len=9, seed=1)
    assert a.n_windows == 1000 // 10
    starts = [a.start(i) for i in range(a.n_windows)]
    assert sorted(starts) == [i * 10 for i in range(a.n_windows)]
    assert starts == [b.start(i) for i in range(b.n_windows)]
    assert starts != [c.start(i) for i in range(c.n_windows)]

def test_doc_ids_from_tokens():
    import torch
    from rankfile.data import doc_ids_from_tokens
    x = torch.tensor([[5, 6, 0, 7, 8, 0, 9]])
    assert doc_ids_from_tokens(x).tolist() == [[0, 0, 0, 1, 1, 1, 2]]

def test_make_batch_shapes_and_shift(tmp_path):
    import torch
    from rankfile.data import FixedOrderSampler, make_batch
    stream = TokenStream(_mk(tmp_path, [200]))
    samp = FixedOrderSampler(stream.total_tokens, seq_len=8, seed=0)
    x, y, d = make_batch(stream, samp, position=0, micro_batch=4, seq_len=8, device="cpu")
    assert x.shape == y.shape == d.shape == (4, 8) and x.dtype == torch.long
    assert torch.equal(y[:, :-1], x[:, 1:])
    x2, _, _ = make_batch(stream, samp, position=4, micro_batch=4, seq_len=8, device="cpu")
    assert not torch.equal(x, x2)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: 3 FAIL with `ImportError`

- [ ] **Step 3: Implement** (append to `data.py`)

```python
class FixedOrderSampler:
    """Seeded permutation of non-overlapping windows of seq_len+1 tokens."""

    def __init__(self, total_tokens: int, seq_len: int, seed: int):
        self.stride = seq_len + 1
        self.n_windows = total_tokens // self.stride
        self.perm = np.random.default_rng(seed).permutation(self.n_windows)

    def start(self, i: int) -> int:
        if i >= self.n_windows:
            raise IndexError(f"window {i} >= {self.n_windows}; data exhausted")
        return int(self.perm[i]) * self.stride


def doc_ids_from_tokens(x: torch.Tensor) -> torch.Tensor:
    """Document index per token; an EOT token belongs to the document it ends."""
    eot = (x == EOT_ID).long()
    return torch.cumsum(eot, dim=1) - eot


def make_batch(stream: TokenStream, sampler: FixedOrderSampler, position: int, micro_batch: int,
               seq_len: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    buf = np.stack([stream.window(sampler.start(position + j), seq_len + 1) for j in range(micro_batch)])
    t = torch.from_numpy(buf.astype(np.int64))
    x, y = t[:, :-1], t[:, 1:]
    d = doc_ids_from_tokens(x)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), d.to(device, non_blocking=True)
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/data.py tests/test_data.py
git commit -m "data: seeded fixed-order window sampler, doc_ids, make_batch"
```

---

### Task 3: prepare_data.py (FineWeb-Edu → shards)

**Files:**
- Create: `scripts/prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Produces: `data/fineweb_edu/train_XXXX.bin` (100M tokens each) and `data/fineweb_edu/val_0000.bin` (20M tokens); a function `shard_documents(doc_iter, tok, out_dir, shard_tokens, max_tokens, val_tokens) -> dict` (returns counts) that the test drives with fake docs.
- Consumes: `rankfile.tokenizer.load_tokenizer`, `encode_docs`; `rankfile.data.write_shard`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prepare_data.py
import numpy as np
from rankfile.tokenizer import train_tokenizer, EOT_ID
from rankfile.data import list_shards, TokenStream

def test_shard_documents_splits_val_then_train(tmp_path):
    from scripts.prepare_data import shard_documents
    tok = train_tokenizer(["alpha beta gamma delta " * 50], vocab_size=300, out_path=tmp_path / "t.json")
    docs = (f"doc {i} alpha beta gamma delta " * 10 for i in range(400))
    stats = shard_documents(docs, tok, tmp_path / "out", shard_tokens=5000, max_tokens=20000, val_tokens=3000)
    val, train = list_shards(tmp_path / "out", "val"), list_shards(tmp_path / "out", "train")
    assert len(val) == 1 and len(train) >= 3
    vs, ts = TokenStream(val), TokenStream(train)
    assert vs.total_tokens >= 3000 and 15000 <= ts.total_tokens <= 21000
    assert (ts.window(0, ts.total_tokens) == EOT_ID).sum() > 100
    assert stats["train_tokens"] == ts.total_tokens
```

Add `tests/../scripts/__init__.py` (empty) so `scripts.prepare_data` imports.

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_prepare_data.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.prepare_data`

- [ ] **Step 3: Implement**

```python
# scripts/prepare_data.py
"""Stream FineWeb-Edu (sample-10BT), tokenize, write uint16 shards.

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


def shard_documents(docs: Iterable[str], tok, out_dir: str | Path, shard_tokens: int, max_tokens: int,
                    val_tokens: int) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buf: list[np.ndarray] = []
    buf_n = 0
    split, shard_idx, total = "val", 0, 0
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
    stats = shard_documents((r["text"] for r in ds), load_tokenizer(a.tokenizer), a.out,
                            a.shard_tokens, a.max_tokens, a.val_tokens)
    print(stats, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_prepare_data.py -v`
Expected: 1 passed

- [ ] **Step 5: Smoke the real pipeline with a small budget** (network; ~2 min)

Run: `set HF_HOME=D:\hf_cache && .venv\Scripts\python.exe scripts/prepare_data.py --out data/fineweb_edu_smoke --max-tokens 50000000 --shard-tokens 25000000 --val-tokens 5000000`
Expected: prints stats with `train_tokens` ≥ 50M; `data/fineweb_edu_smoke/` has `val_0000.bin` and `train_0000.bin`, `train_0001.bin`. Plan 4's smoke run uses this directory.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_data.py scripts/__init__.py tests/test_prepare_data.py
git commit -m "data: FineWeb-Edu streaming tokenizer into val/train shards"
```

---

### Task 4: Full data build (ask the user first)

**Files:** none new.

- [ ] **Step 1: Ask the user for permission**, stating: "This streams ~4.2B tokens of FineWeb-Edu, writes ~8.4 GB to `data/fineweb_edu/`, and takes roughly 60–90 minutes of CPU. OK to start?"

- [ ] **Step 2: On approval, run**

```
set HF_HOME=D:\hf_cache
.venv\Scripts\python.exe scripts/prepare_data.py --out data/fineweb_edu --max-tokens 4200000000
```

- [ ] **Step 3: Verify**

```python
.venv\Scripts\python.exe -c "from rankfile.data import *; t=TokenStream(list_shards('data/fineweb_edu','train')); v=TokenStream(list_shards('data/fineweb_edu','val')); print(t.total_tokens/1e9,'B train', v.total_tokens/1e6,'M val')"
```

Expected: ≥ 4.1B train, ≥ 20M val. Record the exact counts in `CLAUDE.md` §11 decision log with the date.

- [ ] **Step 4: Commit the decision-log line**

```bash
git add CLAUDE.md
git commit -m "docs: record FineWeb-Edu shard build (token counts)"
```

---

## Self-review

- **Spec coverage:** same data/same order (Task 2 permutation by seed), P3 continuing past 2.5B (permutation covers 4.1B; position simply keeps increasing), intra-doc masking inputs (`doc_ids`), 2048 context (`seq_len` parameter), FineWeb-Edu (Task 3).
- **Placeholders:** none.
- **Type consistency:** `make_batch` returns `(x, y, doc_ids)` as `LongTensor[micro_batch, seq_len]`; `position` is an int counting windows; `EOT_ID` imported from `rankfile.tokenizer`, matching Plan 1.
