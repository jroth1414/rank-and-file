r"""Build Python code shards for the continued-pretraining task.

Dataset: codeparrot/codeparrot-clean (data-files based, streams under
datasets>=4), field "content", filtered to permissive licenses on the
"license" field: {mit, apache-2.0, bsd-3-clause, bsd-2-clause, isc}.
Rows with a missing, None, or empty license are dropped, not kept.
(codeparrot/github-code-clean and bigcode/the-stack-smol were probed first
and rejected: the former is a script-based dataset unsupported on
datasets>=4, the latter is gated and requires an authenticated HF account.)

Usage: set HF_HOME=D:\hf_cache && python scripts/prepare_code.py --out data/code
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from rankfile.tokenizer import load_tokenizer
from scripts.prepare_data import shard_documents

PERMISSIVE_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"}


def _keep(row: dict) -> bool:
    """True iff row carries an explicit permissive license (Ruling E: no license => drop)."""
    license_ = row.get("license")
    if not license_:
        return False
    return str(license_).strip().lower() in PERMISSIVE_LICENSES


def python_docs(max_docs: int | None = None) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    n = 0
    for row in ds:
        if max_docs is not None and n >= max_docs:
            break
        if not _keep(row):
            continue
        n += 1
        yield row["content"]


def build_code_shards(docs: Iterable[str], tok, out_dir: str | Path, train_tokens: int, val_tokens: int,
                      shard_tokens: int, tokenizer_path: str | Path | None = None) -> dict:
    return shard_documents(docs, tok, out_dir, shard_tokens=shard_tokens, max_tokens=train_tokens,
                           val_tokens=val_tokens, tokenizer_path=tokenizer_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/code")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--train-tokens", type=int, default=100_000_000)
    ap.add_argument("--val-tokens", type=int, default=5_000_000)
    ap.add_argument("--force", action="store_true", help="overwrite an out dir that already has shards")
    a = ap.parse_args()

    out_dir = Path(a.out)
    if not a.force and out_dir.exists() and any(out_dir.glob("*.bin")):
        print(
            f"{out_dir} already contains .bin shards; pass --force to overwrite",
            file=sys.stderr,
        )
        raise SystemExit(1)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    stats = build_code_shards(python_docs(), load_tokenizer(a.tokenizer), a.out, a.train_tokens, a.val_tokens,
                              25_000_000, tokenizer_path=a.tokenizer)
    print(stats)


if __name__ == "__main__":
    main()
