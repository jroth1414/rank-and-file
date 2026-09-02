import hashlib
import json

import pytest

from rankfile.data import TokenStream, list_shards
from rankfile.tokenizer import EOT_ID, train_tokenizer


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


def test_shard_documents_writes_manifest_matching_shard_sizes(tmp_path):
    from scripts.prepare_data import shard_documents
    tokenizer_path = tmp_path / "t.json"
    tok = train_tokenizer(["alpha beta gamma delta " * 50], vocab_size=300, out_path=tokenizer_path)
    docs = (f"doc {i} alpha beta gamma delta " * 10 for i in range(400))
    out_dir = tmp_path / "out"
    stats = shard_documents(
        docs, tok, out_dir, shard_tokens=5000, max_tokens=20000, val_tokens=3000,
        tokenizer_path=tokenizer_path,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["val_tokens"] == stats["val_tokens"]
    assert manifest["train_tokens"] == stats["train_tokens"]
    assert manifest["docs"] == stats["docs"]
    assert manifest["tokenizer_sha256"] == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    for name, tokens in manifest["shards"].items():
        assert tokens == (out_dir / name).stat().st_size // 2


def test_shard_documents_manifest_tokenizer_sha_defaults_to_none(tmp_path):
    from scripts.prepare_data import shard_documents
    tok = train_tokenizer(["alpha beta gamma delta " * 50], vocab_size=300, out_path=tmp_path / "t.json")
    docs = (f"doc {i} alpha beta gamma delta " * 10 for i in range(400))
    out_dir = tmp_path / "out"
    shard_documents(docs, tok, out_dir, shard_tokens=5000, max_tokens=20000, val_tokens=3000)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["tokenizer_sha256"] is None


def test_shard_documents_raises_when_source_dries_up_before_val(tmp_path):
    from scripts.prepare_data import shard_documents
    tok = train_tokenizer(["alpha beta gamma delta " * 50], vocab_size=300, out_path=tmp_path / "t.json")
    docs = iter(["alpha beta gamma delta"])
    with pytest.raises(ValueError):
        shard_documents(docs, tok, tmp_path / "out", shard_tokens=5000, max_tokens=20000, val_tokens=3000)
