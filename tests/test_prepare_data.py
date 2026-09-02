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
