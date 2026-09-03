from rankfile.data import TokenStream, list_shards
from rankfile.tokenizer import train_tokenizer
from scripts.prepare_code import build_code_shards


def test_build_code_shards_from_iterator(tmp_path):
    tok = train_tokenizer(["def f(x):\n    return x\n" * 100], vocab_size=300, out_path=tmp_path / "t.json")
    docs = (f"def f{i}(x):\n    return x + {i}\n" * 20 for i in range(300))
    stats = build_code_shards(docs, tok, tmp_path / "code", train_tokens=20000, val_tokens=2000, shard_tokens=8000)
    assert TokenStream(list_shards(tmp_path / "code", "val")).total_tokens >= 2000
    assert TokenStream(list_shards(tmp_path / "code", "train")).total_tokens >= 20000
    assert stats["docs"] > 0
