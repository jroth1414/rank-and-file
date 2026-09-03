from rankfile.data import TokenStream, list_shards
from rankfile.tokenizer import train_tokenizer
from scripts.prepare_code import _keep, build_code_shards, python_docs


def test_build_code_shards_from_iterator(tmp_path):
    tok = train_tokenizer(["def f(x):\n    return x\n" * 100], vocab_size=300, out_path=tmp_path / "t.json")
    docs = (f"def f{i}(x):\n    return x + {i}\n" * 20 for i in range(300))
    stats = build_code_shards(docs, tok, tmp_path / "code", train_tokens=20000, val_tokens=2000, shard_tokens=8000)
    assert TokenStream(list_shards(tmp_path / "code", "val")).total_tokens >= 2000
    assert TokenStream(list_shards(tmp_path / "code", "train")).total_tokens >= 20000
    assert stats["docs"] > 0


def test_keep_requires_explicit_permissive_license():
    assert _keep({"license": "MIT"})
    assert _keep({"license": " mit "})
    assert not _keep({"license": None})
    assert not _keep({"license": ""})
    assert not _keep({"license": "gpl-3.0"})
    assert not _keep({})


def test_python_docs_max_docs_counts_yielded_not_scanned(monkeypatch):
    rows = [
        {"license": "MIT", "content": "a"},
        {"license": "gpl-3.0", "content": "b"},
        {"license": "apache-2.0", "content": "c"},
        {"license": None, "content": "d"},
        {"license": "bsd-3-clause", "content": "e"},
    ]

    def fake_load_dataset(*args, **kwargs):
        return iter(rows)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    docs = list(python_docs(max_docs=2))
    assert docs == ["a", "c"]
