import json

import numpy as np
import pytest

from rankfile.data import TokenStream, list_shards, write_shard


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

def test_write_shard_is_atomic_and_round_trips(tmp_path):
    p = tmp_path / "train_0000.bin"
    write_shard(np.arange(10, dtype=np.uint16), p)
    assert not (tmp_path / "train_0000.bin.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []
    s = TokenStream([p])
    assert s.window(0, 10).tolist() == list(range(10))

def test_write_shard_rejects_wrong_dtype(tmp_path):
    with pytest.raises(TypeError):
        write_shard(np.arange(10, dtype=np.int32), tmp_path / "train_0000.bin")

def test_token_stream_verifies_manifest_counts(tmp_path):
    paths = _mk(tmp_path, [10, 5])
    (tmp_path / "manifest.json").write_text(
        json.dumps({"shards": {"train_0000.bin": 10, "train_0001.bin": 999}})
    )
    with pytest.raises(ValueError):
        TokenStream(paths)

def test_token_stream_manifest_ignores_unlisted_shards(tmp_path):
    paths = _mk(tmp_path, [10, 5])
    (tmp_path / "manifest.json").write_text(json.dumps({"shards": {"train_0000.bin": 10}}))
    s = TokenStream(paths)
    assert s.total_tokens == 15

def test_token_stream_works_with_no_manifest(tmp_path):
    paths = _mk(tmp_path, [10, 5])
    s = TokenStream(paths)
    assert s.total_tokens == 15

def test_sampler_start_rejects_negative_index():
    from rankfile.data import FixedOrderSampler
    s = FixedOrderSampler(total_tokens=1000, seq_len=9, seed=0)
    with pytest.raises(IndexError):
        s.start(-1)

def test_list_shards_sorted_by_split(tmp_path):
    _mk(tmp_path, [3, 3])
    write_shard(np.zeros(3, dtype=np.uint16), tmp_path / "val_0000.bin")
    assert [p.name for p in list_shards(tmp_path, "train")] == ["train_0000.bin", "train_0001.bin"]
    assert [p.name for p in list_shards(tmp_path, "val")] == ["val_0000.bin"]

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
