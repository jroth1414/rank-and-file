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

def test_list_shards_sorted_by_split(tmp_path):
    _mk(tmp_path, [3, 3])
    write_shard(np.zeros(3, dtype=np.uint16), tmp_path / "val_0000.bin")
    assert [p.name for p in list_shards(tmp_path, "train")] == ["train_0000.bin", "train_0001.bin"]
    assert [p.name for p in list_shards(tmp_path, "val")] == ["val_0000.bin"]
