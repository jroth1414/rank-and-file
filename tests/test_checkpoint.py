import torch

from rankfile.checkpoint import (
    load_checkpoint,
    load_model_state,
    read_latest,
    save_checkpoint,
    write_latest,
)
from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers


def _mk():
    torch.manual_seed(0)
    m = Transformer(
        ModelConfig(
            vocab_size=64,
            n_layer=1,
            d_model=16,
            n_head=2,
            n_kv_head=1,
            head_dim=8,
            d_ff=32,
            max_seq_len=8,
        )
    )
    return m, build_optimizers(m, "muon", lr=1e-3, weight_decay=0.1)


def test_roundtrip_restores_everything(tmp_path):
    m, opts = _mk()
    x = torch.randint(0, 64, (2, 8))
    m.loss(x, x).backward()
    [o.step() for o in opts]
    torch.manual_seed(123)
    torch.rand(1)  # advance RNG so state is non-trivial
    p = tmp_path / "ckpt_0000001.pt"
    save_checkpoint(
        p, m, opts, step=1, position=16, tokens_seen=32, extra={"note": "hi"}
    )
    r1 = torch.rand(3)
    m2, opts2 = _mk()
    meta = load_checkpoint(p, m2, opts2)
    assert meta["step"] == 1 and meta["position"] == 16 and meta["tokens_seen"] == 32
    assert meta["extra"]["note"] == "hi"
    for a, b in zip(m.parameters(), m2.parameters(), strict=True):
        assert torch.equal(a, b)
    assert torch.equal(torch.rand(3), r1)  # RNG restored to the state at save time
    buf = opts2[0].state[next(iter(opts2[0].param_groups[0]["params"]))][
        "momentum_buffer"
    ]
    assert buf.abs().sum() > 0


def test_latest_pointer(tmp_path):
    assert read_latest(tmp_path) is None
    write_latest(tmp_path, "ckpt_0000005.pt")
    assert read_latest(tmp_path) == tmp_path / "ckpt_0000005.pt"


def test_write_latest_is_atomic_and_leaves_no_tmp(tmp_path):
    write_latest(tmp_path, "ckpt_0000005.pt")
    assert not (tmp_path / "latest.tmp").exists()
    assert (tmp_path / "latest.txt").read_text(encoding="utf-8") == "ckpt_0000005.pt"
    write_latest(tmp_path, "ckpt_0000009.pt")  # overwrite path also leaves no .tmp
    assert not (tmp_path / "latest.tmp").exists()
    assert read_latest(tmp_path) == tmp_path / "ckpt_0000009.pt"


def test_load_model_state_only(tmp_path):
    m, opts = _mk()
    p = tmp_path / "c.pt"
    save_checkpoint(p, m, opts, 0, 0, 0, {})
    sd = load_model_state(p)
    assert "blocks.0.attn.q.weight" in sd and sd["embed.weight"].shape == (64, 16)
