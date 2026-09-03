import json
import random

import numpy as np
import pytest
import torch

from rankfile.checkpoint import save_checkpoint, write_latest
from rankfile.config import to_yaml
from rankfile.data import write_shard
from rankfile.finetune import FinetuneConfig, _bucketed_batches, finetune
from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers
from rankfile.tokenizer import train_tokenizer


def _parent(tmp_path):
    # vocab 512 so a real byte-level tokenizer (>= 257 ids) fits; shard tokens stay < 62
    mc = ModelConfig(vocab_size=512, n_layer=2, d_model=32, n_head=2, n_kv_head=1,
                      head_dim=16, d_ff=64, max_seq_len=32)
    m = Transformer(mc)
    run = tmp_path / "runs" / "parent"
    run.mkdir(parents=True)
    to_yaml(mc, run / "model.yaml")
    save_checkpoint(run / "ckpt_0000001.pt", m, build_optimizers(m, "adamw", 1e-3, 0.1),
                     1, 0, 0, {})
    write_latest(run, "ckpt_0000001.pt")
    (run / "DONE").write_text("1 0 4.0")
    for d in ("pre", "code"):
        seq = [(i * 7 + 3) % 61 + 1 if d == "pre" else (i * 5 + 1) % 61 + 1 for i in range(20000)]
        for i in range(0, 20000, 50):
            seq[i] = 0
        write_shard(np.array(seq, dtype=np.uint16), tmp_path / d / "train_0000.bin")
        write_shard(np.array(seq[:3000], dtype=np.uint16), tmp_path / d / "val_0000.bin")
    return run


def _cfg(tmp_path, run, **kw):
    return FinetuneConfig(parent=str(run), out_root=str(tmp_path / "runs"),
                          data_dir_code=str(tmp_path / "code"), data_dir_pre=str(tmp_path / "pre"),
                          seq_len=32, micro_batch=2, batch_tokens=128, train_tokens=128 * 30,
                          eval_windows=8, compile=False, **kw)


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_full_code_ft_improves_code_loss_and_saves_model(tmp_path):
    run = _parent(tmp_path)
    out = finetune(_cfg(tmp_path, run, method="full", task="code", lr=3e-3), device="cpu")
    r = json.loads((out / "results.json").read_text())
    assert r["after"]["code_val_loss"] < r["before"]["code_val_loss"] and (out / "model.pt").exists()
    assert out.name == "parent__full_code"


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_lora_code_ft_saves_lora_only(tmp_path):
    run = _parent(tmp_path)
    out = finetune(_cfg(tmp_path, run, method="lora", rank=2, alpha=4, task="code", lr=1e-2),
                    device="cpu")
    assert (out / "lora.pt").exists() and not (out / "model.pt").exists()
    assert out.name == "parent__lora2_code"
    sd = torch.load(out / "lora.pt", weights_only=False)
    assert sd["blocks.0.attn.q"]["A"].shape[0] == 2


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_sup_ft_runs_on_fake_examples(tmp_path, monkeypatch):
    run = _parent(tmp_path)
    train_tokenizer(["Review: good bad\nSentiment: positive negative " * 50],
                     vocab_size=300, out_path=tmp_path / "t.json")
    import rankfile.finetune as ft
    fake = {"sst2": ([{"sentence": "good", "label": 1}, {"sentence": "bad", "label": 0}] * 8)}
    monkeypatch.setattr(ft, "_sup_data", lambda cfg: {"sst2": (fake["sst2"], fake["sst2"][:4])})
    out = finetune(_cfg(tmp_path, run, method="full", task="sup", lr=1e-3, epochs=1, sup_batch=4,
                        tokenizer=str(tmp_path / "t.json"), sup_max_len=32), device="cpu")
    r = json.loads((out / "results.json").read_text())
    assert "sst2_acc" in r["after"] and 0.0 <= r["after"]["sup_acc_mean"] <= 1.0
    assert r["after"]["sst2_n"] == 4


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_lora_alpha_defaults_to_twice_rank(tmp_path):
    run = _parent(tmp_path)
    out = finetune(_cfg(tmp_path, run, method="lora", rank=2, alpha=None, task="code", lr=1e-2),
                    device="cpu")
    r = json.loads((out / "results.json").read_text())
    assert r["alpha"] == 4.0
    sd = torch.load(out / "lora.pt", weights_only=False)
    assert sd["blocks.0.attn.q"]["scale"] == 2.0


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_code_ft_raises_when_stream_too_small(tmp_path):
    run = _parent(tmp_path)
    cfg = _cfg(tmp_path, run, method="full", task="code", lr=1e-3)
    cfg.train_tokens = 128 * 10_000  # far more windows than the tiny fixture stream has
    with pytest.raises(ValueError, match="code stream has"):
        finetune(cfg, device="cpu")


def test_bucketed_batches_cut_padding_versus_plain_permutation():
    rng = random.Random(0)
    # 3200 = 2 * (50 * 32): an exact multiple of both the mega-batch and the batch
    # size, so nothing is dropped and bucketed/unbucketed cover the same index set.
    examples = [([0] * rng.randint(5, 200), []) for _ in range(3200)]

    def padding_fraction(batches):
        pad, padded = 0, 0
        for b in batches:
            lens = [len(examples[j][0]) for j in b]
            m = max(lens)
            pad += sum(m - length for length in lens)
            padded += m * len(lens)
        return pad / padded

    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(examples), generator=g).tolist()
    bucketed = _bucketed_batches(examples, order, batch_size=32, generator=g)
    unbucketed = [order[i:i + 32] for i in range(0, len(order) - 32 + 1, 32)]

    assert sorted(j for b in bucketed for j in b) == sorted(j for b in unbucketed for j in b)
    assert padding_fraction(bucketed) < padding_fraction(unbucketed)
