import json

import numpy as np

from rankfile.data import TokenStream, write_shard
from rankfile.model import ModelConfig, Transformer
from rankfile.train import MetricsLog, TrainConfig, derived, evaluate, init_run_dir


def test_derived_steps_and_accum():
    cfg = TrainConfig(total_tokens=524_288 * 10, batch_tokens=524_288, micro_batch=8, seq_len=2048)
    d = derived(cfg)
    assert d["steps_total"] == 10 and d["accum"] == 32

def test_init_run_dir_writes_resolved_configs(tmp_path):
    cfg = TrainConfig(name="t", out_root=str(tmp_path))
    run = init_run_dir(cfg, ModelConfig(n_layer=1))
    assert (run / "config.resolved.yaml").exists() and (run / "model.yaml").exists() and (run / "git.txt").exists()
    log = MetricsLog(run); log.write(step=1, loss=2.0); log.write(step=2, loss=1.5)
    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert lines[1]["loss"] == 1.5 and "time" in lines[0]

def test_evaluate_is_deterministic_and_finite(tmp_path):
    write_shard(np.random.default_rng(0).integers(1, 64, 5000).astype(np.uint16), tmp_path / "val_0000.bin")
    vs = TokenStream([tmp_path / "val_0000.bin"])
    m = Transformer(ModelConfig(vocab_size=64, n_layer=1, d_model=16, n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=16))
    a = evaluate(m.loss, vs, seq_len=16, micro_batch=4, n_windows=8, device="cpu")
    b = evaluate(m.loss, vs, seq_len=16, micro_batch=4, n_windows=8, device="cpu")
    assert a == b and np.isfinite(a) and 3.0 < a < 6.0

def _tiny_data(tmp_path, n=40000):
    # a learnable pattern: token t+1 = (t*7 + 3) % 61, with EOT every 50 tokens
    seq = [(i * 7 + 3) % 61 + 1 for i in range(n)]
    for i in range(0, n, 50): seq[i] = 0
    write_shard(np.array(seq, dtype=np.uint16), tmp_path / "train_0000.bin")
    write_shard(np.array(seq[:5000], dtype=np.uint16), tmp_path / "val_0000.bin")

def _tiny_cfgs(tmp_path, **kw):
    mc = ModelConfig(vocab_size=64, n_layer=2, d_model=32, n_head=2, n_kv_head=1, head_dim=16, d_ff=64, max_seq_len=32)
    tc = TrainConfig(name="tiny", optimizer=kw.pop("optimizer", "adamw"), peak_lr=3e-3, total_tokens=32 * 4 * 40,
                     batch_tokens=32 * 4, micro_batch=2, seq_len=32, data_dir=str(tmp_path), out_root=str(tmp_path / "runs"),
                     eval_every_tokens=32 * 4 * 20, eval_windows=8, compile=False, log_every_steps=1, ckpt_every_minutes=1e9,
                     keep_every_tokens=kw.pop("keep_every_tokens", 32 * 4 * 20), **kw)
    return mc, tc

def test_train_reduces_loss_and_writes_done(tmp_path):
    from rankfile.train import train
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path)
    run = train(tc, mc, device="cpu")
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    losses = [l["loss"] for l in lines if "loss" in l]
    assert losses[-1] < losses[0] * 0.9, (losses[0], losses[-1])
    assert (run / "DONE").exists() and any("val_loss" in l for l in lines)
    assert (run / "ckpt_0000020.pt").exists() and (run / "ckpt_0000040.pt").exists()

def test_muon_arm_trains(tmp_path):
    from rankfile.train import train
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path, optimizer="muon")
    run = train(tc, mc, device="cpu")
    assert (run / "DONE").exists()

def test_permanent_checkpoints_survive_rotation(tmp_path):
    # batch_tokens=128, keep_every_tokens=300 does not divide evenly (unlike the
    # real configs' 524288 into 250_000_000): this is the regression case for the
    # modulus bug, where `tokens_seen % keep_every_tokens == 0` never fires.
    from rankfile.train import train
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path, keep_every_tokens=300)
    assert tc.batch_tokens == 128
    run = train(tc, mc, device="cpu")
    expected = {
        s for s in range(1, 41)
        if (s * tc.batch_tokens) // tc.keep_every_tokens
        > ((s - 1) * tc.batch_tokens) // tc.keep_every_tokens
    }
    assert expected == {3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29, 31, 33, 36, 38, 40}
    for s in expected:
        assert (run / f"ckpt_{s:07d}.pt").exists(), s
    all_ckpts = sorted(run.glob("ckpt_*.pt"))
    non_permanent = [p for p in all_ckpts if int(p.stem.split("_")[1]) not in expected]
    assert len(non_permanent) <= 2, non_permanent

def test_rotate_checkpoints_removes_stale_tmp_files(tmp_path):
    from rankfile.train import _rotate_checkpoints
    run = tmp_path
    (run / "ckpt_0000001.tmp").write_bytes(b"stale")
    tc = TrainConfig(batch_tokens=128, keep_every_tokens=0)
    _rotate_checkpoints(run, tc)
    assert not (run / "ckpt_0000001.tmp").exists()

def test_init_run_dir_preserves_existing_config_on_resume(tmp_path):
    cfg = TrainConfig(name="t2", out_root=str(tmp_path), peak_lr=1e-3)
    run = init_run_dir(cfg, ModelConfig(n_layer=1))
    (run / "config.resolved.yaml").write_text("SENTINEL: true\n", encoding="utf-8")
    (run / "model.yaml").write_text("SENTINEL: true\n", encoding="utf-8")
    cfg2 = TrainConfig(name="t2", out_root=str(tmp_path), peak_lr=999.0)  # would differ if rewritten
    run2 = init_run_dir(cfg2, ModelConfig(n_layer=1))
    assert "SENTINEL" in (run2 / "config.resolved.yaml").read_text()
    assert "SENTINEL" in (run2 / "model.yaml").read_text()
    git_lines = (run2 / "git.txt").read_text().strip().splitlines()
    assert len(git_lines) == 2

def test_adamw_lr_scale_defaults_and_is_recorded(tmp_path):
    import yaml
    cfg = TrainConfig(name="t3", out_root=str(tmp_path))
    assert cfg.adamw_lr_scale == 1.0
    run = init_run_dir(cfg, ModelConfig(n_layer=1))
    resolved = yaml.safe_load((run / "config.resolved.yaml").read_text())
    assert resolved["adamw_lr_scale"] == 1.0

def test_cli_builds_name_and_runs(tmp_path, monkeypatch):
    import sys

    from rankfile.config import to_yaml
    from rankfile.train import main
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path); tc.name = ""
    to_yaml(mc, tmp_path / "m30.yaml"); to_yaml(tc, tmp_path / "adamw.yaml")
    monkeypatch.setattr(sys, "argv", ["train", "--model", str(tmp_path / "m30.yaml"), "--train", str(tmp_path / "adamw.yaml"),
                                      "--seed", "1", "--device", "cpu", "--max-steps", "2", "--set", "log_every_steps=1"])
    run = main()
    assert run.name == "p1_adamw_m30_s1" and (run / "metrics.jsonl").exists()
