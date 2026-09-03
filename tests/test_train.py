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
