import json
from pathlib import Path

import pytest

from rankfile.config import from_dict, load_yaml
from rankfile.model import ModelConfig
from rankfile.train import TrainConfig, train

pytestmark = pytest.mark.gpu

@pytest.mark.skipif(not Path("data/fineweb_edu_smoke").exists(), reason="smoke shards not built (Plan 2 Task 3)")
def test_m30_smoke_end_to_end(tmp_path):
    tc = from_dict(TrainConfig, load_yaml("configs/train/smoke.yaml")); tc.out_root = str(tmp_path)
    mc = from_dict(ModelConfig, load_yaml("configs/model/m30.yaml"))
    run = train(tc, mc)
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [l for l in lines if "loss" in l]
    assert (run / "DONE").exists() and len(steps) == 20
    assert steps[-1]["loss"] < steps[0]["loss"]
    assert max(l["mem_gib"] for l in steps) < 14.0
    assert steps[-1]["tok_per_s"] > 50_000, steps[-1]
