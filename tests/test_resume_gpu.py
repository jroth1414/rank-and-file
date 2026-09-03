"""GPU resume across a real process boundary: fused AdamW, torch.compile, FlexAttention
doc masking, all exercised together (the CPU resume test in test_resume.py runs with
compile off and no doc mask, so it never exercises this combination)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rankfile.config import from_dict, load_yaml
from rankfile.model import ModelConfig
from rankfile.train import TrainConfig, train

pytestmark = pytest.mark.gpu

SMOKE_DATA = Path("data/fineweb_edu_smoke")


@pytest.mark.skipif(not SMOKE_DATA.exists(), reason="smoke shards not built (Plan 2 Task 3)")
def test_resume_on_gpu_with_fused_adamw_compile_and_doc_mask(tmp_path):
    model_cfg = from_dict(ModelConfig, load_yaml("configs/model/m30.yaml"))
    train_cfg = from_dict(TrainConfig, load_yaml("configs/train/smoke.yaml"))
    train_cfg.name = "gpu_resume"
    train_cfg.out_root = str(tmp_path)
    train_cfg.total_tokens = 10 * 524_288  # 10 steps
    train_cfg.ckpt_every_minutes = 1e9  # only step==stop_at or permanent boundaries save

    run = train(train_cfg, model_cfg, max_steps=4)
    assert not (run / "DONE").exists()
    assert list(run.glob("ckpt_*.pt")), "expected a checkpoint at the max_steps boundary"

    run = train(train_cfg, model_cfg)  # resumes from latest.txt to completion
    assert (run / "DONE").exists()

    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    loss_lines = [line for line in lines if "loss" in line]
    steps = [line["step"] for line in loss_lines]
    assert steps == list(range(1, 11)), steps
    losses = [line["loss"] for line in loss_lines]
    print(f"\nGPU resume step losses: {losses}")
    assert all(math.isfinite(loss) for loss in losses), losses
    assert any(line.get("event") == "resume" for line in lines)
