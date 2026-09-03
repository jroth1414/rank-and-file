"""Pretraining loop: gradient accumulation, WSD schedule, eval, resumable checkpoints, compile."""
from __future__ import annotations

import argparse  # noqa: F401 used in Task 3
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from rankfile.checkpoint import (  # noqa: F401 used in Task 3
    load_checkpoint,
    read_latest,
    save_checkpoint,
    unwrap,
    write_latest,
)
from rankfile.config import (  # noqa: F401 used in Task 3
    apply_overrides,
    from_dict,
    load_yaml,
    to_yaml,
)
from rankfile.data import (  # noqa: F401 used in Task 3
    FixedOrderSampler,
    TokenStream,
    list_shards,
    make_batch,
)
from rankfile.model import ModelConfig, Transformer, doc_block_mask  # noqa: F401 used in Task 3
from rankfile.optim.build import build_optimizers, set_lr  # noqa: F401 used in Task 3
from rankfile.schedule import wsd_lr  # noqa: F401 used in Task 3


@dataclass
class TrainConfig:
    name: str = ""                 # run name, e.g. p2_muon_m124_s0
    arm: str = "p1"
    optimizer: str = "adamw"       # adamw | muon
    peak_lr: float = 2e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    total_tokens: int = 2_500_000_000
    batch_tokens: int = 524_288
    micro_batch: int = 8
    seq_len: int = 2048
    seed: int = 0
    data_dir: str = "data/fineweb_edu"
    warmup_frac: float = 0.02
    decay_frac: float = 0.2
    grad_clip: float = 1.0
    eval_every_tokens: int = 50_000_000
    eval_windows: int = 512        # 512 windows x 2048 = 1M val tokens
    ckpt_every_minutes: float = 60.0
    keep_every_tokens: int = 250_000_000   # permanent checkpoints for later analysis
    compile: bool = True
    use_doc_mask: bool = True
    mem_ceiling_gib: float = 14.0
    log_every_steps: int = 10
    out_root: str = "runs"


def derived(cfg: TrainConfig) -> dict:
    per_micro = cfg.micro_batch * cfg.seq_len
    msg = "batch_tokens must be a multiple of micro_batch*seq_len"
    assert cfg.batch_tokens % per_micro == 0, msg
    steps = cfg.total_tokens // cfg.batch_tokens
    accum = cfg.batch_tokens // per_micro
    return {"steps_total": steps, "accum": accum}


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def init_run_dir(cfg: TrainConfig, model_cfg: ModelConfig) -> Path:
    run = Path(cfg.out_root) / cfg.name
    run.mkdir(parents=True, exist_ok=True)
    to_yaml(cfg, run / "config.resolved.yaml")
    to_yaml(model_cfg, run / "model.yaml")
    (run / "git.txt").write_text(_git_hash() + "\n", encoding="utf-8")
    return run


class MetricsLog:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "metrics.jsonl"

    def write(self, **kv) -> None:
        kv.setdefault("time", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv) + "\n")


@torch.no_grad()
def evaluate(
    loss_fn,
    val_stream: TokenStream,
    seq_len: int,
    micro_batch: int,
    n_windows: int,
    device,
    use_doc_mask: bool = True,
) -> float:
    sampler = FixedOrderSampler(val_stream.total_tokens, seq_len, seed=0)
    n_windows = min(n_windows, sampler.n_windows)
    total, count = 0.0, 0
    for pos in range(0, n_windows - micro_batch + 1, micro_batch):
        x, y, d = make_batch(val_stream, sampler, pos, micro_batch, seq_len, device)
        bm = doc_block_mask(d) if use_doc_mask else None
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(str(device).startswith("cuda"))
        ):
            total += loss_fn(x, y, block_mask=bm).item()
        count += 1
    return total / max(1, count)
