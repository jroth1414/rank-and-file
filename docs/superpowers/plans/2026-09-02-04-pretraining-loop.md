# Plan 4: Pretraining Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python -m rankfile.train` trains any arm from a model config plus a train config, with gradient accumulation, WSD schedule, JSONL metrics, periodic validation, hourly resumable checkpoints, a memory-ceiling guard, and `torch.compile`; plus a sequential run queue and the arm configs.

**Architecture:** `checkpoint.py` owns the on-disk format (model + optimizers + step + data position + RNG). `train.py` owns `TrainConfig`, the loop, evaluation, and the CLI. `scripts/queue.py` runs a text file of commands sequentially, skipping runs whose `DONE` marker exists. The loop builds the FlexAttention block mask outside the compiled loss so compile sees a plain tensor argument.

**Tech Stack:** torch 2.14 (`torch.compile`, autocast bf16, `clip_grad_norm_`), pyyaml, pytest.

**Spec:** `proposal.md` §4.2 (schedule, batch ≈0.5M tokens, arms), §5 (micro-batch 8, ≤14 GiB, hourly checkpoints). `CLAUDE.md` §5 (WDDM trap), §6 (run dir contents, naming, checkpoints), §7 (`test_resume.py`, `test_smoke.py`), §8 (smoke before scale, ask before long runs).

## Global Constraints

- Python 3.11 in `.venv`; run as `.venv\Scripts\python.exe ...`.
- Micro-batch 8 × 2048 on m124; accumulate to 524,288 tokens per optimizer step (32 micro-steps). Assert `max_memory_allocated ≤ 14 GiB` after the first few steps.
- Every run dir has `config.resolved.yaml`, `model.yaml`, `git.txt`, `metrics.jsonl`, checkpoints, `latest.txt`, and `DONE` on completion.
- Resume must be exact: same loss trajectory as an uninterrupted run (bitwise on CPU eager).
- Never launch a multi-hour run without asking the user. The m30 smoke run is the only run this plan launches unprompted.
- Commit prefix `train:`; no AI attribution trailers.

## Notes from the Plan 1 final review (read before Task 3)

- `doc_block_mask` at B=8, T=2048 costs about 4.5 ms and a 320 MiB transient per call (measured 2026-09-02). Over 32 micro-steps that is ~144 ms per optimizer step, about 2%. Acceptable, but pass `_compile=True` to `create_block_mask` inside `doc_block_mask` if profiling shows it matters, and never build the mask inside the compiled loss.
- `Transformer.forward` materializes full `[B,T,V]` logits (8.6 GB fp32 at the m124 shape). Only `Transformer.loss` is memory-safe at scale; evaluation must go through `loss`.
- `Transformer.loss` accepts `ignore_index: int = -100` and divides by the count of non-ignored targets; pretraining never uses it, Plan 5's supervised task does.
- Run `.venv\Scripts\python.exe -m ruff check .` before every commit; `src/` is held to 100 columns, `tests/` and `scripts/` have E501/E702/E401 ignored.

## Notes from the Plan 2 final review (read before Task 1 and Task 3)

- The data permutation depends on `(total_tokens, seq_len, seed)`. If a shard is added, truncated, or rebuilt, `total_tokens` changes and the whole order changes silently. **`save_checkpoint` must store `train_stream.total_tokens`, `cfg.seed`, and `cfg.seq_len` in `extra`, and `train()` must raise on resume if any of them differ** from the live stream and config. Add a test to `test_resume.py`: resuming against a directory with one extra shard raises.
- Data exhaustion: `FixedOrderSampler.start` raises `IndexError` past `n_windows`. That is intended (explicit failure beats silent wraparound). The smoke directory has only 24,403 windows (~50M tokens); the smoke config's 20 steps × 32 micro-batches × 8 = 5,120 windows is fine.
- Targets at EOT positions (the next document's first token) stay in the loss. No `ignore_index` for pretraining; the fraction is ~0.1% and identical across arms.
- `data/fineweb_edu/manifest.json` (written by `prepare_data.py`) records per-shard token counts and the tokenizer hash; `TokenStream` verifies shard sizes against it when present. A mismatch raises before training starts.

## Consumed interfaces (from Plans 1–3)

- `rankfile.config`: `load_yaml`, `apply_overrides`, `from_dict`, `to_yaml`.
- `rankfile.model`: `ModelConfig`, `Transformer` with `.loss(idx, targets, doc_ids=None, block_mask=None)`, `doc_block_mask(doc_ids)`, `.num_params()`.
- `rankfile.data`: `list_shards(dir, split)`, `TokenStream(paths)`, `FixedOrderSampler(total_tokens, seq_len, seed)`, `make_batch(stream, sampler, position, micro_batch, seq_len, device) -> (x, y, doc_ids)`.
- `rankfile.optim.build`: `build_optimizers(model, name, lr, weight_decay, betas) -> list`, `set_lr(opts, lr)`, `optimizer_state_dicts(opts)`, `load_optimizer_state_dicts(opts, sds)`.
- `rankfile.schedule.wsd_lr(step, total_steps, peak_lr, warmup_frac=0.02, decay_frac=0.2, final_ratio=0.0)`; warmup is at least one step whenever `warmup_frac > 0`.
- `rankfile.optim.build.build_optimizers(model, name, lr, weight_decay, betas=(0.9, 0.95), adamw_lr_scale=1.0)`; `set_lr(opts, lr)` multiplies each group's stored `lr_scale`, so call it with the schedule's absolute value every step.

---

### Task 1: Checkpoint format

**Files:**
- Create: `src/rankfile/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Produces: `save_checkpoint(path: Path, model, opts, step: int, position: int, tokens_seen: int, extra: dict) -> None`; `load_checkpoint(path: Path, model, opts=None) -> dict` returning `{"step", "position", "tokens_seen", "extra"}` and restoring model, optimizers, and RNG; `unwrap(model)` returning `model._orig_mod` if compiled else `model`; `write_latest(run_dir, ckpt_name)`, `read_latest(run_dir) -> Path | None`; `load_model_state(path) -> dict[str, Tensor]` (just the weights, for analysis).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checkpoint.py
import torch
from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers
from rankfile.checkpoint import save_checkpoint, load_checkpoint, write_latest, read_latest, load_model_state

def _mk():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(vocab_size=64, n_layer=1, d_model=16, n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=8))
    return m, build_optimizers(m, "muon", lr=1e-3, weight_decay=0.1)

def test_roundtrip_restores_everything(tmp_path):
    m, opts = _mk()
    x = torch.randint(0, 64, (2, 8))
    m.loss(x, x).backward(); [o.step() for o in opts]
    torch.manual_seed(123); torch.rand(1)  # advance RNG so state is non-trivial
    p = tmp_path / "ckpt_0000001.pt"
    save_checkpoint(p, m, opts, step=1, position=16, tokens_seen=32, extra={"note": "hi"})
    r1 = torch.rand(3)
    m2, opts2 = _mk()
    meta = load_checkpoint(p, m2, opts2)
    assert meta["step"] == 1 and meta["position"] == 16 and meta["tokens_seen"] == 32 and meta["extra"]["note"] == "hi"
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(torch.rand(3), r1)  # RNG restored to the state at save time
    buf = opts2[0].state[next(iter(opts2[0].param_groups[0]["params"]))]["momentum_buffer"]
    assert buf.abs().sum() > 0

def test_latest_pointer(tmp_path):
    assert read_latest(tmp_path) is None
    write_latest(tmp_path, "ckpt_0000005.pt")
    assert read_latest(tmp_path) == tmp_path / "ckpt_0000005.pt"

def test_load_model_state_only(tmp_path):
    m, opts = _mk()
    p = tmp_path / "c.pt"; save_checkpoint(p, m, opts, 0, 0, 0, {})
    sd = load_model_state(p)
    assert "blocks.0.attn.q.weight" in sd and sd["embed.weight"].shape == (64, 16)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/checkpoint.py
"""Checkpoint format: model, optimizers, step, data position, RNG. Exact resume."""
from __future__ import annotations

from pathlib import Path

import torch

from rankfile.optim.build import load_optimizer_state_dicts, optimizer_state_dicts


def unwrap(model):
    return getattr(model, "_orig_mod", model)


def save_checkpoint(path: str | Path, model, opts, step: int, position: int, tokens_seen: int, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap(model).state_dict(),
        "optimizers": optimizer_state_dicts(opts),
        "step": step, "position": position, "tokens_seen": tokens_seen, "extra": extra,
        "rng": {"cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic on the same volume; a crash mid-write leaves the old file intact


def load_checkpoint(path: str | Path, model, opts=None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(state["model"])
    if opts is not None:
        load_optimizer_state_dicts(opts, state["optimizers"])
    torch.set_rng_state(state["rng"]["cpu"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return {k: state[k] for k in ("step", "position", "tokens_seen", "extra")}


def load_model_state(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def write_latest(run_dir: str | Path, ckpt_name: str) -> None:
    (Path(run_dir) / "latest.txt").write_text(ckpt_name, encoding="utf-8")


def read_latest(run_dir: str | Path) -> Path | None:
    p = Path(run_dir) / "latest.txt"
    if not p.exists():
        return None
    return Path(run_dir) / p.read_text(encoding="utf-8").strip()
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_checkpoint.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/checkpoint.py tests/test_checkpoint.py
git commit -m "train: atomic checkpoint save/load with optimizer, position, and RNG state"
```

---

### Task 2: TrainConfig, run directory, metrics logger, evaluation

**Files:**
- Create: `src/rankfile/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `TrainConfig` dataclass (fields below); `derived(cfg) -> dict` with `steps_total`, `accum`; `init_run_dir(cfg, model_cfg) -> Path` writing `config.resolved.yaml`, `model.yaml`, `git.txt`; `MetricsLog(run_dir)` with `.write(**kv)` appending one JSON line; `evaluate(model_loss_fn, val_stream, seq_len, micro_batch, n_windows, device) -> float` (mean loss over the first `n_windows` sequential windows of the val stream, with doc masks).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train.py
import json, numpy as np, torch
from pathlib import Path
from rankfile.train import TrainConfig, derived, init_run_dir, MetricsLog, evaluate
from rankfile.model import ModelConfig, Transformer
from rankfile.data import write_shard, TokenStream

def test_derived_steps_and_accum():
    cfg = TrainConfig(total_tokens=524_288 * 10, batch_tokens=524_288, micro_batch=8, seq_len=2048)
    d = derived(cfg)
    assert d["steps_total"] == 10 and d["accum"] == 32

def test_init_run_dir_writes_resolved_configs(tmp_path):
    cfg = TrainConfig(name="t", out_root=str(tmp_path))
    run = init_run_dir(cfg, ModelConfig(n_layer=1))
    assert (run / "config.resolved.yaml").exists() and (run / "model.yaml").exists() and (run / "git.txt").exists()
    log = MetricsLog(run); log.write(step=1, loss=2.0); log.write(step=2, loss=1.5)
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert lines[1]["loss"] == 1.5 and "time" in lines[0]

def test_evaluate_is_deterministic_and_finite(tmp_path):
    write_shard(np.random.default_rng(0).integers(1, 64, 5000).astype(np.uint16), tmp_path / "val_0000.bin")
    vs = TokenStream([tmp_path / "val_0000.bin"])
    m = Transformer(ModelConfig(vocab_size=64, n_layer=1, d_model=16, n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=16))
    a = evaluate(m.loss, vs, seq_len=16, micro_batch=4, n_windows=8, device="cpu")
    b = evaluate(m.loss, vs, seq_len=16, micro_batch=4, n_windows=8, device="cpu")
    assert a == b and np.isfinite(a) and 3.0 < a < 6.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/train.py
"""Pretraining loop: gradient accumulation, WSD schedule, eval, resumable checkpoints, compile."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rankfile.checkpoint import load_checkpoint, read_latest, save_checkpoint, unwrap, write_latest
from rankfile.config import apply_overrides, from_dict, load_yaml, to_yaml
from rankfile.data import FixedOrderSampler, TokenStream, list_shards, make_batch
from rankfile.model import ModelConfig, Transformer, doc_block_mask
from rankfile.optim.build import build_optimizers, set_lr
from rankfile.schedule import wsd_lr


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
    assert cfg.batch_tokens % per_micro == 0, "batch_tokens must be a multiple of micro_batch*seq_len"
    return {"steps_total": cfg.total_tokens // cfg.batch_tokens, "accum": cfg.batch_tokens // per_micro}


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
def evaluate(loss_fn, val_stream: TokenStream, seq_len: int, micro_batch: int, n_windows: int, device,
             use_doc_mask: bool = True) -> float:
    sampler = FixedOrderSampler(val_stream.total_tokens, seq_len, seed=0)
    n_windows = min(n_windows, sampler.n_windows)
    total, count = 0.0, 0
    for pos in range(0, n_windows - micro_batch + 1, micro_batch):
        x, y, d = make_batch(val_stream, sampler, pos, micro_batch, seq_len, device)
        bm = doc_block_mask(d) if use_doc_mask else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(str(device).startswith("cuda"))):
            total += loss_fn(x, y, block_mask=bm).item()
        count += 1
    return total / max(1, count)
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/train.py tests/test_train.py
git commit -m "train: TrainConfig, run directory, JSONL metrics, validation loop"
```

---

### Task 3: The training loop with resume

**Files:**
- Modify: `src/rankfile/train.py`
- Test: `tests/test_train.py` (append), `tests/test_resume.py`

**Interfaces:**
- Produces: `train(cfg: TrainConfig, model_cfg: ModelConfig, device: str | None = None, max_steps: int | None = None) -> Path` (run dir). Resumes automatically from `latest.txt` if present. Writes `DONE` when `step == steps_total`. `max_steps` stops early without writing `DONE` (used by tests).
- Checkpoint policy: save when `ckpt_every_minutes` elapsed, when `tokens_seen % keep_every_tokens == 0` (permanent, named `ckpt_{step:07d}.pt`), and at the end. Non-permanent checkpoints older than the two most recent are deleted by the loop (this is the loop's own rotation, permitted by CLAUDE.md §8 rule 4 because it is designed behavior, not an agent deleting results).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train.py (append)
def _tiny_data(tmp_path, n=40000):
    rng = np.random.default_rng(0)
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
                     keep_every_tokens=32 * 4 * 20, **kw)
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
```

```python
# tests/test_resume.py
import json, numpy as np, torch
from rankfile.train import train
from tests.test_train import _tiny_data, _tiny_cfgs

def test_resume_matches_uninterrupted_run_bitwise(tmp_path):
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path)
    tc.name = "straight"; run_a = train(tc, mc, device="cpu")
    tc.name = "resumed"
    run_b = train(tc, mc, device="cpu", max_steps=17)      # stops after 17 steps, checkpoint written
    assert not (run_b / "DONE").exists()
    run_b = train(tc, mc, device="cpu")                    # resumes from latest.txt
    assert (run_b / "DONE").exists()
    la = [json.loads(l) for l in (run_a / "metrics.jsonl").read_text().splitlines() if "loss" in l]
    lb = [json.loads(l) for l in (run_b / "metrics.jsonl").read_text().splitlines() if "loss" in l]
    assert [l["step"] for l in la] == [l["step"] for l in lb]
    assert [l["loss"] for l in la] == [l["loss"] for l in lb]
    sa = torch.load(run_a / "ckpt_0000040.pt", weights_only=False)["model"]
    sb = torch.load(run_b / "ckpt_0000040.pt", weights_only=False)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train.py tests/test_resume.py -v`
Expected: 3 FAIL with `ImportError: train`

- [ ] **Step 3: Implement** (append to `train.py`)

```python
def _permanent(tokens_seen: int, cfg: TrainConfig) -> bool:
    return cfg.keep_every_tokens > 0 and tokens_seen % cfg.keep_every_tokens == 0


def _rotate_checkpoints(run: Path, cfg: TrainConfig, keep_recent: int = 2) -> None:
    ckpts = sorted(run.glob("ckpt_*.pt"))
    temp = [p for p in ckpts if not _permanent(int(p.stem.split("_")[1]) * cfg.batch_tokens, cfg)]
    for p in temp[:-keep_recent]:
        p.unlink()


def _save(run: Path, cfg: TrainConfig, model, opts, step: int, position: int, tokens_seen: int) -> None:
    name = f"ckpt_{step:07d}.pt"
    save_checkpoint(run / name, model, opts, step, position, tokens_seen, {"arm": cfg.arm, "optimizer": cfg.optimizer})
    write_latest(run, name)
    _rotate_checkpoints(run, cfg)


def train(cfg: TrainConfig, model_cfg: ModelConfig, device: str | None = None, max_steps: int | None = None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = str(device).startswith("cuda")
    d = derived(cfg)
    run = init_run_dir(cfg, model_cfg)
    log = MetricsLog(run)
    torch.manual_seed(cfg.seed)
    model = Transformer(model_cfg).to(device)
    opts = build_optimizers(model, cfg.optimizer, cfg.peak_lr, cfg.weight_decay, (cfg.beta1, cfg.beta2))
    train_stream = TokenStream(list_shards(cfg.data_dir, "train"))
    val_stream = TokenStream(list_shards(cfg.data_dir, "val"))
    sampler = FixedOrderSampler(train_stream.total_tokens, cfg.seq_len, cfg.seed)
    step, position, tokens_seen = 0, 0, 0
    latest = read_latest(run)
    if latest is not None and latest.exists():
        meta = load_checkpoint(latest, model, opts)
        step, position, tokens_seen = meta["step"], meta["position"], meta["tokens_seen"]
        log.write(event="resume", step=step, tokens=tokens_seen, ckpt=latest.name)
    loss_fn = torch.compile(model.loss, dynamic=False) if (cfg.compile and on_cuda) else model.loss
    total, nonemb = model.num_params()
    log.write(event="start", params=total, non_embedding=nonemb, steps_total=d["steps_total"], accum=d["accum"], device=str(device))
    last_ckpt_time = time.time()
    next_eval = ((tokens_seen // cfg.eval_every_tokens) + 1) * cfg.eval_every_tokens
    t_log = time.time()
    stop_at = d["steps_total"] if max_steps is None else min(d["steps_total"], step + max_steps)
    while step < stop_at:
        lr = wsd_lr(step, d["steps_total"], cfg.peak_lr, cfg.warmup_frac, cfg.decay_frac)
        set_lr(opts, lr)
        loss_acc = torch.zeros((), device=device)
        for _ in range(d["accum"]):
            x, y, dids = make_batch(train_stream, sampler, position, cfg.micro_batch, cfg.seq_len, device)
            position += cfg.micro_batch
            bm = doc_block_mask(dids) if cfg.use_doc_mask else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_cuda):
                loss = loss_fn(x, y, block_mask=bm) / d["accum"]
            loss.backward()
            loss_acc += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += cfg.batch_tokens
        if step % cfg.log_every_steps == 0 or step == 1:
            now = time.time()
            mem = torch.cuda.max_memory_allocated() / 2**30 if on_cuda else 0.0
            log.write(step=step, tokens=tokens_seen, loss=loss_acc.item(), lr=lr, grad_norm=float(gnorm),
                      tok_per_s=cfg.batch_tokens * cfg.log_every_steps / max(1e-9, now - t_log), mem_gib=mem)
            t_log = now
            if on_cuda and step >= 3 and mem > cfg.mem_ceiling_gib:
                raise RuntimeError(f"peak memory {mem:.1f} GiB exceeds ceiling {cfg.mem_ceiling_gib}; WDDM would page to RAM")
        if tokens_seen >= next_eval:
            vl = evaluate(loss_fn, val_stream, cfg.seq_len, cfg.micro_batch, cfg.eval_windows, device, cfg.use_doc_mask)
            log.write(step=step, tokens=tokens_seen, val_loss=vl)
            next_eval += cfg.eval_every_tokens
        due = (time.time() - last_ckpt_time) / 60 >= cfg.ckpt_every_minutes
        if due or _permanent(tokens_seen, cfg) or step == stop_at:
            _save(run, cfg, model, opts, step, position, tokens_seen)
            last_ckpt_time = time.time()
    if step >= d["steps_total"]:
        vl = evaluate(loss_fn, val_stream, cfg.seq_len, cfg.micro_batch, cfg.eval_windows, device, cfg.use_doc_mask)
        log.write(step=step, tokens=tokens_seen, val_loss=vl, event="final")
        (run / "DONE").write_text(f"{step} {tokens_seen} {vl:.6f}\n", encoding="utf-8")
    return run
```

Note on determinism for the resume test: on CPU eager with no dropout, AdamW (non-fused) and Muon are deterministic, so straight and resumed trajectories match bitwise. `build_optimizers` only sets `fused=True` when CUDA is available, so CPU uses the deterministic path.

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train.py tests/test_resume.py -v`
Expected: 6 passed (the tiny runs take ~20 s each)

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/train.py tests/test_train.py tests/test_resume.py
git commit -m "train: accumulation loop with WSD, eval, hourly checkpoints, exact resume"
```

---

### Task 4: CLI

**Files:**
- Modify: `src/rankfile/train.py` (add `main`)
- Test: `tests/test_train.py` (append)

**Interfaces:**
- Produces: `python -m rankfile.train --model configs/model/m124.yaml --train configs/train/muon.yaml --seed 0 [--name NAME] [--set key=value ...] [--max-steps N] [--device cpu]`. If `--name` is omitted it is `{arm}_{optimizer}_{model_basename}_s{seed}`.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_builds_name_and_runs(tmp_path, monkeypatch):
    import sys
    from rankfile.train import main
    from rankfile.config import to_yaml
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path); tc.name = ""
    to_yaml(mc, tmp_path / "m30.yaml"); to_yaml(tc, tmp_path / "adamw.yaml")
    monkeypatch.setattr(sys, "argv", ["train", "--model", str(tmp_path / "m30.yaml"), "--train", str(tmp_path / "adamw.yaml"),
                                      "--seed", "1", "--device", "cpu", "--max-steps", "2", "--set", "log_every_steps=1"])
    run = main()
    assert run.name == "p1_adamw_m30_s1" and (run / "metrics.jsonl").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train.py::test_cli_builds_name_and_runs -v`
Expected: FAIL with `ImportError: main`

- [ ] **Step 3: Implement** (append)

```python
def main() -> Path:
    ap = argparse.ArgumentParser(description="rank-and-file pretraining")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[], help="override train config key=value")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    model_cfg = from_dict(ModelConfig, load_yaml(a.model))
    td = apply_overrides(load_yaml(a.train), a.set)
    cfg = from_dict(TrainConfig, td)
    if a.seed is not None:
        cfg.seed = a.seed
    if a.name:
        cfg.name = a.name
    if not cfg.name:
        cfg.name = f"{cfg.arm}_{cfg.optimizer}_{Path(a.model).stem}_s{cfg.seed}"
    return train(cfg, model_cfg, device=a.device, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/train.py tests/test_train.py
git commit -m "train: CLI with config overrides and derived run names"
```

---

### Task 5: Train configs and the m30 smoke run (GPU)

**Files:**
- Create: `configs/train/smoke.yaml`, `configs/train/adamw.yaml`, `configs/train/muon.yaml`, `configs/train/p3_adamw.yaml`, `configs/train/sweep_adamw.yaml`, `configs/train/sweep_muon.yaml`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: the config files. Peak learning rates in `adamw.yaml`/`muon.yaml` are placeholders equal to the middle of the sweep grid and **must be replaced with the sweep winner before P1/P2 launch** (record in CLAUDE.md decision log).

- [ ] **Step 1: Write the configs**

```yaml
# configs/train/smoke.yaml — m30 on the 50M-token smoke shards; ~2 minutes on the 5070 Ti.
name: smoke_adamw_m30_s0
arm: smoke
optimizer: adamw
peak_lr: 3.0e-3
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
total_tokens: 10485760        # 20 steps
batch_tokens: 524288
micro_batch: 8
seq_len: 2048
seed: 0
data_dir: data/fineweb_edu_smoke
warmup_frac: 0.1
decay_frac: 0.2
grad_clip: 1.0
eval_every_tokens: 5242880
eval_windows: 64
ckpt_every_minutes: 1.0
keep_every_tokens: 5242880
compile: true
use_doc_mask: true
mem_ceiling_gib: 14.0
log_every_steps: 1
out_root: runs
```

```yaml
# configs/train/adamw.yaml — arm P1. peak_lr is the sweep placeholder; replace with the sweep winner.
name: ""
arm: p1
optimizer: adamw
peak_lr: 2.0e-3
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
total_tokens: 2500000000
batch_tokens: 524288
micro_batch: 8
seq_len: 2048
seed: 0
data_dir: data/fineweb_edu
warmup_frac: 0.02
decay_frac: 0.2
grad_clip: 1.0
eval_every_tokens: 50000000
eval_windows: 512
ckpt_every_minutes: 60.0
keep_every_tokens: 250000000
compile: true
use_doc_mask: true
mem_ceiling_gib: 14.0
log_every_steps: 10
out_root: runs
```

`configs/train/muon.yaml`: identical to `adamw.yaml` except `arm: p2`, `optimizer: muon`.
`configs/train/p3_adamw.yaml`: identical to `adamw.yaml` except `arm: p3`, `total_tokens: 3750000000`.
`configs/train/sweep_adamw.yaml`: identical to `adamw.yaml` except `arm: sweep`, `total_tokens: 200000000`, `keep_every_tokens: 0`, `eval_every_tokens: 25000000`.
`configs/train/sweep_muon.yaml`: same as `sweep_adamw.yaml` with `optimizer: muon`.

- [ ] **Step 2: Write the smoke test**

```python
# tests/test_smoke.py
import json, pytest, torch
from pathlib import Path
from rankfile.config import load_yaml, from_dict
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
```

- [ ] **Step 3: Run the smoke test**

Run: `.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v -s`
Expected: PASS in a few minutes. If `tok_per_s` is below 50k on m30, something is wrong with compile or the doc mask; do not proceed to Task 6 until fixed.

- [ ] **Step 4: Commit**

```bash
git add configs/train/ tests/test_smoke.py
git commit -m "train: arm, sweep, and smoke configs; m30 end-to-end GPU smoke test"
```

---

### Task 6: m124 throughput check and run queue

**Files:**
- Create: `scripts/queue.py`, `configs/queue/sweep.txt`, `configs/queue/core.txt`
- Test: `tests/test_queue.py`

**Interfaces:**
- Produces: `python scripts/queue.py configs/queue/sweep.txt` runs each non-comment line as a shell command in order; parses `--name X` from the line; skips if `runs/X/DONE` exists; on non-zero exit retries once (the run resumes from its checkpoint) and then moves on, logging to `runs/queue.log`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_queue.py
import sys
from pathlib import Path
from scripts.queue import run_queue

def test_queue_skips_done_and_retries_once(tmp_path):
    (tmp_path / "runs" / "done_run").mkdir(parents=True); (tmp_path / "runs" / "done_run" / "DONE").write_text("x")
    py = sys.executable
    lines = [
        f'{py} -c "print(1)" --name done_run',
        f'{py} -c "import sys; sys.exit(1)" --name failing',
        f'{py} -c "open(r\'{tmp_path}/ok.txt\',\'w\').write(\'hi\')" --name ok',
    ]
    q = tmp_path / "q.txt"; q.write_text("# comment\n" + "\n".join(lines) + "\n")
    summary = run_queue(q, runs_root=tmp_path / "runs")
    assert summary == {"done_run": "skipped", "failing": "failed", "ok": "ok"}
    assert (tmp_path / "ok.txt").read_text() == "hi"
    assert "failing" in (tmp_path / "runs" / "queue.log").read_text()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_queue.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/queue.py
"""Run commands from a text file sequentially; skip DONE runs; retry once on failure.

Usage: python scripts/queue.py configs/queue/core.txt
Each line: a full command containing --name <run_name>. Lines starting with # are ignored.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import time
from pathlib import Path


def _name(line: str) -> str:
    m = re.search(r"--name\s+(\S+)", line)
    if not m:
        raise ValueError(f"queue line has no --name: {line}")
    return m.group(1)


def run_queue(queue_file: Path, runs_root: Path = Path("runs")) -> dict[str, str]:
    runs_root.mkdir(parents=True, exist_ok=True)
    logf = open(runs_root / "queue.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    summary: dict[str, str] = {}
    for raw in Path(queue_file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = _name(line)
        if (runs_root / name / "DONE").exists():
            log(f"skip {name}: DONE exists"); summary[name] = "skipped"; continue
        status = "failed"
        for attempt in (1, 2):
            log(f"start {name} (attempt {attempt}): {line}")
            rc = subprocess.call(shlex.split(line, posix=False))
            if rc == 0:
                status = "ok"; log(f"ok {name}"); break
            log(f"exit {rc} for {name}")
        summary[name] = status
    logf.close()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("queue_file"); ap.add_argument("--runs-root", default="runs")
    a = ap.parse_args()
    print(run_queue(Path(a.queue_file), Path(a.runs_root)))
```

Queue files (the python path is the venv's):

```
# configs/queue/sweep.txt — 6 x 200M tokens, ~1 h each at 60k tok/s. Ask the user before launching.
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/sweep_adamw.yaml --seed 0 --name sweep_adamw_lr1e-3 --set peak_lr=1e-3
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/sweep_adamw.yaml --seed 0 --name sweep_adamw_lr2e-3 --set peak_lr=2e-3
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/sweep_adamw.yaml --seed 0 --name sweep_adamw_lr4e-3 --set peak_lr=4e-3
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/sweep_muon.yaml --seed 0 --name sweep_muon_lr1e-3 --set peak_lr=1e-3
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/sweep_muon.yaml --seed 0 --name sweep_muon_lr2e-3 --set peak_lr=2e-3
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/sweep_muon.yaml --seed 0 --name sweep_muon_lr4e-3 --set peak_lr=4e-3
```

```
# configs/queue/core.txt — P1 s0, P2 s0, P3. Set peak_lr in adamw.yaml/muon.yaml to the sweep winners first.
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/adamw.yaml --seed 0 --name p1_adamw_m124_s0
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/muon.yaml --seed 0 --name p2_muon_m124_s0
.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/p3_adamw.yaml --seed 0 --name p3_adamw_m124_s0
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_queue.py -v`
Expected: 1 passed

- [ ] **Step 5: m124 throughput check (5 steps, ~2 minutes)**

Run: `.venv\Scripts\python.exe -m rankfile.train --model configs/model/m124.yaml --train configs/train/adamw.yaml --seed 0 --name throughput_check --max-steps 5 --set log_every_steps=1`
Expected: the last `metrics.jsonl` line shows `tok_per_s` ≥ 60,000 and `mem_gib` < 14. Record the number in the commit body. Delete `runs/throughput_check` afterwards (it is a diagnostic, not a result).

- [ ] **Step 6: Commit**

```bash
git add scripts/queue.py configs/queue/ tests/test_queue.py
git commit -m "train: sequential run queue with DONE skipping and one retry; sweep and core queues"
```

---

### Task 7: Launch gate (no code)

- [ ] **Step 1:** Confirm with the user that `pytest` passes, the smoke test passed, and `data/fineweb_edu` is built (Plan 2 Task 4).
- [ ] **Step 2:** Ask: "Launch the learning-rate sweep queue? Six runs of 200M tokens, roughly 5 to 8 hours total on the 5070 Ti. Pause Windows Update first."
- [ ] **Step 3:** On approval: `.venv\Scripts\python.exe scripts/queue.py configs/queue/sweep.txt`. When done, read each run's final `val_loss` from `DONE`, pick the best `peak_lr` per optimizer, write them into `adamw.yaml`, `muon.yaml`, `p3_adamw.yaml`, and add a decision-log line in `CLAUDE.md` §11.
- [ ] **Step 4:** Ask: "Launch the core queue? P1, P2, P3 back to back, roughly 30 to 60 hours." On approval run `configs/queue/core.txt`.

---

## Self-review

- **Spec coverage:** batch 0.5M tokens with micro-batch 8 (Task 2 `derived`), WSD (Task 3), eval on FineWeb-Edu val (Task 2), hourly checkpoints and permanent ones every 250M tokens for later analysis (Task 3), memory ceiling assert (Task 3), compile (Task 3), run dir contents (Task 2), exact resume (Task 3 test), smoke before scale (Task 5), queue (Task 6), ask before long runs (Task 7), matched-loss arm P3 (Task 5 config).
- **Placeholders:** none. `peak_lr` placeholders in configs are explicitly labeled and resolved by Task 7 Step 3.
- **Type consistency:** `model.loss(x, y, block_mask=bm)` matches the Plan 1 amendment (block_mask kwarg). `build_optimizers` returns a list and the loop iterates it. `make_batch` signature matches Plan 2. Checkpoint keys `model`, `optimizers`, `step`, `position`, `tokens_seen` are what Plans 5 and 6 read via `load_model_state`.
