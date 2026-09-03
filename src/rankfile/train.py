"""Pretraining loop: gradient accumulation, WSD schedule, eval, resumable checkpoints, compile."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from rankfile.checkpoint import (
    load_checkpoint,
    read_latest,
    save_checkpoint,
    write_latest,
)
from rankfile.config import (  # noqa: F401 used in Task 3
    apply_overrides,
    from_dict,
    load_yaml,
    to_yaml,
)
from rankfile.data import (
    FixedOrderSampler,
    TokenStream,
    list_shards,
    make_batch,
)
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


def _permanent(tokens_seen: int, cfg: TrainConfig) -> bool:
    return cfg.keep_every_tokens > 0 and tokens_seen % cfg.keep_every_tokens == 0


def _rotate_checkpoints(run: Path, cfg: TrainConfig, keep_recent: int = 2) -> None:
    ckpts = sorted(run.glob("ckpt_*.pt"))
    temp = [p for p in ckpts if not _permanent(int(p.stem.split("_")[1]) * cfg.batch_tokens, cfg)]
    for p in temp[:-keep_recent]:
        p.unlink()


def _save(
    run: Path,
    cfg: TrainConfig,
    model,
    opts,
    step: int,
    position: int,
    tokens_seen: int,
    data_total_tokens: int,
) -> None:
    name = f"ckpt_{step:07d}.pt"
    extra = {
        "arm": cfg.arm,
        "optimizer": cfg.optimizer,
        "total_tokens": data_total_tokens,
        "seed": cfg.seed,
        "seq_len": cfg.seq_len,
    }
    save_checkpoint(run / name, model, opts, step, position, tokens_seen, extra)
    write_latest(run, name)
    _rotate_checkpoints(run, cfg)


def train(
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    device: str | None = None,
    max_steps: int | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = str(device).startswith("cuda")
    d = derived(cfg)
    run = init_run_dir(cfg, model_cfg)
    log = MetricsLog(run)
    torch.manual_seed(cfg.seed)
    model = Transformer(model_cfg).to(device)
    opts = build_optimizers(
        model, cfg.optimizer, cfg.peak_lr, cfg.weight_decay, (cfg.beta1, cfg.beta2)
    )
    train_stream = TokenStream(list_shards(cfg.data_dir, "train"))
    val_stream = TokenStream(list_shards(cfg.data_dir, "val"))
    sampler = FixedOrderSampler(train_stream.total_tokens, cfg.seq_len, cfg.seed)
    step, position, tokens_seen = 0, 0, 0
    latest = read_latest(run)
    if latest is not None and latest.exists():
        meta = load_checkpoint(latest, model, opts)
        extra = meta["extra"]
        live = {
            "total_tokens": train_stream.total_tokens,
            "seed": cfg.seed,
            "seq_len": cfg.seq_len,
        }
        for field, live_value in live.items():
            ckpt_value = extra.get(field)
            if ckpt_value != live_value:
                raise RuntimeError(
                    f"data provenance mismatch on {field!r}: checkpoint has "
                    f"{ckpt_value!r}, live config/data has {live_value!r}"
                )
        step, position, tokens_seen = meta["step"], meta["position"], meta["tokens_seen"]
        log.write(event="resume", step=step, tokens=tokens_seen, ckpt=latest.name)
    loss_fn = torch.compile(model.loss, dynamic=False) if (cfg.compile and on_cuda) else model.loss
    total, nonemb = model.num_params()
    log.write(
        event="start", params=total, non_embedding=nonemb, steps_total=d["steps_total"],
        accum=d["accum"], device=str(device),
    )
    last_ckpt_time = time.time()
    next_eval = ((tokens_seen // cfg.eval_every_tokens) + 1) * cfg.eval_every_tokens
    t_log = time.time()
    log_step_start = step  # for a correct tok/s window on the first log line and after resume
    stop_at = d["steps_total"] if max_steps is None else min(d["steps_total"], step + max_steps)
    while step < stop_at:
        lr = wsd_lr(step, d["steps_total"], cfg.peak_lr, cfg.warmup_frac, cfg.decay_frac)
        set_lr(opts, lr)
        loss_acc = torch.zeros((), device=device)
        for _ in range(d["accum"]):
            x, y, dids = make_batch(
                train_stream, sampler, position, cfg.micro_batch, cfg.seq_len, device
            )
            position += cfg.micro_batch
            # FlexAttention (which doc masking requires) has no CPU backward in this
            # torch build; real training is always CUDA (CLAUDE.md hardware), so gate
            # doc masking on the training loop's own device rather than skip it there.
            bm = doc_block_mask(dids) if (cfg.use_doc_mask and on_cuda) else None
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
        # Read peak memory every step (not just on log steps) so the WDDM paging guard
        # fires immediately rather than up to log_every_steps-1 steps late.
        mem = torch.cuda.max_memory_allocated() / 2**30 if on_cuda else 0.0
        if on_cuda and step >= 3 and mem > cfg.mem_ceiling_gib:
            raise RuntimeError(
                f"peak memory {mem:.1f} GiB exceeds ceiling {cfg.mem_ceiling_gib}; "
                "WDDM would page to RAM"
            )
        if step % cfg.log_every_steps == 0 or step == 1:
            now = time.time()
            steps_elapsed = step - log_step_start
            log.write(
                step=step, tokens=tokens_seen, loss=loss_acc.item(), lr=lr, grad_norm=float(gnorm),
                tok_per_s=cfg.batch_tokens * steps_elapsed / max(1e-9, now - t_log),
                mem_gib=mem,
            )
            t_log = now
            log_step_start = step
        if tokens_seen >= next_eval:
            vl = evaluate(
                loss_fn, val_stream, cfg.seq_len, cfg.micro_batch, cfg.eval_windows, device,
                cfg.use_doc_mask,
            )
            log.write(step=step, tokens=tokens_seen, val_loss=vl)
            next_eval += cfg.eval_every_tokens
        due = (time.time() - last_ckpt_time) / 60 >= cfg.ckpt_every_minutes
        if due or _permanent(tokens_seen, cfg) or step == stop_at:
            _save(run, cfg, model, opts, step, position, tokens_seen, train_stream.total_tokens)
            last_ckpt_time = time.time()
    if step >= d["steps_total"]:
        vl = evaluate(
            loss_fn, val_stream, cfg.seq_len, cfg.micro_batch, cfg.eval_windows, device,
            cfg.use_doc_mask,
        )
        log.write(step=step, tokens=tokens_seen, val_loss=vl, event="final")
        (run / "DONE").write_text(f"{step} {tokens_seen} {vl:.6f}\n", encoding="utf-8")
    return run


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
