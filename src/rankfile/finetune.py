"""Full fine-tuning and LoRA on a pretrained twin, for code CPT or the supervised bundle."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from rankfile.checkpoint import load_model_state, read_latest
from rankfile.config import apply_overrides, from_dict, load_yaml, to_yaml
from rankfile.data import FixedOrderSampler, TokenStream, list_shards, make_batch
from rankfile.lora import apply_lora, lora_state_dict
from rankfile.model import ModelConfig, Transformer, doc_block_mask
from rankfile.schedule import wsd_lr
from rankfile.tasks import (
    SUP_TASKS,
    collate_sup,
    encode_sup,
    format_example,
    load_sup_split,
    score_options,
    sup_loss,
)
from rankfile.tokenizer import load_tokenizer
from rankfile.train import MetricsLog, _git_hash, evaluate


@dataclass
class FinetuneConfig:
    parent: str = ""               # run dir of the pretrained twin
    method: str = "full"           # full | lora
    rank: int = 16
    alpha: float | None = None     # None resolves to 2 * rank at run time
    task: str = "code"             # code | sup
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_frac: float = 0.05
    decay_frac: float = 0.5
    grad_clip: float = 1.0
    # code task
    train_tokens: int = 100_000_000
    batch_tokens: int = 131_072
    micro_batch: int = 8
    seq_len: int = 2048
    data_dir_code: str = "data/code"
    # sup task
    epochs: int = 3
    sup_train_n: int = 8000
    sup_eval_n: int = 1000
    sup_batch: int = 32
    sup_max_len: int = 512
    tokenizer: str = "data/tokenizer.json"
    # shared
    data_dir_pre: str = "data/fineweb_edu"
    eval_windows: int = 512
    mem_ceiling_gib: float = 14.0
    seed: int = 0
    compile: bool = True
    name: str = ""
    out_root: str = "runs"


def _load_parent(cfg: FinetuneConfig, device) -> tuple[Transformer, ModelConfig]:
    parent = Path(cfg.parent)
    mc = from_dict(ModelConfig, load_yaml(parent / "model.yaml"))
    model = Transformer(mc)
    ckpt = read_latest(parent)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"{parent} has no latest.txt / checkpoint")
    model.load_state_dict(load_model_state(ckpt))
    return model.to(device), mc


def _sup_data(cfg: FinetuneConfig) -> dict[str, tuple[list[dict], list[dict]]]:
    return {
        n: (
            load_sup_split(t, t.train_split, cfg.sup_train_n, cfg.seed),
            load_sup_split(t, t.eval_split, cfg.sup_eval_n, cfg.seed),
        )
        for n, t in SUP_TASKS.items()
    }


@torch.no_grad()
def _sup_eval(model, tok, data, device) -> dict[str, float]:
    model.eval()
    out = {}
    for name, (_, ev) in data.items():
        t = SUP_TASKS[name]
        correct = 0
        for r in ev:
            prompt, label = format_example(t, r)
            correct += int(score_options(model, tok, prompt, t.labels, device) == label)
        out[f"{name}_acc"] = correct / len(ev)
        out[f"{name}_n"] = len(ev)
    out["sup_acc_mean"] = sum(v for k, v in out.items() if k.endswith("_acc")) / len(data)
    model.train()
    return out


def _bucketed_batches(
    examples: list[tuple[list[int], list[int]]],
    order: list[int],
    batch_size: int,
    generator: torch.Generator,
    mega_mult: int = 50,
) -> list[list[int]]:
    """Group a seeded permutation into batches of similar encoded length, then shuffle
    the batch order (with the same generator). Sorting inside fixed-size mega-batches
    rather than globally keeps the epoch's example set and batch count identical to a
    plain permutation while cutting most of the padding.
    """
    mega = mega_mult * batch_size
    batches = []
    for i in range(0, len(order), mega):
        chunk = sorted(order[i : i + mega], key=lambda j: len(examples[j][0]))
        for k in range(0, len(chunk) - batch_size + 1, batch_size):
            batches.append(chunk[k : k + batch_size])
    perm = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[p] for p in perm]


def _code_eval(loss_fn, cfg, device) -> float:
    vs = TokenStream(list_shards(cfg.data_dir_code, "val"))
    return evaluate(loss_fn, vs, cfg.seq_len, cfg.micro_batch, cfg.eval_windows, device)


def _pre_eval(loss_fn, cfg, device) -> float:
    vs = TokenStream(list_shards(cfg.data_dir_pre, "val"))
    return evaluate(loss_fn, vs, cfg.seq_len, cfg.micro_batch, cfg.eval_windows, device)


def _mem_guard(on_cuda: bool, done: int, ceiling_gib: float) -> float:
    """Read peak CUDA memory after an optimizer step; raise past the ceiling once warm.

    `done` is the count of optimizer steps taken so far (1-indexed). The first two
    steps include one-time compile transients, so the ceiling only applies from the
    3rd step on, and the peak is reset once at that point (mirrors rankfile.train).
    """
    mem = torch.cuda.max_memory_allocated() / 2**30 if on_cuda else 0.0
    if on_cuda and done >= 3 and mem > ceiling_gib:
        raise RuntimeError(
            f"peak memory {mem:.1f} GiB exceeds ceiling {ceiling_gib}; WDDM would page to RAM"
        )
    if on_cuda and done == 3:
        torch.cuda.reset_peak_memory_stats()
    return mem


def finetune(cfg: FinetuneConfig, device: str | None = None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = str(device).startswith("cuda")
    parent_name = Path(cfg.parent).name
    tag = "full" if cfg.method == "full" else f"lora{cfg.rank}"
    cfg.name = cfg.name or f"{parent_name}__{tag}_{cfg.task}"
    run = Path(cfg.out_root) / cfg.name
    run.mkdir(parents=True, exist_ok=True)
    to_yaml(cfg, run / "config.resolved.yaml")
    with open(run / "git.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {_git_hash()}\n")
    log = MetricsLog(run)
    torch.manual_seed(cfg.seed)
    model, mc = _load_parent(cfg, device)
    if cfg.method == "lora":
        # apply_lora zero-inits B, so the wrapped model's forward pass is exactly the
        # parent's until the first optimizer step: "before" metrics are unaffected.
        alpha = cfg.alpha if cfg.alpha is not None else 2.0 * cfg.rank
        params = apply_lora(model, cfg.rank, alpha)
    else:
        alpha = None
        params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    loss_fn = torch.compile(model.loss, dynamic=False) if (cfg.compile and on_cuda) else model.loss

    def ac():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_cuda)

    if cfg.task == "code":
        ts = TokenStream(list_shards(cfg.data_dir_code, "train"))
        sampler = FixedOrderSampler(ts.total_tokens, cfg.seq_len, cfg.seed)
        accum = cfg.batch_tokens // (cfg.micro_batch * cfg.seq_len)
        steps = cfg.train_tokens // cfg.batch_tokens
        need = steps * accum * cfg.micro_batch
        if need > sampler.n_windows:
            raise ValueError(f"code stream has {sampler.n_windows} windows, run needs {need}")

    before = {"pre_val_loss": _pre_eval(loss_fn, cfg, device)}
    if cfg.task == "code":
        before["code_val_loss"] = _code_eval(loss_fn, cfg, device)
    elif cfg.task == "sup":
        tok = load_tokenizer(cfg.tokenizer)
        data = _sup_data(cfg)
        before.update(_sup_eval(model, tok, data, device))
    else:
        raise ValueError(f"unknown task {cfg.task!r}")

    t0 = time.time()  # training only: "before"/"after" evals are excluded on both ends
    if cfg.task == "code":
        pos = 0
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = wsd_lr(step, steps, cfg.lr, cfg.warmup_frac, cfg.decay_frac)
            acc = torch.zeros((), device=device)
            for _ in range(accum):
                x, y, d = make_batch(ts, sampler, pos, cfg.micro_batch, cfg.seq_len, device)
                pos += cfg.micro_batch
                # FlexAttention (which doc masking requires) has no CPU backward in this
                # torch build; gate the doc mask on device the same way rankfile.train does.
                bm = doc_block_mask(d) if on_cuda else None
                with ac():
                    loss = loss_fn(x, y, block_mask=bm) / accum
                loss.backward()
                acc += loss.detach()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            mem = _mem_guard(on_cuda, step + 1, cfg.mem_ceiling_gib)
            if step % 10 == 0:
                mem_reserved = torch.cuda.max_memory_reserved() / 2**30 if on_cuda else 0.0
                log.write(step=step, loss=acc.item(), lr=opt.param_groups[0]["lr"],
                          mem_gib=mem, mem_reserved_gib=mem_reserved)
    else:  # sup
        examples = []
        for n, (tr, _) in data.items():
            t = SUP_TASKS[n]
            for r in tr:
                prompt, label = format_example(t, r)
                examples.append(encode_sup(tok, prompt, t.labels[label], cfg.sup_max_len))
        g = torch.Generator().manual_seed(cfg.seed)
        steps = cfg.epochs * (len(examples) // cfg.sup_batch)
        step = 0
        for _ in range(cfg.epochs):
            order = torch.randperm(len(examples), generator=g).tolist()
            for batch_idx in _bucketed_batches(examples, order, cfg.sup_batch, g):
                for pg in opt.param_groups:
                    pg["lr"] = wsd_lr(step, steps, cfg.lr, cfg.warmup_frac, cfg.decay_frac)
                batch = [examples[j] for j in batch_idx]
                ids, mask = collate_sup(batch)
                with ac():
                    loss = sup_loss(model, ids.to(device), mask.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                mem = _mem_guard(on_cuda, step + 1, cfg.mem_ceiling_gib)
                if step % 10 == 0:
                    mem_reserved = torch.cuda.max_memory_reserved() / 2**30 if on_cuda else 0.0
                    log.write(step=step, loss=loss.item(), lr=opt.param_groups[0]["lr"],
                              mem_gib=mem, mem_reserved_gib=mem_reserved)
                step += 1
    train_seconds = time.time() - t0

    if cfg.task == "code":
        after = {"code_val_loss": _code_eval(loss_fn, cfg, device)}
    else:
        after = _sup_eval(model, tok, data, device)
    after["pre_val_loss"] = _pre_eval(loss_fn, cfg, device)
    if cfg.method == "lora":
        torch.save(lora_state_dict(model), run / "lora.pt")
    else:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run / "model.pt")
    parent_ckpt = read_latest(cfg.parent)
    results = {
        "parent": parent_name,
        "parent_ckpt": parent_ckpt.name if parent_ckpt is not None else None,
        "method": cfg.method,
        "rank": cfg.rank if cfg.method == "lora" else None,
        "alpha": alpha,
        "task": cfg.task,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "before": before,
        "after": after,
        "train_seconds": train_seconds,
    }
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return run


def main() -> Path:
    ap = argparse.ArgumentParser(description="rank-and-file fine-tuning")
    ap.add_argument("--parent", required=True)
    ap.add_argument("--method", choices=["full", "lora"], required=True)
    ap.add_argument("--task", choices=["code", "sup"], required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--config", default=None, help="optional YAML with FinetuneConfig fields")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    d = load_yaml(a.config) if a.config else {}
    d.update(parent=a.parent, method=a.method, task=a.task, rank=a.rank)
    if a.lr is not None:
        d["lr"] = a.lr
    base = from_dict(FinetuneConfig, {})
    merged = apply_overrides({**base.__dict__, **d}, a.set)
    cfg = from_dict(FinetuneConfig, merged)
    if a.name:
        cfg.name = a.name
    return finetune(cfg, device=a.device)


if __name__ == "__main__":
    main()
