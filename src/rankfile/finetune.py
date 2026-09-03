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
from rankfile.train import MetricsLog, evaluate


@dataclass
class FinetuneConfig:
    parent: str = ""               # run dir of the pretrained twin
    method: str = "full"           # full | lora
    rank: int = 16
    alpha: float = 32.0
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
    eval_windows: int = 256
    seed: int = 0
    compile: bool = True
    name: str = ""
    out_root: str = "runs"


def _load_parent(cfg: FinetuneConfig, device) -> tuple[Transformer, ModelConfig]:
    parent = Path(cfg.parent)
    mc = from_dict(ModelConfig, load_yaml(parent / "model.yaml"))
    model = Transformer(mc)
    model.load_state_dict(load_model_state(read_latest(parent)))
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
    out["sup_acc_mean"] = sum(v for k, v in out.items() if k.endswith("_acc")) / len(data)
    model.train()
    return out


def _code_eval(loss_fn, cfg, device) -> float:
    vs = TokenStream(list_shards(cfg.data_dir_code, "val"))
    return evaluate(loss_fn, vs, cfg.seq_len, cfg.micro_batch, cfg.eval_windows, device)


def _pre_eval(loss_fn, cfg, device) -> float:
    vs = TokenStream(list_shards(cfg.data_dir_pre, "val"))
    return evaluate(loss_fn, vs, cfg.seq_len, cfg.micro_batch, cfg.eval_windows, device)


def finetune(cfg: FinetuneConfig, device: str | None = None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = str(device).startswith("cuda")
    parent_name = Path(cfg.parent).name
    tag = "full" if cfg.method == "full" else f"lora{cfg.rank}"
    cfg.name = cfg.name or f"{parent_name}__{tag}_{cfg.task}"
    run = Path(cfg.out_root) / cfg.name
    run.mkdir(parents=True, exist_ok=True)
    to_yaml(cfg, run / "config.resolved.yaml")
    log = MetricsLog(run)
    torch.manual_seed(cfg.seed)
    model, mc = _load_parent(cfg, device)
    if cfg.method == "lora":
        params = apply_lora(model, cfg.rank, cfg.alpha)
    else:
        params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    loss_fn = torch.compile(model.loss, dynamic=False) if (cfg.compile and on_cuda) else model.loss

    def ac():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_cuda)

    before = {"pre_val_loss": _pre_eval(loss_fn, cfg, device)}
    t0 = time.time()
    if cfg.task == "code":
        before["code_val_loss"] = _code_eval(loss_fn, cfg, device)
        ts = TokenStream(list_shards(cfg.data_dir_code, "train"))
        sampler = FixedOrderSampler(ts.total_tokens, cfg.seq_len, cfg.seed)
        accum = cfg.batch_tokens // (cfg.micro_batch * cfg.seq_len)
        steps = cfg.train_tokens // cfg.batch_tokens
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
            if step % 10 == 0:
                log.write(step=step, loss=acc.item(), lr=opt.param_groups[0]["lr"])
        after = {"code_val_loss": _code_eval(loss_fn, cfg, device)}
    elif cfg.task == "sup":
        tok = load_tokenizer(cfg.tokenizer)
        data = _sup_data(cfg)
        before.update(_sup_eval(model, tok, data, device))
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
            for i in range(0, len(order) - cfg.sup_batch + 1, cfg.sup_batch):
                for pg in opt.param_groups:
                    pg["lr"] = wsd_lr(step, steps, cfg.lr, cfg.warmup_frac, cfg.decay_frac)
                batch = [examples[j] for j in order[i:i + cfg.sup_batch]]
                ids, mask = collate_sup(batch)
                with ac():
                    loss = sup_loss(model, ids.to(device), mask.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % 10 == 0:
                    log.write(step=step, loss=loss.item(), lr=opt.param_groups[0]["lr"])
                step += 1
        after = _sup_eval(model, tok, data, device)
    else:
        raise ValueError(f"unknown task {cfg.task!r}")
    after["pre_val_loss"] = _pre_eval(loss_fn, cfg, device)
    if cfg.method == "lora":
        torch.save(lora_state_dict(model), run / "lora.pt")
    else:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run / "model.pt")
    results = {
        "parent": parent_name,
        "method": cfg.method,
        "rank": cfg.rank if cfg.method == "lora" else None,
        "task": cfg.task,
        "lr": cfg.lr,
        "before": before,
        "after": after,
        "train_seconds": time.time() - t0,
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
