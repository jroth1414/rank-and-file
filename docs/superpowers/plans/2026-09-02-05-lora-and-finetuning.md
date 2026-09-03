# Plan 5: LoRA and Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune any pretrained twin with full fine-tuning or hand-written LoRA at a given rank on two task types, code continued pretraining and a supervised classification bundle, and write a `results.json` per run holding the target metric, the forgetting metric, and the weights needed for spectral analysis.

**Architecture:** `lora.py` wraps `nn.Linear` layers in place and exposes the delta `B·A·scale`. `tasks.py` supplies data for the two task types: code reuses the Plan 2 shard format; the supervised bundle builds prompt/label examples with a loss mask and scores classification by comparing label-word log-probabilities. `finetune.py` owns `FinetuneConfig`, both training loops, evaluation, and the CLI. `scripts/prepare_code.py` builds the code shards.

**Tech Stack:** torch 2.14, `datasets`, pytest.

**Spec:** `proposal.md` §4.3 (methods, ranks 4/16/64, targets = all attention and MLP projections, two tasks, LoRA gap metric, forgetting). `CLAUDE.md` §2.2, §6 (no PEFT library), §7 `test_lora.py`.

## Global Constraints

- Python 3.11 in `.venv`; run as `.venv\Scripts\python.exe ...`.
- LoRA targets: `q, k, v, o, gate, up, down` in every block. Embeddings are never adapted.
- **LoRA alpha = 2·rank** (constant scale 2.0 at every rank), per the CLAUDE.md decision log; `FinetuneConfig.alpha=None` means "2·rank".
- Same fine-tuning hyperparameters across twins for a given (method, task): learning rates live in `configs/finetune/{full,lora}_{code,sup}.yaml`, swept once on P1 s0 (`configs/queue/ft_sweep.txt`, 12 runs), then frozen.
- The fine-tuning loops carry the same memory guard as pretraining (ceiling raise from step 3, peak reset at step 3, `mem_gib` logged), write `git.txt`, and record `parent_ckpt`, `alpha`, `seed` in `results.json`. Supervised batches are length-bucketed within shuffled mega-batches of 50 batches.
- Every fine-tune run dir has `config.resolved.yaml`, `results.json`, and either `model.pt` (full) or `lora.pt` (LoRA).
- Commit prefix `lora:` / `ft:` / `tasks:`; no AI attribution trailers.

## Consumed interfaces

- `rankfile.model.Transformer`, `ModelConfig`, `.loss(idx, targets, doc_ids=None, block_mask=None)`, `.forward(idx)`, `doc_block_mask`.
- `rankfile.checkpoint`: `load_model_state(path) -> dict`, `read_latest(run_dir)`, `unwrap`.
- `rankfile.data`: `write_shard`, `list_shards`, `TokenStream`, `FixedOrderSampler`, `make_batch`.
- `rankfile.tokenizer`: `load_tokenizer`, `encode_docs`, `EOT_ID`.
- `rankfile.train`: `evaluate(loss_fn, val_stream, seq_len, micro_batch, n_windows, device, use_doc_mask)`, `MetricsLog`.
- `rankfile.schedule.wsd_lr`.
- `rankfile.config`: `load_yaml`, `from_dict`, `apply_overrides`, `to_yaml`.

---

### Task 1: LoRA layers

**Files:**
- Create: `src/rankfile/lora.py`
- Test: `tests/test_lora.py`

**Interfaces:**
- Produces: `LoRALinear(base: nn.Linear, r: int, alpha: float)` with `.A: Parameter[r, in]` (Kaiming-uniform), `.B: Parameter[out, r]` (zeros), `.scale = alpha / r`, `.delta() -> Tensor[out, in]`; `TARGETS = ("q", "k", "v", "o", "gate", "up", "down")`; `apply_lora(model, r, alpha, targets=TARGETS) -> list[Parameter]` (freezes all base params, wraps targets, returns trainable A/B); `lora_state_dict(model) -> dict[str, dict]` mapping wrapped-module name → `{"A", "B", "scale"}`; `merge_lora(model) -> None` (folds deltas into base weights and unwraps); `lora_deltas(model) -> dict[str, Tensor]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lora.py
import torch
from rankfile.model import ModelConfig, Transformer
from rankfile.lora import LoRALinear, apply_lora, lora_state_dict, merge_lora, lora_deltas, TARGETS

def _m():
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=64, n_layer=2, d_model=32, n_head=2, n_kv_head=1, head_dim=16, d_ff=64, max_seq_len=16))

def test_zero_init_is_identity():
    base = torch.nn.Linear(8, 6, bias=False)
    l = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(l(x), base(x))
    assert l.scale == 2.0

def test_delta_rank_and_merge_equivalence():
    m = _m(); x = torch.randint(0, 64, (2, 16))
    params = apply_lora(m, r=4, alpha=8)
    assert all(p.requires_grad for p in params) and not m.embed.weight.requires_grad
    for p in params: p.data.normal_()
    out_lora = m(x)
    d = lora_deltas(m)
    assert set(n.split(".")[-1] for n in d) == set(TARGETS)
    assert all(torch.linalg.matrix_rank(v) <= 4 for v in d.values())
    merge_lora(m)
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())
    assert torch.allclose(m(x), out_lora, atol=1e-5)

def test_lora_state_dict_shapes():
    m = _m(); apply_lora(m, r=3, alpha=3)
    sd = lora_state_dict(m)
    assert "blocks.0.attn.q.weight" not in sd and "blocks.0.attn.q" in sd
    assert sd["blocks.0.mlp.down"]["A"].shape == (3, 64) and sd["blocks.0.mlp.down"]["B"].shape == (32, 3)
    assert len(sd) == 2 * len(TARGETS)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lora.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/lora.py
"""LoRA (Hu et al. 2021) implemented by hand: W x + (B A x) * alpha/r, B zero-initialized."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

TARGETS = ("q", "k", "v", "o", "gate", "up", "down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def delta(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale


def apply_lora(model: nn.Module, r: int, alpha: float, targets: tuple[str, ...] = TARGETS) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    for blk in model.blocks:
        for parent in (blk.attn, blk.mlp):
            for name in targets:
                if hasattr(parent, name) and isinstance(getattr(parent, name), nn.Linear):
                    wrapped = LoRALinear(getattr(parent, name), r, alpha)
                    setattr(parent, name, wrapped)
                    trainable += [wrapped.A, wrapped.B]
    return trainable


def _lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_state_dict(model: nn.Module) -> dict[str, dict]:
    return {n: {"A": m.A.detach().cpu().clone(), "B": m.B.detach().cpu().clone(), "scale": m.scale}
            for n, m in _lora_modules(model)}


def lora_deltas(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: m.delta().detach() for n, m in _lora_modules(model)}


@torch.no_grad()
def merge_lora(model: nn.Module) -> None:
    for name, mod in _lora_modules(model):
        mod.base.weight.add_(mod.delta().to(mod.base.weight.dtype))
        parent_name, attr = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, mod.base)
    for p in model.parameters():
        p.requires_grad_(True)
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_lora.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/lora.py tests/test_lora.py
git commit -m "lora: hand-written LoRA wrappers with delta, merge, and state dict"
```

---

### Task 2: Code task data (`prepare_code.py`)

**Files:**
- Create: `scripts/prepare_code.py`
- Test: `tests/test_prepare_code.py`

**Interfaces:**
- Produces: `data/code/train_*.bin` (100M tokens) and `data/code/val_0000.bin` (5M) in the Plan 2 shard format, built from permissively licensed Python; reuses `scripts.prepare_data.shard_documents`. A function `python_docs(max_docs: int | None) -> Iterator[str]` that streams `codeparrot/github-code-clean` filtered to Python with MIT/Apache/BSD licenses. If that dataset is unavailable, fall back to `bigcode/the-stack-smol` config `data/python` and say so in the decision log.

- [ ] **Step 1: Write the failing test** (offline; drives `shard_documents` with fake code docs)

```python
# tests/test_prepare_code.py
from rankfile.tokenizer import train_tokenizer
from rankfile.data import list_shards, TokenStream
from scripts.prepare_code import build_code_shards

def test_build_code_shards_from_iterator(tmp_path):
    tok = train_tokenizer(["def f(x):\n    return x\n" * 100], vocab_size=300, out_path=tmp_path / "t.json")
    docs = (f"def f{i}(x):\n    return x + {i}\n" * 20 for i in range(300))
    stats = build_code_shards(docs, tok, tmp_path / "code", train_tokens=20000, val_tokens=2000, shard_tokens=8000)
    assert TokenStream(list_shards(tmp_path / "code", "val")).total_tokens >= 2000
    assert TokenStream(list_shards(tmp_path / "code", "train")).total_tokens >= 20000
    assert stats["docs"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_prepare_code.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/prepare_code.py
"""Build Python code shards for the continued-pretraining task.

Usage: set HF_HOME=D:\hf_cache && python scripts/prepare_code.py --out data/code
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from rankfile.tokenizer import load_tokenizer
from scripts.prepare_data import shard_documents


def python_docs(max_docs: int | None = None) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("codeparrot/github-code-clean", streaming=True, split="train",
                      languages=["Python"], licenses=["mit", "apache-2.0", "bsd-3-clause"], trust_remote_code=True)
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        yield row["code"]


def build_code_shards(docs: Iterable[str], tok, out_dir: str | Path, train_tokens: int, val_tokens: int,
                      shard_tokens: int) -> dict:
    return shard_documents(docs, tok, out_dir, shard_tokens=shard_tokens, max_tokens=train_tokens, val_tokens=val_tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/code")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--train-tokens", type=int, default=100_000_000)
    ap.add_argument("--val-tokens", type=int, default=5_000_000)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    print(build_code_shards(python_docs(), load_tokenizer(a.tokenizer), a.out, a.train_tokens, a.val_tokens, 25_000_000))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then build the real shards** (network, ~10 min)

Run: `.venv\Scripts\python.exe -m pytest tests/test_prepare_code.py -v` → 1 passed
Run: `set HF_HOME=D:\hf_cache && .venv\Scripts\python.exe scripts/prepare_code.py`
Expected: prints stats with `train_tokens` ≥ 100M.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_code.py tests/test_prepare_code.py
git commit -m "tasks: Python code shards for continued pretraining"
```

---

### Task 3: Supervised bundle (SST-2, BoolQ, AG News)

**Files:**
- Create: `src/rankfile/tasks.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces: `SUP_TASKS: dict[str, TaskDef]` where `TaskDef(name, hf_path, hf_config, train_split, eval_split, template: Callable[[dict], str], labels: list[str], label_key: str)`; `format_example(task, row) -> tuple[str, int]` (prompt, label index); `encode_sup(tok, prompt, label_text, max_len) -> tuple[list[int], list[int]]` (ids, loss_mask with 1 on label tokens only); `SupBatch = collate_sup(examples, pad_id=EOT_ID) -> (ids[B,L], mask[B,L])`; `score_options(model, tok, prompt, labels, device) -> int` (argmax of summed label log-prob); `load_sup_split(task, split, n, seed) -> list[dict]` (balanced subsample, cached to `data/sup/{task}_{split}.json`).

Label words carry a leading space so they tokenize as whole words after the prompt colon.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tasks.py
import torch
from rankfile.tokenizer import train_tokenizer, EOT_ID
from rankfile.tasks import SUP_TASKS, format_example, encode_sup, collate_sup, score_options
from rankfile.model import ModelConfig, Transformer

def _tok(tmp_path):
    return train_tokenizer(["Review: great movie\nSentiment: positive negative yes no world sports business technology " * 50],
                           vocab_size=400, out_path=tmp_path / "t.json")

def test_templates_and_labels():
    p, y = format_example(SUP_TASKS["sst2"], {"sentence": "great movie", "label": 1})
    assert p.endswith("Sentiment:") and y == 1 and SUP_TASKS["sst2"].labels == [" negative", " positive"]
    p, y = format_example(SUP_TASKS["boolq"], {"passage": "Sky is blue.", "question": "is the sky blue", "answer": True})
    assert "Question:" in p and y == 1
    p, y = format_example(SUP_TASKS["ag_news"], {"text": "Stocks rose.", "label": 2})
    assert p.endswith("Topic:") and SUP_TASKS["ag_news"].labels[2] == " business"

def test_encode_masks_prompt_only(tmp_path):
    tok = _tok(tmp_path)
    ids, mask = encode_sup(tok, "Review: great\nSentiment:", " positive", max_len=32)
    assert len(ids) == len(mask) and sum(mask) >= 1 and mask[0] == 0
    assert ids[-1] == EOT_ID and mask[-1] == 1  # EOT after the label is trained too

def test_collate_pads_right(tmp_path):
    tok = _tok(tmp_path)
    a = encode_sup(tok, "Review: x\nSentiment:", " positive", 32); b = encode_sup(tok, "Review: a much longer review here\nSentiment:", " negative", 32)
    ids, mask = collate_sup([a, b], pad_id=EOT_ID)
    assert ids.shape == mask.shape and ids.shape[0] == 2 and ids.shape[1] == max(len(a[0]), len(b[0]))
    assert mask[0, len(a[0]):].sum() == 0

def test_score_options_returns_index(tmp_path):
    tok = _tok(tmp_path)
    m = Transformer(ModelConfig(vocab_size=400, n_layer=1, d_model=16, n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=64)).eval()
    idx = score_options(m, tok, "Review: great\nSentiment:", [" negative", " positive"], device="cpu")
    assert idx in (0, 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tasks.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/tasks.py
"""Supervised bundle: SST-2, BoolQ, AG News as LM prompts scored by label-word log-probs."""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from rankfile.tokenizer import EOT_ID


@dataclass
class TaskDef:
    name: str
    hf_path: str
    hf_config: str | None
    train_split: str
    eval_split: str
    template: Callable[[dict], str]
    labels: list[str]
    label_key: str


def _trunc(s: str, n: int = 1500) -> str:
    return s if len(s) <= n else s[:n]


SUP_TASKS: dict[str, TaskDef] = {
    "sst2": TaskDef("sst2", "stanfordnlp/sst2", None, "train", "validation",
                    lambda r: f"Review: {_trunc(r['sentence']).strip()}\nSentiment:", [" negative", " positive"], "label"),
    "boolq": TaskDef("boolq", "google/boolq", None, "train", "validation",
                     lambda r: f"Passage: {_trunc(r['passage'])}\nQuestion: {r['question'].strip()}?\nAnswer:", [" no", " yes"], "answer"),
    "ag_news": TaskDef("ag_news", "fancyzhx/ag_news", None, "train", "test",
                       lambda r: f"Article: {_trunc(r['text'])}\nTopic:", [" world", " sports", " business", " technology"], "label"),
}


def format_example(task: TaskDef, row: dict) -> tuple[str, int]:
    return task.template(row), int(row[task.label_key])


def encode_sup(tok, prompt: str, label_text: str, max_len: int) -> tuple[list[int], list[int]]:
    p = tok.encode(prompt).ids
    l = tok.encode(label_text).ids + [EOT_ID]
    p = p[-(max_len - len(l)):] if len(p) + len(l) > max_len else p   # keep the end of long prompts
    return p + l, [0] * len(p) + [1] * len(l)


def collate_sup(examples: list[tuple[list[int], list[int]]], pad_id: int = EOT_ID) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), L), dtype=torch.long)
    for i, (x, m) in enumerate(examples):
        ids[i, :len(x)] = torch.tensor(x); mask[i, :len(m)] = torch.tensor(m)
    return ids, mask


def sup_loss(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Next-token loss on positions whose *target* is a label token."""
    logits = model(ids[:, :-1]).float()
    tgt, m = ids[:, 1:], mask[:, 1:].float()
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view_as(m)
    return (nll * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def score_options(model, tok, prompt: str, labels: list[str], device) -> int:
    p = tok.encode(prompt).ids
    scores = []
    for lab in labels:
        l = tok.encode(lab).ids
        ids = torch.tensor([p + l], device=device)
        logp = F.log_softmax(model(ids[:, :-1]).float(), dim=-1)[0]
        scores.append(sum(logp[len(p) - 1 + j, l[j]].item() for j in range(len(l))))
    return int(torch.tensor(scores).argmax())


def load_sup_split(task: TaskDef, split: str, n: int, seed: int, cache_dir: str | Path = "data/sup") -> list[dict]:
    cache = Path(cache_dir) / f"{task.name}_{split}_{n}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from datasets import load_dataset

    ds = load_dataset(task.hf_path, task.hf_config, split=split)
    rows = [dict(r) for r in ds]
    by_label: dict[int, list[dict]] = {}
    for r in rows:
        by_label.setdefault(int(r[task.label_key]), []).append(r)
    rng = random.Random(seed)
    per = max(1, n // len(by_label))
    out: list[dict] = []
    for lab, rs in sorted(by_label.items()):
        rng.shuffle(rs); out += rs[:per]
    rng.shuffle(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tasks.py -v`
Expected: 4 passed

- [ ] **Step 5: Download and cache the real splits** (network, ~2 min)

```
set HF_HOME=D:\hf_cache
.venv\Scripts\python.exe -c "from rankfile.tasks import *; [load_sup_split(t,t.train_split,8000,0) for t in SUP_TASKS.values()]; [load_sup_split(t,t.eval_split,1000,0) for t in SUP_TASKS.values()]; print('ok')"
```

- [ ] **Step 6: Commit**

```bash
git add src/rankfile/tasks.py tests/test_tasks.py
git commit -m "tasks: SST-2, BoolQ, AG News as masked LM prompts with label-word scoring"
```

---

### Task 4: Fine-tuning loop and CLI

**Files:**
- Create: `src/rankfile/finetune.py`
- Test: `tests/test_finetune.py`

**Interfaces:**
- Produces: `FinetuneConfig` (below); `finetune(cfg: FinetuneConfig, device=None) -> Path`; `python -m rankfile.finetune --parent runs/p2_muon_m124_s0 --method lora --rank 16 --task code [--lr 5e-4] [--name N] [--set k=v]`. Run dir `runs/{name}` with `results.json`:

```json
{"parent": "p2_muon_m124_s0", "method": "lora", "rank": 16, "task": "code", "lr": 5e-4,
 "before": {"code_val_loss": 2.9, "pre_val_loss": 3.1},
 "after":  {"code_val_loss": 1.7, "pre_val_loss": 3.3},
 "train_seconds": 1234.0}
```
For `task == "sup"`, the metric keys are `sst2_acc`, `boolq_acc`, `ag_news_acc`, `sup_acc_mean`. Weights: `model.pt` (full state dict) for `full`, `lora.pt` (from `lora_state_dict`) for `lora`.

- [ ] **Step 1: Write the failing tests** (CPU, tiny model, synthetic data)

```python
# tests/test_finetune.py
import json, numpy as np, torch
from pathlib import Path
from rankfile.model import ModelConfig, Transformer
from rankfile.config import to_yaml
from rankfile.checkpoint import save_checkpoint, write_latest
from rankfile.optim.build import build_optimizers
from rankfile.data import write_shard
from rankfile.tokenizer import train_tokenizer
from rankfile.finetune import FinetuneConfig, finetune

def _parent(tmp_path):
    # vocab 512 so a real byte-level tokenizer (>= 257 ids) fits; shard tokens stay < 62
    mc = ModelConfig(vocab_size=512, n_layer=2, d_model=32, n_head=2, n_kv_head=1, head_dim=16, d_ff=64, max_seq_len=32)
    m = Transformer(mc); run = tmp_path / "runs" / "parent"; run.mkdir(parents=True)
    to_yaml(mc, run / "model.yaml")
    save_checkpoint(run / "ckpt_0000001.pt", m, build_optimizers(m, "adamw", 1e-3, 0.1), 1, 0, 0, {}); write_latest(run, "ckpt_0000001.pt")
    (run / "DONE").write_text("1 0 4.0")
    rng = np.random.default_rng(0)
    for d in ("pre", "code"):
        seq = [(i * 7 + 3) % 61 + 1 if d == "pre" else (i * 5 + 1) % 61 + 1 for i in range(20000)]
        for i in range(0, 20000, 50): seq[i] = 0
        write_shard(np.array(seq, dtype=np.uint16), tmp_path / d / "train_0000.bin")
        write_shard(np.array(seq[:3000], dtype=np.uint16), tmp_path / d / "val_0000.bin")
    return run

def _cfg(tmp_path, run, **kw):
    return FinetuneConfig(parent=str(run), out_root=str(tmp_path / "runs"), data_dir_code=str(tmp_path / "code"),
                          data_dir_pre=str(tmp_path / "pre"), seq_len=32, micro_batch=2, batch_tokens=128, train_tokens=128 * 30,
                          eval_windows=8, compile=False, **kw)

def test_full_code_ft_improves_code_loss_and_saves_model(tmp_path):
    run = _parent(tmp_path)
    out = finetune(_cfg(tmp_path, run, method="full", task="code", lr=3e-3), device="cpu")
    r = json.loads((out / "results.json").read_text())
    assert r["after"]["code_val_loss"] < r["before"]["code_val_loss"] and (out / "model.pt").exists()
    assert out.name == "parent__full_code"

def test_lora_code_ft_saves_lora_only(tmp_path):
    run = _parent(tmp_path)
    out = finetune(_cfg(tmp_path, run, method="lora", rank=2, alpha=4, task="code", lr=1e-2), device="cpu")
    assert (out / "lora.pt").exists() and not (out / "model.pt").exists() and out.name == "parent__lora2_code"
    sd = torch.load(out / "lora.pt", weights_only=False); assert sd["blocks.0.attn.q"]["A"].shape[0] == 2

def test_sup_ft_runs_on_fake_examples(tmp_path, monkeypatch):
    run = _parent(tmp_path)
    tok = train_tokenizer(["Review: good bad\nSentiment: positive negative " * 50], vocab_size=300, out_path=tmp_path / "t.json")
    import rankfile.finetune as ft
    fake = {"sst2": ([{"sentence": "good", "label": 1}, {"sentence": "bad", "label": 0}] * 8)}
    monkeypatch.setattr(ft, "_sup_data", lambda cfg: {"sst2": (fake["sst2"], fake["sst2"][:4])})
    out = finetune(_cfg(tmp_path, run, method="full", task="sup", lr=1e-3, epochs=1, sup_batch=4,
                        tokenizer=str(tmp_path / "t.json"), sup_max_len=32), device="cpu")
    r = json.loads((out / "results.json").read_text())
    assert "sst2_acc" in r["after"] and 0.0 <= r["after"]["sup_acc_mean"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/finetune.py
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
from rankfile.tasks import SUP_TASKS, collate_sup, encode_sup, format_example, load_sup_split, score_options, sup_loss
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
    return {n: (load_sup_split(t, t.train_split, cfg.sup_train_n, cfg.seed), load_sup_split(t, t.eval_split, cfg.sup_eval_n, cfg.seed))
            for n, t in SUP_TASKS.items()}


@torch.no_grad()
def _sup_eval(model, tok, data, device) -> dict[str, float]:
    model.eval()
    out = {}
    for name, (_, ev) in data.items():
        t = SUP_TASKS[name]
        correct = sum(score_options(model, tok, *format_example(t, r)[:1], t.labels, device) == format_example(t, r)[1] for r in ev)
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
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_cuda)  # noqa: E731

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
                x, y, d = make_batch(ts, sampler, pos, cfg.micro_batch, cfg.seq_len, device); pos += cfg.micro_batch
                with ac():
                    loss = loss_fn(x, y, block_mask=doc_block_mask(d)) / accum
                loss.backward(); acc += loss.detach()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step(); opt.zero_grad(set_to_none=True)
            if step % 10 == 0:
                log.write(step=step, loss=acc.item(), lr=opt.param_groups[0]["lr"])
        after = {"code_val_loss": _code_eval(loss_fn, cfg, device)}
    elif cfg.task == "sup":
        tok = load_tokenizer(cfg.tokenizer)
        data = _sup_data(cfg)
        before.update(_sup_eval(model, tok, data, device))
        examples = [encode_sup(tok, *format_example(SUP_TASKS[n], r)[:1], SUP_TASKS[n].labels[format_example(SUP_TASKS[n], r)[1]], cfg.sup_max_len)
                    for n, (tr, _) in data.items() for r in tr]
        g = torch.Generator().manual_seed(cfg.seed)
        steps = cfg.epochs * (len(examples) // cfg.sup_batch)
        step = 0
        for _ in range(cfg.epochs):
            order = torch.randperm(len(examples), generator=g).tolist()
            for i in range(0, len(order) - cfg.sup_batch + 1, cfg.sup_batch):
                for pg in opt.param_groups:
                    pg["lr"] = wsd_lr(step, steps, cfg.lr, cfg.warmup_frac, cfg.decay_frac)
                ids, mask = collate_sup([examples[j] for j in order[i:i + cfg.sup_batch]])
                with ac():
                    loss = sup_loss(model, ids.to(device), mask.to(device))
                loss.backward(); torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step(); opt.zero_grad(set_to_none=True)
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
    results = {"parent": parent_name, "method": cfg.method, "rank": cfg.rank if cfg.method == "lora" else None,
               "task": cfg.task, "lr": cfg.lr, "before": before, "after": after, "train_seconds": time.time() - t0}
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
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/finetune.py tests/test_finetune.py
git commit -m "ft: full and LoRA fine-tuning on code CPT and supervised bundle with results.json"
```

---

### Task 5: Fine-tuning configs, LR sweep, and grid queue generator

**Files:**
- Create: `configs/finetune/full.yaml`, `configs/finetune/lora.yaml`, `scripts/ft_grid.py`, `configs/queue/ft_sweep.txt`
- Test: `tests/test_ft_grid.py`

**Interfaces:**
- Produces: `scripts/ft_grid.py --parents runs/p1_adamw_m124_s0 runs/p2_muon_m124_s0 runs/p3_adamw_m124_s0 --out configs/queue/ft_core.txt` writing one queue line per (parent × method × task) with the frozen learning rates from `configs/finetune/*.yaml`. `ft_sweep.txt` sweeps `lr` on P1 s0 only: full ∈ {3e-5, 1e-4, 3e-4}, LoRA r16 ∈ {3e-4, 1e-3, 3e-3}, on the code task; the winner is copied to `full.yaml` and `lora.yaml` and recorded in the decision log.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ft_grid.py
from scripts.ft_grid import grid_lines

def test_grid_lines_cover_all_cells():
    lines = grid_lines(["runs/p1_adamw_m124_s0", "runs/p2_muon_m124_s0"], ranks=[4, 16, 64], tasks=["code", "sup"],
                       full_lr=1e-4, lora_lr=1e-3, py=".venv\\Scripts\\python.exe")
    assert len(lines) == 2 * (1 + 3) * 2
    assert any("--method full --task code --lr 0.0001 --name p1_adamw_m124_s0__full_code" in l for l in lines)
    assert any("--method lora --task sup --rank 64 --lr 0.001 --name p2_muon_m124_s0__lora64_sup" in l for l in lines)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ft_grid.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```yaml
# configs/finetune/full.yaml — lr is the sweep placeholder; replace with the ft_sweep winner.
lr: 1.0e-4
weight_decay: 0.0
```

```yaml
# configs/finetune/lora.yaml — lr is the sweep placeholder; replace with the ft_sweep winner.
lr: 1.0e-3
alpha: 32.0
weight_decay: 0.0
```

```python
# scripts/ft_grid.py
"""Emit the fine-tuning grid as a queue file: parents x {full, lora r4/16/64} x {code, sup}."""
from __future__ import annotations

import argparse
from pathlib import Path

from rankfile.config import load_yaml


def grid_lines(parents: list[str], ranks: list[int], tasks: list[str], full_lr: float, lora_lr: float, py: str) -> list[str]:
    lines = []
    for parent in parents:
        pname = Path(parent).name
        for task in tasks:
            lines.append(f"{py} -m rankfile.finetune --parent {parent} --method full --task {task} --lr {full_lr} --name {pname}__full_{task}")
            for r in ranks:
                lines.append(f"{py} -m rankfile.finetune --parent {parent} --method lora --task {task} --rank {r} --lr {lora_lr} --name {pname}__lora{r}_{task}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", nargs="+", type=int, default=[4, 16, 64])
    ap.add_argument("--tasks", nargs="+", default=["code", "sup"])
    a = ap.parse_args()
    full_lr = load_yaml("configs/finetune/full.yaml")["lr"]
    lora_lr = load_yaml("configs/finetune/lora.yaml")["lr"]
    lines = grid_lines(a.parents, a.ranks, a.tasks, full_lr, lora_lr, ".venv\\Scripts\\python.exe")
    Path(a.out).write_text("# generated by scripts/ft_grid.py\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines to {a.out}")


if __name__ == "__main__":
    main()
```

```
# configs/queue/ft_sweep.txt — learning-rate sweep on P1 s0, code task only. Each run < 1 h.
.venv\Scripts\python.exe -m rankfile.finetune --parent runs/p1_adamw_m124_s0 --method full --task code --lr 3e-5 --name ftsweep_full_lr3e-5
.venv\Scripts\python.exe -m rankfile.finetune --parent runs/p1_adamw_m124_s0 --method full --task code --lr 1e-4 --name ftsweep_full_lr1e-4
.venv\Scripts\python.exe -m rankfile.finetune --parent runs/p1_adamw_m124_s0 --method full --task code --lr 3e-4 --name ftsweep_full_lr3e-4
.venv\Scripts\python.exe -m rankfile.finetune --parent runs/p1_adamw_m124_s0 --method lora --task code --rank 16 --lr 3e-4 --name ftsweep_lora16_lr3e-4
.venv\Scripts\python.exe -m rankfile.finetune --parent runs/p1_adamw_m124_s0 --method lora --task code --rank 16 --lr 1e-3 --name ftsweep_lora16_lr1e-3
.venv\Scripts\python.exe -m rankfile.finetune --parent runs/p1_adamw_m124_s0 --method lora --task code --rank 16 --lr 3e-3 --name ftsweep_lora16_lr3e-3
```

Note: `scripts/queue.py` skips runs whose `DONE` file exists; fine-tune runs write `results.json` instead. Add to `queue.py`'s skip check: `or (runs_root / name / "results.json").exists()`. Update `tests/test_queue.py` accordingly (a run dir containing `results.json` is skipped).

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ft_grid.py tests/test_queue.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add configs/finetune/ scripts/ft_grid.py configs/queue/ft_sweep.txt tests/test_ft_grid.py scripts/queue.py tests/test_queue.py
git commit -m "ft: fine-tuning configs, LR sweep queue, and grid generator"
```

---

### Task 6: Launch gate (no code)

- [ ] **Step 1:** Requires `runs/p1_adamw_m124_s0/DONE` (Plan 4 Task 7). Run a GPU smoke: `.venv\Scripts\python.exe -m rankfile.finetune --parent runs/p1_adamw_m124_s0 --method lora --task code --rank 16 --lr 1e-3 --name ft_smoke --set train_tokens=2097152` and confirm `results.json` appears and `mem_gib` in metrics stays under 14. Delete `runs/ft_smoke`.
- [ ] **Step 2:** Ask the user: "Launch the fine-tuning LR sweep (6 runs, ~4 hours)?" On approval run `scripts/queue.py configs/queue/ft_sweep.txt`. Pick the lr with the lowest `after.code_val_loss` per method; write into `configs/finetune/full.yaml` and `lora.yaml`; add a decision-log line.
- [ ] **Step 3:** Generate the grid: `.venv\Scripts\python.exe scripts/ft_grid.py --parents runs/p1_adamw_m124_s0 runs/p2_muon_m124_s0 runs/p3_adamw_m124_s0 --out configs/queue/ft_core.txt`. Ask: "Launch the 24-run fine-tuning grid (~9 to 15 hours)?" On approval run it.

---

## Self-review

- **Spec coverage:** full FT and LoRA at 4/16/64 on all projections (Task 1, Task 5), code CPT on 100M Python tokens with held-out loss (Tasks 2, 4), supervised bundle with accuracy (Tasks 3, 4), forgetting as pre-val-loss delta (Task 4 `before/after.pre_val_loss`), LR swept once on one twin and reused (Task 5), weights saved for spectral analysis (Task 4).
- **Placeholders:** none; lr placeholders in YAML are labeled and resolved by Task 6.
- **Type consistency:** `results.json` keys (`before`, `after`, `code_val_loss`, `pre_val_loss`, `*_acc`, `sup_acc_mean`, `method`, `rank`, `task`, `parent`) are what Plan 6 reads. `lora.pt` format is `{module_name: {"A","B","scale"}}` and `model.pt` is a plain state dict; Plan 6 depends on both. `model.loss(..., block_mask=...)` matches Plans 1 and 4.
