# Plan 1: Model and Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Qwen3-style decoder (`rankfile.model.Transformer`) with config loading, intra-document masking via FlexAttention, a chunked loss, parameter grouping for Muon, the m30/m124 configs, and a 32k byte-level BPE tokenizer.

**Architecture:** One file per responsibility: `config.py` (YAML → dataclass with `key=value` overrides), `model.py` (the transformer, single file, no framework), `tokenizer.py` (train/load/encode wrappers around the `tokenizers` library). The model exposes `forward`, `loss`, and two parameter lists (`hidden_matrix_params`, `other_params`) that Plan 3 uses to route weights to Muon or AdamW.

**Tech Stack:** torch 2.14 (SDPA with `enable_gqa`, `torch.nn.attention.flex_attention`), `tokenizers`, `pyyaml`, pytest.

**Spec:** `proposal.md` §4.1, `CLAUDE.md` §4 (architecture table), §6 (layout), §7 (required tests).

## Global Constraints

- Python 3.11 in `.venv`; run as `.venv\Scripts\python.exe -m pytest ...` from repo root.
- torch 2.14.0+cu130; flash SDPA unavailable, cuDNN SDPA and FlexAttention available.
- Architecture is fixed: RMSNorm pre-norm, SwiGLU, no biases, RoPE θ=10000, GQA 12/4 heads of dim 64, QK-norm, tied embeddings, vocab 32768, context 2048. m124 = 12 layers × 768; m30 = 4 layers × 256.
- `EOT_ID = 0` is `<|endoftext|>`. Documents are separated by it.
- GPU tests marked `@pytest.mark.gpu`, skipped without CUDA.
- Commit prefix `model:` / `config:` / `tok:`; no AI attribution trailers.

---

### Task 1: Config loading with overrides

**Files:**
- Create: `src/rankfile/config.py`
- Test: `tests/test_config.py`
- Modify: `pyproject.toml` (add pytest marker)

**Interfaces:**
- Produces: `load_yaml(path: str | Path) -> dict`; `apply_overrides(d: dict, overrides: list[str]) -> dict` where each override is `"key=value"` and value is parsed as YAML (so `lr=1e-3` → float, `compile=false` → bool); `from_dict(cls, d: dict)` which raises `KeyError` on unknown keys; `to_yaml(obj, path)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
from dataclasses import dataclass
import pytest
from rankfile.config import load_yaml, apply_overrides, from_dict, to_yaml

@dataclass
class Cfg:
    lr: float = 1e-3
    name: str = "x"
    compile: bool = True

def test_roundtrip_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    to_yaml(Cfg(lr=2e-3), p)
    d = load_yaml(p)
    assert d == {"lr": 2e-3, "name": "x", "compile": True}

def test_overrides_parse_yaml_scalars():
    d = apply_overrides({"lr": 1e-3, "compile": True}, ["lr=5e-4", "compile=false"])
    assert d["lr"] == 5e-4 and d["compile"] is False

def test_override_unknown_key_rejected():
    with pytest.raises(KeyError):
        apply_overrides({"lr": 1.0}, ["nope=1"])

def test_from_dict_rejects_unknown():
    assert from_dict(Cfg, {"lr": 0.5}).lr == 0.5
    with pytest.raises(KeyError):
        from_dict(Cfg, {"bogus": 1})
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rankfile.config'`

- [ ] **Step 3: Implement**

```python
# src/rankfile/config.py
"""YAML config loading with key=value overrides into dataclasses."""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return {} if d is None else dict(d)


def to_yaml(obj: Any, path: str | Path) -> None:
    d = asdict(obj) if is_dataclass(obj) else dict(obj)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    out = dict(d)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        if k not in out:
            raise KeyError(f"unknown config key {k!r}; known: {sorted(out)}")
        out[k] = yaml.safe_load(v)
    return out


def from_dict(cls: type[T], d: dict) -> T:
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**d)
```

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["gpu: requires a CUDA device"]
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/config.py tests/test_config.py pyproject.toml
git commit -m "config: YAML loading with key=value overrides into dataclasses"
```

---

### Task 2: Transformer core (SDPA path)

**Files:**
- Create: `src/rankfile/model.py`
- Test: `tests/test_model.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `ModelConfig` dataclass; `Transformer(cfg).forward(idx: LongTensor[B,T], doc_ids: LongTensor[B,T] | None = None) -> FloatTensor[B,T,V]`; `RMSNorm`; `precompute_rope(head_dim, max_seq_len, theta) -> (cos, sin)` each `[1,1,T,head_dim//2]`; `apply_rope(x[B,H,T,D], cos, sin)`.
- Consumes: nothing.

- [ ] **Step 1: conftest with a CUDA skip marker and a tiny config fixture**

```python
# tests/conftest.py
import pytest
import torch

def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)

@pytest.fixture
def tiny_cfg():
    from rankfile.model import ModelConfig
    return ModelConfig(vocab_size=512, n_layer=2, d_model=64, n_head=4, n_kv_head=2,
                       head_dim=16, d_ff=128, max_seq_len=64)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_model.py
import torch
import pytest
from rankfile.model import ModelConfig, Transformer, RMSNorm, precompute_rope, apply_rope

def test_forward_shape(tiny_cfg):
    m = Transformer(tiny_cfg)
    idx = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    out = m(idx)
    assert out.shape == (2, 16, tiny_cfg.vocab_size)

def test_tied_embeddings_share_storage(tiny_cfg):
    m = Transformer(tiny_cfg)
    assert m.lm_head_weight().data_ptr() == m.embed.weight.data_ptr()

def test_gqa_kv_projection_width(tiny_cfg):
    m = Transformer(tiny_cfg)
    blk = m.blocks[0].attn
    assert blk.k.weight.shape == (tiny_cfg.n_kv_head * tiny_cfg.head_dim, tiny_cfg.d_model)
    assert blk.q.weight.shape == (tiny_cfg.n_head * tiny_cfg.head_dim, tiny_cfg.d_model)

def test_no_biases(tiny_cfg):
    m = Transformer(tiny_cfg)
    assert all("bias" not in n for n, _ in m.named_parameters())

def test_rmsnorm_unit_scale():
    n = RMSNorm(8)
    x = torch.randn(3, 8) * 5
    y = n(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3), atol=1e-4)

def test_rope_preserves_norm():
    cos, sin = precompute_rope(16, 32, 10000.0)
    x = torch.randn(1, 2, 32, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)

def test_causal_no_future_leak(tiny_cfg):
    m = Transformer(tiny_cfg).eval()
    idx = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    a = m(idx)[0, :6]
    idx2 = idx.clone(); idx2[0, 6:] = 1
    b = m(idx2)[0, :6]
    assert torch.allclose(a, b, atol=1e-5)

def test_init_scales(tiny_cfg):
    m = Transformer(tiny_cfg)
    assert abs(m.embed.weight.std().item() - tiny_cfg.init_std) < 0.005
    o = m.blocks[0].attn.o.weight.std().item()
    expected = tiny_cfg.init_std / (2 * tiny_cfg.n_layer) ** 0.5
    assert abs(o - expected) < 0.005
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rankfile.model'`

- [ ] **Step 4: Implement the model (SDPA path only; doc masking comes in Task 3)**

```python
# src/rankfile/model.py
"""Qwen3-style decoder: RMSNorm pre-norm, SwiGLU, RoPE, GQA, QK-norm, tied embeddings.

Single file, no framework. See CLAUDE.md §4 for the architecture table and why
each choice was made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 12
    d_model: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    head_dim: int = 64
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    init_std: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv)  # [T, D/2]
    return freqs.cos()[None, None], freqs.sin()[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B,H,T,D]; cos/sin: [1,1,>=T,D/2]. Interleaved-pair rotation, computed in fp32."""
    T = x.shape[2]
    cos, sin = cos[:, :, :T].to(x.dtype), sin[:, :, :T].to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_norm(self.q(x).view(B, T, self.n_head, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(B, T, self.n_kv_head, self.head_dim)).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if block_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            y = _flex(q, k, v, block_mask)
        return self.o(y.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim))


def _flex(q, k, v, block_mask):  # replaced in Task 3
    raise NotImplementedError("document masking is implemented in Task 3")


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, block_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, block_mask)
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.rms_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # scale residual-writing projections by 1/sqrt(2L) as in GPT-2/Llama
        for blk in self.blocks:
            for w in (blk.attn.o.weight, blk.mlp.down.weight):
                nn.init.normal_(w, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def lm_head_weight(self) -> torch.Tensor:
        return self.embed.weight if self.cfg.tie_embeddings else self.lm_head.weight

    def hidden(self, idx: torch.Tensor, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        block_mask = None if doc_ids is None else doc_block_mask(doc_ids)
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin, block_mask)
        return self.norm_f(x)

    def forward(self, idx: torch.Tensor, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(self.hidden(idx, doc_ids), self.lm_head_weight())


def doc_block_mask(doc_ids: torch.Tensor):  # implemented in Task 3
    raise NotImplementedError
```

- [ ] **Step 5: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add src/rankfile/model.py tests/test_model.py tests/conftest.py
git commit -m "model: Qwen3-style transformer with GQA, QK-norm, RoPE, tied embeddings"
```

---

### Task 3: Intra-document masking with FlexAttention

**Files:**
- Modify: `src/rankfile/model.py` (replace `_flex` and `doc_block_mask`)
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Produces: `doc_block_mask(doc_ids: LongTensor[B,T]) -> BlockMask` (causal AND same-document); `Transformer.forward(idx, doc_ids)` uses it when `doc_ids` is given.
- Semantics of `doc_ids`: token t belongs to document `doc_ids[b,t]`; ids are non-decreasing along T. Plan 2 produces them as `cumsum(x == EOT_ID) - (x == EOT_ID)` so each EOT token belongs to the document it ends.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:

```python
def test_doc_mask_blocks_cross_document_attention(tiny_cfg):
    """Output for doc 2 with masking must equal running doc 2 alone (up to fp error)."""
    torch.manual_seed(0)
    m = Transformer(tiny_cfg).eval()
    T = 16
    idx = torch.randint(1, tiny_cfg.vocab_size, (1, T))
    doc_ids = torch.zeros(1, T, dtype=torch.long); doc_ids[0, 8:] = 1
    with torch.no_grad():
        masked = m(idx, doc_ids)[0, 8:]
        alone = m(idx[:, 8:], pos_offset=8)[0]   # same RoPE positions as inside the window
    assert torch.allclose(masked, alone, atol=1e-4), (masked - alone).abs().max()

def test_doc_mask_equals_causal_when_single_doc(tiny_cfg):
    torch.manual_seed(0)
    m = Transformer(tiny_cfg).eval()
    idx = torch.randint(1, tiny_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        a = m(idx)
        b = m(idx, torch.zeros(2, 16, dtype=torch.long))
    assert torch.allclose(a, b, atol=1e-4)
```

Note: RoPE positions are absolute within the window, so "doc 2 alone" only matches
if the reference run uses the same positions. That is why `forward` gains a
`pos_offset` argument in this task, and why the loop in Plan 4 builds the block
mask itself: `hidden`/`forward`/`loss` all accept an optional prebuilt
`block_mask` so the training loop can call `doc_block_mask` outside the
compiled region and pass the result in.

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -k doc_mask -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement**

Replace the two stubs in `src/rankfile/model.py`:

```python
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


def _flex(q, k, v, block_mask: BlockMask):
    return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)


def doc_block_mask(doc_ids: torch.Tensor) -> BlockMask:
    """Causal AND same-document block mask for FlexAttention. doc_ids: [B,T] long."""
    B, T = doc_ids.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    return create_block_mask(mask_mod, B, None, T, T, device=doc_ids.device)
```

and add `pos_offset` plumbing:

```python
    def hidden(self, idx, doc_ids=None, block_mask=None, pos_offset: int = 0):
        if block_mask is None and doc_ids is not None:
            block_mask = doc_block_mask(doc_ids)
        T = idx.shape[1]
        cos = self.rope_cos[:, :, pos_offset:pos_offset + T]
        sin = self.rope_sin[:, :, pos_offset:pos_offset + T]
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin, block_mask)
        return self.norm_f(x)

    def forward(self, idx, doc_ids=None, block_mask=None, pos_offset: int = 0):
        return F.linear(self.hidden(idx, doc_ids, block_mask, pos_offset), self.lm_head_weight())
```

**Interface produced for later plans:** `forward(idx, doc_ids=None, block_mask=None, pos_offset=0)`. Callers pass either `doc_ids` (mask built inside) or a prebuilt `block_mask` (Plan 4 does this so the mask is built outside `torch.compile`).

`apply_rope` already slices `cos[:, :, :T]`, which is still correct after the offset slice.

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: 10 passed (flex_attention runs its eager fallback on CPU; slow but fine at T=16)

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/model.py tests/test_model.py
git commit -m "model: intra-document causal masking via FlexAttention block mask"
```

---

### Task 4: Chunked loss, parameter groups, parameter counts

**Files:**
- Modify: `src/rankfile/model.py`
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Produces: `Transformer.loss(idx, targets, doc_ids=None, block_mask=None, chunk: int = 512) -> Tensor` (mean CE over all tokens, computed on fp32 logits in chunks along T so the full `[B,T,V]` fp32 logits are never materialized; `block_mask` is forwarded to `hidden`); `Transformer.hidden_matrix_params() -> list[nn.Parameter]` (every 2-D weight inside `blocks`, i.e. what Muon optimizes); `Transformer.other_params() -> list[nn.Parameter]` (embedding, `lm_head` if untied, all RMSNorm weights); `Transformer.num_params() -> tuple[int, int]` (total, non-embedding).

- [ ] **Step 1: Write the failing tests**

```python
def test_loss_matches_unchunked(tiny_cfg):
    torch.manual_seed(0)
    m = Transformer(tiny_cfg)
    idx = torch.randint(0, tiny_cfg.vocab_size, (2, 16)); tgt = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    ref = torch.nn.functional.cross_entropy(m(idx).float().view(-1, tiny_cfg.vocab_size), tgt.view(-1))
    assert torch.allclose(m.loss(idx, tgt, chunk=5), ref, atol=1e-5)

def test_param_groups_partition(tiny_cfg):
    m = Transformer(tiny_cfg)
    hidden, other = m.hidden_matrix_params(), m.other_params()
    ids = {id(p) for p in m.parameters()}
    assert {id(p) for p in hidden} | {id(p) for p in other} == ids
    assert not ({id(p) for p in hidden} & {id(p) for p in other})
    assert all(p.ndim == 2 for p in hidden)
    assert id(m.embed.weight) in {id(p) for p in other}

def test_num_params_m124_shape():
    m = Transformer(ModelConfig())
    total, nonemb = m.num_params()
    assert abs(total - 100.7e6) < 0.5e6 and abs(nonemb - 75.5e6) < 0.5e6
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -k "loss or param" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement** (add methods to `Transformer`)

```python
    def loss(self, idx, targets, doc_ids=None, block_mask=None, chunk: int = 512) -> torch.Tensor:
        h = self.hidden(idx, doc_ids, block_mask)          # [B,T,D]
        w = self.lm_head_weight()
        B, T, _ = h.shape
        total = h.new_zeros((), dtype=torch.float32)
        for s in range(0, T, chunk):
            logits = F.linear(h[:, s:s + chunk], w).float()  # [B,c,V] fp32, one chunk at a time
            total = total + F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets[:, s:s + chunk].reshape(-1), reduction="sum")
        return total / (B * T)

    def hidden_matrix_params(self) -> list[nn.Parameter]:
        return [p for n, p in self.blocks.named_parameters() if p.ndim == 2]

    def other_params(self) -> list[nn.Parameter]:
        hidden = {id(p) for p in self.hidden_matrix_params()}
        return [p for p in self.parameters() if id(p) not in hidden]

    def num_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel() + (0 if self.cfg.tie_embeddings else self.lm_head.weight.numel())
        return total, total - emb
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/model.py tests/test_model.py
git commit -m "model: chunked cross-entropy, Muon/AdamW parameter groups, param counts"
```

---

### Task 5: Model configs m30 and m124

**Files:**
- Create: `configs/model/m124.yaml`, `configs/model/m30.yaml`
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Produces: the two YAML files, loadable with `from_dict(ModelConfig, load_yaml(path))`.

- [ ] **Step 1: Write the failing test**

```python
def test_model_configs_load_and_have_expected_sizes():
    from rankfile.config import load_yaml, from_dict
    m124 = from_dict(ModelConfig, load_yaml("configs/model/m124.yaml"))
    m30 = from_dict(ModelConfig, load_yaml("configs/model/m30.yaml"))
    assert (m124.n_layer, m124.d_model, m124.n_head, m124.n_kv_head) == (12, 768, 12, 4)
    assert m30.vocab_size == m124.vocab_size == 32768 and m30.max_seq_len == 2048
    total, _ = Transformer(m30).num_params()
    assert total < 25e6
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -k configs -v`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Write the configs**

```yaml
# configs/model/m124.yaml — the real model (CLAUDE.md §4). ~100.7M params, 75.5M non-embedding.
vocab_size: 32768
n_layer: 12
d_model: 768
n_head: 12
n_kv_head: 4
head_dim: 64
d_ff: 2048
max_seq_len: 2048
rope_theta: 10000.0
rms_eps: 1.0e-6
tie_embeddings: true
init_std: 0.02
```

```yaml
# configs/model/m30.yaml — smoke-test model. Same tokenizer and context as m124; must run end to end in minutes.
vocab_size: 32768
n_layer: 4
d_model: 256
n_head: 4
n_kv_head: 2
head_dim: 64
d_ff: 704
max_seq_len: 2048
rope_theta: 10000.0
rms_eps: 1.0e-6
tie_embeddings: true
init_std: 0.02
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add configs/model/m124.yaml configs/model/m30.yaml tests/test_model.py
git commit -m "config: m124 and m30 model configs"
```

---

### Task 6: GPU sanity test for the real shape

**Files:**
- Test: `tests/test_model_gpu.py`

**Interfaces:**
- Consumes: `Transformer`, `ModelConfig`, `doc_block_mask`.

- [ ] **Step 1: Write the test**

```python
# tests/test_model_gpu.py
import pytest, torch
from rankfile.model import ModelConfig, Transformer

pytestmark = pytest.mark.gpu

def test_m124_forward_backward_under_memory_ceiling():
    torch.cuda.reset_peak_memory_stats()
    m = Transformer(ModelConfig()).cuda()
    idx = torch.randint(0, 32768, (8, 2048), device="cuda"); tgt = torch.randint(0, 32768, (8, 2048), device="cuda")
    doc = torch.zeros(8, 2048, dtype=torch.long, device="cuda"); doc[:, 1024:] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m.loss(idx, tgt, doc)
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert torch.cuda.max_memory_allocated() / 2**30 < 14.0
    assert abs(loss.item() - 10.4) < 1.0  # ln(32768)=10.4 at init
```

- [ ] **Step 2: Run**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model_gpu.py -v`
Expected: PASS. If the memory assertion fails, lower `chunk` in `loss()` to 256; do not shrink the model.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_gpu.py
git commit -m "model: GPU test for m124 loss, doc mask, and memory ceiling"
```

---

### Task 7: Tokenizer train / load / encode

**Files:**
- Create: `src/rankfile/tokenizer.py`, `scripts/train_tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces: `EOT = "<|endoftext|>"`, `EOT_ID = 0`; `train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: Path) -> Tokenizer`; `load_tokenizer(path) -> Tokenizer`; `encode_docs(tok, texts: list[str]) -> list[np.ndarray]` where each array is `uint16` token ids **with `EOT_ID` appended**; `decode(tok, ids) -> str`.
- Plan 2 consumes `encode_docs` and `EOT_ID`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tokenizer.py
import numpy as np
from rankfile.tokenizer import EOT_ID, train_tokenizer, load_tokenizer, encode_docs, decode

CORPUS = ["the quick brown fox jumps over the lazy dog. " * 20, "def f(x):\n    return x * 2\n" * 20,
          "Numbers 1234 and symbols !@# and unicode é ñ 中文 " * 20]

def test_train_load_roundtrip(tmp_path):
    p = tmp_path / "tok.json"
    tok = train_tokenizer(CORPUS, vocab_size=600, out_path=p)
    tok2 = load_tokenizer(p)
    assert tok2.get_vocab_size() == 600
    assert tok2.token_to_id("<|endoftext|>") == EOT_ID == 0
    s = "unicode é ñ 中文 and code def f(x):"
    assert decode(tok2, tok2.encode(s).ids) == s

def test_encode_docs_appends_eot_and_is_uint16(tmp_path):
    tok = train_tokenizer(CORPUS, vocab_size=600, out_path=tmp_path / "t.json")
    arrs = encode_docs(tok, ["hello world", "second doc"])
    assert len(arrs) == 2 and all(a.dtype == np.uint16 for a in arrs)
    assert all(a[-1] == EOT_ID for a in arrs)
    assert all((a[:-1] != EOT_ID).all() for a in arrs)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/tokenizer.py
"""32k byte-level BPE trained on the pretraining data. EOT is id 0 and ends every document."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
EOT_ID = 0


def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: str | Path) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    assert tok.token_to_id(EOT) == EOT_ID
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode_docs(tok: Tokenizer, texts: list[str]) -> list[np.ndarray]:
    out = []
    for enc in tok.encode_batch(texts):
        ids = np.asarray(enc.ids + [EOT_ID], dtype=np.uint16)
        out.append(ids)
    return out


def decode(tok: Tokenizer, ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=False)
```

```python
# scripts/train_tokenizer.py
"""Train the 32k tokenizer on a slice of FineWeb-Edu.

Usage: python scripts/train_tokenizer.py --out data/tokenizer.json --docs 200000
"""
import argparse, itertools, os
from datasets import load_dataset
from rankfile.tokenizer import train_tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tokenizer.json")
    ap.add_argument("--docs", type=int, default=200_000)
    ap.add_argument("--vocab", type=int, default=32768)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    texts = (r["text"] for r in itertools.islice(ds, a.docs))
    tok = train_tokenizer(texts, a.vocab, a.out)
    print(f"saved {a.out}: vocab {tok.get_vocab_size()}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py -v`
Expected: 2 passed

- [ ] **Step 5: Train the real tokenizer** (network; ~5 minutes; writes `data/tokenizer.json`, which is gitignored)

Run: `set HF_HOME=D:\hf_cache && .venv\Scripts\python.exe scripts/train_tokenizer.py`
Expected: `saved data/tokenizer.json: vocab 32768`

- [ ] **Step 6: Commit**

```bash
git add src/rankfile/tokenizer.py scripts/train_tokenizer.py tests/test_tokenizer.py
git commit -m "tok: byte-level BPE training, loading, and document encoding with EOT"
```

---

## Self-review

- **Spec coverage:** §4.1 architecture (Task 2), intra-document masking (Task 3), tied embeddings and 32k tokenizer (Tasks 2, 7), parameter groups needed by Plan 3 (Task 4), m30 smoke config (Task 5), memory ceiling (Task 6). Positional encoding θ=10000 per CLAUDE.md §4.
- **Placeholders:** none; every step has code.
- **Type consistency:** `doc_ids` is `LongTensor[B,T]` everywhere; `encode_docs` returns `uint16` arrays ending in `EOT_ID=0`; `hidden_matrix_params`/`other_params` names match what Plan 3 consumes.
