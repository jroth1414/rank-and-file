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
