"""Environment sanity check for rank-and-file.

Run this before any long job. It verifies the things that were found to matter
on this machine (RTX 5070 Ti, native Windows, PyTorch cu130 + triton-windows):

  1. GPU is sm_120 and bf16 works; bf16 matmul throughput is in the expected range.
  2. Which SDPA backends have kernels (flash has none in the Windows build; cuDNN does).
  3. torch.compile works (Triton importable) and the m124-shaped model hits the
     expected compiled throughput at micro-batch 8 without spilling into system RAM.
  4. FlexAttention with a document mask compiles and matches a dense-mask reference.

Usage:  python scripts/env_check.py [--quick]
Exit code is non-zero if any hard requirement fails.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECT_SM = (12, 0)
EXPECT_MATMUL_TFLOPS_MIN = 80.0  # measured 100 on 2026-09-01
EXPECT_COMPILED_TOKS_MIN = 60_000  # measured 80k on 2026-09-01
MEM_CEILING_GIB = 14.0  # above this, WDDM starts paging to system RAM silently

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def check_device() -> None:
    print("[device]")
    if not torch.cuda.is_available():
        fail("CUDA not available")
        return
    p = torch.cuda.get_device_properties(0)
    ok(f"{p.name} | sm_{p.major}{p.minor} | {p.total_memory / 2**30:.1f} GiB | {p.multi_processor_count} SMs")
    ok(f"torch {torch.__version__} | cuda {torch.version.cuda} | cudnn {torch.backends.cudnn.version()}")
    if (p.major, p.minor) != EXPECT_SM:
        fail(f"expected sm_{EXPECT_SM[0]}{EXPECT_SM[1]}")
    if not torch.cuda.is_bf16_supported():
        fail("bf16 not supported")
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        a @ b
    torch.cuda.synchronize()
    tflops = 2 * n**3 / ((time.perf_counter() - t) / 10) / 1e12
    (ok if tflops >= EXPECT_MATMUL_TFLOPS_MIN else fail)(f"bf16 matmul {tflops:.0f} TFLOPS (expect >= {EXPECT_MATMUL_TFLOPS_MIN:.0f})")
    del a, b


def check_sdpa() -> None:
    print("[sdpa backends]")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(8, 12, 2048, 64, device="cuda", dtype=torch.bfloat16)
    usable = []
    for name, be in [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ]:
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
                torch.cuda.synchronize()
            ok(f"{name}: kernel available")
            usable.append(name)
        except RuntimeError:
            print(f"  info  {name}: no kernel in this build")
    if "cudnn" not in usable and "efficient" not in usable:
        fail("no fast SDPA backend available")


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class _RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.w, 1e-6)


class _Block(nn.Module):
    """Qwen3-shaped block matching configs/model/m124: D=768, H=12, KV=4, HD=64, FF=2048."""

    D, H, KVH, HD, FF = 768, 12, 4, 64, 2048

    def __init__(self):
        super().__init__()
        D, H, KVH, HD, FF = self.D, self.H, self.KVH, self.HD, self.FF
        self.n1, self.n2 = _RMSNorm(D), _RMSNorm(D)
        self.q, self.k, self.v = nn.Linear(D, H * HD, bias=False), nn.Linear(D, KVH * HD, bias=False), nn.Linear(D, KVH * HD, bias=False)
        self.o = nn.Linear(H * HD, D, bias=False)
        self.qn, self.kn = _RMSNorm(HD), _RMSNorm(HD)
        self.gate, self.up, self.down = nn.Linear(D, FF, bias=False), nn.Linear(D, FF, bias=False), nn.Linear(FF, D, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.qn(self.q(h).view(B, T, self.H, self.HD)).transpose(1, 2)
        k = self.kn(self.k(h).view(B, T, self.KVH, self.HD)).transpose(1, 2)
        v = self.v(h).view(B, T, self.KVH, self.HD).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, self.H * self.HD))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class _Model(nn.Module):
    V, L, T = 32768, 12, 2048

    def __init__(self):
        super().__init__()
        D, HD = _Block.D, _Block.HD
        self.emb = nn.Embedding(self.V, D)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([_Block() for _ in range(self.L)])
        self.nf = _RMSNorm(D)
        inv = 1.0 / (10000 ** (torch.arange(0, HD, 2).float() / HD))
        fr = torch.outer(torch.arange(self.T).float(), inv)
        self.register_buffer("cos", fr.cos()[None, None])
        self.register_buffer("sin", fr.sin()[None, None])

    def forward(self, idx: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = F.linear(self.nf(x), self.emb.weight)
        return F.cross_entropy(logits.float().view(-1, self.V), tgt.view(-1))


def check_compile_throughput(micro_batch: int = 8) -> None:
    print(f"[compile throughput, m124 shape, micro-batch {micro_batch}x2048]")
    try:
        import triton  # noqa: F401

        ok(f"triton {triton.__version__} importable")
    except ImportError:
        fail("triton not importable; install triton-windows (Windows) or triton (Linux)")
        return
    torch.cuda.reset_peak_memory_stats()
    m = _Model().cuda()
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    cm = torch.compile(m)
    idx = torch.randint(0, m.V, (micro_batch, m.T), device="cuda")
    tgt = torch.randint(0, m.V, (micro_batch, m.T), device="cuda")

    def step() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = cm(idx, tgt)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    t0 = time.perf_counter()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    ok(f"compiled in {time.perf_counter() - t0:.0f}s ({n_params / 1e6:.0f}M params)")
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        loss = step()
    torch.cuda.synchronize()
    toks = micro_batch * m.T / ((time.perf_counter() - t) / 10)
    peak = torch.cuda.max_memory_allocated() / 2**30
    (ok if toks >= EXPECT_COMPILED_TOKS_MIN else fail)(f"{toks / 1e3:.0f}k tok/s -> {1e9 / toks / 3600:.1f} h per 1B tokens (expect >= {EXPECT_COMPILED_TOKS_MIN / 1e3:.0f}k)")
    (ok if peak <= MEM_CEILING_GIB else fail)(f"peak memory {peak:.1f} GiB (ceiling {MEM_CEILING_GIB} GiB before WDDM paging)")
    if not torch.isfinite(loss):
        fail("non-finite loss")


def check_flex() -> None:
    print("[flex attention, document mask, GQA]")
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, H, KVH, T, D = 4, 12, 4, 2048, 64
    doc = (torch.arange(T, device="cuda") // 512).expand(B, T)

    def mask_mod(b, h, q, k):  # noqa: ANN001
        return (q >= k) & (doc[b, q] == doc[b, k])

    bm = create_block_mask(mask_mod, B, None, T, T, device="cuda")
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, KVH, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, KVH, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fa = torch.compile(flex_attention)
    try:
        out = fa(q, k, v, block_mask=bm, enable_gqa=True)
        out.sum().backward()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        fail(f"flex_attention failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        return
    bi = torch.arange(B, device="cuda")[:, None, None, None]
    qi = torch.arange(T, device="cuda")[None, None, :, None]
    ki = torch.arange(T, device="cuda")[None, None, None, :]
    dense = mask_mod(bi, torch.zeros(1, 1, 1, 1, dtype=torch.long, device="cuda"), qi, ki)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=dense, enable_gqa=True)
    diff = (out - ref).abs().max().item()
    (ok if diff < 2e-2 else fail)(f"matches dense-mask SDPA, max abs diff {diff:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the compile-throughput benchmark")
    args = ap.parse_args()
    check_device()
    if not torch.cuda.is_available():
        return 1
    check_sdpa()
    check_flex()
    if not args.quick:
        check_compile_throughput()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
