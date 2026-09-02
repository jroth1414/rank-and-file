# Plan 3: Optimizers and Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A from-scratch Muon optimizer following Liu et al. (2025) "Muon is Scalable" (Newton–Schulz orthogonalization, decoupled weight decay, RMS-matched update scaling), a builder that routes parameters to Muon or AdamW per arm, and the warmup-stable-decay schedule.

**Architecture:** `optim/muon.py` holds `zeropower_via_newtonschulz5` and the `Muon` optimizer. `optim/build.py` turns `(model, optimizer_name, lr, weight_decay, betas)` into a list of optimizers: `["adamw"]` → one fused AdamW over all params; `["muon"]` → Muon over `model.hidden_matrix_params()` and AdamW over `model.other_params()`. `schedule.py` is a pure function of step. Implementing Muon by hand is a course requirement (CLAUDE.md §6: no optimizer libraries).

**Tech Stack:** torch 2.14, pytest.

**Spec:** `proposal.md` §4.2 (AdamW betas 0.9/0.95, wd 0.1; Muon momentum 0.95 Nesterov with Liu et al. corrections; embeddings and norms on AdamW; WSD 2%/20%). `CLAUDE.md` §3 rule 4, §7 `test_muon.py` requirements.

## Global Constraints

- Python 3.11 in `.venv`; run as `.venv\Scripts\python.exe ...`.
- Muon applies only to 2-D weight matrices inside transformer blocks; 1-D params and embeddings go to AdamW. Both arms use the same weight decay 0.1.
- Update scaling: Liu et al. use `0.2 * sqrt(max(fan_out, fan_in))` so Muon's update RMS matches AdamW's, letting both arms share a learning-rate grid.
- Commit prefix `muon:` / `sched:`; no AI attribution trailers.

---

### Task 1: Newton–Schulz orthogonalization

**Files:**
- Create: `src/rankfile/optim/muon.py`
- Test: `tests/test_muon.py`

**Interfaces:**
- Produces: `zeropower_via_newtonschulz5(G: Tensor[m,n], steps: int = 5) -> Tensor[m,n]` returning an approximate orthogonal factor `U V^T` of `G`'s SVD; runs in bf16 internally on CUDA, fp32 on CPU.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_muon.py
import torch, pytest
from rankfile.optim.muon import zeropower_via_newtonschulz5

@pytest.mark.parametrize("shape", [(128, 32), (32, 128), (96, 64)])
def test_newton_schulz_singular_values_near_one(shape):
    """The Muon quintic (3.4445, -4.7750, 2.0315) trades exactness for speed: after 5 steps the
    singular values sit in roughly [0.7, 1.3], not at 1. Test that band, not U^T U == I."""
    torch.manual_seed(0)
    G = torch.randn(*shape)
    X = zeropower_via_newtonschulz5(G, steps=5)
    assert X.shape == G.shape
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.5 and s.max() < 1.5, (s.min().item(), s.max().item())

def test_newton_schulz_preserves_singular_directions():
    torch.manual_seed(0)
    G = torch.randn(48, 24)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    X = zeropower_via_newtonschulz5(G, steps=5)
    assert torch.allclose(X, U @ Vh, atol=0.3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_muon.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/optim/muon.py
"""Muon (Jordan et al. 2024) with the scaling and weight-decay fixes of Liu et al. 2025.

Muon replaces each 2-D weight's momentum-smoothed gradient with an approximate
orthogonalization computed by five quintic Newton–Schulz iterations, then scales
the update so its RMS matches what AdamW would produce (0.2 * sqrt(max(m, n))).
"""
from __future__ import annotations

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = _NS_COEFFS
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)  # spectral norm <= 1 so the iteration converges
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_muon.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/optim/muon.py tests/test_muon.py
git commit -m "muon: quintic Newton-Schulz orthogonalization"
```

---

### Task 2: The Muon optimizer

**Files:**
- Modify: `src/rankfile/optim/muon.py`
- Test: `tests/test_muon.py` (append)

**Interfaces:**
- Produces: `Muon(params, lr: float, momentum: float = 0.95, nesterov: bool = True, weight_decay: float = 0.1, ns_steps: int = 5)`, a `torch.optim.Optimizer` that raises `ValueError` if any param is not 2-D. Update per param: `buf = momentum*buf + g; g' = g + momentum*buf if nesterov else buf; O = NS(g') * 0.2*sqrt(max(m,n)); p -= lr*(O + weight_decay*p)`.

- [ ] **Step 1: Write the failing tests**

```python
from rankfile.optim.muon import Muon

def test_muon_rejects_non_2d():
    with pytest.raises(ValueError):
        Muon([torch.nn.Parameter(torch.zeros(4))], lr=0.1)

def test_muon_update_rms_is_scaled():
    """One step from zero momentum: update = lr * 0.2*sqrt(max(m,n)) * orth(g); RMS ≈ lr*0.2*sqrt(max/ min?)"""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(64, 32))
    opt = Muon([p], lr=1.0, momentum=0.0, nesterov=False, weight_decay=0.0)
    p.grad = torch.randn(64, 32)
    opt.step()
    # orth(g) has 32 unit singular values over a 64x32 matrix -> RMS = sqrt(32/(64*32)) = 1/8
    expected_rms = 0.2 * (64 ** 0.5) * (1 / 8)
    assert abs(p.pow(2).mean().sqrt().item() - expected_rms) / expected_rms < 0.2

def test_muon_weight_decay_is_decoupled():
    p = torch.nn.Parameter(torch.ones(8, 8))
    opt = Muon([p], lr=0.5, weight_decay=0.1)
    p.grad = torch.zeros(8, 8)
    opt.step()
    assert torch.allclose(p, torch.full((8, 8), 1 - 0.5 * 0.1))

def test_muon_reduces_quadratic_loss():
    """Sign-like orthogonalized descent without momentum converges to within one step of the target."""
    torch.manual_seed(0)
    target = torch.randn(16, 16)
    p = torch.nn.Parameter(torch.zeros(16, 16))
    opt = Muon([p], lr=0.3, momentum=0.0, nesterov=False, weight_decay=0.0)
    losses = []
    for _ in range(100):
        loss = (p - target).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1] < 0.2 * losses[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_muon.py -v`
Expected: 4 new FAIL with `ImportError: Muon`

- [ ] **Step 3: Implement** (append)

```python
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.1, ns_steps: int = 5):
        params = list(params)
        for p in params:
            if p.ndim != 2:
                raise ValueError(f"Muon only handles 2-D params, got shape {tuple(p.shape)}")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(mu).add_(g)
                upd = g.add(buf, alpha=mu) if group["nesterov"] else buf
                O = zeropower_via_newtonschulz5(upd, group["ns_steps"])
                O = O * (0.2 * max(p.shape[0], p.shape[1]) ** 0.5)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(O, alpha=-lr)
        return loss
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_muon.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/optim/muon.py tests/test_muon.py
git commit -m "muon: optimizer with Nesterov momentum, RMS-matched scaling, decoupled decay"
```

---

### Task 3: Optimizer builder

**Files:**
- Create: `src/rankfile/optim/build.py`
- Test: `tests/test_build_optim.py`

**Interfaces:**
- Consumes: `Transformer.hidden_matrix_params()`, `Transformer.other_params()` from Plan 1; `Muon`.
- Produces: `build_optimizers(model, name: str, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95)) -> list[torch.optim.Optimizer]`; `set_lr(opts: list, lr: float) -> None`; `optimizer_state_dicts(opts) -> list[dict]`; `load_optimizer_state_dicts(opts, sds) -> None`. `name` is `"adamw"` or `"muon"`.
- Weight decay on AdamW is applied to 2-D params only (embedding included, per Llama/OLMo practice) and not to RMSNorm gains, in **both** arms, so the only difference between arms is Muon-vs-AdamW on the hidden matrices.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_optim.py
import torch
from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers, set_lr
from rankfile.optim.muon import Muon

def _model():
    return Transformer(ModelConfig(vocab_size=128, n_layer=2, d_model=32, n_head=2, n_kv_head=1, head_dim=16, d_ff=64, max_seq_len=16))

def test_adamw_arm_single_optimizer_covers_all_params():
    m = _model()
    opts = build_optimizers(m, "adamw", lr=1e-3, weight_decay=0.1)
    assert len(opts) == 1 and isinstance(opts[0], torch.optim.AdamW)
    covered = {id(p) for g in opts[0].param_groups for p in g["params"]}
    assert covered == {id(p) for p in m.parameters()}
    wd = {g["weight_decay"] for g in opts[0].param_groups}
    assert wd == {0.0, 0.1}

def test_muon_arm_routes_hidden_matrices_to_muon():
    m = _model()
    opts = build_optimizers(m, "muon", lr=1e-3, weight_decay=0.1)
    assert [type(o) for o in opts] == [Muon, torch.optim.AdamW]
    muon_ids = {id(p) for g in opts[0].param_groups for p in g["params"]}
    assert muon_ids == {id(p) for p in m.hidden_matrix_params()}
    adam_ids = {id(p) for g in opts[1].param_groups for p in g["params"]}
    assert id(m.embed.weight) in adam_ids and not (muon_ids & adam_ids)

def test_set_lr_applies_everywhere():
    opts = build_optimizers(_model(), "muon", lr=1e-3, weight_decay=0.1)
    set_lr(opts, 5e-4)
    assert all(g["lr"] == 5e-4 for o in opts for g in o.param_groups)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_build_optim.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/optim/build.py
"""Route parameters to AdamW or Muon per arm. The only difference between arms is
which optimizer updates the 2-D hidden matrices; everything else is identical."""
from __future__ import annotations

import torch

from rankfile.optim.muon import Muon


def _adamw(groups: list[dict], lr: float, betas: tuple[float, float]) -> torch.optim.AdamW:
    on_cuda = all(p.is_cuda for g in groups for p in g["params"])
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=on_cuda)  # fused only for CUDA params; CPU stays deterministic


def build_optimizers(model, name: str, lr: float, weight_decay: float,
                     betas: tuple[float, float] = (0.9, 0.95)) -> list[torch.optim.Optimizer]:
    hidden = model.hidden_matrix_params()
    other = model.other_params()
    other_decay = [p for p in other if p.ndim >= 2]   # embeddings
    other_nodecay = [p for p in other if p.ndim < 2]  # RMSNorm gains
    if name == "adamw":
        groups = [dict(params=hidden + other_decay, weight_decay=weight_decay),
                  dict(params=other_nodecay, weight_decay=0.0)]
        return [_adamw(groups, lr, betas)]
    if name == "muon":
        groups = [dict(params=other_decay, weight_decay=weight_decay),
                  dict(params=other_nodecay, weight_decay=0.0)]
        return [Muon(hidden, lr=lr, weight_decay=weight_decay), _adamw(groups, lr, betas)]
    raise ValueError(f"unknown optimizer {name!r}; use 'adamw' or 'muon'")


def set_lr(opts: list[torch.optim.Optimizer], lr: float) -> None:
    for o in opts:
        for g in o.param_groups:
            g["lr"] = lr


def optimizer_state_dicts(opts: list[torch.optim.Optimizer]) -> list[dict]:
    return [o.state_dict() for o in opts]


def load_optimizer_state_dicts(opts: list[torch.optim.Optimizer], sds: list[dict]) -> None:
    assert len(opts) == len(sds)
    for o, sd in zip(opts, sds):
        o.load_state_dict(sd)
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_build_optim.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/optim/build.py tests/test_build_optim.py
git commit -m "muon: optimizer builder routing hidden matrices to Muon or AdamW per arm"
```

---

### Task 4: Warmup-stable-decay schedule

**Files:**
- Create: `src/rankfile/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Produces: `wsd_lr(step: int, total_steps: int, peak_lr: float, warmup_frac: float = 0.02, decay_frac: float = 0.2, final_ratio: float = 0.0) -> float`. Linear warmup from 0 to peak over `warmup_frac*total`, constant, then linear decay to `final_ratio*peak` over the last `decay_frac*total` steps. Plan 4 calls it every step and applies via `set_lr`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schedule.py
from rankfile.schedule import wsd_lr

def test_wsd_phases():
    T, peak = 1000, 1e-3
    assert wsd_lr(0, T, peak) == 0.0
    assert abs(wsd_lr(10, T, peak) - peak * 10 / 20) < 1e-12   # warmup is 2% = 20 steps
    assert wsd_lr(20, T, peak) == peak
    assert wsd_lr(500, T, peak) == peak
    assert wsd_lr(800, T, peak) == peak                        # decay starts at 80%
    assert abs(wsd_lr(900, T, peak) - peak * 0.5) < 1e-12
    assert wsd_lr(1000, T, peak) == 0.0

def test_wsd_final_ratio():
    assert abs(wsd_lr(1000, 1000, 1e-3, final_ratio=0.1) - 1e-4) < 1e-12
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/schedule.py
"""Warmup-stable-decay learning-rate schedule (linear warmup, flat, linear decay)."""


def wsd_lr(step: int, total_steps: int, peak_lr: float, warmup_frac: float = 0.02,
           decay_frac: float = 0.2, final_ratio: float = 0.0) -> float:
    warm = int(round(total_steps * warmup_frac))
    decay_start = total_steps - int(round(total_steps * decay_frac))
    if step < warm:
        return peak_lr * step / warm
    if step < decay_start:
        return peak_lr
    frac = (step - decay_start) / max(1, total_steps - decay_start)
    frac = min(1.0, frac)
    return peak_lr * (1.0 - frac * (1.0 - final_ratio))
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_schedule.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/schedule.py tests/test_schedule.py
git commit -m "sched: warmup-stable-decay learning-rate schedule"
```

---

### Task 5: GPU check that Muon steps the real model

**Files:**
- Test: `tests/test_muon_gpu.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_muon_gpu.py
import pytest, torch, time
from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers

pytestmark = pytest.mark.gpu

def test_muon_step_on_m124_is_fast_and_finite():
    m = Transformer(ModelConfig()).cuda()
    opts = build_optimizers(m, "muon", lr=1e-3, weight_decay=0.1)
    idx = torch.randint(0, 32768, (4, 2048), device="cuda"); tgt = torch.randint(0, 32768, (4, 2048), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m.loss(idx, tgt).backward()
    torch.cuda.synchronize(); t = time.perf_counter()
    for o in opts: o.step()
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    assert dt < 0.5, f"Muon step took {dt:.2f}s; Newton-Schulz should be ~ms on 124M"
    assert all(torch.isfinite(p).all() for p in m.parameters())
```

- [ ] **Step 2: Run**

Run: `.venv\Scripts\python.exe -m pytest tests/test_muon_gpu.py -v`
Expected: PASS. Record the measured step time in the commit body.

- [ ] **Step 3: Commit**

```bash
git add tests/test_muon_gpu.py
git commit -m "muon: GPU step-time test on m124"
```

---

## Self-review

- **Spec coverage:** Muon hyperparameters (momentum 0.95, Nesterov, wd, update scaling) Task 2; embeddings/norms on AdamW Task 3; AdamW betas Task 3; WSD 2%/20% Task 4; the "only difference is the optimizer" control is enforced by the shared weight-decay grouping in Task 3.
- **Placeholders:** none.
- **Type consistency:** `build_optimizers` returns a `list`; Plan 4 must iterate it for `step()`/`zero_grad()` and use `optimizer_state_dicts`/`load_optimizer_state_dicts` for checkpoints. `hidden_matrix_params`/`other_params` names match Plan 1 Task 4.
