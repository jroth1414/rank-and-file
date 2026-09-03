import time

import pytest
import torch

from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers

pytestmark = pytest.mark.gpu

def test_muon_step_on_m124_is_fast_and_finite():
    m = Transformer(ModelConfig()).cuda()
    opts = build_optimizers(m, "muon", lr=1e-3, weight_decay=0.1)
    idx = torch.randint(0, 32768, (4, 2048), device="cuda"); tgt = torch.randint(0, 32768, (4, 2048), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m.loss(idx, tgt).backward()

    # Untimed warmup step: absorbs CUDA context/kernel-cache setup so it doesn't pollute
    # the timing below. Grads persist (no zero_grad/backward between steps), so repeated
    # o.step() calls keep operating on the same gradients.
    for o in opts:
        o.step()

    n_trials = 5
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize(); t = time.perf_counter()
        for o in opts:
            o.step()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    times.sort()
    median_dt = times[n_trials // 2]

    assert median_dt < 0.5, f"Muon step took {median_dt:.2f}s median; Newton-Schulz should be ~ms on 124M"
    assert all(torch.isfinite(p).all() for p in m.parameters())
    print(f"\nMuon step median time over {n_trials} trials: {median_dt:.4f}s")

    del m, opts, idx, tgt
    torch.cuda.empty_cache()
