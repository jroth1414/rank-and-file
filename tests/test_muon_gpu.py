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
    torch.cuda.synchronize(); t = time.perf_counter()
    for o in opts:
        o.step()
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    assert dt < 0.5, f"Muon step took {dt:.2f}s; Newton-Schulz should be ~ms on 124M"
    assert all(torch.isfinite(p).all() for p in m.parameters())
    print(f"\nMuon step time: {dt:.4f}s")
