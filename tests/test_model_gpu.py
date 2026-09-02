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
