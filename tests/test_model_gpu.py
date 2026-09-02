import pytest, torch
from rankfile.model import ModelConfig, Transformer, doc_block_mask

pytestmark = pytest.mark.gpu

def test_m124_compiled_loss_under_memory_ceiling():
    """Mirrors the training loop: block mask built outside, loss compiled. Eager flex would use ~18 GiB."""
    torch.cuda.reset_peak_memory_stats()
    m = Transformer(ModelConfig()).cuda()
    idx = torch.randint(0, 32768, (8, 2048), device="cuda"); tgt = torch.randint(0, 32768, (8, 2048), device="cuda")
    doc = torch.zeros(8, 2048, dtype=torch.long, device="cuda"); doc[:, 1024:] = 1
    bm = doc_block_mask(doc)
    loss_fn = torch.compile(m.loss, dynamic=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_fn(idx, tgt, block_mask=bm)
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nPeak memory: {peak:.2f} GiB, Loss: {loss.item():.4f}")
    assert torch.isfinite(loss)
    assert peak < 14.0, f"peak {peak:.1f} GiB"
    assert abs(loss.item() - 10.4) < 1.0, loss.item()  # ln(32768)=10.4 at init
