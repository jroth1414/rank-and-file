import pytest
import torch

from rankfile.optim.muon import zeropower_via_newtonschulz5


@pytest.mark.parametrize("shape", [(128, 32), (32, 128), (96, 64)])
def test_newton_schulz_singular_values_near_one(shape):
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
