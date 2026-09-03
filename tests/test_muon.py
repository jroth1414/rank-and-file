import pytest
import torch

from rankfile.optim.muon import Muon, zeropower_via_newtonschulz5


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


def test_muon_rejects_non_2d():
    with pytest.raises(ValueError):
        Muon([torch.nn.Parameter(torch.zeros(4))], lr=0.1)


def test_muon_update_rms_is_scaled():
    """One step from zero momentum: update = lr * 0.2*sqrt(max(m,n)) * orth(g). For a 64x32
    gradient, orth(g) has 32 unit singular values, so its RMS is sqrt(32/(64*32)) = 1/8, and the
    update RMS is lr * 0.2*sqrt(64) * 1/8 = 0.2*lr."""
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
    """Sign-like orthogonalized descent without momentum drives the loss below 20% of its
    initial value within 100 steps."""
    torch.manual_seed(0)
    target = torch.randn(16, 16)
    p = torch.nn.Parameter(torch.zeros(16, 16))
    opt = Muon([p], lr=0.3, momentum=0.0, nesterov=False, weight_decay=0.0)
    losses = []
    for _ in range(100):
        loss = (p - target).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1] < 0.2 * losses[0]


def test_muon_update_is_invariant_to_gradient_scale():
    """Newton-Schulz orthogonalization is (approximately) scale-invariant, so scaling the
    gradient should not change the update: two optimizers fed g and 32*g for three steps
    should stay bit-identical."""
    torch.manual_seed(0)
    g = torch.randn(16, 16)
    p1 = torch.nn.Parameter(torch.zeros(16, 16))
    p2 = torch.nn.Parameter(torch.zeros(16, 16))
    opt1 = Muon([p1], lr=0.1, momentum=0.95)
    opt2 = Muon([p2], lr=0.1, momentum=0.95)
    for _ in range(3):
        p1.grad = g.clone()
        p2.grad = (32 * g).clone()
        opt1.step()
        opt2.step()
    assert torch.equal(p1, p2)


def test_muon_rejects_param_group_dicts():
    with pytest.raises(ValueError):
        Muon([{"params": [torch.nn.Parameter(torch.zeros(4, 4))]}], lr=0.1)
