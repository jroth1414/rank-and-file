import torch

from rankfile.spectra import (
    effective_rank,
    matrix_report,
    singular_values,
    stable_rank,
    subspace_overlap,
    top_r_energy,
)


def test_effective_rank_identity_and_rank_one():
    assert abs(effective_rank(singular_values(torch.eye(10))) - 10.0) < 1e-4
    u = torch.randn(10, 1); v = torch.randn(1, 7)
    assert abs(effective_rank(singular_values(u @ v)) - 1.0) < 1e-3

def test_stable_rank():
    s = torch.tensor([2.0, 1.0, 1.0])
    assert abs(stable_rank(s) - 6.0 / 4.0) < 1e-6

def test_top_r_energy_of_rank_r_matrix_is_one():
    W = torch.randn(20, 3) @ torch.randn(3, 15)
    s = singular_values(W)
    assert abs(top_r_energy(s, 3) - 1.0) < 1e-5 and top_r_energy(s, 1) < 1.0

def test_subspace_overlap_bounds():
    B = torch.randn(12, 2); A = torch.randn(2, 9)
    delta_in = B @ A
    assert abs(subspace_overlap(delta_in, B) - 1.0) < 1e-5
    Q, _ = torch.linalg.qr(torch.cat([B, torch.randn(12, 10)], 1))
    delta_out = Q[:, 2:5] @ torch.randn(3, 9)   # orthogonal to colspace(B)
    assert subspace_overlap(delta_out, B) < 1e-5

def test_matrix_report_parses_names():
    r = matrix_report("blocks.3.mlp.down.weight", torch.randn(32, 64))
    assert r["layer"] == 3 and r["module"] == "mlp.down" and r["rows"] == 32 and r["cols"] == 64
    assert 1.0 <= r["erank"] <= 32 and 0 < r["top4"] <= r["top16"] <= 1.0
