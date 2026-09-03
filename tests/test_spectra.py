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


def test_effective_rank_zero_matrix():
    """effective_rank on all-zero matrix returns 0.0."""
    s = torch.zeros(5)
    assert effective_rank(s) == 0.0


def test_stable_rank_zero_spectrum():
    """stable_rank on zero spectrum returns 0.0."""
    s = torch.zeros(5)
    assert stable_rank(s) == 0.0


def test_subspace_overlap_zero_B():
    """subspace_overlap with B=zeros returns 0.0."""
    delta = torch.randn(10, 5)
    B = torch.zeros(10, 3)
    assert subspace_overlap(delta, B) == 0.0


def test_subspace_overlap_zero_delta():
    """subspace_overlap with zero delta returns 0.0."""
    B = torch.randn(10, 3)
    delta = torch.zeros(10, 5)
    assert subspace_overlap(delta, B) == 0.0


def test_subspace_overlap_rank_deficient_B():
    """subspace_overlap handles rank-deficient B correctly (duplicated column)."""
    B_single = torch.randn(12, 1)
    B_dup = torch.cat([B_single, B_single], 1)
    delta = torch.randn(12, 5)
    overlap_single = subspace_overlap(delta, B_single)
    overlap_dup = subspace_overlap(delta, B_dup)
    assert abs(overlap_single - overlap_dup) < 1e-5
