"""Test ft_grid: fine-tuning grid generator."""
from scripts.ft_grid import grid_lines


def test_grid_lines_cover_all_cells():
    lines = grid_lines(
        ["runs/p1_adamw_m124_s0", "runs/p2_muon_m124_s0"],
        ranks=[4, 16, 64],
        tasks=["code", "sup"],
        full_lr=1e-4,
        lora_lr=1e-3,
        py=".venv\\Scripts\\python.exe",
    )
    assert len(lines) == 2 * (1 + 3) * 2
    assert any(
        "--method full --task code --lr 0.0001 --name p1_adamw_m124_s0__full_code"
        in l
        for l in lines
    )
    assert any(
        "--method lora --task sup --rank 64 --lr 0.001 --name p2_muon_m124_s0__lora64_sup"
        in l
        for l in lines
    )
