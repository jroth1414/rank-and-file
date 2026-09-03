import subprocess
import sys

from scripts.results_table import results_table
from tests.test_plot import _csv


def test_results_table_has_rows(tmp_path):
    a = tmp_path / "analysis"
    _csv(a / "finetune_results.csv", [dict(
        run="p1__lora4_code", parent="p1_adamw_m30_s0", task="code", method="lora", rank=4, lr=1e-4,
        optimizer="adamw", arm="p1", seed=0, metric_before=-3.0, metric_full=-2.0, metric_after=-2.5,
        recovered=0.5, before_mismatch=0.0, forgetting=0.1, alpha="", parent_ckpt="",
    )])
    _csv(a / "pretrained_spectra.csv", [dict(
        run="p1_adamw_m30_s0", optimizer="adamw", arm="p1", seed=0, val_loss=3.0, name="n", layer=0,
        module="attn.q", rows=8, cols=8, erank=5.0, erank_norm=0.625, srank=2.0, fro=1.0, top4=0.5,
        top16=0.8, top64=1.0,
    )])
    _csv(a / "delta_spectra.csv", [dict(
        run="p1__full_code", parent="p1_adamw_m30_s0", task="code", lr=1e-4, optimizer="adamw", arm="p1",
        seed=0, name="n", layer=0, module="attn.q", rows=8, cols=8, erank=3.0, erank_norm=0.375, srank=2.0,
        fro=1.0, top4=0.5, top16=0.8, top64=1.0,
    )])
    md = results_table(a)
    assert "| p1_adamw_m30_s0 | code | lora4 |" in md and "0.500" in md
    assert "| p1_adamw_m30_s0 | p1 | adamw | 0 | 3.0000 |" in md
    assert "| p1_adamw_m30_s0 | code | 3.00 |" in md
    assert "delta W" in md and "Δ" not in md
    assert (a / "results.md").exists()


def test_results_table_full_ft_and_undefined_recovered(tmp_path):
    a = tmp_path / "analysis"
    _csv(a / "finetune_results.csv", [
        dict(run="p1__full_code", parent="p1_adamw_m30_s0", task="code", method="full", rank="", lr=1e-4,
             optimizer="adamw", arm="p1", seed=0, metric_before=-3.0, metric_full=-2.0, metric_after=-2.5,
             recovered="nan", before_mismatch=0.0, forgetting=0.1, alpha="", parent_ckpt=""),
        dict(run="p1__lora64_code", parent="p1_adamw_m30_s0", task="code", method="lora", rank=64, lr=2e-4,
             optimizer="adamw", arm="p1", seed=0, metric_before=-3.0, metric_full=-2.0, metric_after=-2.4,
             recovered=0.5, before_mismatch=0.0, forgetting=0.1, alpha="", parent_ckpt=""),
    ])
    _csv(a / "pretrained_spectra.csv", [dict(
        run="p1_adamw_m30_s0", optimizer="adamw", arm="p1", seed=0, val_loss=3.0, name="n", layer=0,
        module="attn.q", rows=8, cols=8, erank=5.0, erank_norm=0.625, srank=2.0, fro=1.0, top4=0.5,
        top16=0.8, top64=1.0,
    )])
    # Empty delta_spectra.csv (header only)
    a.mkdir(parents=True, exist_ok=True)
    with open(a / "delta_spectra.csv", "w", newline="") as f:
        f.write("run,parent,task,lr,optimizer,arm,seed,name,layer,module,rows,cols,erank,erank_norm,srank,fro,top4,top16,top64\n")

    md = results_table(a)

    # Check full-FT row renders with method "full" and an empty (undefined) recovered cell.
    assert "| p1_adamw_m30_s0 | code | full |" in md
    assert "nan" not in md
    full_line = next(line for line in md.splitlines() if line.startswith("| p1_adamw_m30_s0 | code | full |"))
    assert full_line == "| p1_adamw_m30_s0 | code | full | -3.0000 | -2.0000 | -2.5000 |  | 0.1000 | 0.0001 |"
    # Check lora64 row
    assert "| p1_adamw_m30_s0 | code | lora64 |" in md and "0.500" in md
    # Check rows are ordered: full before lora64
    full_idx = md.index("| p1_adamw_m30_s0 | code | full |")
    lora_idx = md.index("| p1_adamw_m30_s0 | code | lora64 |")
    assert full_idx < lora_idx
    # Check results.md is written
    assert (a / "results.md").exists()


def test_results_table_never_pools_across_arms(tmp_path):
    a = tmp_path / "analysis"
    _csv(a / "finetune_results.csv", [dict(
        run="p1__full_code", parent="p1_adamw_m30_s0", task="code", method="full", rank="", lr=1e-4,
        optimizer="adamw", arm="p1", seed=0, metric_before=-3.0, metric_full=-2.0, metric_after=-2.0,
        recovered="nan", before_mismatch=0.0, forgetting=0.1, alpha="", parent_ckpt="",
    )])
    _csv(a / "pretrained_spectra.csv", [
        dict(run="p1_adamw_m30_s0", optimizer="adamw", arm="p1", seed=0, val_loss=3.5, name="n", layer=0,
             module="attn.q", rows=8, cols=8, erank=5.0, erank_norm=0.625, srank=2.0, fro=1.0, top4=0.5,
             top16=0.8, top64=1.0),
        dict(run="p3_adamw_m30_s0", optimizer="adamw", arm="p3", seed=0, val_loss=3.2, name="n", layer=0,
             module="attn.q", rows=8, cols=8, erank=6.0, erank_norm=0.75, srank=2.5, fro=1.0, top4=0.5,
             top16=0.8, top64=1.0),
    ])
    a.mkdir(parents=True, exist_ok=True)
    with open(a / "delta_spectra.csv", "w", newline="") as f:
        f.write("run,parent,task,lr,optimizer,arm,seed,name,layer,module,rows,cols,erank,erank_norm,srank,fro,top4,top16,top64\n")
    md = results_table(a)
    # both runs get their own row (not averaged into one "adamw" row).
    lines = [line for line in md.splitlines() if line.startswith("| p1_adamw_m30_s0 |") or line.startswith("| p3_adamw_m30_s0 |")]
    h1_lines = [line for line in lines if " p1 | adamw | 0 | " in line or " p3 | adamw | 0 | " in line]
    assert len(h1_lines) == 2


def test_results_table_cli_exits_zero(tmp_path):
    a = tmp_path / "analysis"
    _csv(a / "finetune_results.csv", [dict(
        run="p1__full_code", parent="p1_adamw_m30_s0", task="code", method="full", rank="", lr=1e-4,
        optimizer="adamw", arm="p1", seed=0, metric_before=-3.0, metric_full=-2.0, metric_after=-2.5,
        recovered="nan", before_mismatch=0.0, forgetting=0.1, alpha="", parent_ckpt="",
    )])
    _csv(a / "pretrained_spectra.csv", [dict(
        run="p1_adamw_m30_s0", optimizer="adamw", arm="p1", seed=0, val_loss=3.0, name="n", layer=0,
        module="attn.q", rows=8, cols=8, erank=5.0, erank_norm=0.625, srank=2.0, fro=1.0, top4=0.5,
        top16=0.8, top64=1.0,
    )])
    a.mkdir(parents=True, exist_ok=True)
    with open(a / "delta_spectra.csv", "w", newline="") as f:
        f.write("run,parent,task,lr,optimizer,arm,seed,name,layer,module,rows,cols,erank,erank_norm,srank,fro,top4,top16,top64\n")
    proc = subprocess.run(
        [sys.executable, "scripts/results_table.py", "--analysis", str(a)],
        capture_output=True,
    )
    assert proc.returncode == 0
