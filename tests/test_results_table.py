from scripts.results_table import results_table
from tests.test_plot import _csv


def test_results_table_has_rows(tmp_path):
    a = tmp_path / "analysis"
    _csv(a / "finetune_results.csv", [dict(run="p1__lora4_code", parent="p1_adamw_m30_s0", task="code", method="lora", rank=4,
         metric_before=-3.0, metric_full=-2.0, metric_after=-2.5, recovered=0.5, forgetting=0.1)])
    _csv(a / "pretrained_spectra.csv", [dict(run="p1_adamw_m30_s0", optimizer="adamw", arm="p1", seed=0, val_loss=3.0, name="n", layer=0,
         module="attn.q", rows=8, cols=8, erank=5.0, srank=2.0, fro=1.0, top4=0.5, top16=0.8, top64=1.0)])
    _csv(a / "delta_spectra.csv", [dict(run="p1__full_code", parent="p1_adamw_m30_s0", task="code", name="n", layer=0, module="attn.q",
         rows=8, cols=8, erank=3.0, srank=2.0, fro=1.0, top4=0.5, top16=0.8, top64=1.0)])
    md = results_table(a)
    assert "| p1_adamw_m30_s0 | code | lora4 |" in md and "0.500" in md and "adamw" in md
    assert (a / "results.md").exists()


def test_results_table_full_ft_and_empty_delta(tmp_path):
    a = tmp_path / "analysis"
    _csv(a / "finetune_results.csv", [
        dict(run="p1__full_code", parent="p1_adamw_m30_s0", task="code", method="full", rank="",
             metric_before=-3.0, metric_full=-2.0, metric_after=-2.5, recovered="nan", forgetting=0.1),
        dict(run="p1__lora64_code", parent="p1_adamw_m30_s0", task="code", method="lora", rank=64,
             metric_before=-3.0, metric_full=-2.0, metric_after=-2.4, recovered=0.5, forgetting=0.1)
    ])
    _csv(a / "pretrained_spectra.csv", [dict(run="p1_adamw_m30_s0", optimizer="adamw", arm="p1", seed=0, val_loss=3.0, name="n", layer=0,
         module="attn.q", rows=8, cols=8, erank=5.0, srank=2.0, fro=1.0, top4=0.5, top16=0.8, top64=1.0)])
    # Empty delta_spectra.csv (header only)
    a.mkdir(parents=True, exist_ok=True)
    with open(a / "delta_spectra.csv", "w", newline="") as f:
        f.write("run,parent,task,name,layer,module,rows,cols,erank,srank,fro,top4,top16,top64\n")

    md = results_table(a)

    # Check full-FT row renders with method "full" and empty recovered cell
    assert "| p1_adamw_m30_s0 | code | full |" in md
    # Check that "nan" does not appear in the markdown
    assert "nan" not in md
    # Check full row has empty recovered cell (between two pipes with just spaces)
    assert "| full | -2.5000 |  | 0.1000 |" in md
    # Check lora64 row
    assert "| p1_adamw_m30_s0 | code | lora64 |" in md and "0.500" in md
    # Check rows are ordered: full before lora64
    full_idx = md.index("| p1_adamw_m30_s0 | code | full |")
    lora_idx = md.index("| p1_adamw_m30_s0 | code | lora64 |")
    assert full_idx < lora_idx
    # Check results.md is written
    assert (a / "results.md").exists()
