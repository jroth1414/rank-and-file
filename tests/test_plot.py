import csv

import pytest

from scripts.plot import make_figures


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_make_figures_writes_all(tmp_path):
    a = tmp_path / "analysis"
    fields = dict(rows=8, cols=8, srank=2.0, fro=1.0, top4=0.5, top16=0.8, top64=1.0)
    _csv(
        a / "pretrained_spectra.csv",
        [
            dict(
                run=r,
                optimizer=o,
                arm=r[:2],
                seed=0,
                val_loss=3.0,
                name=f"blocks.{l}.attn.q.weight",
                layer=l,
                module="attn.q",
                erank=5 + l,
                **fields,
            )
            for r, o in [("p1_adamw_m30_s0", "adamw"), ("p2_muon_m30_s0", "muon")]
            for l in range(2)
        ],
    )
    _csv(
        a / "delta_spectra.csv",
        [
            dict(
                run=f"{p}__full_{t}",
                parent=p,
                task=t,
                name=f"blocks.{l}.attn.q.weight",
                layer=l,
                module="attn.q",
                erank=3 + l,
                **fields,
            )
            for p in ("p1_adamw_m30_s0", "p2_muon_m30_s0")
            for t in ("code", "sup")
            for l in range(2)
        ],
    )
    _csv(
        a / "lora_overlap.csv",
        [dict(run="x", parent="p1_adamw_m30_s0", task="code", rank=4, name="blocks.0.attn.q", overlap=0.3)],
    )
    _csv(
        a / "finetune_results.csv",
        [
            dict(
                run=f"{p}__{m}_{t}",
                parent=p,
                task=t,
                method=("full" if m == "full" else "lora"),
                rank=("" if m == "full" else m[4:]),
                metric_before=0,
                metric_full=1,
                metric_after=0.5,
                recovered=("nan" if m == "full" else 0.5),
                forgetting=0.1,
            )
            for p in ("p1_adamw_m30_s0", "p2_muon_m30_s0")
            for t in ("code", "sup")
            for m in ("full", "lora4", "lora16", "lora64")
        ],
    )
    out = make_figures(a, tmp_path / "figs")
    names = {p.name for p in out}
    for stem in ("fig_erank_by_layer", "fig_delta_erank", "fig_lora_gap", "fig_energy_vs_recovered", "fig_forgetting"):
        assert f"{stem}.png" in names and f"{stem}.pdf" in names


@pytest.mark.filterwarnings("error::UserWarning")
def test_make_figures_all_nan_task_panel_has_no_legend_warning(tmp_path):
    # One task's LoRA rows all have recovered = nan (e.g. the metric never
    # ran for that task), so its fig_lora_gap panel has no plotted lines.
    # An unconditional ax.legend() there emits "No artists with labels found
    # to put in legend" as a UserWarning; with the marker above that would
    # fail the test, so make_figures must guard the legend call and still
    # write all ten figure files.
    a = tmp_path / "analysis"
    fields = dict(rows=8, cols=8, srank=2.0, fro=1.0, top4=0.5, top16=0.8, top64=1.0)
    _csv(
        a / "pretrained_spectra.csv",
        [
            dict(
                run=r,
                optimizer=o,
                arm=r[:2],
                seed=0,
                val_loss=3.0,
                name=f"blocks.{l}.attn.q.weight",
                layer=l,
                module="attn.q",
                erank=5 + l,
                **fields,
            )
            for r, o in [("p1_adamw_m30_s0", "adamw"), ("p2_muon_m30_s0", "muon")]
            for l in range(2)
        ],
    )
    _csv(
        a / "delta_spectra.csv",
        [
            dict(
                run=f"{p}__full_{t}",
                parent=p,
                task=t,
                name=f"blocks.{l}.attn.q.weight",
                layer=l,
                module="attn.q",
                erank=3 + l,
                **fields,
            )
            for p in ("p1_adamw_m30_s0", "p2_muon_m30_s0")
            for t in ("code", "sup")
            for l in range(2)
        ],
    )
    _csv(
        a / "finetune_results.csv",
        [
            dict(
                run=f"{p}__{m}_{t}",
                parent=p,
                task=t,
                method=("full" if m == "full" else "lora"),
                rank=("" if m == "full" else m[4:]),
                metric_before=0,
                metric_full=1,
                metric_after=0.5,
                # every "code" LoRA row is nan; "sup" is normal so its panel
                # still gets a legend.
                recovered=("nan" if (m == "full" or t == "code") else 0.5),
                forgetting=0.1,
            )
            for p in ("p1_adamw_m30_s0", "p2_muon_m30_s0")
            for t in ("code", "sup")
            for m in ("full", "lora4", "lora16", "lora64")
        ],
    )
    out = make_figures(a, tmp_path / "figs")
    names = {p.name for p in out}
    for stem in ("fig_erank_by_layer", "fig_delta_erank", "fig_lora_gap", "fig_energy_vs_recovered", "fig_forgetting"):
        assert f"{stem}.png" in names and f"{stem}.pdf" in names
