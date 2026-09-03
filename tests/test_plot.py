import csv

import pytest

from scripts.plot import make_figures

FIELDS = dict(rows=8, cols=8, srank=2.0, fro=1.0, top4=0.5, top16=0.8, top64=1.0)
PARENTS = [("p1_adamw_m30_s0", "adamw", "p1", 0), ("p2_muon_m30_s0", "muon", "p2", 1)]


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _pretrained_rows():
    return [
        dict(
            run=r, optimizer=o, arm=arm, seed=seed, val_loss=3.0,
            name=f"blocks.{l}.attn.q.weight", layer=l, module="attn.q",
            erank=5 + l, erank_norm=(5 + l) / 8, **FIELDS,
        )
        for r, o, arm, seed in PARENTS
        for l in range(2)
    ]


def _delta_rows():
    return [
        dict(
            run=f"{p}__full_{t}", parent=p, task=t, lr=1e-4, optimizer=o, arm=arm, seed=seed,
            name=f"blocks.{l}.attn.q.weight", layer=l, module="attn.q",
            erank=3 + l, erank_norm=(3 + l) / 8, **FIELDS,
        )
        for p, o, arm, seed in PARENTS
        for t in ("code", "sup")
        for l in range(2)
    ]


def _finetune_rows():
    return [
        dict(
            run=f"{p}__{m}_{t}", parent=p, task=t, method=("full" if m == "full" else "lora"),
            rank=("" if m == "full" else m[4:]), lr=1e-4, optimizer=o, arm=arm, seed=seed,
            metric_before=0, metric_full=1, metric_after=0.5,
            recovered=("nan" if m == "full" else 0.5), before_mismatch=0.0, forgetting=0.1,
            alpha="", parent_ckpt="",
        )
        for p, o, arm, seed in PARENTS
        for t in ("code", "sup")
        for m in ("full", "lora4", "lora16", "lora64")
    ]


def _overlap_rows():
    return [
        dict(
            run=f"{p}__lora{r}_{t}", parent=p, task=t, rank=r, name="blocks.0.attn.q",
            lr=1e-4, optimizer=o, arm=arm, seed=seed,
            overlap=0.3, overlap_two_sided=0.2, chance_one=r / 8, chance_two=(r * r) / (8 * 8),
        )
        for p, o, arm, seed in PARENTS
        for t in ("code", "sup")
        for r in (4, 16, 64)
    ]


def _traj_rows():
    return [
        dict(
            run=r, step=step, tokens=step * 100, val_loss=5.0 - step, optimizer=o, arm=arm, seed=seed,
            name=f"blocks.{l}.attn.q.weight", layer=l, module="attn.q",
            erank=3 + l, erank_norm=(3 + l) / 8, **FIELDS,
        )
        for r, o, arm, seed in PARENTS
        for step in (1, 2, 3)
        for l in range(2)
    ]


def _write_core_csvs(a):
    _csv(a / "pretrained_spectra.csv", _pretrained_rows())
    _csv(a / "delta_spectra.csv", _delta_rows())
    _csv(a / "finetune_results.csv", _finetune_rows())


def test_make_figures_writes_all(tmp_path):
    a = tmp_path / "analysis"
    _write_core_csvs(a)
    _csv(a / "lora_overlap.csv", _overlap_rows())
    out = make_figures(a, tmp_path / "figs")
    names = {p.name for p in out}
    for stem in (
        "fig_erank_by_layer", "fig_delta_erank", "fig_lora_gap", "fig_energy_vs_recovered",
        "fig_forgetting", "fig_overlap",
    ):
        assert f"{stem}.png" in names and f"{stem}.pdf" in names
    # no trajectory CSV -> no trajectory figure, but nothing else breaks
    assert "fig_erank_vs_loss.png" not in names


def test_make_figures_with_trajectory_csv(tmp_path):
    a = tmp_path / "analysis"
    _write_core_csvs(a)
    _csv(a / "lora_overlap.csv", _overlap_rows())
    _csv(a / "pretrained_spectra_traj.csv", _traj_rows())
    out = make_figures(a, tmp_path / "figs")
    names = {p.name for p in out}
    assert "fig_erank_vs_loss.png" in names and "fig_erank_vs_loss.pdf" in names


def test_make_figures_unknown_optimizer_raises(tmp_path):
    a = tmp_path / "analysis"
    rows = _pretrained_rows()
    for r in rows:
        if r["run"] == "p1_adamw_m30_s0":
            r["optimizer"] = "sgd"
    _csv(a / "pretrained_spectra.csv", rows)
    _csv(a / "delta_spectra.csv", _delta_rows())
    _csv(a / "finetune_results.csv", _finetune_rows())
    with pytest.raises(ValueError, match="sgd"):
        make_figures(a, tmp_path / "figs")


@pytest.mark.filterwarnings("error::UserWarning")
def test_make_figures_all_nan_task_panel_has_no_legend_warning(tmp_path):
    # One task's LoRA rows all have recovered = nan (e.g. the metric never
    # ran for that task), so its fig_lora_gap panel has no plotted lines.
    # An unconditional ax.legend() there emits "No artists with labels found
    # to put in legend" as a UserWarning; with the marker above that would
    # fail the test, so make_figures must guard the legend call and still
    # write all the figure files.
    a = tmp_path / "analysis"
    _csv(a / "pretrained_spectra.csv", _pretrained_rows())
    _csv(a / "delta_spectra.csv", _delta_rows())
    rows = [
        dict(
            run=f"{p}__{m}_{t}", parent=p, task=t, method=("full" if m == "full" else "lora"),
            rank=("" if m == "full" else m[4:]), lr=1e-4, optimizer=o, arm=arm, seed=seed,
            metric_before=0, metric_full=1, metric_after=0.5,
            # every "code" LoRA row is nan; "sup" is normal so its panel still gets a legend.
            recovered=("nan" if (m == "full" or t == "code") else 0.5),
            before_mismatch=0.0, forgetting=0.1, alpha="", parent_ckpt="",
        )
        for p, o, arm, seed in PARENTS
        for t in ("code", "sup")
        for m in ("full", "lora4", "lora16", "lora64")
    ]
    _csv(a / "finetune_results.csv", rows)
    out = make_figures(a, tmp_path / "figs")
    names = {p.name for p in out}
    for stem in ("fig_erank_by_layer", "fig_delta_erank", "fig_lora_gap", "fig_energy_vs_recovered", "fig_forgetting"):
        assert f"{stem}.png" in names and f"{stem}.pdf" in names
