"""Every paper figure, generated from runs/analysis/*.csv. Never edit figures by hand.

Usage: python scripts/plot.py --analysis runs/analysis --out paper/figures
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = {"adamw": "#1E6E8C", "muon": "#B24A2E"}
MODULES = ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"]


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _opt(run: str) -> str:
    return "muon" if "_muon_" in run else "adamw"


def _save(fig, fig_dir: Path, stem: str) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / f"{stem}.png", fig_dir / f"{stem}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return paths


def fig_erank_by_layer(pre: list[dict], fig_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, len(MODULES), figsize=(3 * len(MODULES), 3), sharey=False)
    by = defaultdict(lambda: defaultdict(dict))
    for r in pre:
        by[r["module"]][r["run"]][int(r["layer"])] = float(r["erank"])
    for ax, mod in zip(axes, MODULES, strict=False):
        for run, layers in sorted(by[mod].items()):
            xs = sorted(layers)
            ax.plot(xs, [layers[x] for x in xs], marker="o", ms=3, color=COLOR[_opt(run)], label=run, alpha=0.8)
        ax.set_title(mod)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("effective rank")
    axes[0].legend(fontsize=6)
    fig.suptitle("H1: effective rank of pretrained weights by layer")
    return _save(fig, fig_dir, "fig_erank_by_layer")


def fig_delta_erank(delta: list[dict], fig_dir: Path) -> list[Path]:
    tasks = sorted({r["task"] for r in delta})
    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(4 * max(1, len(tasks)), 3), squeeze=False)
    for ax, task in zip(axes[0], tasks, strict=False):
        by = defaultdict(lambda: defaultdict(list))
        for r in delta:
            if r["task"] == task:
                by[r["parent"]][int(r["layer"])].append(float(r["erank"]))
        for parent, layers in sorted(by.items()):
            xs = sorted(layers)
            ax.plot(
                xs,
                [sum(layers[x]) / len(layers[x]) for x in xs],
                marker="o",
                ms=3,
                color=COLOR[_opt(parent)],
                label=parent,
            )
        ax.set_title(f"task: {task}")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean effective rank of ΔW")
        ax.legend(fontsize=6)
    fig.suptitle("H2: effective rank of the full fine-tuning update")
    return _save(fig, fig_dir, "fig_delta_erank")


def _lora_rows(res: list[dict]) -> list[dict]:
    return [r for r in res if r["method"] == "lora" and r["rank"] not in ("", "None")]


def fig_lora_gap(
    res: list[dict],
    fig_dir: Path,
    key: str = "recovered",
    stem: str = "fig_lora_gap",
    ylabel: str = "fraction of full-FT gain recovered",
) -> list[Path]:
    rows = _lora_rows(res)
    tasks = sorted({r["task"] for r in rows})
    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(4 * max(1, len(tasks)), 3), squeeze=False)
    for ax, task in zip(axes[0], tasks, strict=False):
        by = defaultdict(dict)
        for r in rows:
            if r["task"] == task and not math.isnan(float(r[key])):
                by[r["parent"]][int(r["rank"])] = float(r[key])
        for parent, pts in sorted(by.items()):
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], marker="o", color=COLOR[_opt(parent)], label=parent)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("LoRA rank")
        ax.set_ylabel(ylabel)
        ax.set_title(f"task: {task}")
        ax.legend(fontsize=6)
    fig.suptitle("H3: LoRA gap versus rank" if key == "recovered" else "forgetting versus rank")
    return _save(fig, fig_dir, stem)


def fig_energy_vs_recovered(delta: list[dict], res: list[dict], fig_dir: Path) -> list[Path]:
    energy = defaultdict(list)
    for r in delta:
        for k in (4, 16, 64):
            energy[(r["parent"], r["task"], k)].append(float(r[f"top{k}"]))
    fig, ax = plt.subplots(figsize=(4, 4))
    for r in _lora_rows(res):
        key = (r["parent"], r["task"], int(r["rank"]))
        if key in energy and not math.isnan(float(r["recovered"])):
            x = sum(energy[key]) / len(energy[key])
            ax.scatter(
                x,
                float(r["recovered"]),
                color=COLOR[_opt(r["parent"])],
                marker="o" if r["task"] == "code" else "s",
            )
            ax.annotate(f"r{r['rank']}", (x, float(r["recovered"])), fontsize=6)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
    ax.set_xlabel("mean top-r energy of full-FT ΔW")
    ax.set_ylabel("recovered fraction at rank r")
    ax.set_title("does ΔW spectrum predict the LoRA gap?")
    return _save(fig, fig_dir, "fig_energy_vs_recovered")


def make_figures(analysis_dir: Path, fig_dir: Path) -> list[Path]:
    analysis_dir, fig_dir = Path(analysis_dir), Path(fig_dir)
    pre, delta, res = (
        _read(analysis_dir / "pretrained_spectra.csv"),
        _read(analysis_dir / "delta_spectra.csv"),
        _read(analysis_dir / "finetune_results.csv"),
    )
    out = []
    out += fig_erank_by_layer(pre, fig_dir)
    out += fig_delta_erank(delta, fig_dir)
    out += fig_lora_gap(res, fig_dir)
    out += fig_energy_vs_recovered(delta, res, fig_dir)
    out += fig_lora_gap(res, fig_dir, key="forgetting", stem="fig_forgetting", ylabel="increase in FineWeb-Edu val loss")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="runs/analysis")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    for p in make_figures(Path(a.analysis), Path(a.out)):
        print(p)
