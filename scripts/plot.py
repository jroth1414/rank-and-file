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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

# Colour-blind-safe, greyscale-separable pair (Wong 2011).
COLOR = {"adamw": "#0072B2", "muon": "#E69F00"}
LINESTYLE = {"p1": "-", "p2": "-", "p3": "--"}
MARKER = {"0": "o", "1": "^"}
MODULES = ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"]
RANKS = (4, 16, 64)


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _style(row: dict) -> dict:
    """color/linestyle/marker for a CSV row, keyed by its optimizer/arm/seed columns."""
    opt = row["optimizer"]
    if opt not in COLOR:
        raise ValueError(f"unknown optimizer {opt!r}")
    return {
        "color": COLOR[opt],
        "linestyle": LINESTYLE.get(row.get("arm"), "-"),
        "marker": MARKER.get(str(row.get("seed", "0")), "o"),
    }


def _rank(a: np.ndarray) -> np.ndarray:
    """Ranks of `a` (average rank for ties), 0-indexed."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, implemented with numpy ranks (no scipy)."""
    if len(xs) < 2:
        return math.nan
    rx, ry = _rank(np.asarray(xs, dtype=float)), _rank(np.asarray(ys, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _log2_rank_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(RANKS))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("LoRA rank")


def _save(fig, fig_dir: Path, stem: str) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / f"{stem}.png", fig_dir / f"{stem}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return paths


def fig_erank_by_layer(pre: list[dict], fig_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5), sharey=True, layout="constrained")
    axes_flat = axes.flatten()
    by: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    meta: dict[str, dict] = {}
    for r in pre:
        by[r["module"]][r["run"]][int(r["layer"])] = float(r["erank"])
        meta[r["run"]] = r
    for ax, mod in zip(axes_flat, MODULES, strict=False):
        for run, layers in sorted(by[mod].items()):
            xs = sorted(layers)
            ax.plot(xs, [layers[x] for x in xs], ms=4, label=run, alpha=0.85, **_style(meta[run]))
        ax.set_title(mod)
        ax.set_xlabel("layer")
    axes_flat[0].set_ylabel("effective rank")
    axes_flat[4].set_ylabel("effective rank")
    axes_flat[len(MODULES)].axis("off")  # unused 8th slot
    handles: dict[str, object] = {}
    for ax in axes_flat[: len(MODULES)]:
        hs, ls = ax.get_legend_handles_labels()
        for h, label in zip(hs, ls, strict=True):
            handles.setdefault(label, h)
    if handles:
        fig.legend(list(handles.values()), list(handles), loc="center", bbox_to_anchor=(0.875, 0.25), fontsize=7)
    fig.suptitle("H1: effective rank of pretrained weights by layer")
    return _save(fig, fig_dir, "fig_erank_by_layer")


def fig_delta_erank(delta: list[dict], fig_dir: Path) -> list[Path]:
    tasks = sorted({r["task"] for r in delta})
    fig, axes = plt.subplots(
        1, max(1, len(tasks)), figsize=(4 * max(1, len(tasks)), 3.6), squeeze=False, layout="constrained"
    )
    for ax, task in zip(axes[0], tasks, strict=False):
        by = defaultdict(lambda: defaultdict(list))
        meta: dict[str, dict] = {}
        for r in delta:
            if r["task"] == task:
                by[r["parent"]][int(r["layer"])].append(float(r["erank"]))
                meta[r["parent"]] = r
        for parent, layers in sorted(by.items()):
            xs = sorted(layers)
            ax.plot(
                xs,
                [sum(layers[x]) / len(layers[x]) for x in xs],
                ms=4,
                label=parent,
                **_style(meta[parent]),
            )
        ax.set_title(f"task: {task}")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean effective rank of ΔW")
        if ax.get_legend_handles_labels()[0]:
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
    tasks = sorted({r["task"] for r in res})
    fig, axes = plt.subplots(
        1, max(1, len(tasks)), figsize=(4 * max(1, len(tasks)), 3.6), squeeze=False, layout="constrained"
    )
    full_rows = [r for r in res if r["method"] == "full"]
    for ax, task in zip(axes[0], tasks, strict=False):
        by = defaultdict(dict)
        meta: dict[str, dict] = {}
        for r in rows:
            if r["task"] == task and not math.isnan(float(r[key])):
                by[r["parent"]][int(r["rank"])] = float(r[key])
                meta[r["parent"]] = r
        for parent, pts in sorted(by.items()):
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], label=parent, **_style(meta[parent]))
        if stem == "fig_forgetting":
            for r in full_rows:
                if r["task"] != task:
                    continue
                v = float(r["forgetting"])
                if math.isnan(v):
                    continue
                ax.axhline(v, color=_style(r)["color"], linestyle=_style(r)["linestyle"], lw=1, alpha=0.5)
        _log2_rank_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"task: {task}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    fig.suptitle("H3: LoRA gap versus rank" if key == "recovered" else "forgetting versus rank")
    return _save(fig, fig_dir, stem)


def fig_energy_vs_recovered(delta: list[dict], res: list[dict], fig_dir: Path) -> list[Path]:
    energy = defaultdict(list)
    for r in delta:
        for k in RANKS:
            energy[(r["parent"], r["task"], k)].append(float(r[f"top{k}"]))
    fig, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    groups: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    meta: dict[str, dict] = {}
    all_x: list[float] = []
    all_y: list[float] = []
    for r in _lora_rows(res):
        key = (r["parent"], r["task"], int(r["rank"]))
        if key in energy and not math.isnan(float(r["recovered"])):
            x = sum(energy[key]) / len(energy[key])
            y = float(r["recovered"])
            groups[(r["parent"], r["task"])].append((int(r["rank"]), x, y))
            meta[r["parent"]] = r
            all_x.append(x)
            all_y.append(y)
    for (parent, task), pts in groups.items():
        pts = sorted(pts)
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        ax.plot(xs, ys, color="0.65", lw=0.8, alpha=0.7, zorder=1)
        ax.scatter(xs, ys, color=_style(meta[parent])["color"], marker="o" if task == "code" else "s", zorder=2)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR["adamw"], markersize=7, label="adamw"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR["muon"], markersize=7, label="muon"),
        Line2D([0], [0], marker="o", color="0.3", linestyle="none", markersize=6, label="code"),
        Line2D([0], [0], marker="s", color="0.3", linestyle="none", markersize=6, label="sup"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="best")
    ax.set_xlabel("mean top-r energy of full-FT ΔW")
    ax.set_ylabel("recovered fraction at rank r")
    rho = _spearman(all_x, all_y)
    rho_str = f"{rho:.2f}" if not math.isnan(rho) else "n/a"
    ax.set_title(f"does ΔW spectrum predict the LoRA gap? (ρ = {rho_str})")
    return _save(fig, fig_dir, "fig_energy_vs_recovered")


def fig_overlap(ov: list[dict], fig_dir: Path) -> list[Path]:
    tasks = sorted({r["task"] for r in ov})
    fig, axes = plt.subplots(
        1, max(1, len(tasks)), figsize=(4 * max(1, len(tasks)), 3.6), squeeze=False, layout="constrained"
    )
    for ax, task in zip(axes[0], tasks, strict=False):
        by = defaultdict(lambda: defaultdict(list))
        chance_by = defaultdict(list)
        meta: dict[str, dict] = {}
        for r in ov:
            if r["task"] != task:
                continue
            by[r["parent"]][int(r["rank"])].append(float(r["overlap_two_sided"]))
            chance_by[int(r["rank"])].append(float(r["chance"]))
            meta[r["parent"]] = r
        for parent, pts in sorted(by.items()):
            xs = sorted(pts)
            ax.plot(xs, [sum(pts[x]) / len(pts[x]) for x in xs], label=parent, **_style(meta[parent]))
        if chance_by:
            xs = sorted(chance_by)
            ax.plot(
                xs, [sum(chance_by[x]) / len(chance_by[x]) for x in xs],
                linestyle=":", color="gray", lw=1.2, label="chance",
            )
        _log2_rank_axis(ax)
        ax.set_ylabel("two-sided subspace overlap")
        ax.set_title(f"task: {task}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    fig.suptitle("LoRA subspace overlap with the full fine-tuning update")
    return _save(fig, fig_dir, "fig_overlap")


def fig_erank_vs_loss(traj: list[dict], fig_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    by: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[str, dict] = {}
    for r in traj:
        vl = float(r["val_loss"])
        if math.isnan(vl):
            continue
        by[r["run"]][vl].append(float(r["erank_norm"]))
        meta[r["run"]] = r
    for run, pts in sorted(by.items()):
        xs = sorted(pts)
        ax.plot(xs, [sum(pts[x]) / len(pts[x]) for x in xs], ms=4, label=run, **_style(meta[run]))
    ax.invert_xaxis()  # loss falls as training progresses -> progress reads left to right
    ax.set_xlabel("validation loss")
    ax.set_ylabel("mean erank_norm")
    ax.set_title("H1: effective rank versus training progress")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    return _save(fig, fig_dir, "fig_erank_vs_loss")


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

    ov_path = analysis_dir / "lora_overlap.csv"
    if ov_path.exists():
        ov = _read(ov_path)
        if ov:
            out += fig_overlap(ov, fig_dir)

    traj_path = analysis_dir / "pretrained_spectra_traj.csv"
    if traj_path.exists():
        traj = _read(traj_path)
        if traj:
            out += fig_erank_vs_loss(traj, fig_dir)

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="runs/analysis")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    for p in make_figures(Path(a.analysis), Path(a.out)):
        print(p)
