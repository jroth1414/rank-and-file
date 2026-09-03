"""Markdown result tables from runs/analysis CSVs. Usage: python scripts/results_table.py --analysis runs/analysis"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def _read(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _fmt(v: str, spec: str = ".4f") -> str:
    """Format a numeric CSV cell, or "" for a NaN (undefined) value."""
    x = float(v)
    return "" if math.isnan(x) else format(x, spec)


def _pretraining_table(pre: list[dict]) -> list[str]:
    """H1: one row per pretraining run. Never pooled across arms or optimizers."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in pre:
        by_run[r["run"]].append(r)
    lines = [
        "| run | arm | optimizer | seed | val_loss | mean erank | mean erank_norm | mean srank |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run, rows in sorted(by_run.items()):
        r0 = rows[0]
        erank = _mean([float(r["erank"]) for r in rows])
        erank_norm = _mean([float(r["erank_norm"]) for r in rows])
        srank = _mean([float(r["srank"]) for r in rows])
        lines.append(
            f"| {run} | {r0['arm']} | {r0['optimizer']} | {r0['seed']} | {float(r0['val_loss']):.4f} | "
            f"{erank:.2f} | {erank_norm:.3f} | {srank:.2f} |"
        )
    return lines


def _delta_table(delta: list[dict]) -> list[str]:
    """H2: one row per (parent, task); never pooled across parents."""
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in delta:
        by_cell[(r["parent"], r["task"])].append(r)
    lines = ["| parent | task | mean erank (delta W) | mean erank_norm (delta W) |", "|---|---|---|---|"]
    for (parent, task), rows in sorted(by_cell.items()):
        erank = _mean([float(r["erank"]) for r in rows])
        erank_norm = _mean([float(r["erank_norm"]) for r in rows])
        lines.append(f"| {parent} | {task} | {erank:.2f} | {erank_norm:.3f} |")
    return lines


def _finetune_table(res: list[dict]) -> list[str]:
    """H3: one row per fine-tune run."""
    lines = [
        "| parent | task | method | metric before | metric full | metric after | recovered | forgetting | lr |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(res, key=lambda r: (r["parent"], r["task"], r["method"], int(r["rank"] or 0))):
        method = "full" if r["method"] == "full" else f"lora{r['rank']}"
        lines.append(
            f"| {r['parent']} | {r['task']} | {method} | {_fmt(r['metric_before'])} | "
            f"{_fmt(r['metric_full'])} | {_fmt(r['metric_after'])} | {_fmt(r['recovered'], '.3f')} | "
            f"{_fmt(r['forgetting'])} | {float(r['lr']):.4g} |"
        )
    return lines


def results_table(analysis_dir: Path) -> str:
    analysis_dir = Path(analysis_dir)
    res = _read(analysis_dir / "finetune_results.csv")
    pre = _read(analysis_dir / "pretrained_spectra.csv")
    delta = _read(analysis_dir / "delta_spectra.csv")

    lines: list[str] = []
    lines += _finetune_table(res)
    lines += [""]
    lines += _pretraining_table(pre)
    lines += [""]
    lines += _delta_table(delta)

    md = "\n".join(lines) + "\n"
    (analysis_dir / "results.md").write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="runs/analysis")
    print(results_table(Path(ap.parse_args().analysis)))
