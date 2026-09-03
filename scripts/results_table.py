"""Markdown result tables from runs/analysis CSVs. Usage: python scripts/results_table.py --analysis runs/analysis"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _read(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def results_table(analysis_dir: Path) -> str:
    analysis_dir = Path(analysis_dir)
    res = _read(analysis_dir / "finetune_results.csv")
    lines = ["| parent | task | method | metric after | recovered | forgetting |", "|---|---|---|---|---|---|"]
    for r in sorted(res, key=lambda r: (r["parent"], r["task"], r["method"], int(r["rank"] or 0))):
        method = "full" if r["method"] == "full" else f"lora{r['rank']}"
        rec = "" if r["recovered"] in ("nan", "") else f"{float(r['recovered']):.3f}"
        lines.append(f"| {r['parent']} | {r['task']} | {method} | {float(r['metric_after']):.4f} | {rec} | {float(r['forgetting']):.4f} |")
    pre = _read(analysis_dir / "pretrained_spectra.csv")
    by = defaultdict(list)
    for r in pre:
        by[r["optimizer"]].append(float(r["erank"]))
    lines += ["", "| optimizer | mean effective rank (pretrained) |", "|---|---|"]
    lines += [f"| {o} | {sum(v) / len(v):.2f} |" for o, v in sorted(by.items())]
    delta = _read(analysis_dir / "delta_spectra.csv")
    byd = defaultdict(list)
    for r in delta:
        byd[(r["parent"], r["task"])].append(float(r["erank"]))
    lines += ["", "| parent | task | mean effective rank (ΔW) |", "|---|---|---|"]
    lines += [f"| {p} | {t} | {sum(v) / len(v):.2f} |" for (p, t), v in sorted(byd.items())]
    md = "\n".join(lines) + "\n"
    (analysis_dir / "results.md").write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", default="runs/analysis")
    print(results_table(Path(ap.parse_args().analysis)))
