"""Turn runs/ into CSVs for the paper: pretrained spectra, full-FT delta spectra, LoRA overlap, LoRA gap.

Usage: python scripts/analyze.py --runs runs --out runs/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from rankfile.checkpoint import load_model_state, read_latest
from rankfile.config import load_yaml
from rankfile.spectra import matrix_report, subspace_overlap

REPORT_FIELDS = ["name", "layer", "module", "rows", "cols", "erank", "srank", "fro", "top4", "top16", "top64"]


def _write(path: Path, rows: list[dict], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _block_matrices(sd: dict) -> dict[str, torch.Tensor]:
    return {k: v for k, v in sd.items() if k.startswith("blocks.") and v.ndim == 2}


def _metric(task: str, d: dict) -> float:
    return -d["code_val_loss"] if task == "code" else d["sup_acc_mean"]


def analyze(runs_root: Path, out_dir: Path) -> dict[str, Path]:
    runs_root, out_dir = Path(runs_root), Path(out_dir)
    pre_rows, delta_rows, ov_rows, res_rows = [], [], [], []
    pre_states: dict[str, dict] = {}
    for run in sorted(p for p in runs_root.iterdir() if (p / "DONE").exists()):
        cfg = load_yaml(run / "config.resolved.yaml")
        val_loss = float(run.joinpath("DONE").read_text().split()[2])
        sd = load_model_state(read_latest(run))
        pre_states[run.name] = sd
        for k, W in _block_matrices(sd).items():
            pre_rows.append({"run": run.name, "optimizer": cfg.get("optimizer"), "arm": cfg.get("arm"),
                              "seed": cfg.get("seed"), "val_loss": val_loss, **matrix_report(k, W)})

    ft_runs = sorted(p for p in runs_root.iterdir() if (p / "results.json").exists())
    results = {p.name: json.loads((p / "results.json").read_text()) for p in ft_runs}

    # Full-FT deltas and per-(parent, task) full-run lookup, needed by both the
    # LoRA-overlap and LoRA-gap passes below.
    full_deltas: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    full_by_cell: dict[tuple[str, str], dict] = {}
    for p in ft_runs:
        r = results[p.name]
        if r["method"] != "full":
            continue
        pre = pre_states.get(r["parent"]) or load_model_state(read_latest(runs_root / r["parent"]))
        ft = torch.load(p / "model.pt", weights_only=False)
        deltas = {k: ft[k].float() - pre[k].float() for k in _block_matrices(pre)}
        full_deltas[(r["parent"], r["task"])] = deltas
        full_by_cell[(r["parent"], r["task"])] = r
        for k, D in deltas.items():
            delta_rows.append({"run": p.name, "parent": r["parent"], "task": r["task"], **matrix_report(k, D)})

    for p in ft_runs:
        r = results[p.name]
        if r["method"] != "lora":
            continue
        deltas = full_deltas.get((r["parent"], r["task"]))
        if deltas is None:
            continue
        lsd = torch.load(p / "lora.pt", weights_only=False)
        for name, ab in lsd.items():
            ov_rows.append({"run": p.name, "parent": r["parent"], "task": r["task"], "rank": r["rank"],
                             "name": name, "overlap": subspace_overlap(deltas[name + ".weight"], ab["B"])})

    # LoRA gap: recovered = (lora.after - full.before) / (full.after - full.before), using
    # the matching full-FT run's own "before" as the single reference for the (parent, task)
    # cell. The LoRA run's own "before" must agree with it (same parent checkpoint, same
    # eval) within 1e-3, or the two runs aren't comparable.
    for p in ft_runs:
        r = results[p.name]
        before = _metric(r["task"], r["before"])
        after = _metric(r["task"], r["after"])
        full = full_by_cell.get((r["parent"], r["task"]))
        m_full = _metric(r["task"], full["after"]) if full is not None else math.nan
        if r["method"] == "full" or full is None:
            recovered = math.nan
        else:
            full_before = _metric(r["task"], full["before"])
            if abs(before - full_before) > 1e-3:
                raise ValueError(
                    f"{p.name}: before metric {before} disagrees with full-FT before {full_before} "
                    f"for parent {r['parent']!r} task {r['task']!r}"
                )
            recovered = (after - full_before) / (m_full - full_before)
        res_rows.append({
            "run": p.name, "parent": r["parent"], "task": r["task"], "method": r["method"], "rank": r["rank"],
            "metric_before": before, "metric_full": m_full, "metric_after": after, "recovered": recovered,
            "forgetting": r["after"]["pre_val_loss"] - r["before"]["pre_val_loss"],
            "alpha": r.get("alpha"), "parent_ckpt": r.get("parent_ckpt"),
        })

    return {
        "pretrained_spectra": _write(out_dir / "pretrained_spectra.csv", pre_rows,
                                      ["run", "optimizer", "arm", "seed", "val_loss", *REPORT_FIELDS]),
        "delta_spectra": _write(out_dir / "delta_spectra.csv", delta_rows,
                                 ["run", "parent", "task", *REPORT_FIELDS]),
        "lora_overlap": _write(out_dir / "lora_overlap.csv", ov_rows,
                                ["run", "parent", "task", "rank", "name", "overlap"]),
        "finetune_results": _write(out_dir / "finetune_results.csv", res_rows,
                                    ["run", "parent", "task", "method", "rank", "metric_before", "metric_full",
                                     "metric_after", "recovered", "forgetting", "alpha", "parent_ckpt"]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/analysis")
    a = ap.parse_args()
    for k, v in analyze(Path(a.runs), Path(a.out)).items():
        print(k, v)
