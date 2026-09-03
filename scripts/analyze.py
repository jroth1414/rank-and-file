"""Turn runs/ into CSVs for the paper: pretrained spectra, full-FT delta spectra, LoRA overlap, LoRA gap.

Usage: python scripts/analyze.py --runs runs --out runs/analysis
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

from rankfile.checkpoint import load_model_state, read_latest
from rankfile.config import load_yaml
from rankfile.spectra import matrix_report, subspace_overlap, subspace_overlap_two_sided

REPORT_FIELDS = [
    "name", "layer", "module", "rows", "cols",
    "erank", "erank_norm", "srank", "fro", "top4", "top16", "top64",
]
PARENT_META = ["optimizer", "arm", "seed"]
# Canonical fine-tune run naming from scripts/ft_grid.py: {parent}__{full|lora<r>}_{code|sup}.
# Sweep runs (e.g. ftsweep_full_lr1e-4) never match this and are excluded.
FT_NAME_RE = re.compile(r"^(?P<parent>.+)__(?:full|lora\d+)_(?:code|sup)$")
BEFORE_MISMATCH_WARN = 1e-3
BEFORE_MISMATCH_RAISE = 1e-2


def _write(path: Path, rows: list[dict], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _block_matrices(sd: dict) -> dict[str, torch.Tensor]:
    return {k: v for k, v in sd.items() if k.startswith("blocks.") and v.ndim == 2}


def _metric(task: str, d: dict, run: str) -> float:
    key = "code_val_loss" if task == "code" else "sup_acc_mean"
    if key not in d:
        raise KeyError(f"{run}: results.json lacks {key}")
    return -d[key] if task == "code" else d[key]


def _parent_meta(cfg: dict) -> dict[str, object]:
    return {"optimizer": cfg.get("optimizer"), "arm": cfg.get("arm"), "seed": cfg.get("seed")}


def _read_metrics(run: Path) -> list[dict]:
    p = run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _val_loss_at_or_before(metrics: list[dict], tokens: int) -> float:
    """val_loss from the metrics.jsonl line with the largest tokens <= `tokens`; nan if none."""
    candidates = [m for m in metrics if "val_loss" in m and "tokens" in m and m["tokens"] <= tokens]
    if not candidates:
        return math.nan
    return float(max(candidates, key=lambda m: m["tokens"])["val_loss"])


def _included_pretraining_runs(
    runs_root: Path, arms: tuple[str, ...], include: str | None
) -> dict[str, tuple[Path, dict]]:
    """Pretraining runs (DONE, with a usable latest checkpoint) whose config.resolved.yaml
    `arm` is in `arms`, or, when `include` is given, whose directory name matches that glob
    instead of the arm check. A DONE run without a usable latest.txt is skipped with a warning."""
    out: dict[str, tuple[Path, dict]] = {}
    for run in sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "DONE").exists()):
        if include is not None:
            if not fnmatch.fnmatch(run.name, include):
                continue
        else:
            cfg_check = load_yaml(run / "config.resolved.yaml")
            if cfg_check.get("arm") not in arms:
                continue
        latest = read_latest(run)
        if latest is None:
            print(f"warning: {run.name}: DONE but no usable latest.txt, skipping")
            continue
        out[run.name] = (run, load_yaml(run / "config.resolved.yaml"))
    return out


def _included_ft_runs(
    runs_root: Path, included_pre: dict[str, tuple[Path, dict]]
) -> tuple[list[Path], dict[str, dict]]:
    """Fine-tune run dirs whose name matches the canonical pattern and whose parent is
    included, plus their parsed results.json. Raises ValueError on a duplicate
    (parent, task, method, rank) cell claimed by two different run dirs."""
    ft_runs: list[Path] = []
    results: dict[str, dict] = {}
    seen_cells: dict[tuple[str, str, str, str], str] = {}
    for p in sorted(x for x in runs_root.iterdir() if x.is_dir() and (x / "results.json").exists()):
        m = FT_NAME_RE.match(p.name)
        if m is None or m.group("parent") not in included_pre:
            continue
        r = json.loads((p / "results.json").read_text())
        cell = (r["parent"], r["task"], r["method"], str(r.get("rank")))
        if cell in seen_cells:
            raise ValueError(
                f"duplicate fine-tune cell {cell}: {seen_cells[cell]!r} and {p.name!r} both claim it"
            )
        seen_cells[cell] = p.name
        ft_runs.append(p)
        results[p.name] = r
    return ft_runs, results


def analyze(
    runs_root: Path,
    out_dir: Path,
    arms: tuple[str, ...] = ("p1", "p2", "p3"),
    include: str | None = None,
    allow_before_mismatch: bool = False,
    all_ckpts: bool = False,
) -> dict[str, Path]:
    runs_root, out_dir = Path(runs_root), Path(out_dir)
    included_pre = _included_pretraining_runs(runs_root, arms, include)

    pre_rows: list[dict] = []
    for run_name, (run, cfg) in included_pre.items():
        val_loss = float(run.joinpath("DONE").read_text().split()[2])
        sd = load_model_state(read_latest(run))
        for k, W in _block_matrices(sd).items():
            pre_rows.append({"run": run_name, "val_loss": val_loss, **_parent_meta(cfg), **matrix_report(k, W)})
        del sd

    traj_rows: list[dict] = []
    if all_ckpts:
        for run_name, (run, cfg) in included_pre.items():
            batch_tokens = cfg.get("batch_tokens")
            metrics = _read_metrics(run)
            for ckpt in sorted(run.glob("ckpt_*.pt")):
                step = int(ckpt.stem.split("_")[1])
                tokens = step * batch_tokens
                val_loss = _val_loss_at_or_before(metrics, tokens)
                sd = load_model_state(ckpt)
                for k, W in _block_matrices(sd).items():
                    traj_rows.append({
                        "run": run_name, "step": step, "tokens": tokens, "val_loss": val_loss,
                        **_parent_meta(cfg), **matrix_report(k, W),
                    })
                del sd

    ft_runs, results = _included_ft_runs(runs_root, included_pre)
    ft_by_parent: dict[str, list[Path]] = defaultdict(list)
    for p in ft_runs:
        ft_by_parent[results[p.name]["parent"]].append(p)

    # Full-FT deltas and per-(parent, task) full-run lookup, needed by both the
    # LoRA-overlap and LoRA-gap passes below. Processed one parent at a time so only
    # one parent's (block-matrix-only) state dict is ever resident at once.
    delta_rows: list[dict] = []
    ov_rows: list[dict] = []
    full_by_cell: dict[tuple[str, str], tuple[str, dict]] = {}

    for parent, cells in sorted(ft_by_parent.items()):
        parent_run, parent_cfg = included_pre[parent]
        meta = _parent_meta(parent_cfg)
        pre_blocks = _block_matrices(load_model_state(read_latest(parent_run)))

        deltas_by_task: dict[str, dict[str, torch.Tensor]] = {}
        for p in cells:
            r = results[p.name]
            if r["method"] != "full":
                continue
            full_by_cell[(r["parent"], r["task"])] = (p.name, r)
            ft_sd = torch.load(p / "model.pt", weights_only=False)
            deltas = {k: ft_sd[k].float() - pre_blocks[k].float() for k in pre_blocks}
            deltas_by_task[r["task"]] = deltas
            for k, D in deltas.items():
                delta_rows.append({
                    "run": p.name, "parent": r["parent"], "task": r["task"], "lr": r.get("lr"),
                    **meta, **matrix_report(k, D),
                })
            del ft_sd

        for p in cells:
            r = results[p.name]
            if r["method"] != "lora":
                continue
            deltas = deltas_by_task.get(r["task"])
            if deltas is None:
                continue
            lsd = torch.load(p / "lora.pt", weights_only=False)
            for name, ab in lsd.items():
                D = deltas[name + ".weight"]
                ov_rows.append({
                    "run": p.name, "parent": r["parent"], "task": r["task"], "rank": r["rank"],
                    "name": name, "lr": r.get("lr"), **meta,
                    "overlap": subspace_overlap(D, ab["B"]),
                    "overlap_two_sided": subspace_overlap_two_sided(D, ab["B"], ab["A"]),
                    "chance": r["rank"] / min(D.shape[0], D.shape[1]),
                })
            del lsd

        del pre_blocks, deltas_by_task

    # LoRA gap: recovered = (lora.after - full.before) / (full.after - full.before), using
    # the matching full-FT run's own "before" as the single reference for the (parent, task)
    # cell. The LoRA run's own "before" is compared against it as before_mismatch: a warning
    # above 1e-3, a hard error above 1e-2 unless allow_before_mismatch.
    res_rows: list[dict] = []
    for p in ft_runs:
        r = results[p.name]
        meta = _parent_meta(included_pre[r["parent"]][1])
        before = _metric(r["task"], r["before"], p.name)
        after = _metric(r["task"], r["after"], p.name)
        full = full_by_cell.get((r["parent"], r["task"]))
        full_name, full_r = full if full is not None else (None, None)
        m_full = _metric(r["task"], full_r["after"], full_name) if full is not None else math.nan
        before_mismatch = 0.0 if r["method"] == "full" else math.nan
        if r["method"] == "full" or full is None:
            recovered = math.nan
        else:
            full_before = _metric(r["task"], full_r["before"], full_name)
            before_mismatch = before - full_before
            if abs(before_mismatch) > BEFORE_MISMATCH_WARN:
                print(
                    f"warning: {p.name}: before metric {before} disagrees with full-FT before "
                    f"{full_before} for parent {r['parent']!r} task {r['task']!r} "
                    f"(mismatch {before_mismatch:+.4g})"
                )
            if abs(before_mismatch) > BEFORE_MISMATCH_RAISE and not allow_before_mismatch:
                raise ValueError(
                    f"{p.name}: before metric {before} disagrees with full-FT before {full_before} "
                    f"for parent {r['parent']!r} task {r['task']!r} (mismatch {before_mismatch:+.4g})"
                )
            # A full-FT run whose metric did not improve (m_full <= full_before) makes the
            # "fraction of full-FT improvement recovered" undefined, not infinite/negative.
            recovered = math.nan if m_full <= full_before else (after - full_before) / (m_full - full_before)
        res_rows.append({
            "run": p.name, "parent": r["parent"], "task": r["task"], "method": r["method"], "rank": r["rank"],
            "lr": r.get("lr"), **meta,
            "metric_before": before, "metric_full": m_full, "metric_after": after,
            "recovered": recovered, "before_mismatch": before_mismatch,
            "forgetting": r["after"]["pre_val_loss"] - r["before"]["pre_val_loss"],
            "alpha": r.get("alpha"), "parent_ckpt": r.get("parent_ckpt"),
        })

    out = {
        "pretrained_spectra": _write(
            out_dir / "pretrained_spectra.csv", pre_rows, ["run", *PARENT_META, "val_loss", *REPORT_FIELDS]
        ),
        "delta_spectra": _write(
            out_dir / "delta_spectra.csv", delta_rows,
            ["run", "parent", "task", "lr", *PARENT_META, *REPORT_FIELDS],
        ),
        "lora_overlap": _write(
            out_dir / "lora_overlap.csv", ov_rows,
            ["run", "parent", "task", "rank", "name", "lr", *PARENT_META, "overlap", "overlap_two_sided", "chance"],
        ),
        "finetune_results": _write(
            out_dir / "finetune_results.csv", res_rows,
            ["run", "parent", "task", "method", "rank", "lr", *PARENT_META, "metric_before", "metric_full",
             "metric_after", "recovered", "before_mismatch", "forgetting", "alpha", "parent_ckpt"],
        ),
    }
    if all_ckpts:
        out["pretrained_spectra_traj"] = _write(
            out_dir / "pretrained_spectra_traj.csv", traj_rows,
            ["run", "step", "tokens", "val_loss", *PARENT_META, *REPORT_FIELDS],
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/analysis")
    ap.add_argument("--arms", nargs="+", default=["p1", "p2", "p3"])
    ap.add_argument("--include", default=None)
    ap.add_argument("--allow-before-mismatch", action="store_true")
    ap.add_argument("--all-ckpts", action="store_true")
    a = ap.parse_args()
    written = analyze(
        Path(a.runs), Path(a.out), tuple(a.arms), a.include, a.allow_before_mismatch, a.all_ckpts
    )
    for k, v in written.items():
        print(k, v)
