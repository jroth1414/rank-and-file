# Plan 6: Spectral Analysis and Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pure measurement functions for singular-value spectra, an analysis script that turns run directories into CSVs (pretrained spectra, full-FT update spectra, LoRA subspace overlap, LoRA-gap table), and a plot script that produces every paper figure from those CSVs.

**Architecture:** `spectra.py` contains only pure functions on tensors so each is unit-tested against matrices with known spectra. `scripts/analyze.py` walks `runs/`, applies them, and writes `runs/analysis/*.csv`. `scripts/plot.py` reads the CSVs and writes `paper/figures/*.png` and `.pdf`. No figure is ever produced any other way.

**Tech Stack:** torch (`linalg.svdvals`), numpy, csv, matplotlib, pytest.

**Spec:** `proposal.md` §4.4 (effective rank, stable rank, top-r energy, subspace overlap, LoRA gap), §2 hypotheses H1–H3. `CLAUDE.md` §2.3, §7 `test_spectra.py`, §8 rule 7 (figures only from `scripts/plot.py`).

## Global Constraints

- Python 3.11 in `.venv`; run as `.venv\Scripts\python.exe ...`.
- Effective rank is the Roy–Vetterli definition: `exp(H(p))` with `p_i = s_i / Σ s_j` over singular values.
- LoRA gap is reported as **recovered fraction** `(m_lora − m_base) / (m_full − m_base)` where `m` is the task metric with sign chosen so higher is better (negative loss for code, accuracy for sup).
- Analysis reads only `runs/`; it never modifies run directories.
- Commit prefix `spectra:` / `plot:`; no AI attribution trailers.

## Consumed interfaces

- `rankfile.checkpoint.load_model_state(path) -> dict[str, Tensor]`, `read_latest(run_dir)`.
- Pretraining run dir: `DONE`, `latest.txt`, `ckpt_*.pt`, `config.resolved.yaml` (has `optimizer`, `arm`, `seed`).
- Fine-tune run dir: `results.json` (keys `parent, method, rank, task, before, after`), `model.pt` (full) or `lora.pt` (`{name: {"A","B","scale"}}`).
- Weight names: `blocks.{i}.attn.{q,k,v,o}.weight`, `blocks.{i}.mlp.{gate,up,down}.weight`. LoRA module names are the same without `.weight`.

---

### Task 1: Spectrum measures

**Files:**
- Create: `src/rankfile/spectra.py`
- Test: `tests/test_spectra.py`

**Interfaces:**
- Produces: `singular_values(W: Tensor) -> Tensor` (fp32, descending); `effective_rank(s: Tensor) -> float`; `stable_rank(s: Tensor) -> float`; `top_r_energy(s: Tensor, r: int) -> float`; `subspace_overlap(delta: Tensor[out,in], B: Tensor[out,r]) -> float` = fraction of `‖delta‖_F²` captured by projecting onto `colspace(B)`; `matrix_report(name: str, W: Tensor, ranks=(4, 16, 64)) -> dict` with keys `name, layer, module, rows, cols, erank, srank, fro, top4, top16, top64`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spectra.py
import math, torch
from rankfile.spectra import singular_values, effective_rank, stable_rank, top_r_energy, subspace_overlap, matrix_report

def test_effective_rank_identity_and_rank_one():
    assert abs(effective_rank(singular_values(torch.eye(10))) - 10.0) < 1e-4
    u = torch.randn(10, 1); v = torch.randn(1, 7)
    assert abs(effective_rank(singular_values(u @ v)) - 1.0) < 1e-3

def test_stable_rank():
    s = torch.tensor([2.0, 1.0, 1.0])
    assert abs(stable_rank(s) - 6.0 / 4.0) < 1e-6

def test_top_r_energy_of_rank_r_matrix_is_one():
    W = torch.randn(20, 3) @ torch.randn(3, 15)
    s = singular_values(W)
    assert abs(top_r_energy(s, 3) - 1.0) < 1e-5 and top_r_energy(s, 1) < 1.0

def test_subspace_overlap_bounds():
    B = torch.randn(12, 2); A = torch.randn(2, 9)
    delta_in = B @ A
    assert abs(subspace_overlap(delta_in, B) - 1.0) < 1e-5
    Q, _ = torch.linalg.qr(torch.cat([B, torch.randn(12, 10)], 1))
    delta_out = Q[:, 2:5] @ torch.randn(3, 9)   # orthogonal to colspace(B)
    assert subspace_overlap(delta_out, B) < 1e-5

def test_matrix_report_parses_names():
    r = matrix_report("blocks.3.mlp.down.weight", torch.randn(32, 64))
    assert r["layer"] == 3 and r["module"] == "mlp.down" and r["rows"] == 32 and r["cols"] == 64
    assert 1.0 <= r["erank"] <= 32 and 0 < r["top4"] <= r["top16"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_spectra.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/rankfile/spectra.py
"""Pure spectral measurements: effective rank, stable rank, top-r energy, LoRA subspace overlap."""
from __future__ import annotations

import re

import torch

_NAME = re.compile(r"blocks\.(\d+)\.(attn|mlp)\.(\w+)(?:\.weight)?$")


def singular_values(W: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(W.detach().float().cpu())


def effective_rank(s: torch.Tensor) -> float:
    s = s[s > 0]
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def stable_rank(s: torch.Tensor) -> float:
    return float((s * s).sum() / (s[0] * s[0]))


def top_r_energy(s: torch.Tensor, r: int) -> float:
    e = s * s
    return float(e[:r].sum() / e.sum())


def subspace_overlap(delta: torch.Tensor, B: torch.Tensor) -> float:
    delta, B = delta.detach().float().cpu(), B.detach().float().cpu()
    Q, _ = torch.linalg.qr(B)                       # orthonormal basis of colspace(B)
    proj = Q @ (Q.T @ delta)
    return float((proj * proj).sum() / (delta * delta).sum())


def matrix_report(name: str, W: torch.Tensor, ranks: tuple[int, ...] = (4, 16, 64)) -> dict:
    m = _NAME.search(name)
    layer, module = (int(m.group(1)), f"{m.group(2)}.{m.group(3)}") if m else (-1, name)
    s = singular_values(W)
    out = {"name": name, "layer": layer, "module": module, "rows": W.shape[0], "cols": W.shape[1],
           "erank": effective_rank(s), "srank": stable_rank(s), "fro": float(s.norm())}
    for r in ranks:
        out[f"top{r}"] = top_r_energy(s, r)
    return out
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_spectra.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/rankfile/spectra.py tests/test_spectra.py
git commit -m "spectra: effective rank, stable rank, top-r energy, LoRA subspace overlap"
```

---

### Task 2: analyze.py — run directories to CSVs

**Files:**
- Create: `scripts/analyze.py`
- Test: `tests/test_analyze.py`

**Interfaces:**
- Produces: `analyze(runs_root: Path, out_dir: Path) -> dict[str, Path]` writing:
  - `pretrained_spectra.csv`: one row per (run, matrix): `run, optimizer, arm, seed, val_loss` + `matrix_report` fields. Runs = dirs with `DONE`.
  - `delta_spectra.csv`: one row per (full-FT run, matrix) on `W_ft − W_pre`: `run, parent, task` + report fields.
  - `lora_overlap.csv`: one row per (LoRA run, matrix): `run, parent, task, rank, overlap` where `overlap = subspace_overlap(delta_full, B)` with `delta_full` from the matching full-FT run (same parent and task).
  - `finetune_results.csv`: one row per fine-tune run: `run, parent, task, method, rank, metric_before, metric_full, metric_after, recovered, forgetting`. `metric` is `-code_val_loss` for code and `sup_acc_mean` for sup; `recovered` is `(after − before)/(full − before)`, `NaN` for full runs; `forgetting = after.pre_val_loss − before.pre_val_loss`.
- Also a CLI: `python scripts/analyze.py --runs runs --out runs/analysis`.

- [ ] **Step 1: Write the failing test** (builds fake run dirs with a tiny model)

```python
# tests/test_analyze.py
import csv, json, math, torch
from pathlib import Path
from rankfile.model import ModelConfig, Transformer
from rankfile.config import to_yaml
from rankfile.checkpoint import save_checkpoint, write_latest
from rankfile.optim.build import build_optimizers
from rankfile.lora import apply_lora, lora_state_dict
from scripts.analyze import analyze

MC = ModelConfig(vocab_size=64, n_layer=1, d_model=16, n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=8)

def _pre(root, name, opt):
    torch.manual_seed(0); m = Transformer(MC); run = root / name; run.mkdir(parents=True)
    to_yaml(MC, run / "model.yaml"); to_yaml({"optimizer": opt, "arm": name[:2], "seed": 0}, run / "config.resolved.yaml")
    save_checkpoint(run / "ckpt_0000001.pt", m, build_optimizers(m, opt, 1e-3, 0.1), 1, 0, 0, {}); write_latest(run, "ckpt_0000001.pt")
    (run / "DONE").write_text("1 0 3.5\n"); return m

def _ft(root, parent, method, task, m, rank=None, before=3.0, after=2.0):
    name = f"{parent}__{'full' if method=='full' else f'lora{rank}'}_{task}"; run = root / name; run.mkdir()
    if method == "full":
        m2 = Transformer(MC); m2.load_state_dict(m.state_dict())
        with torch.no_grad():
            for p in m2.blocks.parameters(): p.add_(0.1 * torch.randn_like(p))
        torch.save(m2.state_dict(), run / "model.pt")
    else:
        m2 = Transformer(MC); m2.load_state_dict(m.state_dict()); apply_lora(m2, rank, 2 * rank)
        for n, mod in m2.named_modules():
            if hasattr(mod, "B"): mod.B.data.normal_()
        torch.save(lora_state_dict(m2), run / "lora.pt")
    key = "code_val_loss" if task == "code" else "sup_acc_mean"
    json.dump({"parent": parent, "method": method, "rank": rank, "task": task, "lr": 1e-4,
               "before": {key: before, "pre_val_loss": 3.5}, "after": {key: after, "pre_val_loss": 3.6}}, open(run / "results.json", "w"))

def test_analyze_writes_all_csvs(tmp_path):
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw"); m2 = _pre(root, "p2_muon_m30_s0", "muon")
    _ft(root, "p1_adamw_m30_s0", "full", "code", m1, before=3.0, after=2.0)
    _ft(root, "p1_adamw_m30_s0", "lora", "code", m1, rank=2, before=3.0, after=2.5)
    _ft(root, "p2_muon_m30_s0", "full", "sup", m2, before=0.5, after=0.8)
    out = analyze(root, tmp_path / "analysis")
    pre = list(csv.DictReader(open(out["pretrained_spectra"])))
    assert {r["run"] for r in pre} == {"p1_adamw_m30_s0", "p2_muon_m30_s0"} and len(pre) == 2 * 7
    assert all(float(r["erank"]) >= 1 for r in pre) and pre[0]["val_loss"] == "3.5"
    delta = list(csv.DictReader(open(out["delta_spectra"])))
    assert {r["run"] for r in delta} == {"p1_adamw_m30_s0__full_code", "p2_muon_m30_s0__full_sup"}
    ov = list(csv.DictReader(open(out["lora_overlap"])))
    assert len(ov) == 7 and all(0.0 <= float(r["overlap"]) <= 1.0 + 1e-6 for r in ov)
    res = {r["run"]: r for r in csv.DictReader(open(out["finetune_results"]))}
    assert abs(float(res["p1_adamw_m30_s0__lora2_code"]["recovered"]) - 0.5) < 1e-9
    assert math.isnan(float(res["p1_adamw_m30_s0__full_code"]["recovered"]))
    assert abs(float(res["p2_muon_m30_s0__full_sup"]["forgetting"]) - 0.1) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/analyze.py
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
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
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
        sd = load_model_state(read_latest(run)); pre_states[run.name] = sd
        for k, W in _block_matrices(sd).items():
            pre_rows.append({"run": run.name, "optimizer": cfg.get("optimizer"), "arm": cfg.get("arm"), "seed": cfg.get("seed"),
                             "val_loss": val_loss, **matrix_report(k, W)})
    ft_runs = sorted(p for p in runs_root.iterdir() if (p / "results.json").exists())
    results = {p.name: json.loads((p / "results.json").read_text()) for p in ft_runs}
    full_deltas: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for p in ft_runs:
        r = results[p.name]
        if r["method"] != "full":
            continue
        pre = pre_states.get(r["parent"]) or load_model_state(read_latest(runs_root / r["parent"]))
        ft = torch.load(p / "model.pt", weights_only=False)
        deltas = {k: ft[k].float() - pre[k].float() for k in _block_matrices(pre)}
        full_deltas[(r["parent"], r["task"])] = deltas
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
            ov_rows.append({"run": p.name, "parent": r["parent"], "task": r["task"], "rank": r["rank"], "name": name,
                            "overlap": subspace_overlap(deltas[name + ".weight"], ab["B"])})
    for p in ft_runs:
        r = results[p.name]
        before, after = _metric(r["task"], r["before"]), _metric(r["task"], r["after"])
        full = next((results[q.name] for q in ft_runs if results[q.name]["method"] == "full"
                     and results[q.name]["parent"] == r["parent"] and results[q.name]["task"] == r["task"]), None)
        m_full = _metric(r["task"], full["after"]) if full else math.nan
        recovered = math.nan if r["method"] == "full" or not full else (after - before) / (m_full - before)
        res_rows.append({"run": p.name, "parent": r["parent"], "task": r["task"], "method": r["method"], "rank": r["rank"],
                         "metric_before": before, "metric_full": m_full, "metric_after": after, "recovered": recovered,
                         "forgetting": r["after"]["pre_val_loss"] - r["before"]["pre_val_loss"]})
    return {
        "pretrained_spectra": _write(out_dir / "pretrained_spectra.csv", pre_rows, ["run", "optimizer", "arm", "seed", "val_loss", *REPORT_FIELDS]),
        "delta_spectra": _write(out_dir / "delta_spectra.csv", delta_rows, ["run", "parent", "task", *REPORT_FIELDS]),
        "lora_overlap": _write(out_dir / "lora_overlap.csv", ov_rows, ["run", "parent", "task", "rank", "name", "overlap"]),
        "finetune_results": _write(out_dir / "finetune_results.csv", res_rows,
                                   ["run", "parent", "task", "method", "rank", "metric_before", "metric_full", "metric_after", "recovered", "forgetting"]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="runs/analysis")
    a = ap.parse_args()
    for k, v in analyze(Path(a.runs), Path(a.out)).items():
        print(k, v)
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze.py tests/test_analyze.py
git commit -m "spectra: analyze.py writes pretrained, delta, overlap, and LoRA-gap CSVs"
```

---

### Task 3: plot.py — the paper figures

**Files:**
- Create: `scripts/plot.py`
- Test: `tests/test_plot.py`

**Interfaces:**
- Produces: `make_figures(analysis_dir: Path, fig_dir: Path) -> list[Path]` writing, as both PNG and PDF:
  - `fig_erank_by_layer`: effective rank vs layer, one panel per module type (q, k, v, o, gate, up, down), one line per pretraining run coloured by optimizer. Tests H1.
  - `fig_delta_erank`: effective rank of ΔW vs layer per task, lines per parent. Tests H2.
  - `fig_lora_gap`: recovered fraction vs LoRA rank (log x), one panel per task, lines per parent with seeds as points. Tests H3.
  - `fig_energy_vs_recovered`: scatter of ΔW mean top-r energy (x) against recovered fraction at rank r (y), one marker per (parent, task, r). The mechanism plot.
  - `fig_forgetting`: forgetting vs rank per task and parent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot.py
import csv
from pathlib import Path
from scripts.plot import make_figures

def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def test_make_figures_writes_all(tmp_path):
    a = tmp_path / "analysis"
    fields = dict(rows=8, cols=8, srank=2.0, fro=1.0, top4=0.5, top16=0.8, top64=1.0)
    _csv(a / "pretrained_spectra.csv", [dict(run=r, optimizer=o, arm=r[:2], seed=0, val_loss=3.0, name=f"blocks.{l}.attn.q.weight",
         layer=l, module="attn.q", erank=5 + l, **fields) for r, o in [("p1_adamw_m30_s0", "adamw"), ("p2_muon_m30_s0", "muon")] for l in range(2)])
    _csv(a / "delta_spectra.csv", [dict(run=f"{p}__full_{t}", parent=p, task=t, name=f"blocks.{l}.attn.q.weight", layer=l, module="attn.q",
         erank=3 + l, **fields) for p in ("p1_adamw_m30_s0", "p2_muon_m30_s0") for t in ("code", "sup") for l in range(2)])
    _csv(a / "lora_overlap.csv", [dict(run="x", parent="p1_adamw_m30_s0", task="code", rank=4, name="blocks.0.attn.q", overlap=0.3)])
    _csv(a / "finetune_results.csv", [dict(run=f"{p}__{m}_{t}", parent=p, task=t, method=("full" if m == "full" else "lora"),
         rank=("" if m == "full" else m[4:]), metric_before=0, metric_full=1, metric_after=0.5, recovered=("nan" if m == "full" else 0.5), forgetting=0.1)
         for p in ("p1_adamw_m30_s0", "p2_muon_m30_s0") for t in ("code", "sup") for m in ("full", "lora4", "lora16", "lora64")])
    out = make_figures(a, tmp_path / "figs")
    names = {p.name for p in out}
    for stem in ("fig_erank_by_layer", "fig_delta_erank", "fig_lora_gap", "fig_energy_vs_recovered", "fig_forgetting"):
        assert f"{stem}.png" in names and f"{stem}.pdf" in names
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_plot.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/plot.py
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
    for ax, mod in zip(axes, MODULES):
        for run, layers in sorted(by[mod].items()):
            xs = sorted(layers); ax.plot(xs, [layers[x] for x in xs], marker="o", ms=3, color=COLOR[_opt(run)], label=run, alpha=0.8)
        ax.set_title(mod); ax.set_xlabel("layer")
    axes[0].set_ylabel("effective rank"); axes[0].legend(fontsize=6)
    fig.suptitle("H1: effective rank of pretrained weights by layer")
    return _save(fig, fig_dir, "fig_erank_by_layer")


def fig_delta_erank(delta: list[dict], fig_dir: Path) -> list[Path]:
    tasks = sorted({r["task"] for r in delta})
    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(4 * max(1, len(tasks)), 3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        by = defaultdict(lambda: defaultdict(list))
        for r in delta:
            if r["task"] == task:
                by[r["parent"]][int(r["layer"])].append(float(r["erank"]))
        for parent, layers in sorted(by.items()):
            xs = sorted(layers); ax.plot(xs, [sum(layers[x]) / len(layers[x]) for x in xs], marker="o", ms=3, color=COLOR[_opt(parent)], label=parent)
        ax.set_title(f"task: {task}"); ax.set_xlabel("layer"); ax.set_ylabel("mean effective rank of ΔW"); ax.legend(fontsize=6)
    fig.suptitle("H2: effective rank of the full fine-tuning update")
    return _save(fig, fig_dir, "fig_delta_erank")


def _lora_rows(res: list[dict]) -> list[dict]:
    return [r for r in res if r["method"] == "lora" and r["rank"] not in ("", "None")]


def fig_lora_gap(res: list[dict], fig_dir: Path, key: str = "recovered", stem: str = "fig_lora_gap", ylabel: str = "fraction of full-FT gain recovered") -> list[Path]:
    rows = _lora_rows(res)
    tasks = sorted({r["task"] for r in rows})
    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(4 * max(1, len(tasks)), 3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        by = defaultdict(dict)
        for r in rows:
            if r["task"] == task and not math.isnan(float(r[key])):
                by[r["parent"]][int(r["rank"])] = float(r[key])
        for parent, pts in sorted(by.items()):
            xs = sorted(pts); ax.plot(xs, [pts[x] for x in xs], marker="o", color=COLOR[_opt(parent)], label=parent)
        ax.set_xscale("log", base=2); ax.set_xlabel("LoRA rank"); ax.set_ylabel(ylabel); ax.set_title(f"task: {task}"); ax.legend(fontsize=6)
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
            x = sum(energy[key]) / len(energy[key]); ax.scatter(x, float(r["recovered"]), color=COLOR[_opt(r["parent"])],
                                                              marker="o" if r["task"] == "code" else "s")
            ax.annotate(f"r{r['rank']}", (x, float(r["recovered"])), fontsize=6)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
    ax.set_xlabel("mean top-r energy of full-FT ΔW"); ax.set_ylabel("recovered fraction at rank r")
    ax.set_title("does ΔW spectrum predict the LoRA gap?")
    return _save(fig, fig_dir, "fig_energy_vs_recovered")


def make_figures(analysis_dir: Path, fig_dir: Path) -> list[Path]:
    analysis_dir, fig_dir = Path(analysis_dir), Path(fig_dir)
    pre, delta, res = _read(analysis_dir / "pretrained_spectra.csv"), _read(analysis_dir / "delta_spectra.csv"), _read(analysis_dir / "finetune_results.csv")
    out = []
    out += fig_erank_by_layer(pre, fig_dir)
    out += fig_delta_erank(delta, fig_dir)
    out += fig_lora_gap(res, fig_dir)
    out += fig_energy_vs_recovered(delta, res, fig_dir)
    out += fig_lora_gap(res, fig_dir, key="forgetting", stem="fig_forgetting", ylabel="increase in FineWeb-Edu val loss")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", default="runs/analysis"); ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    for p in make_figures(Path(a.analysis), Path(a.out)):
        print(p)
```

- [ ] **Step 4: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_plot.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/plot.py tests/test_plot.py
git commit -m "plot: H1-H3 figures and mechanism scatter from analysis CSVs"
```

---

### Task 4: End-to-end on real runs and the results table

**Files:**
- Create: `scripts/results_table.py`
- Test: `tests/test_results_table.py`

**Interfaces:**
- Produces: `results_table(analysis_dir) -> str` returning a Markdown table with one row per (parent, task, method/rank) showing `metric_after`, `recovered`, `forgetting`, plus a second table of mean effective rank per optimizer for pretrained weights and for ΔW. Printed to stdout and written to `runs/analysis/results.md` for pasting into the paper.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_results_table.py
from pathlib import Path
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_results_table.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/results_table.py
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
```

- [ ] **Step 4: Run tests, then the real pipeline**

Run: `.venv\Scripts\python.exe -m pytest tests/test_results_table.py -v` → 1 passed

When the core grid has finished (Plan 5 Task 6):

```
.venv\Scripts\python.exe scripts/analyze.py --runs runs --out runs/analysis
.venv\Scripts\python.exe scripts/plot.py --analysis runs/analysis --out paper/figures
.venv\Scripts\python.exe scripts/results_table.py --analysis runs/analysis
```

Expected: five figures in `paper/figures/`, `runs/analysis/results.md` printed. Report the H1, H2, H3 numbers to the user plainly, including any that contradict the hypotheses, and add a decision-log line in `CLAUDE.md` §11 with the headline numbers.

- [ ] **Step 5: Commit**

```bash
git add scripts/results_table.py tests/test_results_table.py paper/figures/
git commit -m "spectra: results tables; first figures from the core grid"
```

---

## Self-review

- **Spec coverage:** effective rank and stable rank of pretrained matrices (Task 1, 2 → H1), ΔW spectra (Task 2 → H2), top-r energy vs LoRA gap (Task 2, 3 → H3 mechanism), subspace overlap (Tasks 1, 2), recovered fraction and forgetting (Task 2), figures only from script (Task 3), report tables (Task 4).
- **Placeholders:** none.
- **Type consistency:** `results.json` keys match Plan 5 (`before/after`, `code_val_loss`, `sup_acc_mean`, `pre_val_loss`, `method`, `rank`, `task`, `parent`). LoRA name → weight name mapping is `name + ".weight"`, matching Plan 5's `lora_state_dict` keys (module names) and Plan 1's parameter names. `DONE` third field is the final val loss, matching Plan 4 Task 3.
