import csv
import json
import math
import shutil

import pytest
import torch

from rankfile.checkpoint import save_checkpoint, write_latest
from rankfile.config import to_yaml
from rankfile.lora import apply_lora, lora_state_dict
from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers
from scripts.analyze import analyze

MC = ModelConfig(vocab_size=64, n_layer=1, d_model=16, n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=8)


def _pre(root, name, opt, arm=None, batch_tokens=64):
    torch.manual_seed(0); m = Transformer(MC); run = root / name; run.mkdir(parents=True)
    to_yaml(MC, run / "model.yaml")
    to_yaml({"optimizer": opt, "arm": arm if arm is not None else name[:2], "seed": 0,
              "batch_tokens": batch_tokens}, run / "config.resolved.yaml")
    save_checkpoint(run / "ckpt_0000001.pt", m, build_optimizers(m, opt, 1e-3, 0.1), 1, 0, 0, {}); write_latest(run, "ckpt_0000001.pt")
    (run / "DONE").write_text("1 0 3.5\n"); return m


def _ft(root, parent, method, task, m, rank=None, before=3.0, after=2.0, name=None):
    name = name or f"{parent}__{'full' if method=='full' else f'lora{rank}'}_{task}"; run = root / name; run.mkdir()
    if method == "full":
        m2 = Transformer(MC); m2.load_state_dict(m.state_dict())
        with torch.no_grad():
            for p in m2.blocks.parameters(): p.add_(0.1 * torch.randn_like(p))
        torch.save(m2.state_dict(), run / "model.pt")
    else:
        m2 = Transformer(MC); m2.load_state_dict(m.state_dict()); apply_lora(m2, rank, 2 * rank)
        for _n, mod in m2.named_modules():
            if hasattr(mod, "B"): mod.B.data.normal_()
        torch.save(lora_state_dict(m2), run / "lora.pt")
    key = "code_val_loss" if task == "code" else "sup_acc_mean"
    json.dump({"parent": parent, "method": method, "rank": rank, "task": task, "lr": 1e-4,
               "before": {key: before, "pre_val_loss": 3.5}, "after": {key: after, "pre_val_loss": 3.6}}, open(run / "results.json", "w"))
    return run


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
    assert all(0 < float(r["erank_norm"]) <= 1.0 + 1e-6 for r in pre)
    delta = list(csv.DictReader(open(out["delta_spectra"])))
    assert {r["run"] for r in delta} == {"p1_adamw_m30_s0__full_code", "p2_muon_m30_s0__full_sup"}
    assert all(r["lr"] == "0.0001" and r["optimizer"] in ("adamw", "muon") for r in delta)
    ov = list(csv.DictReader(open(out["lora_overlap"])))
    assert len(ov) == 7 and all(0.0 <= float(r["overlap"]) <= 1.0 + 1e-6 for r in ov)
    assert all(0.0 <= float(r["overlap_two_sided"]) <= 1.0 + 1e-6 for r in ov)
    assert all(abs(float(r["chance"]) - 2 / min(16, 16)) < 1e-9 or float(r["chance"]) > 0 for r in ov)
    assert all(r["optimizer"] == "adamw" and r["arm"] == "p1" and r["seed"] == "0" for r in ov)
    res = {r["run"]: r for r in csv.DictReader(open(out["finetune_results"]))}
    assert abs(float(res["p1_adamw_m30_s0__lora2_code"]["recovered"]) - 0.5) < 1e-9
    assert math.isnan(float(res["p1_adamw_m30_s0__full_code"]["recovered"]))
    assert abs(float(res["p2_muon_m30_s0__full_sup"]["forgetting"]) - 0.1) < 1e-9
    assert res["p1_adamw_m30_s0__full_code"]["optimizer"] == "adamw"
    assert res["p1_adamw_m30_s0__lora2_code"]["lr"] == "0.0001"
    # alpha/parent_ckpt columns exist and are empty strings for results.json without them
    assert res["p1_adamw_m30_s0__lora2_code"]["alpha"] == ""
    assert res["p1_adamw_m30_s0__lora2_code"]["parent_ckpt"] == ""


def test_analyze_mismatched_before_raises(tmp_path):
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw")
    _ft(root, "p1_adamw_m30_s0", "full", "code", m1, before=3.0, after=2.0)
    _ft(root, "p1_adamw_m30_s0", "lora", "code", m1, rank=2, before=3.5, after=2.5)
    with pytest.raises(ValueError, match="p1_adamw_m30_s0__lora2_code"):
        analyze(root, tmp_path / "analysis")


def test_analyze_before_mismatch_warns_below_raise_threshold(tmp_path, capsys):
    # task="sup" so the metric is the raw accuracy value (no -loss sign flip), which
    # keeps before_mismatch's sign easy to reason about in the test.
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw")
    _ft(root, "p1_adamw_m30_s0", "full", "sup", m1, before=0.5, after=0.6)
    # mismatch of 5e-3: above the 1e-3 warn threshold, below the 1e-2 raise threshold.
    _ft(root, "p1_adamw_m30_s0", "lora", "sup", m1, rank=2, before=0.505, after=0.55)
    out = analyze(root, tmp_path / "analysis")
    assert "warning" in capsys.readouterr().out
    res = {r["run"]: r for r in csv.DictReader(open(out["finetune_results"]))}
    assert abs(float(res["p1_adamw_m30_s0__lora2_sup"]["before_mismatch"]) - 0.005) < 1e-9


def test_analyze_allow_before_mismatch_suppresses_raise(tmp_path):
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw")
    _ft(root, "p1_adamw_m30_s0", "full", "sup", m1, before=0.5, after=0.6)
    _ft(root, "p1_adamw_m30_s0", "lora", "sup", m1, rank=2, before=0.6, after=0.55)
    out = analyze(root, tmp_path / "analysis", allow_before_mismatch=True)
    res = {r["run"]: r for r in csv.DictReader(open(out["finetune_results"]))}
    assert abs(float(res["p1_adamw_m30_s0__lora2_sup"]["before_mismatch"]) - 0.1) < 1e-9


def test_analyze_recovered_is_nan_when_full_ft_did_not_move(tmp_path):
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw")
    _ft(root, "p1_adamw_m30_s0", "full", "code", m1, before=3.0, after=3.0)
    _ft(root, "p1_adamw_m30_s0", "lora", "code", m1, rank=2, before=3.0, after=2.5)
    out = analyze(root, tmp_path / "analysis")
    res = {r["run"]: r for r in csv.DictReader(open(out["finetune_results"]))}
    assert math.isnan(float(res["p1_adamw_m30_s0__lora2_code"]["recovered"]))


def test_analyze_recovered_is_nan_when_full_ft_got_worse(tmp_path):
    # task="code" metric is -loss (higher is better); a full-FT loss that goes UP
    # (3.0 -> 3.5) means the metric went down, i.e. full FT made things worse.
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw")
    _ft(root, "p1_adamw_m30_s0", "full", "code", m1, before=3.0, after=3.5)
    _ft(root, "p1_adamw_m30_s0", "lora", "code", m1, rank=2, before=3.0, after=2.5)
    out = analyze(root, tmp_path / "analysis")
    res = {r["run"]: r for r in csv.DictReader(open(out["finetune_results"]))}
    assert math.isnan(float(res["p1_adamw_m30_s0__lora2_code"]["recovered"]))


def test_analyze_skips_pretraining_run_missing_latest(tmp_path, capsys):
    root = tmp_path / "runs"
    run = root / "p1_adamw_m30_s0"; run.mkdir(parents=True)
    to_yaml({"optimizer": "adamw", "arm": "p1", "seed": 0, "batch_tokens": 64}, run / "config.resolved.yaml")
    (run / "DONE").write_text("1 0 3.5\n")  # no checkpoint / latest.txt written
    out = analyze(root, tmp_path / "analysis")
    pre = list(csv.DictReader(open(out["pretrained_spectra"])))
    assert pre == []
    assert "warning" in capsys.readouterr().out


def test_analyze_excludes_non_arm_and_malformed_ft_names(tmp_path):
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw")
    _ft(root, "p1_adamw_m30_s0", "full", "code", m1, before=3.0, after=2.0)
    # a sweep pretraining run (arm "sweep") must not enter the CSVs by default.
    _pre(root, "sweep_adamw_lr1e-3", "adamw", arm="sweep")
    # a sweep fine-tune run dir that does not match the canonical naming pattern.
    sweep_ft = root / "ftsweep_full_lr1e-4"; sweep_ft.mkdir()
    (sweep_ft / "results.json").write_text("not valid json but never parsed", encoding="utf-8")
    out = analyze(root, tmp_path / "analysis")
    pre = list(csv.DictReader(open(out["pretrained_spectra"])))
    assert {r["run"] for r in pre} == {"p1_adamw_m30_s0"}
    delta = list(csv.DictReader(open(out["delta_spectra"])))
    assert {r["run"] for r in delta} == {"p1_adamw_m30_s0__full_code"}


def test_analyze_include_glob_overrides_arms(tmp_path):
    root = tmp_path / "runs"
    _pre(root, "p1_adamw_m30_s0", "adamw")
    _pre(root, "sweep_adamw_lr1e-3", "adamw", arm="sweep")
    out = analyze(root, tmp_path / "analysis", include="sweep_*")
    pre = list(csv.DictReader(open(out["pretrained_spectra"])))
    assert {r["run"] for r in pre} == {"sweep_adamw_lr1e-3"}


def test_analyze_duplicate_cell_raises(tmp_path):
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw")
    _ft(root, "p1_adamw_m30_s0", "lora", "code", m1, rank=4, before=3.0, after=2.5)
    # A second, differently-named run dir (still matching the canonical pattern) whose
    # results.json claims the same (parent, task, method, rank) cell.
    dup = root / "p1_adamw_m30_s0__lora04_code"
    shutil.copytree(root / "p1_adamw_m30_s0__lora4_code", dup)
    with pytest.raises(ValueError, match="duplicate"):
        analyze(root, tmp_path / "analysis")


def test_analyze_all_ckpts_writes_trajectory(tmp_path):
    root = tmp_path / "runs"
    m1 = _pre(root, "p1_adamw_m30_s0", "adamw", batch_tokens=100)
    run = root / "p1_adamw_m30_s0"
    opts = build_optimizers(m1, "adamw", 1e-3, 0.1)
    save_checkpoint(run / "ckpt_0000002.pt", m1, opts, 2, 0, 0, {})
    save_checkpoint(run / "ckpt_0000004.pt", m1, opts, 4, 0, 0, {})
    metrics = [
        {"step": 1, "tokens": 100, "val_loss": 4.0},
        {"step": 2, "tokens": 200, "val_loss": 3.5},
        {"step": 3, "tokens": 300, "val_loss": 3.0},
        {"step": 4, "tokens": 400, "val_loss": 2.5},
        {"step": 10, "loss": 1.0},  # a training-loss-only line, no val_loss/tokens pairing issue
    ]
    with open(run / "metrics.jsonl", "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    out = analyze(root, tmp_path / "analysis", all_ckpts=True)
    assert "pretrained_spectra_traj" in out
    traj = list(csv.DictReader(open(out["pretrained_spectra_traj"])))
    # _pre() itself writes ckpt_0000001.pt (the run's "latest"), so all three permanent
    # checkpoints - steps 1, 2, 4 - are included.
    steps = {int(r["step"]) for r in traj}
    assert steps == {1, 2, 4}
    by_step = {int(r["step"]): r for r in traj}
    assert int(by_step[1]["tokens"]) == 100 and abs(float(by_step[1]["val_loss"]) - 4.0) < 1e-9
    assert int(by_step[2]["tokens"]) == 200 and abs(float(by_step[2]["val_loss"]) - 3.5) < 1e-9
    assert int(by_step[4]["tokens"]) == 400 and abs(float(by_step[4]["val_loss"]) - 2.5) < 1e-9
    assert all(r["arm"] == "p1" and r["optimizer"] == "adamw" for r in traj)


def test_analyze_no_trajectory_csv_without_all_ckpts(tmp_path):
    root = tmp_path / "runs"
    _pre(root, "p1_adamw_m30_s0", "adamw")
    out = analyze(root, tmp_path / "analysis")
    assert "pretrained_spectra_traj" not in out
