import csv
import json
import math

import pytest
import torch

from rankfile.checkpoint import save_checkpoint, write_latest
from rankfile.config import to_yaml
from rankfile.lora import apply_lora, lora_state_dict
from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers
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
        for _n, mod in m2.named_modules():
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
