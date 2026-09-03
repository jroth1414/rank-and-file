import json

import numpy as np
import pytest
import torch

from rankfile.data import write_shard
from rankfile.train import train
from tests.test_train import _tiny_cfgs, _tiny_data


def test_resume_matches_uninterrupted_run_bitwise(tmp_path):
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path)
    tc.name = "straight"; run_a = train(tc, mc, device="cpu")
    tc.name = "resumed"
    run_b = train(tc, mc, device="cpu", max_steps=17)      # stops after 17 steps, checkpoint written
    assert not (run_b / "DONE").exists()
    run_b = train(tc, mc, device="cpu")                    # resumes from latest.txt
    assert (run_b / "DONE").exists()
    # parse first, then filter on the dict key -- "loss" in the raw line text would
    # also match "val_loss" lines and KeyError below
    la = [json.loads(l) for l in (run_a / "metrics.jsonl").read_text().splitlines()]
    lb = [json.loads(l) for l in (run_b / "metrics.jsonl").read_text().splitlines()]
    la = [l for l in la if "loss" in l]
    lb = [l for l in lb if "loss" in l]
    assert [l["step"] for l in la] == [l["step"] for l in lb]
    assert [l["loss"] for l in la] == [l["loss"] for l in lb]
    sa = torch.load(run_a / "ckpt_0000040.pt", weights_only=False)["model"]
    sb = torch.load(run_b / "ckpt_0000040.pt", weights_only=False)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_resume_rejects_changed_data_provenance(tmp_path):
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path)
    tc.name = "provenance"
    run = train(tc, mc, device="cpu", max_steps=5)
    assert not (run / "DONE").exists()
    write_shard(np.zeros(500, dtype=np.uint16), tmp_path / "train_0001.bin")
    with pytest.raises(RuntimeError, match="provenance"):
        train(tc, mc, device="cpu")


def test_resume_rejects_changed_peak_lr(tmp_path):
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path)
    tc.name = "peak_lr_provenance"
    run = train(tc, mc, device="cpu", max_steps=5)
    assert not (run / "DONE").exists()
    tc.peak_lr = tc.peak_lr * 2
    with pytest.raises(RuntimeError, match="provenance"):
        train(tc, mc, device="cpu")


def test_train_refuses_restart_when_latest_missing_but_checkpoints_exist(tmp_path):
    _tiny_data(tmp_path)
    mc, tc = _tiny_cfgs(tmp_path)
    tc.name = "orphaned_latest"
    run = train(tc, mc, device="cpu", max_steps=5)
    assert not (run / "DONE").exists()
    assert list(run.glob("ckpt_*.pt"))
    (run / "latest.txt").unlink()
    with pytest.raises(RuntimeError, match="refusing to restart"):
        train(tc, mc, device="cpu")
