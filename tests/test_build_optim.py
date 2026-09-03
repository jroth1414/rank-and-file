import pytest
import torch

from rankfile.model import ModelConfig, Transformer
from rankfile.optim.build import build_optimizers, set_lr
from rankfile.optim.muon import Muon


def _model():
    return Transformer(
        ModelConfig(
            vocab_size=128,
            n_layer=2,
            d_model=32,
            n_head=2,
            n_kv_head=1,
            head_dim=16,
            d_ff=64,
            max_seq_len=16,
        )
    )


def test_adamw_arm_single_optimizer_covers_all_params():
    m = _model()
    opts = build_optimizers(m, "adamw", lr=1e-3, weight_decay=0.1)
    assert len(opts) == 1 and isinstance(opts[0], torch.optim.AdamW)
    covered = {id(p) for g in opts[0].param_groups for p in g["params"]}
    assert covered == {id(p) for p in m.parameters()}
    wd = {g["weight_decay"] for g in opts[0].param_groups}
    assert wd == {0.0, 0.1}


def test_muon_arm_routes_hidden_matrices_to_muon():
    m = _model()
    opts = build_optimizers(m, "muon", lr=1e-3, weight_decay=0.1)
    assert [type(o) for o in opts] == [Muon, torch.optim.AdamW]
    muon_ids = {id(p) for g in opts[0].param_groups for p in g["params"]}
    assert muon_ids == {id(p) for p in m.hidden_matrix_params()}
    adam_ids = {id(p) for g in opts[1].param_groups for p in g["params"]}
    assert id(m.embed.weight) in adam_ids and not (muon_ids & adam_ids)


def test_set_lr_applies_everywhere():
    opts = build_optimizers(_model(), "muon", lr=1e-3, weight_decay=0.1)
    set_lr(opts, 5e-4)
    assert all(g["lr"] == 5e-4 for o in opts for g in o.param_groups)


def test_set_lr_respects_per_group_adamw_scale():
    opts = build_optimizers(_model(), "muon", lr=1e-3, weight_decay=0.1, adamw_lr_scale=0.5)
    set_lr(opts, 2e-3)
    muon_opt, adamw_opt = opts
    assert all(g["lr"] == 2e-3 for g in muon_opt.param_groups)
    assert all(g["lr"] == 1e-3 for g in adamw_opt.param_groups)


def test_both_arms_decay_the_same_parameters():
    m = _model()

    def decayed(arm):
        return {
            id(p)
            for o in build_optimizers(m, arm, lr=1e-3, weight_decay=0.1)
            for g in o.param_groups
            if g["weight_decay"] != 0.0
            for p in g["params"]
        }

    assert decayed("adamw") == decayed("muon")


def test_build_optimizers_rejects_unknown_arm():
    with pytest.raises(ValueError):
        build_optimizers(_model(), "sgd", lr=1e-3, weight_decay=0.1)
