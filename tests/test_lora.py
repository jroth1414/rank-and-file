import torch

from rankfile.lora import TARGETS, LoRALinear, apply_lora, lora_deltas, lora_state_dict, merge_lora
from rankfile.model import ModelConfig, Transformer


def _m():
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=64, n_layer=2, d_model=32, n_head=2, n_kv_head=1, head_dim=16, d_ff=64, max_seq_len=16))


def test_zero_init_is_identity():
    base = torch.nn.Linear(8, 6, bias=False)
    l = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(l(x), base(x))
    assert l.scale == 2.0


def test_delta_rank_and_merge_equivalence():
    m = _m(); x = torch.randint(0, 64, (2, 16))
    params = apply_lora(m, r=4, alpha=8)
    assert all(p.requires_grad for p in params) and not m.embed.weight.requires_grad
    for p in params: p.data.normal_()
    out_lora = m(x)
    d = lora_deltas(m)
    assert set(n.split(".")[-1] for n in d) == set(TARGETS)
    assert all(torch.linalg.matrix_rank(v) <= 4 for v in d.values())
    merge_lora(m)
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())
    assert torch.allclose(m(x), out_lora, atol=1e-5)


def test_lora_state_dict_shapes():
    m = _m(); apply_lora(m, r=3, alpha=3)
    sd = lora_state_dict(m)
    assert "blocks.0.attn.q.weight" not in sd and "blocks.0.attn.q" in sd
    assert sd["blocks.0.mlp.down"]["A"].shape == (3, 64) and sd["blocks.0.mlp.down"]["B"].shape == (32, 3)
    assert len(sd) == 2 * len(TARGETS)
