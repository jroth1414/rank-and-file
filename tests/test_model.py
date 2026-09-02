import torch
import pytest
from rankfile.model import ModelConfig, Transformer, RMSNorm, precompute_rope, apply_rope

def test_forward_shape(tiny_cfg):
    m = Transformer(tiny_cfg)
    idx = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    out = m(idx)
    assert out.shape == (2, 16, tiny_cfg.vocab_size)

def test_tied_embeddings_share_storage(tiny_cfg):
    m = Transformer(tiny_cfg)
    assert m.lm_head_weight().data_ptr() == m.embed.weight.data_ptr()

def test_gqa_kv_projection_width(tiny_cfg):
    m = Transformer(tiny_cfg)
    blk = m.blocks[0].attn
    assert blk.k.weight.shape == (tiny_cfg.n_kv_head * tiny_cfg.head_dim, tiny_cfg.d_model)
    assert blk.q.weight.shape == (tiny_cfg.n_head * tiny_cfg.head_dim, tiny_cfg.d_model)

def test_no_biases(tiny_cfg):
    m = Transformer(tiny_cfg)
    assert all("bias" not in n for n, _ in m.named_parameters())

def test_rmsnorm_unit_scale():
    n = RMSNorm(8)
    x = torch.randn(3, 8) * 5
    y = n(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3), atol=1e-4)

def test_rope_preserves_norm():
    cos, sin = precompute_rope(16, 32, 10000.0)
    x = torch.randn(1, 2, 32, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)

def test_causal_no_future_leak(tiny_cfg):
    m = Transformer(tiny_cfg).eval()
    idx = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    a = m(idx)[0, :6]
    idx2 = idx.clone(); idx2[0, 6:] = 1
    b = m(idx2)[0, :6]
    assert torch.allclose(a, b, atol=1e-5)

def test_init_scales(tiny_cfg):
    m = Transformer(tiny_cfg)
    assert abs(m.embed.weight.std().item() - tiny_cfg.init_std) < 0.005
    o = m.blocks[0].attn.o.weight.std().item()
    expected = tiny_cfg.init_std / (2 * tiny_cfg.n_layer) ** 0.5
    assert abs(o - expected) < 0.005

@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_doc_mask_blocks_cross_document_attention(tiny_cfg):
    """Output for doc 2 with masking must equal running doc 2 alone (up to fp error)."""
    torch.manual_seed(0)
    m = Transformer(tiny_cfg).eval()
    T = 16
    idx = torch.randint(1, tiny_cfg.vocab_size, (1, T))
    doc_ids = torch.zeros(1, T, dtype=torch.long); doc_ids[0, 8:] = 1
    with torch.no_grad():
        masked = m(idx, doc_ids)[0, 8:]
        alone = m(idx[:, 8:], pos_offset=8)[0]   # same RoPE positions as inside the window
    assert torch.allclose(masked, alone, atol=1e-4), (masked - alone).abs().max()

@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_doc_mask_equals_causal_when_single_doc(tiny_cfg):
    torch.manual_seed(0)
    m = Transformer(tiny_cfg).eval()
    idx = torch.randint(1, tiny_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        a = m(idx)
        b = m(idx, torch.zeros(2, 16, dtype=torch.long))
    assert torch.allclose(a, b, atol=1e-4)
