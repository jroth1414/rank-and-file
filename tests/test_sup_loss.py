import torch
import torch.nn.functional as F

from rankfile.model import ModelConfig, Transformer
from rankfile.tasks import sup_loss

CFG = ModelConfig(vocab_size=97, n_layer=2, d_model=32, n_head=4, n_kv_head=2, head_dim=8, d_ff=64,
                  max_seq_len=32)


def _reference(model, ids, mask):
    """Full-vocab-logits formulation: the definition sup_loss must reproduce."""
    logits = model(ids[:, :-1]).float()
    tgt, m = ids[:, 1:], mask[:, 1:].float()
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none")
    return (nll.view_as(m) * m).sum() / m.sum().clamp(min=1)


def _batch():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, CFG.vocab_size, (3, 20), generator=g)
    mask = torch.zeros_like(ids)
    mask[0, 18:] = 1
    mask[1, 10] = 1
    mask[2, 5:7] = 1
    return ids, mask


def test_sup_loss_matches_full_logits_value_and_grad():
    torch.manual_seed(0)
    model = Transformer(CFG)
    ids, mask = _batch()
    ref = _reference(model, ids, mask)
    ref.backward()
    ref_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    got = sup_loss(model, ids, mask)
    got.backward()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
    for g, r in zip((p.grad for p in model.parameters()), ref_grads, strict=True):
        torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-6)


def test_sup_loss_never_builds_full_vocab_logits():
    # Full-vocab logits at sup_batch 32 x 512 were ~2 GiB fp32 and pushed full FT
    # past the 14 GiB ceiling; only label positions may be projected to the vocab.
    torch.manual_seed(0)
    model = Transformer(CFG)
    ids, mask = _batch()
    shapes = []
    hook = model.register_forward_hook(lambda *_: shapes.append("forward"))
    sup_loss(model, ids, mask)
    hook.remove()
    assert shapes == [], "sup_loss must not call model.forward (full [B,T,V] logits)"


def test_sup_loss_empty_mask_is_zero():
    torch.manual_seed(0)
    model = Transformer(CFG)
    ids, _ = _batch()
    assert sup_loss(model, ids, torch.zeros_like(ids)).item() == 0.0
