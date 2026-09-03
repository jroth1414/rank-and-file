from rankfile.model import ModelConfig, Transformer
from rankfile.tasks import SUP_TASKS, collate_sup, encode_sup, format_example, score_options
from rankfile.tokenizer import EOT_ID, train_tokenizer


def _tok(tmp_path):
    return train_tokenizer(["Review: great movie\nSentiment: positive negative yes no world sports business technology " * 50],
                           vocab_size=400, out_path=tmp_path / "t.json")

def test_templates_and_labels():
    p, y = format_example(SUP_TASKS["sst2"], {"sentence": "great movie", "label": 1})
    assert p.endswith("Sentiment:") and y == 1 and SUP_TASKS["sst2"].labels == [" negative", " positive"]
    p, y = format_example(SUP_TASKS["boolq"], {"passage": "Sky is blue.", "question": "is the sky blue", "answer": True})
    assert "Question:" in p and y == 1
    p, y = format_example(SUP_TASKS["ag_news"], {"text": "Stocks rose.", "label": 2})
    assert p.endswith("Topic:") and SUP_TASKS["ag_news"].labels[2] == " business"

def test_encode_masks_prompt_only(tmp_path):
    tok = _tok(tmp_path)
    ids, mask = encode_sup(tok, "Review: great\nSentiment:", " positive", max_len=32)
    assert len(ids) == len(mask) and sum(mask) >= 1 and mask[0] == 0
    assert ids[-1] == EOT_ID and mask[-1] == 1  # EOT after the label is trained too

def test_collate_pads_right(tmp_path):
    tok = _tok(tmp_path)
    a = encode_sup(tok, "Review: x\nSentiment:", " positive", 32); b = encode_sup(tok, "Review: a much longer review here\nSentiment:", " negative", 32)
    ids, mask = collate_sup([a, b], pad_id=EOT_ID)
    assert ids.shape == mask.shape and ids.shape[0] == 2 and ids.shape[1] == max(len(a[0]), len(b[0]))
    assert mask[0, len(a[0]):].sum() == 0

def test_score_options_returns_index(tmp_path):
    tok = _tok(tmp_path)
    m = Transformer(ModelConfig(vocab_size=400, n_layer=1, d_model=16, n_head=2, n_kv_head=1, head_dim=8, d_ff=32, max_seq_len=64)).eval()
    idx = score_options(m, tok, "Review: great\nSentiment:", [" negative", " positive"], device="cpu")
    assert idx in (0, 1)
