import numpy as np
from rankfile.tokenizer import EOT_ID, train_tokenizer, load_tokenizer, encode_docs, decode

CORPUS = ["the quick brown fox jumps over the lazy dog. " * 20, "def f(x):\n    return x * 2\n" * 20,
          "Numbers 1234 and symbols !@# and unicode é ñ 中文 " * 20]

def test_train_load_roundtrip(tmp_path):
    p = tmp_path / "tok.json"
    tok = train_tokenizer(CORPUS, vocab_size=600, out_path=p)
    tok2 = load_tokenizer(p)
    assert tok2.get_vocab_size() == tok.get_vocab_size() > 256  # loaded == trained; larger than the byte alphabet + EOT
    assert tok2.token_to_id("<|endoftext|>") == EOT_ID == 0
    s = "unicode é ñ 中文 and code def f(x):"
    assert decode(tok2, tok2.encode(s).ids) == s

def test_encode_docs_appends_eot_and_is_uint16(tmp_path):
    tok = train_tokenizer(CORPUS, vocab_size=600, out_path=tmp_path / "t.json")
    arrs = encode_docs(tok, ["hello world", "second doc"])
    assert len(arrs) == 2 and all(a.dtype == np.uint16 for a in arrs)
    assert all(a[-1] == EOT_ID for a in arrs)
    assert all((a[:-1] != EOT_ID).all() for a in arrs)
