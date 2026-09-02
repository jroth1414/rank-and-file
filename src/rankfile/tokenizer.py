"""32k byte-level BPE trained on the pretraining data. EOT is id 0 and ends every document."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
EOT_ID = 0


def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: str | Path) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.token_to_id(EOT) != EOT_ID:
        raise ValueError(f"{EOT!r} must be id {EOT_ID}, got {tok.token_to_id(EOT)}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode_docs(tok: Tokenizer, texts: list[str]) -> list[np.ndarray]:
    if tok.get_vocab_size() > 65535:
        raise ValueError(f"vocab size {tok.get_vocab_size()} does not fit in uint16")
    # add_special_tokens=False alone does not stop a literal "<|endoftext|>" in the text
    # from being matched as the special token (tokenizers==0.23.1); encode_special_tokens
    # is the switch that forces it through ordinary byte-level BPE instead. It does not
    # persist across save/load, so it must be set here on every call.
    tok.encode_special_tokens = True
    out = []
    for enc in tok.encode_batch(texts, add_special_tokens=False):
        ids = np.asarray(enc.ids + [EOT_ID], dtype=np.uint16)
        out.append(ids)
    return out


def decode(tok: Tokenizer, ids: Iterable[int] | np.ndarray) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=False)
