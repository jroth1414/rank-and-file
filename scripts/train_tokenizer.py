"""Train the 32k tokenizer on a slice of FineWeb-Edu.

Usage: python scripts/train_tokenizer.py --out data/tokenizer.json --docs 200000
"""
import argparse, itertools, os
from datasets import load_dataset
from rankfile.tokenizer import train_tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tokenizer.json")
    ap.add_argument("--docs", type=int, default=200_000)
    ap.add_argument("--vocab", type=int, default=32768)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    texts = (r["text"] for r in itertools.islice(ds, a.docs))
    tok = train_tokenizer(texts, a.vocab, a.out)
    print(f"saved {a.out}: vocab {tok.get_vocab_size()}")

if __name__ == "__main__":
    main()
