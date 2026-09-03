"""Supervised bundle: SST-2, BoolQ, AG News as LM prompts scored by label-word log-probs."""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from rankfile.tokenizer import EOT_ID


@dataclass
class TaskDef:
    name: str
    hf_path: str
    hf_config: str | None
    train_split: str
    eval_split: str
    template: Callable[[dict], str]
    labels: list[str]
    label_key: str


def _trunc(s: str, n: int = 1500) -> str:
    return s if len(s) <= n else s[:n]


def _sst2_template(r: dict) -> str:
    return f"Review: {_trunc(r['sentence']).strip()}\nSentiment:"


def _boolq_template(r: dict) -> str:
    return f"Passage: {_trunc(r['passage'])}\nQuestion: {r['question'].strip()}?\nAnswer:"


def _ag_news_template(r: dict) -> str:
    return f"Article: {_trunc(r['text'])}\nTopic:"


SUP_TASKS: dict[str, TaskDef] = {
    "sst2": TaskDef(
        "sst2", "stanfordnlp/sst2", None, "train", "validation",
        _sst2_template, [" negative", " positive"], "label",
    ),
    "boolq": TaskDef(
        "boolq", "google/boolq", None, "train", "validation",
        _boolq_template, [" no", " yes"], "answer",
    ),
    "ag_news": TaskDef(
        "ag_news", "fancyzhx/ag_news", None, "train", "test",
        _ag_news_template, [" world", " sports", " business", " technology"], "label",
    ),
}


def format_example(task: TaskDef, row: dict) -> tuple[str, int]:
    return task.template(row), int(row[task.label_key])


def encode_sup(tok, prompt: str, label_text: str, max_len: int) -> tuple[list[int], list[int]]:
    p = tok.encode(prompt).ids
    label_ids = tok.encode(label_text).ids + [EOT_ID]
    if len(p) + len(label_ids) > max_len:
        p = p[-(max_len - len(label_ids)):]  # keep the end of long prompts
    return p + label_ids, [0] * len(p) + [1] * len(label_ids)


def collate_sup(
    examples: list[tuple[list[int], list[int]]], pad_id: int = EOT_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), length), dtype=torch.long)
    for i, (x, m) in enumerate(examples):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(m)] = torch.tensor(m)
    return ids, mask


def sup_loss(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Next-token loss on positions whose *target* is a label token."""
    logits = model(ids[:, :-1]).float()
    tgt, m = ids[:, 1:], mask[:, 1:].float()
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none",
    ).view_as(m)
    return (nll * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def score_options(model, tok, prompt: str, labels: list[str], device) -> int:
    p = tok.encode(prompt).ids
    scores = []
    for lab in labels:
        label_ids = tok.encode(lab).ids
        ids = torch.tensor([p + label_ids], device=device)
        logp = F.log_softmax(model(ids[:, :-1]).float(), dim=-1)[0]
        scores.append(sum(logp[len(p) - 1 + j, label_ids[j]].item() for j in range(len(label_ids))))
    return int(torch.tensor(scores).argmax())


def load_sup_split(
    task: TaskDef, split: str, n: int, seed: int, cache_dir: str | Path = "data/sup",
) -> list[dict]:
    cache = Path(cache_dir) / f"{task.name}_{split}_{n}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from datasets import load_dataset

    ds = load_dataset(task.hf_path, task.hf_config, split=split)
    rows = [dict(r) for r in ds]
    by_label: dict[int, list[dict]] = {}
    for r in rows:
        by_label.setdefault(int(r[task.label_key]), []).append(r)
    rng = random.Random(seed)
    per = max(1, n // len(by_label))
    out: list[dict] = []
    for _lab, rs in sorted(by_label.items()):
        rng.shuffle(rs)
        out += rs[:per]
    rng.shuffle(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out
