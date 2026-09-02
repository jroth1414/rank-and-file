# rank-and-file

**Does Muon pretraining change what LoRA can learn?**

Matched pairs of small Qwen3-style language models are pretrained with AdamW
and with Muon on identical data, then fine-tuned with full fine-tuning and
with LoRA at ranks 4, 16, and 64. The question is whether the optimizer used
in pretraining changes how much a low-rank adapter can recover, and whether
the singular-value spectrum of the full fine-tuning update predicts it.

Semester research project for EN.705.743 *ChatGPT from Scratch: Building and
Training Large Language Models*, Johns Hopkins University, Fall 2026.

- Proposal and full experimental design: [`proposal.md`](proposal.md)
- Engineering conventions and decision log: [`CLAUDE.md`](CLAUDE.md)

## Status

Proposal stage. Code scaffold only; nothing trains yet.

## Layout

```
configs/      model, training, and fine-tuning YAML configs
src/rankfile/ model, Muon, LoRA, training and fine-tuning loops, spectral analysis
scripts/      data prep, tokenizer training, run queue, analysis, plotting
tests/        pytest suite
docs/course/  course outline and rubric
paper/        write-up and generated figures
runs/         (gitignored) one directory per run
data/         (gitignored) tokenized shards, tokenizer, task data
```

## Setup

All experiments run on a single RTX 5070 Ti (16 GB) on native Windows 11.
`torch.compile` works through the `triton-windows` package; WSL2 is not needed.

```powershell
uv venv .venv --python 3.11
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch
uv pip install -e ".[dev]"
.venv\Scripts\python.exe scripts\env_check.py   # verifies GPU, SDPA backends, compile, FlexAttention
.venv\Scripts\python.exe -m pytest
```

Verified stack (2026-09-01): torch 2.14.0+cu130, triton-windows 3.8.0,
driver 610.88. Compiled throughput for the 124M model is about 80k tokens per
second at micro-batch 8, or 3.5 hours per billion tokens.

## Reproducing a run (planned)

```bash
python scripts/prepare_data.py --config configs/model/m124.yaml
python -m rankfile.train --model configs/model/m124.yaml --train configs/train/muon.yaml --seed 0
python -m rankfile.finetune --parent runs/p2_muon_m124_s0 --method configs/finetune/lora_r16.yaml --task code
python scripts/analyze.py --runs runs/ --out runs/spectra.csv
python scripts/plot.py
```
