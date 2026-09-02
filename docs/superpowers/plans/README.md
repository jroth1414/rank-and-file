# rank-and-file implementation plans

Six plans, one per subsystem. Each is self-contained: it restates the
interfaces it consumes and produces, so an agent can execute one plan in a
fresh context without reading the others. Execute in order; each plan's
"Consumes" block names exactly what earlier plans must have delivered.

| # | Plan | Delivers | Depends on |
|---|---|---|---|
| 1 | `2026-09-02-01-model-and-tokenizer.md` | `rankfile/config.py`, `rankfile/model.py`, `rankfile/tokenizer.py`, `configs/model/*.yaml`, `scripts/train_tokenizer.py` | — |
| 2 | `2026-09-02-02-data-pipeline.md` | `rankfile/data.py`, `scripts/prepare_data.py`, tokenized shards in `data/` | 1 |
| 3 | `2026-09-02-03-optimizers-and-schedule.md` | `rankfile/optim/muon.py`, `rankfile/optim/build.py`, `rankfile/schedule.py` | 1 |
| 4 | `2026-09-02-04-pretraining-loop.md` | `rankfile/checkpoint.py`, `rankfile/train.py`, `scripts/queue.py`, `configs/train/*.yaml`, m30 smoke run | 1, 2, 3 |
| 5 | `2026-09-02-05-lora-and-finetuning.md` | `rankfile/lora.py`, `rankfile/tasks.py`, `rankfile/finetune.py`, `scripts/prepare_code.py`, `configs/finetune/*.yaml` | 1, 2, 4 |
| 6 | `2026-09-02-06-spectral-analysis-and-plots.md` | `rankfile/spectra.py`, `scripts/analyze.py`, `scripts/plot.py` | 4, 5 |

The stretch attention-head analysis (`rankfile/heads.py`) has no plan yet.
Write one only after Plan 6 is complete and the core grid has run.

## Rules that apply to every plan

Copied from `CLAUDE.md`; every task inherits them.

- Python 3.11 in `.venv` (uv). Run everything as `.venv\Scripts\python.exe ...`
  from the repo root. **Never use the global Python**; its torch is broken.
- torch 2.14.0+cu130, triton-windows 3.8. `torch.compile` works. Flash SDPA has
  no kernel; cuDNN SDPA and FlexAttention do.
- Micro-batch **8 × 2048** max on the 124M model. Peak allocated memory must
  stay **≤ 14 GiB**; WDDM pages to system RAM instead of raising OOM.
- Tests: `pytest` must pass before every commit that touches `src/`.
  GPU tests are marked `@pytest.mark.gpu` and skipped when CUDA is absent.
- Commits: imperative subject ≤ 72 chars, area prefix (`model:`, `data:`,
  `muon:`, `train:`, `lora:`, `spectra:`, `docs:`). **No AI attribution
  trailers of any kind.** Author is the repo-local git identity (John Roth).
- Never edit a config a completed run used. Never delete under `runs/` or
  `data/`. Never launch a multi-hour run without asking the user.
- No new dependencies beyond `pyproject.toml` without asking.
- Run names: `{arm}_{opt}_{size}_s{seed}`; fine-tunes: `{parent}__{method}_{task}`.

## Shared vocabulary

- **m30**: smoke-test model (4 layers, 256 wide). Must run end to end in minutes.
- **m124**: the real model (12 layers, 768 wide, ~100.7M params with tied 32k embeddings).
- **EOT_ID = 0**: the `<|endoftext|>` token id; documents in shards are separated by it.
- **doc_ids**: per-token document index within a window, used for intra-document masking.
- **Arm**: P1 (AdamW 2.5B), P2 (Muon 2.5B), P3 (AdamW 3.75B).
