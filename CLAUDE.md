# CLAUDE.md — rank-and-file

Research codebase for the EN.705.743 (JHU, Fall 2026) semester project:
**"Does Muon Pretraining Change What LoRA Can Learn?"**

Read this file fully before touching anything. `proposal.md` is the scientific
source of truth; this file is the engineering source of truth. If they
disagree on a scientific question (arms, hypotheses, metrics), `proposal.md`
wins and this file should be updated. If they disagree on an engineering
question (layout, conventions, tooling), this file wins.

---

## 1. What this project is

We pretrain matched pairs ("twins") of small Qwen3-style language models,
one with AdamW and one with Muon, on identical data with identical
architecture, then ask whether the optimizer changes what a low-rank adapter
(LoRA) can learn afterwards. The mechanism we test is spectral: Muon's
orthogonalized updates should leave flatter singular-value spectra in the
weights, the full fine-tuning update on a Muon twin should have higher
effective rank, and LoRA at low rank should therefore recover less of the
full fine-tuning improvement on Muon twins than on AdamW twins.

Three hypotheses, in the order they get tested:

| ID | Claim | Tested by |
|---|---|---|
| H1 | Muon twins have higher effective rank in projection matrices at matched validation loss | spectra of pretrained checkpoints |
| H2 | Full fine-tuning update ΔW has higher effective rank on Muon twins | spectra of (W_ft − W_pre) |
| H3 | LoRA at low rank recovers a smaller fraction of full-FT improvement on Muon twins; gap closes with rank | fine-tuning grid, LoRA-gap curves |

The deliverables are a presentation on **2026-12-01** and a paper on
**2026-12-08**. The intended venue after the course is an ICLR 2027
workshop (deadlines expected early February 2027).

## 2. Experimental design (summary; details in proposal.md §4)

### 2.1 Pretraining arms

| Arm | Optimizer | Size | Tokens | Seeds | Role |
|---|---|---|---|---|---|
| P1 | AdamW | 124M-class | 2.5B | 2 | baseline twin |
| P2 | Muon | 124M-class | 2.5B | 2 | Muon twin, matched tokens |
| P3 | AdamW | 124M-class | ~3.75B | 1 | AdamW trained to P2's validation loss |

P3 is the **primary comparison** for every mechanistic claim: it separates
"Muon made a better-trained model" from "Muon made a different model."
Second seeds train only after the first fine-tuning grid is complete.

### 2.2 Fine-tuning grid

- Methods: full fine-tuning; LoRA r ∈ {4, 16, 64} on all attention and MLP projections.
- Tasks: (a) code continued pretraining, 100M tokens Python, metric = held-out code loss;
  (b) supervised bundle SST-2 + BoolQ + AG News as LM prompts with label words, metric = accuracy.
- Forgetting: increase in FineWeb-Edu validation loss after fine-tuning.
- Primary metric: **LoRA gap** = fraction of full-FT improvement recovered by LoRA at rank r.
- Learning rate per method is swept once on one twin and reused on the other. Never tune per twin.

### 2.3 Spectral analysis

- Effective rank (entropy of normalized spectrum) and stable rank of every projection matrix.
- Same for ΔW = W_ft − W_pre from each full-FT run.
- Top-r energy fraction of ΔW for r ∈ {4, 16, 64}, compared against measured LoRA gap.

### 2.4 Stretch (only if core is on schedule)

Attention-head analysis on the same checkpoints: induction and previous-token
scores, sink mass, head redundancy. Lives in `src/rankfile/heads.py`. Do not
start this before the core grid has run.

## 3. Non-negotiable experimental controls

These are what make the twins "matched." Changing any of them between arms
invalidates the comparison. If you think one must change, stop and ask.

1. **Same architecture, same config file, same tokenizer** for every arm.
2. **Same data, same order.** Data shards are pre-tokenized once; the shard
   order is fixed by seed and shared across P1/P2 (P3 continues past it).
3. **Same schedule shape:** warmup-stable-decay, 2% warmup, 20% decay, batch ≈ 0.5M tokens, weight decay 0.1.
4. **Optimizer is the only difference** between P1 and P2. Muon covers 2-D
   hidden weight matrices; embeddings and norm gains use AdamW in both arms
   (standard Muon practice). Peak LR is chosen per optimizer by the sweep in
   `configs/train/sweep_*.yaml` and then frozen.
5. **Same fine-tuning hyperparameters across twins** for a given method.
6. **Checkpoint naming and seeds are deterministic** (see §6).

## 4. Architecture: Qwen3 recipe at 124M

| Component | Choice | Why |
|---|---|---|
| Norm | RMSNorm, pre-norm | Qwen3 |
| FFN | SwiGLU, hidden 2048 | Qwen3 |
| Biases | none | Qwen3 |
| Positions | RoPE | Qwen3 |
| Attention | GQA, 12 query heads / 4 KV heads, head dim 64 | Qwen3 uses 128-dim heads; at width 768 that leaves 6 heads, so 64 |
| QK-norm | RMSNorm on q and k per head | Qwen3 |
| Embeddings | tied | Qwen3-0.6B ties |
| Tokenizer | byte-level BPE, 32k vocab, trained on FineWeb-Edu | Qwen3's 151k vocab would be >100M params at this width |
| Layers / width | 12 / 768 | ~75M non-embedding, ~100M total |
| Context | 2048, intra-document masking | |

A `m30` config (fewer layers, narrower; measured ~11.3M total, ~3M non-embedding)
exists for smoke tests only. It must run the full pipeline end to end in minutes.
**No long run is ever launched without a passing `m30` smoke run of the same code.**

The optimizer-coupled recipes (modded-nanogpt, nanochat: value embeddings,
U-net skips, ReLU², etc.) are deliberately **not** used. Do not add their
tricks.

## 5. Hardware and environment (verified 2026-09-01)

Everything below was measured on this machine with `scripts/env_check.py`.
**Run that script before any long job**; it fails loudly if the stack drifted.

- **Machine:** Windows 11 Pro 26200, Ryzen 5 7500X3D (6c/12t), 31 GB RAM.
  Drives: **D: 2 TB free** (repo, venv, `data/`, `runs/` all live here),
  C: 63 GB free (only the Inductor cache in `%TEMP%\torchinductor_Admin`).
- **GPU:** single RTX 5070 Ti, 16 GB, Blackwell **sm_120**, driver 610.88 (CUDA 13.3 capable),
  **WDDM mode** (it also drives the display; ~1.6 GB is taken by the desktop).
  Measured **100 TFLOPS** bf16 matmul with fp32 accumulate.
- **Stack: native Windows, no WSL2.** `.venv` (uv, Python 3.11.9) with
  **torch 2.14.0+cu130** and **triton-windows 3.8.0**. `torch.compile` works.
  Install: `uv venv .venv --python 3.11 && uv pip install --index-url https://download.pytorch.org/whl/cu130 torch && uv pip install -e ".[dev]"`.
  The global Python has a broken torch install (missing `sympy`); never use it.
- **Measured throughput, m124 shape, micro-batch 8 × 2048, AdamW fused:**

  | Mode | tok/s | peak mem | h per 1B tokens |
  |---|---|---|---|
  | eager | 39k | 14.5 GiB | 7.1 |
  | `torch.compile` | **80k** | **7.6 GiB** | **3.5** |

  Plan with the compiled number. Real training adds data loading, Muon's
  Newton–Schulz, logging, and checkpoints, so expect 60–80k in practice.
- **Attention backends:** flash SDPA has **no kernel** in the Windows build.
  cuDNN SDPA works and is fastest (1.2 ms for B16 H12 T2048 D64 causal);
  memory-efficient works (2.3 ms). `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`
  dispatches correctly. **FlexAttention** with a document block mask and GQA
  compiles, runs at 2.8 ms fwd+bwd (B8), and matches a dense-mask reference.
  Use FlexAttention for intra-document masking.
- **WDDM memory trap:** Windows lets CUDA oversubscribe into system RAM
  instead of raising OOM. Micro-batch 16 eager "worked" at 27.6 GiB and ran
  at 3.5k tok/s, 11× slower. **Keep peak allocated ≤ 14 GiB**, use micro-batch
  8 with gradient accumulation to reach ~0.5M tokens per step, and have the
  training loop assert `torch.cuda.max_memory_allocated()` stays under the ceiling.
  The fp32 logits for cross-entropy are the largest single tensor at 32k vocab;
  compile fuses much of that, and a chunked loss is the fallback.
- **Sleep/restart:** AC sleep and hibernate timeouts are 0 (never). Windows
  Update active hours are 11:00–05:00, so an automatic restart can land
  between 05:00 and 11:00. Before a multi-day run, pause updates
  (Settings → Windows Update → Pause) or extend active hours.
- **TDR** (GPU watchdog) is at the 2 s default. Training kernels are ms-scale;
  if a Newton–Schulz or loss kernel ever trips it, raise `TdrDelay` rather than
  shrinking the work.
- **Windows long paths** are not enabled (`LongPathsEnabled` unset). Inductor
  cache paths have been fine so far; if compile fails with a path error,
  enable it in the registry or move `TORCHINDUCTOR_CACHE_DIR` to a short path on D:.
- **HF datasets:** streaming from `HuggingFaceFW/fineweb-edu` (sample-10BT)
  and `HuggingFaceTB/finemath` works. The hub warns about symlinks on
  Windows; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or enable Developer Mode.
  Set `HF_HOME` to a directory on D: before the first download.
- **Precision:** bf16 autocast, fp32 master weights and optimizer states.
- Long runs: **checkpoint at least hourly, resume automatically**, launch from
  the queue script so the card is never idle. A crash must cost < 1 h.
- The ~$500 cloud budget is a **reserve for reruns only**, not part of the plan.

## 6. Repository layout and conventions

```
rank-and-file/
├── CLAUDE.md               this file
├── README.md
├── proposal.md             scientific source of truth
├── pyproject.toml          package `rankfile`, deps
├── configs/
│   ├── model/              m30.yaml (smoke), m124.yaml
│   ├── train/              adamw.yaml, muon.yaml, sweep_adamw.yaml, sweep_muon.yaml, p3_matched.yaml
│   └── finetune/           full.yaml, lora_r4.yaml, lora_r16.yaml, lora_r64.yaml, tasks/*.yaml
├── src/rankfile/
│   ├── model.py            Qwen3-style transformer (single file, no framework)
│   ├── tokenizer.py        train/load 32k BPE
│   ├── data.py             shard prep, fixed-order loader, intra-doc masking
│   ├── optim/muon.py       Muon with Newton–Schulz, weight decay, update scaling (Liu et al. 2025)
│   ├── schedule.py         warmup-stable-decay
│   ├── train.py            pretraining loop, checkpoint/resume
│   ├── lora.py             LoRA layers, merge, rank utilities
│   ├── finetune.py         full-FT and LoRA fine-tuning loop
│   ├── tasks.py            code CPT and supervised-bundle data/metrics
│   ├── spectra.py          effective rank, stable rank, top-r energy, ΔW analysis
│   └── heads.py            STRETCH: induction/prev-token/sink scores
├── scripts/
│   ├── prepare_data.py     download + tokenize FineWeb-Edu into shards
│   ├── train_tokenizer.py
│   ├── queue.py            sequential run queue with resume
│   ├── analyze.py          runs spectra over a set of checkpoints → CSV
│   └── plot.py             every paper figure is generated here, never by hand
├── tests/                  pytest; see §7
├── runs/                   gitignored; one dir per run
├── data/                   gitignored; shards, tokenizer, task data
├── docs/course/            course outline and rubric PDFs
└── paper/                  LaTeX/Markdown draft, figures/ (generated)
```

Conventions:

- **Configs are YAML, runs are config + overrides.** Every run directory
  contains the fully resolved config it ran with (`config.resolved.yaml`)
  and the git commit hash. A run that cannot be reproduced from its
  directory is a bug.
- **Run names:** `{arm}_{opt}_{size}_s{seed}`, e.g. `p2_muon_m124_s0`,
  `p3_adamw_m124_s0`. Fine-tuning runs: `{parent}__{method}_{task}`, e.g.
  `p2_muon_m124_s0__lora16_code`. Sweeps: `sweep_{opt}_lr{value}`.
- **Seeds:** seed 0 and seed 1 only. Seed controls init and data order.
  Data shard order is identical across optimizers for the same seed.
- **Logging:** plain JSONL in the run dir (`metrics.jsonl`) plus stdout.
  No mandatory external tracker. If W&B is added it is optional and off by default.
- **Checkpoints:** `ckpt_{step:07d}.pt` with model, optimizer, scheduler,
  RNG states, data position. `latest` symlink or pointer file. Keep the
  final checkpoint and ~10 evenly spaced ones for pretraining runs.
- **No notebooks in `src/`.** Exploratory notebooks go in `notebooks/`
  (gitignored outputs) and anything that becomes a result moves into a script.
- **Code style:** Python 3.11, type hints on public functions, `ruff` for
  lint/format, small pure functions for anything that is a measurement
  (spectra, scores) so it can be unit-tested against known matrices.
- **Dependencies stay minimal:** torch, numpy, tokenizers, datasets, pyyaml,
  safetensors, tqdm, matplotlib. No Lightning, no HF Trainer, no PEFT
  library: LoRA and Muon are implemented here because understanding them is
  the point of the course, and the rubric grades conceptual understanding.

## 6a. Implementation plans

`docs/superpowers/plans/README.md` indexes six self-contained plans, one per
subsystem, in execution order. Each plan restates the interfaces it consumes
and produces so an agent can execute it in a fresh context without the
others. Execute with the superpowers subagent-driven-development or
executing-plans skill, one task at a time, tests before code, commit per task.

## 7. Testing

Run `pytest` before any commit that touches `src/`. Required tests:

- `test_model.py`: forward shapes; GQA head grouping; QK-norm applied; tied
  embedding weight identity; intra-doc mask blocks cross-document attention.
- `test_muon.py`: Newton–Schulz output is approximately orthogonal
  (‖UᵀU − I‖ small); update scaling matches the Liu et al. formula; 1-D
  params are excluded and routed to AdamW.
- `test_lora.py`: merged LoRA weights equal base + BA·scale; rank-r update
  has rank ≤ r; zero-init B gives identity at step 0.
- `test_spectra.py`: effective rank of identity = n; of rank-1 matrix ≈ 1;
  top-r energy of a rank-r matrix = 1.
- `test_resume.py`: train N steps, checkpoint, resume, train M more; loss
  trajectory equals training N+M steps straight through (bitwise on CPU).
- `test_smoke.py`: `m30` config trains 20 steps and fine-tunes 5 steps end to end.

## 8. Workflow rules

1. **Smoke before scale.** `m30` end-to-end pass on the exact code, then launch.
2. **Never start a multi-hour run without asking** the user, and state the
   expected duration. Never kill a running job without asking.
3. **Never modify a config that a completed run used.** Copy it to a new
   file if a variant is needed. Completed runs are immutable.
4. **Do not delete anything under `runs/` or `data/`** without explicit permission.
5. **Report numbers, not adjectives.** "Loss 3.21 vs 3.18" not "slightly better."
   Always give the seed and the arm.
6. **When a result contradicts a hypothesis, say so plainly** and record it in
   the decision log below. Negative results are reportable results.
7. Every figure in `paper/figures/` is produced by `scripts/plot.py` from
   CSVs in `runs/`. No manual figure editing.

## 9. Git

- **The student is the sole author of every commit.** Do not add
  `Co-Authored-By`, `Claude-Session`, or any AI attribution trailer to
  commit messages, PR descriptions, or files. This is a graded academic
  project and authorship must be the student's alone.
- Commit messages: imperative mood, ≤ 72-char subject, body explains *why*.
  Prefix by area when useful: `model:`, `muon:`, `lora:`, `data:`, `spectra:`, `paper:`, `docs:`.
- Never commit `runs/`, `data/`, checkpoints, or anything > 5 MB other than the course PDFs already in `docs/course/`.
- `main` only; feature branches are optional for a solo project. Commit
  small and often; each completed experiment gets a commit that records the
  run name and headline number in the body.

## 10. Timeline

| Week | Dates | Milestone | GPU busy? |
|---|---|---|---|
| 1 | Sep 29 – Oct 5 | WSL2 + torch verified; codebase; tokenizer; m30 smoke; LR sweep overnight | sweep |
| 2 | Oct 6 – Oct 12 | P1 s0, P2 s0 back to back; build FT harness + task data meanwhile | yes, ~1–1.5 days |
| 3 | Oct 13 – Oct 19 | P3; spectra of P1/P2 (H1); FT LR sweeps after P3 | yes, ~0.5–1 day |
| 4 | Oct 20 – Oct 26 | FT grid on P1/P2/P3; LoRA-gap curves (H3); ΔW spectra (H2) | short jobs |
| 5 | Oct 27 – Nov 2 | Second seeds of P1/P2; subspace-overlap analysis | yes |
| 6 | Nov 3 – Nov 9 | FT grid on second seeds; reconcile; stretch decision; final figures | short jobs |
| 7 | Nov 10 – Nov 16 | Paper draft | — |
| 8 | Nov 17 – Nov 22 | Slides | — |
| — | Nov 23 – 29 | Break / buffer | — |
| 9 | Dec 1 | **Presentation** | — |
| 10 | Dec 8 | **Paper** | — |

A complete single-seed result must exist by end of week 4. If throughput is
at the slow end, second seeds are the first thing cut.

## 11. Decision log

Append, never rewrite. Format: date, decision, reason.

- **2026-09-01** — Project chosen over alternatives (logit-control × head
  analysis; diffusion data-constrained mechanism; Muon for masked diffusion)
  after literature checks. Diffusion mechanism was scooped (arXiv 2510.04071);
  Muon full-FT transfer is covered by arXiv 2606.09658, so this project's
  novelty rests on the **LoRA-rank** question and the **spectral predictor**.
- **2026-09-01** — Qwen3 recipe chosen over GPT-2 and over modded-nanogpt/
  nanochat because it was developed under AdamW and is therefore
  optimizer-neutral.
- **2026-09-01** — Scope slimmed: attention-head analysis and 350M twins moved
  to stretch so the core is finishable in weeks 1–4 on one GPU.
- **2026-09-01** — All compute local on the 5070 Ti by the student's choice;
  cloud budget is a reserve. Second seeds scheduled after the first FT grid.
- **2026-09-02** — **One shared peak LR per arm.** In the Muon arm the
  embedding and norm gains (AdamW) use the same peak LR as Muon's hidden
  matrices, as in Moonlight (the 0.2·√max(m,n) scaling matches Muon's update
  RMS to AdamW's so one LR grid serves both). `build_optimizers` exposes
  `adamw_lr_scale` (default 1.0) so the choice is explicit; it stays 1.0
  unless the sweep shows instability, and any change is logged here.
- **2026-09-02** — **Gradient handling is arm-symmetric by setting, not by
  effect.** Accumulated micro-batch losses are averaged (÷32) and global-norm
  clipping at 1.0 applies in both arms. Muon's update is exactly invariant to
  gradient scale, AdamW's is not (ε), so clipping perturbs the arms
  differently; that is inherent to comparing the optimizers and is not
  tuned away. `grad_norm` is logged every step.
- **2026-09-02** — **Newton–Schulz limitation to state in the paper.** Five
  quintic NS steps amplify singular values by at most ≈3.44⁵ ≈ 484×, so
  gradient directions more than ~500× below the top singular value are left
  essentially unorthogonalized (measured: 30% of σ(X) < 0.5 at condition
  number 2.4e3). This is reference Muon behaviour and is kept; H1's
  "flatter spectra" mechanism is weaker than idealized orthogonalization on
  ill-conditioned late-training gradients.
- **2026-09-01** — Environment audit: **native Windows, no WSL2.** torch
  2.14.0+cu130 + triton-windows 3.8 gives working `torch.compile` (80k tok/s,
  7.6 GiB at micro-batch 8) and FlexAttention. Flash SDPA has no Windows kernel;
  cuDNN SDPA is used instead. Global Python's torch is broken (no sympy); the
  project uses `.venv` via uv. Micro-batch capped at 8 because WDDM pages to
  system RAM instead of OOM-ing.

## 12. Key references (full list in proposal.md)

- Liu et al. 2025, *Muon is Scalable for LLM Training*, arXiv:2502.16982 — the Muon variant we implement.
- Hu et al. 2021, *LoRA*. Biderman et al. 2024, *LoRA Learns Less and Forgets Less*.
- arXiv:2606.09658 (June 2026), *Muon Learns More Robust and Transferable Features than Adam* — the closest prior work; we extend it to low-rank adapters.
- arXiv:2603.00742 (March 2026), simplicity bias in Muon — motivates H1.
- Qwen Team 2025, *Qwen3 Technical Report* — architecture.
