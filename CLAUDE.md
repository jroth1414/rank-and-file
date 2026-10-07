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
  Update active hours are 10:00–02:00 (checked 2026-09-15), so an automatic restart can land
  between 02:00 and 10:00. Before a multi-day run, pause updates
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
│   ├── run_queue.py        sequential run queue with resume (not queue.py: it would shadow the stdlib)
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
- **2026-09-03** — **Analysis conventions.** Only arms P1/P2/P3 and canonically
  named fine-tune runs enter the paper's CSVs (sweeps are excluded by
  `scripts/analyze.py`). Tables never pool across arms; H1 is read as P2
  vs P3 at their (adjacent-row) validation losses. Subspace overlap is
  reported one-sided (chance baseline `r/rows`) and two-sided (chance
  baseline `r²/(rows·cols)`).
  Effective rank is Roy–Vetterli on unsquared singular values; top-r energy
  is the squared (Frobenius) fraction; both are also reported normalised by
  `min(rows, cols)` when shapes are pooled. The intermediate-checkpoint
  effective-rank-vs-loss trajectory (`--all-ckpts`) is the primary H1 figure.
- **2026-09-02** — **LoRA alpha = 2·rank, so the adapter scale is 2.0 at every
  rank.** A fixed alpha across r ∈ {4, 16, 64} would vary the effective
  update magnitude 16× and make H3's rank curve a learning-rate artifact.
  Recorded in `results.json` as `alpha`.
- **2026-09-02** — **Fine-tuning learning rate is swept per (method, task)
  on P1 s0 only** and reused on every twin: `configs/finetune/{full,lora}_{code,sup}.yaml`.
  Still never tuned per twin (CLAUDE.md §3 rule 5).
- **2026-09-02** — **Code fine-tuning corpus is `codeparrot/codeparrot-clean`**
  (field `content`, rows whose `license` is one of mit, apache-2.0,
  bsd-3-clause, bsd-2-clause, isc), built to 100.0M train and 5.0M val
  tokens with a manifest. The planned `codeparrot/github-code-clean` is a
  script-based dataset that `datasets` 5.x no longer loads, and
  `bigcode/the-stack-smol` is gated. The corpus is identical across arms,
  so the substitution cannot confound the comparison.
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
- **2026-09-28** — **Peak LR frozen at 4e-3 for both optimizers** (P1, P2, P3).
  Sweep: seed 0, m124, 200M tokens (381 steps), final val_loss:
  AdamW 1e-3 4.9286 | 2e-3 5.0774 | 4e-3 **4.8828** | 8e-3 5.2503;
  Muon 1e-3 3.8017 | 2e-3 3.6843 | 4e-3 **3.6567** | 8e-3 3.6931.
  4e-3 was the top edge of the first grid for both, so one point beyond it
  (8e-3) was run; it lost for both, so 4e-3 is interior. The AdamW response is
  non-monotonic (2e-3 is worse than both 1e-3 and 4e-3 by 0.15–0.19) and
  nearly flat early (val at step 48: 6.94 / 6.97 / 6.99 for 1e-3 / 2e-3 / 4e-3);
  the curves are smooth, grad norm 0.1–0.6, clipping never binds, and the
  optimizer code routes the scheduled LR correctly, so this is read as a
  short-horizon sweep effect, not a bug. Muon's warmup has grad-norm spikes
  (23.1 at step 10 for 1e-3), absorbed by clipping at 1.0. At 200M tokens Muon
  leads AdamW by 1.23 in val_loss (3.66 vs 4.88), so P3's token budget may
  far exceed the 3.75B prior; it is sized from P1's curve after P2 finishes.
  All runs: no non-finite values, peak allocated ≤ 8.23 GiB.
- **2026-09-30** — **AdamW-baseline diagnostics: frozen config kept** (LR 4e-3,
  2% warmup, β₂ 0.95). At matched stable-phase train loss Muon is 1.7× faster
  by step 50 and 3.0–3.3× by steps 200–300, against the 1.4× Wen et al. 2025
  (arXiv:2509.02046) report at 130M for a fully tuned AdamW, so the AdamW twin
  was checked for a handicap. Checkpoints show none in the code: AdamW moves
  the hidden matrices as far as Muon (weight RMS 0.043–0.054 vs 0.050–0.082),
  median √v̂ is 180× ε, RMS of m̂/√v̂ is 0.23 in both arms; what differs is
  ΔW effective rank (layer 3: AdamW 94–487, Muon 241–721). Two one-variable
  tests, seed 0, 200M tokens, LR 4e-3, applied to both arms per §3 rule 4:
  warmup 0.02 → 0.10 (8 → 38 steps): AdamW 4.8828 → 4.8557, Muon 3.6567 →
  3.6797; AdamW-group β₂ 0.95 → 0.98: AdamW 4.8828 → 4.9382, Muon 3.6567 →
  3.6657. Neither closes more than 0.05 of the 1.23 gap, so the gap is read
  as Muon's genuine advantage at this short horizon (381 steps, ~0.08×
  Chinchilla tokens), where Wen et al. also find the largest speedups.
  P1/P2 at 2.5B tokens measure whether it shrinks; P3's design is decided
  after P2 (options: size AdamW to P2's loss, cap at 5B, or loss-match with a
  short Muon run). Grad clip (1.0) and weight decay were not tested.
- **2026-09-30** — **Correction to the entry above: warmup does matter for
  AdamW, and the 200M-token sweep under-represents it.** P1 s0 (2.5B schedule,
  95-step warmup) and `sweep_adamw_lr4e-3` (8-step warmup) share data order,
  model, LR and everything else; at step 190 P1's train loss is 5.509 vs 5.855
  (38-step warmup: 5.942), and P1's val_loss at 150M tokens (4.883, still at
  peak LR) equals the sweep run's final annealed 200M value (4.8828). The
  response is non-monotonic over 8/38/95 steps and not yet explained. So the
  "genuine short-horizon gap" reading is withdrawn, and 4e-3 was chosen under
  a warmup the real runs do not use. Decision: P1/P2 continue unchanged; they
  share an identical schedule, so their gap is the fair measurement. If it
  remains far above ~1.4×, check AdamW LR on the real schedule (2e-3, 8e-3,
  truncated at ~500M tokens and compared step-matched against P1's stable
  phase) before sizing P3. Lesson: sweep warmup in absolute steps, not as a
  fraction of a 12× shorter run.
- **2026-10-01** — **P1 s0 and P2 s0 complete; P3 sized at 4.0B tokens.**
  Final val_loss (seed 0, m124, 2.5B tokens, identical schedule): P1 AdamW
  **3.0939**, P2 Muon **3.0373** (gap 0.057). The gap shrinks monotonically:
  1.164 at 100M, 0.257 at 400M, 0.116 at 1.0B, 0.068 at 2.0B. Stable-phase
  matched-loss speedup of Muon over AdamW is 1.75–2.1× (≈1.8× after step 500),
  vs 1.4× reported by Wen et al. at 130M against a fully tuned AdamW, so no
  AdamW LR re-check before P3: P3 reaches matched loss whatever its LR costs
  in tokens. P3 sizing: power-law fits to P1's stable-phase val_loss (fit
  start 0.3/0.5/0.8/1.0B) plus P1's measured anneal drop (0.171, from 3.2647
  at decay start to 3.0939) give 4.12/3.91/3.77/3.83B; the 1.8× speedup
  heuristic gives 4.5B. Chose **4.0B** (above the fit median to avoid
  undershooting; 4.5B would also exceed the 4.20B train tokens on disk).
  P3's 2% warmup is 153 steps vs 95 for P1/P2 (fixed-fraction schedule per
  proposal §4). Both runs: no non-finite values, peak 8.23 / 7.95 GiB, ~97k tok/s.
- **2026-10-02** — **P3 s0 complete: val_loss 3.0128 at 4.0B tokens**, 0.0245
  *below* P2 s0 (3.0373), so the matched-loss twin slightly overshoots rather
  than matches. The sizing fit was mildly pessimistic: stable-phase val_loss
  at P3's decay start (3.2B) was 3.1949 vs the predicted 3.2081, and P3's anneal
  drop was 0.182 vs P1's 0.171. Kept as is, not re-run: H1 is read on the
  `--all-ckpts` effective-rank-vs-loss trajectory, where loss matching is done
  across checkpoints, and the final-checkpoint comparison P2 vs P3 is reported
  with both losses. Muon's end-of-run token advantage is therefore < 4.0/2.5 =
  1.6×. No non-finite values, peak 8.23 GiB, 97k tok/s, 16 checkpoints.
- **2026-10-02** — **H1 supported on seed 0.** Mean normalized effective rank
  over all 84 block matrices, final checkpoints: P1 0.822 (val 3.094), P2
  **0.875** (3.037), P3 **0.837** (3.013); P2 − P3 = +0.038 although P3 has the
  lower loss. P2 > P3 in all seven module types (+0.013 mlp.down to +0.068
  attn.q). Stable rank differs more: mlp.gate 161.5 vs 32.4, mlp.up 187.0 vs
  104.0, attn.q 51.1 vs 16.9, attn.o 117.2 vs 70.3. On the checkpoint
  trajectory at matched val_loss ≈3.207: P2 0.880 vs P3 0.833. Caveat for the
  paper: the gap narrows with training. AdamW's effective rank rises
  monotonically (P3 0.717 at val 3.954 → 0.837), Muon's falls slightly
  (0.894 at 3.648 → 0.875), so longer training may shrink the H1 effect.
  Single seed; seed 1 is needed before calling it robust.
- **2026-10-03** — **Fine-tuning LRs frozen** (swept on P1 s0, reused on every
  twin): full code **1e-3**, full sup **1e-4**, LoRA code **1e-2**, LoRA sup
  **1e-3** (LoRA swept at r16; alpha = 2r keeps the scale fixed across ranks).
  Code val loss (from 2.0158), forgetting in parentheses: full 3e-5 1.2413
  (+0.055), 1e-4 1.1053 (+0.123), 3e-4 0.9929 (+0.251), 1e-3 0.8970 (+0.635);
  LoRA r16 3e-4 1.3040 (+0.104), 1e-3 1.1684 (+0.211), 3e-3 1.0768 (+0.431),
  1e-2 1.0577 (+0.826). Sup accuracy (from 0.6215): full 3e-5 0.8177, 1e-4
  0.8336, 3e-4 0.8302; LoRA r16 3e-4 0.8184, 1e-3 0.8245, 3e-3 0.7259
  (diverged, forgetting +20.3). Both code winners stayed at the top edge after
  one extension each (user chose to freeze rather than extend again): **full
  code is unbracketed**, LoRA code is flattening (−0.019 for 3.3×). Code CPT
  selected by target loss alone trades heavily into forgetting; the rule is
  the same for every twin, so H3 is unaffected, and forgetting is reported as
  its own outcome. Two engineering fixes made during the sweep: LoRA adapters
  were created on CPU after the parent moved to CUDA (every GPU LoRA forward
  failed; fixed in lora.py), and sup_loss built full [32,511,32768] logits,
  pushing full-FT sup to 14.5 GiB past the guard (now projects only label
  positions; 8.66 GiB). `ftsweep_full_sup_lr3e-5` holds two guard-stopped
  attempts and was re-run as `_r2`; `ftsweep_full_sup_lr1e-4` ran the fixed
  sup_loss but its git.txt records the prior commit e9a5e12 (the fix was
  uncommitted for ~1 min).
- **2026-10-03** — **Seed-0 fine-tuning grid complete (24/24). H2 is
  contradicted; H3 is weak on code, directionally supported on sup, and its
  "gap closes with rank" clause is contradicted; LoRA r64 diverged on code.**
  *H2:* mean normalized effective rank of full-FT ΔW is *lower* on the Muon
  twin than on loss-matched AdamW: code P2 0.757 vs P3 0.776 (P1 0.761); sup
  P2 0.663 vs P3 0.686 (P1 0.666). Stable rank too: code 34.2 vs 46.4, sup 8.2
  vs 13.5; top-r energy is *higher* on P2 (code top16 0.268 vs 0.228; sup
  0.514 vs 0.436). Muon's flatter pretrained spectra (H1) do not carry over to
  a higher-rank update. *H3* (LoRA recovered fraction, P2 vs P3): code r4
  0.744 vs 0.754, r16 0.841 vs 0.852 (P2 lower by ~0.01, single seed, likely
  within noise); sup r4 0.955 vs 0.983, r16 0.966 vs 1.031, r64 0.903 vs
  1.001 (P2 lower by 0.028–0.098, but the gap *grows* with rank). Since P2's
  ΔW is more concentrated yet LoRA recovers less on it, the top-r-energy
  predictor points the wrong way on seed 0. *Forgetting:* full-FT sup forgets
  far more on P2 (+1.256) than P3 (+0.251) or P1 (+0.280) at the same LR.
  *r64 divergence:* LoRA r64 code at the frozen LR 1e-2 (chosen at r16) got
  worse than the base model on all three parents (code val loss 2.670 /
  3.097 / 3.232 from ~1.95–2.02, forgetting +4.7 to +5.7); r64 sup kept
  accuracy but forgot +2.0 to +4.9. With α = 2r the scale is fixed but ‖BA‖
  grows with r, so r64 takes larger steps than r16 at one LR; the r64 code
  cells are not usable for H3 as run. sup accuracy is on 1,000 eval examples
  per task (SE ≈ 1.2 pt per task).
- **2026-10-05** — **LoRA LR is now swept per rank** (P1 s0; supersedes "LoRA
  swept at r16" above), after r64 code diverged at the r16 LR. Code val loss
  (forgetting): r4 3e-3 1.2035 (+0.285), **1e-2 1.1690** (+0.525), 3e-2 3.6912
  (diverged); r64 3e-4 1.1972 (+0.152), 1e-3 1.0627 (+0.309), **3e-3 0.9868**
  (+0.633), 1e-2 2.6702 (diverged). Sup accuracy (forgetting): r4 3e-4 0.8051
  (+0.057), **1e-3 0.8258** (+0.210), 3e-3 0.8260 (+0.991); r64 1e-4 0.8190
  (+0.104), **3e-4 0.8265** (+0.263), 1e-3 0.8184 (+2.080). Frozen: code
  {4: 1e-2, 16: 1e-2, 64: 3e-3}, sup {4: 1e-3, 16: 1e-3, 64: 3e-4}, in
  `configs/finetune/lora_{code,sup}_by_rank.yaml` (the original
  `lora_{code,sup}.yaml` are kept: the first grid used them). r4 sup is a
  deliberate tie-break: 3e-3 led 1e-3 by 0.0002 (noise, top edge) at 4.7×
  the forgetting, so 1e-3 is kept (user decision). The six superseded r64
  cells (P1/P2/P3 × code/sup) were moved, intact, to `runs/_superseded/`
  with the user's permission and re-run under canonical names from
  `configs/queue/ft_core_rank.txt`. The optimal LoRA LR falls with rank at
  fixed α = 2r, consistent with ‖BA‖ growing with r.
- **2026-10-05** — **H3 on seed 0 with per-rank LRs** (corrects the r64 cells
  of the 2026-10-03 entry). LoRA recovered fraction, P2 vs P3: code r4 0.744
  vs 0.754, r16 0.841 vs 0.852, r64 0.914 vs 0.918 (P2 lower by 0.010 /
  0.011 / 0.004: the gap narrows at r64, as H3 predicts, but is ~1 pt at
  most); sup r4 0.955 vs 0.983, r16 0.966 vs 1.031, r64 0.969 vs 1.005 (P2
  lower by 0.028 / 0.065 / 0.036: larger, but not monotone in rank). So H3's
  direction holds in all six rank × task cells on seed 0; its "closes with
  rank" clause holds on code only. Sup forgetting is 2.6–3.9× higher on P2
  than P3 for every method (full +1.256 vs +0.251; LoRA r4 +0.694 vs +0.180,
  r16 +1.085 vs +0.431, r64 +0.579 vs +0.225). H2 stands as contradicted
  (unchanged; full-FT runs were not re-run). Seed 1 is next.
- **2026-10-05** — **Seed 1 pretraining replicates seed 0.** Final val_loss:
  P1 s1 3.0936 (s0 3.0939), P2 s1 3.0350 (s0 3.0373); mean normalized
  effective rank: P1 0.822 (0.822), P2 0.874 (0.875). Seed-to-seed spread
  (≤0.0023 loss, ≤0.001 erank) is ~25–50× smaller than the P1–P2 gap
  (0.057, 0.053). Both runs clean, peak 8.23 / 7.95 GiB. By user decision,
  P3 s1 (same config as P3 s0, 4.0B tokens) and the full 24-run seed-1 grid
  follow (`configs/queue/seed1.txt`), so the matched-loss comparison is
  replicated rather than single-seed.
- **2026-10-07** — **Two-seed results. H1 supported; H2 contradicted; H3 not
  supported; Muon twins forget more on sup.** P3 s1 val_loss 3.0199 (s0
  3.0128). *H1* (mean erank_norm, P2 vs P3): s0 0.875 vs 0.837, s1 0.874 vs
  0.835. *H2* (full-FT ΔW erank_norm, P2 vs P3): code s0 0.757 vs 0.776, s1
  0.756 vs 0.773; sup s0 0.663 vs 0.686, s1 0.657 vs 0.673; stable rank code
  34.2/33.9 vs 46.4/45.2, sup 8.2/8.0 vs 13.5/13.0; Muon's update is lower
  rank and more concentrated on both seeds. *H3* (LoRA recovered fraction,
  P2 vs P3, r4/r16/r64): code s0 0.744/0.841/0.914 vs 0.754/0.852/0.918, s1
  0.757/0.850/0.919 vs 0.754/0.852/0.918; sup s0 0.955/0.966/0.969 vs
  0.983/1.031/1.005, s1 0.971/0.984/1.005 vs 0.951/0.994/0.941. Seed 1
  erases or reverses every seed-0 gap, so the seed-0 "direction holds in all
  six cells" reading (2026-10-05) was noise: no evidence on two seeds that
  LoRA recovers less on Muon twins. *Forgetting* (FineWeb-Edu val loss
  increase, P2 vs P3): sup full FT s0 +1.256 vs +0.251, s1 +1.522 vs +0.270;
  sup LoRA r16 s0 +1.085 vs +0.431, s1 +1.054 vs +0.391; code full FT
  +0.683/+0.678 vs +0.615/+0.619. The forgetting gap is the largest
  replicated fine-tuning effect in the study and was not hypothesized.

## 12. Key references (full list in proposal.md)

- Liu et al. 2025, *Muon is Scalable for LLM Training*, arXiv:2502.16982 — the Muon variant we implement.
- Hu et al. 2021, *LoRA*. Biderman et al. 2024, *LoRA Learns Less and Forgets Less*.
- arXiv:2606.09658 (June 2026), *Muon Learns More Robust and Transferable Features than Adam* — the closest prior work; we extend it to low-rank adapters.
- arXiv:2603.00742 (March 2026), simplicity bias in Muon — motivates H1.
- Qwen Team 2025, *Qwen3 Technical Report* — architecture.
