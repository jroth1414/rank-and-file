# Does Muon Pretraining Change What LoRA Can Learn? Optimizer, Weight Spectra, and Low-Rank Fine-Tunability

**Course:** EN.705.743 ChatGPT from Scratch: Building and Training Large Language Models, Fall 2026
**Student:** [Your name]
**Proposal date:** September 29, 2026
**Deliverables:** Presentation December 1, 2026; research paper December 8, 2026

---

## 1. Problem statement

Muon has become the leading alternative to AdamW for pretraining language models. Instead of scaling each weight entry by its own gradient history, it orthogonalizes the update to every two-dimensional weight matrix, which equalizes step sizes across singular directions. Reported results are a roughly two-fold compute advantage at scale, better tail-token memorization, and, in a June 2026 study, features that transfer better under linear probes and full fine-tuning.

Almost nobody fine-tunes with full fine-tuning. The dominant method is LoRA, which restricts the weight update to a low-rank product. Whether Muon's transfer advantage survives that restriction is untested, and there is a concrete reason to doubt it. Muon's orthogonalized updates spread learning across many singular directions rather than concentrating it in a few, which should give Muon-trained weights a flatter spectrum. If the weight change needed to adapt such a model is also spread across more directions, then a rank-16 adapter captures less of it, and LoRA recovers a smaller fraction of full fine-tuning on a Muon-trained model than on an AdamW-trained one.

This project trains matched pairs of small language models with each optimizer and asks:

1. Does LoRA at a given rank recover the same fraction of the full fine-tuning improvement on each twin, and does forgetting differ?
2. Do the twins differ in the singular-value spectra of their pretrained weights and of their full fine-tuning updates?
3. Does the effective rank of the full fine-tuning update predict the LoRA gap?

The third question links the other two. If it holds, the paper delivers not only a result about Muon and LoRA but a measurable quantity that predicts when a low-rank adapter will fall short, for any pretrained model.

## 2. Hypotheses

- **H1.** Muon twins have flatter singular-value spectra, meaning higher effective rank, in attention and MLP projection matrices than AdamW twins at matched validation loss.
- **H2.** The full fine-tuning update on a Muon twin has higher effective rank than on an AdamW twin for the same task.
- **H3.** LoRA at low rank recovers a smaller fraction of the full fine-tuning improvement on Muon twins than on AdamW twins, and the gap closes as rank increases. Muon's transfer advantage under full fine-tuning narrows or reverses under low-rank adapters.

## 3. Related work

**Muon.** Jordan et al. (2024) introduced Muon in the modded-nanogpt speedrun. Liu et al. (2025) added weight decay and update scaling to make it work at scale and reported about two-fold compute efficiency over AdamW. A September 2025 paper shows Muon outperforms Adam on tail-end associative memory. A March 2026 paper argues from simplicity bias that Muon learns many directions at once instead of following the staged curriculum other optimizers follow. A June 2026 paper shows features learned by Muon are more robust to corruption and transfer better under linear probes and full fine-tuning. A May 2026 paper studies the reverse direction, fine-tuning Adam-pretrained models with Muon, and documents an optimizer-mismatch problem. A May 2026 spectral analysis describes two-phase dynamics in the singular-value distribution during pretraining.

**LoRA and full fine-tuning.** Hu et al. (2021) introduced LoRA. Biderman et al. (2024) showed LoRA learns less and forgets less than full fine-tuning and that full fine-tuning updates have rank 10 to 100 times higher than typical LoRA configurations. Shuttleworth et al. (2024) showed LoRA introduces intruder dimensions absent from full fine-tuning. Thinking Machines (2025) reported that very low rank suffices for reinforcement learning but not for large supervised datasets, and framed the question as one of information capacity.

**Gap.** Every study of LoRA's capacity has used models pretrained with Adam-family optimizers. No work tests whether the pretraining optimizer changes how much a low-rank adapter can recover, or whether the spectral structure it leaves in the weights predicts that. This project addresses both with matched twins that a single student can train.

## 4. Methodology

### 4.1 Base architecture: Qwen3 recipe

The architecture follows the Qwen3 small-model recipe, which was developed under AdamW and is therefore neutral with respect to the optimizer comparison. The modded-nanogpt and nanochat recipes were rejected because their architectural choices, such as value embeddings and U-net skips, were selected by how much they helped under Muon.

The Qwen3 components used:

- Pre-norm with RMSNorm; SwiGLU feed-forward; no biases in any linear layer.
- Rotary position embeddings.
- Grouped-query attention.
- QK-norm: RMSNorm applied to queries and keys per head before the dot product.
- Tied input and output embeddings, as in Qwen3-0.6B.
- Byte-level byte-pair encoding tokenizer.

Two deviations are forced by the model size. Qwen3's 151k vocabulary would put over 100M parameters in the embedding matrix alone, so a 32k tokenizer of the same type is trained on the pretraining data. Qwen3 uses 128-dimensional heads, which would leave only six heads at 768 width, so head dimension is 64 to keep a realistic head count.

| Size | Layers | Hidden | Q / KV heads | Head dim | FFN hidden | Non-embedding params | Total |
|---|---|---|---|---|---|---|---|
| 124M-class | 12 | 768 | 12 / 4 | 64 | 2,048 | ~75M | ~100M |

Sequence length is 2,048 with intra-document attention masking. Data is FineWeb-Edu.

### 4.2 Pretraining arms

Both optimizers use a warmup-stable-decay schedule with 2% warmup and a 20% decay phase, batch size of about 0.5M tokens, bf16 autocast, and weight decay 0.1. AdamW uses betas 0.9 and 0.95. Muon is applied to all two-dimensional hidden weight matrices with momentum 0.95, Nesterov, and the weight-decay and update-scaling corrections from Liu et al. (2025); embeddings and norm gains use AdamW, which is the standard configuration. Peak learning rate for each optimizer is chosen by a sweep of three values at 0.2B tokens, so neither baseline is undertrained.

| Arm | Optimizer | Size | Tokens | Seeds | Purpose |
|---|---|---|---|---|---|
| P1 | AdamW | 124M | 2.5B | 2 | Baseline twin |
| P2 | Muon | 124M | 2.5B | 2 | Muon twin, matched tokens |
| P3 | AdamW | 124M | ~3.75B | 1 | AdamW trained until it matches P2's validation loss |

Arm P3 separates two explanations for any difference. If Muon twins differ from P1 but not from P3, the difference comes from being a better-trained model. If they differ from both, it comes from the optimizer itself. P3 is the primary comparison for every claim.

The first seed of P1 and P2 plus P3 form the core. Second seeds are trained after the first fine-tuning grid is complete, so a full single-seed result exists before any GPU time is spent on replication. A 350M scale check is not planned on local hardware; it remains possible with the cloud budget if the 124M results are clean and time allows.

### 4.3 Fine-tuning grid

Each 124M twin is fine-tuned with full fine-tuning and with LoRA at ranks 4, 16, and 64, applied to all attention and MLP projections. The learning rate for each method is tuned with a three-point sweep on one twin and reused for the other, so tuning cannot favor either optimizer. Two task types cover the range from broad domain shift to narrow supervised learning:

- **Code:** continued pretraining on 100M tokens of Python from a permissively licensed code corpus. Measured by held-out code loss.
- **Supervised bundle:** SST-2, BoolQ, and AG News formatted as language-model prompts with label words, three epochs each. Measured by accuracy.

Forgetting is measured as the increase in FineWeb-Edu validation loss after fine-tuning. The primary metric is the **LoRA gap**: the fraction of the full fine-tuning improvement that LoRA at rank r recovers, plotted against r for each twin and task. H3 predicts the Muon curve sits below the AdamW curve at rank 4 and converges by rank 64.

| Grid dimension | Values |
|---|---|
| Twins | P1, P2, P3 (P1 and P2 at two seeds) |
| Methods | Full fine-tuning; LoRA r = 4, 16, 64 |
| Tasks | Code; supervised bundle |
| Runs | 5 checkpoints x 4 methods x 2 tasks = 40, each under one hour locally |

### 4.4 Spectral analysis

All measurements run on final checkpoints and cost minutes each.

- **Pretrained weight spectra.** Singular values of every projection matrix, summarized by effective rank (entropy of the normalized spectrum) and stable rank. Reported per layer and per matrix type. This tests H1.
- **Full fine-tuning update spectra.** The same summaries applied to the difference between fine-tuned and pretrained weights for each full fine-tuning run. This tests H2 and is the bridge quantity.
- **Subspace overlap.** The fraction of the full fine-tuning update's energy captured by the top-r singular directions, for r = 4, 16, 64, compared directly to the measured LoRA gap at each rank. If effective rank predicts the LoRA gap, these two curves should track each other across twins and tasks. This tests H3's mechanism.

### 4.5 Stretch: attention head analysis

If the core grid is complete on schedule, the same checkpoints support a circuit-level comparison at no additional training cost: induction and previous-token scores per head, attention-sink mass, and head redundancy within a layer. This would test the March 2026 prediction that Muon learns circuits in a less staged way. It is scoped as a stretch goal so that the core result does not depend on it.

## 5. Compute plan

All training and analysis runs on a single RTX 5070 Ti with 16 GB of memory. The card's bf16 matmul rate with fp32 accumulation is about 88 TFLOPS, comparable to an RTX 4090. At the 35 to 50% utilization that small models reach with torch.compile and flash attention, the 124M model trains at roughly 40k to 80k tokens per second, or 3.5 to 7 hours per billion tokens. Memory is not a constraint at this size. Every estimate below is given as a range between those two rates.

| Workload | Tokens | GPU hours |
|---|---|---|
| Learning-rate sweep, 3 values x 2 optimizers at 0.2B | 1.2B | 4 to 8 |
| P1 and P2, first seed | 5B | 18 to 35 |
| P3, matched-loss AdamW | 3.75B | 13 to 26 |
| Fine-tuning learning-rate sweeps | — | ~4 |
| Fine-tuning grid on P1, P2, P3, 24 runs | — | 9 to 15 |
| Spectral analysis | — | < 1 |
| **Core total** | | **48 to 89** |
| P1 and P2, second seed | 5B | 18 to 35 |
| Fine-tuning grid on second seeds, 16 runs | — | 6 to 10 |
| **Total with replication** | | **72 to 134** |

The core is two to four days of continuous GPU time, and the full plan with second seeds is three to six days. Pretraining runs are sequential on one card, so the calendar constraint is that P1, P2, and P3 occupy the machine for most of weeks 2 and 3. Fine-tuning runs are each under an hour and can be scheduled around other use.

Three practices make this workable on a personal machine. Every run checkpoints at least hourly and resumes from the last checkpoint, so a crash or reboot costs under an hour. Runs are launched from a queue script so the card is never idle overnight. Development uses a 30M configuration that exercises the full pipeline in minutes, so no bug is discovered inside a 30-hour run.

The $500 cloud budget is held in reserve. If a pretraining run needs to be redone, or if time allows the 350M scale check, an H100 trains the 124M model in about two hours per run at roughly $6, and the 350M pair for about $25.

Local tooling is PyTorch 2.7 or newer with CUDA 12.8 builds for the Blackwell sm_120 target, run under WSL2 so that torch.compile works.

## 6. Timeline

| Week | Dates | Milestone |
|---|---|---|
| 1 | Sep 29 – Oct 5 | Training codebase with both optimizers; 32k tokenizer; 30M end-to-end test; learning-rate sweep runs overnight |
| 2 | Oct 6 – Oct 12 | P1 and P2 first seeds train back to back; fine-tuning harness with LoRA and task data built while the card is busy |
| 3 | Oct 13 – Oct 19 | P3 trains; spectral analysis of P1 and P2, first test of H1; fine-tuning learning-rate sweeps once P3 finishes |
| 4 | Oct 20 – Oct 26 | Fine-tuning grid on P1, P2, P3; LoRA gap curves; first test of H3; update spectra, test of H2 |
| 5 | Oct 27 – Nov 2 | Second seeds of P1 and P2 train; subspace-overlap analysis on first-seed results |
| 6 | Nov 3 – Nov 9 | Fine-tuning grid on second seeds; reconcile seeds; decision on head-analysis stretch; final figures |
| 7 | Nov 10 – Nov 16 | Paper draft: related work, methods, results |
| 8 | Nov 17 – Nov 22 | Presentation slides; figure polish |
| — | Nov 23 – Nov 29 | Thanksgiving break; buffer |
| 9 | Dec 1 | Presentation |
| 10 | Dec 8 | Final paper |

A complete single-seed result exists at the end of week 4. Weeks 5 and 6 add replication, and if the slow end of the throughput range holds, the second seeds can be trimmed to one optimizer or dropped without losing the core result.

## 7. Risks and mitigations

- **Muon reaches lower loss, so differences reflect training quality rather than the optimizer.** Arm P3 trains AdamW to matched validation loss and is the primary comparison.
- **Differences fall within seed noise.** Second seeds are scheduled for weeks 5 and 6. Differences smaller than seed variance are reported as null results, which is still informative given the June 2026 transfer claim.
- **LoRA results depend on learning-rate tuning.** Each method gets its own sweep, and the same rate is used across twins.
- **H3 could fail while H1 and H2 hold.** That outcome would mean spectral rank does not predict LoRA capacity, which is itself a publishable negative result and is reported as such.
- **Throughput lands at the slow end of the range.** The core still fits in weeks 1 to 4. Second seeds are the first thing cut, and the cloud reserve can absorb any single rerun for about $6.
- **A long local run is interrupted.** Hourly checkpoints with automatic resume, and a run queue so the card is never idle.
- **Blackwell tooling on Windows.** WSL2 and current PyTorch builds; verified in week 1 before any run depends on it.

## 8. Expected contributions

1. The first test of whether the pretraining optimizer changes low-rank fine-tunability, measured as the LoRA gap across ranks and task types, with a matched-loss control.
2. A spectral comparison of Muon-trained and AdamW-trained language models: pretrained weight spectra and full fine-tuning update spectra.
3. A predictor, the effective rank of the full fine-tuning update, that links the two and indicates when a low-rank adapter will fall short.
4. Released code and checkpoints so that other optimizers can be evaluated the same way.

## 9. Course concepts exercised

Module 4 (transformers) covers the Qwen3 architecture and the reasons for each component. Module 6 (training) is the core: two optimizers, schedules, and the matched-loss design. Module 8 (fine-tuning) and Module 9 (parameter-efficient tuning) cover the full fine-tuning and LoRA grid and the capacity argument. Module 3 (multi-head attention) underlies the stretch head analysis.

## 10. Target venue

ICLR 2027 workshops, with deadlines expected in early February 2027, in particular workshops on optimization for deep learning or efficient fine-tuning. ACL 2027 workshops are a second option.

## References

- Biderman et al. (2024). LoRA Learns Less and Forgets Less.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Jordan et al. (2024). Muon: An Optimizer for Hidden Layers in Neural Networks.
- Liu et al. (2025). Muon is Scalable for LLM Training. arXiv:2502.16982.
- Qwen Team (2025). Qwen3 Technical Report.
- Shuttleworth et al. (2024). LoRA vs Full Fine-tuning: An Illusion of Equivalence.
- Thinking Machines (2025). LoRA Without Regret.
- Muon Outperforms Adam in Tail-End Associative Memory Learning (2025). arXiv:2509.26030.
- To Use or Not to Use Muon: How Simplicity Bias in Optimizers Matters (2026). arXiv:2603.00742.
- Can Muon Fine-tune Adam-Pretrained Models? (2026). arXiv:2605.10468.
- The Stability of Singular Distribution: A Spectral Perspective on the Two-Phase Dynamics of Language Model Pre-training (2026). arXiv:2605.26489.
- Muon Learns More Robust and Transferable Features than Adam (2026). arXiv:2606.09658.
