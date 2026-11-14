# Architectural decision register

Each entry: identifier, question, options, choice, reason, date. Changes to a
decision are new entries referencing the old one.

## D-C1 — Python and PyTorch versions
- **Question:** Which Python/PyTorch do the research modules target?
- **Options:** (a) pin to repo CI matrix (Python 3.6–3.8), (b) current Python.
- **Choice:** (b) Python 3.12.3, PyTorch 2.15.0 (CPU build on the dev machine;
  CUDA builds of the same version supported by the same code). Device is a
  configuration value; no `.cuda()` calls in new code.
- **Reason:** The existing code cannot run with modern PyTorch on 3.6–3.8; CI
  matrix is not modified (AGENTS.md R-6).
- **Date:** 2026-11-12.

## U-H1 — Hardware (carried forward)
- **Status:** Dev machine is CPU-only (no `nvidia-smi`). User reports a GPU on
  another machine. Code is device-agnostic (cpu/cuda auto by config). Phase 4
  Small-tier sizing and Phase 6 runs are blocked until the user confirms the
  GPU model and VRAM and runs there.
- **Date:** 2026-11-12.

## U-E1 — Dataset access (carried forward)
- **Status:** `nampdn-ai/tiny-webtext` is gated; no `HF_TOKEN` in the
  environment. User approved building the data pipeline and tests against a
  local fixture now; real schema inspection (U-D1) and any real-data run wait
  for an `HF_TOKEN` from the user.
- **Date:** 2026-11-12.

## D-T1 — Tokenizer
- **Question:** GPT-2 BPE or repository vocabulary for the MLA data path?
- **Options:** (a) GPT-2 byte-level BPE (tiktoken/tokenizers), (b) repository
  WordPiece-style `Vocab`/`Tokenizer`.
- **Choice:** (b) Repository vocabulary — decided by the user, overriding the
  PRD's recommended default.
- **Reason:** User decision. Consequence: the vocabulary file itself is not
  shipped by the repository; the data path takes a `vocab_path` from the
  configuration and tests use a small fixture vocabulary. Building or sourcing
  the real vocabulary file for Tiny WebText is an open item (blocker for real
  data runs), recorded in the log.
- **Date:** 2026-11-12.

## D-SZ1 — Model size
- **Question:** Repository default (d=1024, heads=16) or GPT-2 small
  (d=768, heads=12, L=12, rate=4)?
- **Choice:** GPT-2 small dimensions for the Small tier (user decision).
- **Reason:** Matches PRD §9.4; avoids a 1.5× heavier study.
- **Date:** 2026-11-12.

## D-P1 — Learned positional embedding in primary MLA arm
- **Question:** Keep or drop the learned absolute positional table in the
  primary MLA arm?
- **Choice:** Keep (PRD recommended default). An explicit flag disables it
  for the H4 ablation.
- **Reason:** Changes exactly one thing (attention block) relative to the
  baseline in the primary comparison.
- **Date:** 2026-11-12.

## D-S1 — Attention implementation: explicit vs SDPA
- **Question:** Does `F.scaled_dot_product_attention` accept query/key
  dimension 96 (d_h + d_R = 64 + 32) with value dimension 64, and does its
  boolean mask match the reference after conversion?
- **Evidence (measured 2026-11-12, CPU, torch 2.15.0):** With
  `attn_mask` = NOT(mask) (SDPA True = keep; repository convention
  True = masked), and `scale = 1/sqrt(d_h + d_R)`: max abs diff vs the
  explicit matmul reference with additive -1e4 masking = **4.8e-7**.
  Also agrees with a hard -inf mask reference to the same tolerance.
- **Choice:** The module ships the explicit matmul reference path
  (NFR-01). SDPA is numerically compatible and may be wired in later as a
  checked optimisation with a single documented mask-conversion point; it is
  not wired in now to keep the reference path canonical.
- **Date:** 2026-11-12.

## D-I1 — Initialisation for MLA projections
- **Choice:** Match the baseline: embeddings N(0, 0.02); other linears keep
  PyTorch defaults. DeepSeek-V2's std 0.006 is not adopted.
- **Date:** 2026-11-12.

## Baseline reference capture (T-B1)
- Captured `tests/fixtures/baseline_reference.npz` + `.json` from the
  unmodified tree at commit `7b159669f6225231ee75f5ea110c1be69f4ce8ef`:
  9,472 parameters (layers=2, words=80, dims=16, heads=2, seq_len=100),
  logits SHA-256 `673441ee9b1f86fdf8047dceef7414301dcb7cb049a017dc39781c60e5882467`.
  Regenerated twice with identical results on the same machine.
- **Note (pre-existing, not a defect we fix):** the baseline's
  `MultiHeadAttention` mishandles 1-D inputs during incremental decoding with
  a 2-D past (head/batch axes collapse). Fixture uses shape (1, 1) queries.
  Documented as carried baseline behaviour; the MLA arm is not required to
  reproduce it.
- **Date:** 2026-11-12.

## D-TOL1 — Test tolerances (recorded, not weakened silently)
- **TR-2 accumulation equivalence:** atol 1e-5 on fp32 CPU. Measured max
  diff 7.2e-6 between one batch of 4 and two accumulated batches of 2;
  caused by fp32 summation order, not a defect.
- **T-R1 exact resume:** atol 1e-7 on fp32 CPU; observed behaviour is
  bit-equal in practice at test sizes.
- **T-K1 cache equivalence:** atol 1e-4; observed max diff ~1e-6 at test
  sizes (4.8e-7 for the SDPA check, D-S1).
- **Config fingerprint excludes `steps`:** resuming a k-step checkpoint to
  a larger N with otherwise identical configuration is the T-R1 scenario;
  every other configuration field is verified loudly. Recorded 2026-11-12.
