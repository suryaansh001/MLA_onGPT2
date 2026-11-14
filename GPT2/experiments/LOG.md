# Implementation log

Entries are appended, never rewritten (DOC-1).

## 2026-11-12 — Phase 0 — Repository audit and setup

- **Task:** Phase 0 of IMPLEMENTATION_PLAN.md.
- **Files read:** PRD.md, IMPLEMENTATION_PLAN.md, AGENTS.md,
  REPOSITORY_AUDIT.md (section map), src/gpt2/modeling/{transformer,attention,
  embedding,masking}.py, src/gpt2/data/{vocabulary,tokenization}.py,
  tests/modeling/test_transformer.py, requirements.txt.
- **Environment:** Python 3.12.3; created `venv-research/`; installed torch
  2.15.0+cpu, numpy 2.5.4, regex 2026.11.9, pytest, tqdm. Pins recorded in
  `requirements-research.txt`. No GPU on this machine (no nvidia-smi);
  device-agnostic code required (user confirmed CPU-only here, GPU elsewhere).
- **Branch:** `research/mla` created from 7b15966; working tree initially had
  only the four untracked planning documents.
- **Commands run:**
  - `python -m compileall src tests` → PASS.
  - `pytest tests -q` → **35 passed, 0 failed, 0 skipped** (1.44 s).
  - Baseline fixture captured twice → identical outputs (T-B1 precondition).
- **Files changed (new only):** `requirements-research.txt`,
  `scripts/capture_baseline_fixture.py`, `tests/fixtures/baseline_reference.npz`,
  `tests/fixtures/baseline_reference.json`, `docs/decisions.md`,
  `docs/architecture.md`, `experiments/LOG.md`.
- **Fixture:** 9,472 params; logits SHA-256
  673441ee…82467; commit hash recorded. Reproduces on regeneration.
- **Decisions:** D-C1 (Python 3.12 + torch 2.15, CPU), U-H1 carried forward
  (CPU-only here), U-E1 carried forward (no HF_TOKEN; fixtures approved),
  D-T1 → repository vocabulary (user), D-SZ1 → GPT-2 small dims (user),
  D-P1 → keep learned positional embedding (default), D-I1 → std 0.02 for
  embeddings, PyTorch defaults elsewhere.
- **Blockers:**
  - U-E1/U-D1: gated dataset needs `HF_TOKEN` from the user; real schema
    inspection deferred. Phase 3 code/tests proceed with a local fixture.
  - D-T1 consequence: the repository ships no vocabulary file; the user chose
    the repository vocabulary, so a vocabulary file for the real corpus is
    needed from the user (or an approved build) before any real-data run.
    Fixture vocabulary used for tests meanwhile.
- **CI status:** not checked remotely (no request to query GitHub actions for
  the upstream repo from here); recorded as unverified.
- **Validation criteria:** compileall PASS; suite 35/35; fixture reproduces;
  dataset schema NOT recorded (blocked) — carried forward explicitly.

## 2026-11-12 — Phase 1 — Standalone MLA implementation

- **Task:** Phase 1 of IMPLEMENTATION_PLAN.md.
- **Files read:** src/gpt2/modeling/attention.py (mask convention, additive
  -1e4, Past type), masking.py, transformer.py, tests/modeling/*.py.
- **Files changed (new only):** src/gpt2/modeling/rotary.py,
  src/gpt2/modeling/latent_attention.py,
  tests/modeling/test_latent_attention.py. No existing file changed.
- **Commands run:** pytest tests/modeling/test_latent_attention.py -q →
  **16 passed, 0 failed**; compileall → PASS. Full existing suite unchanged
  (35 passed).
- **Test evidence:** relative-position RoPE property holds (1e-5); rotation
  invertibility R(p)ᵀR(-p)=I holds; gradient flow to every parameter holds
  for all four (use_query_latent × use_latent_norm) combinations;
  dimension errors raise ValueError naming the tensor (h, mask, past[0],
  past[1], rope_head_dims, kv_latent_dims, heads); scale test pins
  1/sqrt(d_h + d_R) and rejects 1/sqrt(d_h); one-hot value probe confirms
  values come from the content pathway only; cache keeps sequence axis at
  -2 in prefill and incremental steps (A-7).
- **D-S1 (measured):** SDPA accepts q/k dim 96 with v dim 64 on CPU;
  boolean attn_mask (True = keep) matches the additive -1e4 reference to
  4.8e-7. SDPA NOT wired into the module; explicit matmul is the reference
  path. Recorded in docs/decisions.md.
- **Fixes during the phase (test bugs, not module bugs):** one-hot probe
  mask sized without a past; masked-position test had the mask convention
  inverted; RoPE pair-norm test paired wrong coordinates; module fix:
  RMSNorm on the query latent is constructed only when both the latent norm
  and the query latent are enabled (otherwise it would be a dangling
  parameter, violating T-4).
- **Blockers:** none.
- **Validation criteria:** all met. Phase 1 complete.

## 2026-11-12 — Phase 2 — GPT-2 integration (selectable attention)

- **Task:** Phase 2 of IMPLEMENTATION_PLAN.md.
- **Files read:** transformer.py (full), latent_attention.py, attention.py,
  utils/fusing.py (LayerNorm = nn.LayerNorm, weight+bias), modeling/__init__.py.
- **Files changed (additive):** src/gpt2/modeling/transformer.py —
  TransformerLayer/Transformer gain `attention` ('mha' default / 'mla'),
  MLA dimension options, and the `positional_embedding` flag (default on);
  forward() skips the positional table when the flag is off. The 'mha'
  branch constructs AttentionLayer exactly as before; no existing call site
  changed. src/gpt2/modeling/__init__.py — exports
  MultiHeadLatentAttention, RMSNorm, apply_rotary_embedding, rotary_tables.
- **Files added:** tests/modeling/test_causal_invariance.py (T-C1, T-C2),
  tests/modeling/test_transformer_mla.py (T-B1, T-3/T-4, T-P1, FR-04).
- **Test evidence:** pytest tests -q → **62 passed, 0 failed, 0 skipped**
  (35 pre-existing + 27 new). compileall → PASS.
  - T-B1: baseline logits, incremental logits and parameter count are
    bit-identical to the Phase 0 fixture (equal(), not allclose), and the
    logits hash matches.
  - T-C1 causal invariance for both arms at positions 0/3/9, with and
    without padding; T-C2 padding invariance for both arms.
  - T-3/T-4: full MLA model trains on CPU; every parameter gets a
    non-zero gradient; the positional-embedding flag is on in this config.
  - T-P1: baseline 123,849,984 and MLA 115,904,640 parameters; per-layer
    attention 2,362,368 (mha) vs 1,700,256 (mla) — all equal to PRD 8.5 to
    the element (vocab 50,257, seq_len 256, d=768, n_h=12, L=12).
  - Positional-embedding flag removes exactly seq_len x dims parameters.
  - MLA cache: first element keeps the sequence axis at -2 through
    incremental steps (offset logic verified, A-7).
- **Test fixes (test bugs):** eval-mode forward returns (logits, past);
  training-mode returns logits; fixture test consumed RNG in the wrong
  order for the second input draw.
- **Review note (Lead):** the transformer.py diff touches only the
  constructor signatures (new optional kwargs with defaults preserving the
  previous behaviour) and two forward branches; verified bit-identical
  baseline by T-B1.
- **Blockers:** none.
- **Validation criteria:** all met. Phase 2 complete.

## 2026-11-12 — Phase 3 — Dataset pipeline (repository vocabulary, fixtures)

- **Task:** Phase 3 of IMPLEMENTATION_PLAN.md (fixture-based; real dataset
  access blocked).
- **Decisions applied:** D-T1 = repository vocabulary (user decision,
  overriding the PRD recommendation); U-D3 consequence: no external
  tokenizer dependency is needed — the repository `Vocab`/`Tokenizer` is
  used, so only `datasets` (deferred until access) could ever be added.
- **Files read:** src/gpt2/data/{vocabulary,tokenization}.py, corpus.py
  (not modified, R-4).
- **Files added:** src/gpt2/data/token_corpus.py,
  tests/data/test_token_corpus.py, docs/dataset.md.
- **Test evidence:** pytest tests -q → **73 passed, 0 failed, 0 skipped**
  (11 new data tests). compileall → PASS.
  - Tokenizer round-trip with separator: PASS.
  - Shard hashes byte-identical across two runs: PASS.
  - Split disjointness over 2,000 documents, deterministic: PASS.
  - Dropped counts hand-computed: record_limit (10 read / 20 dropped),
    token_limit (16 tokens, 1 sequence, >=1 dropped): PASS.
  - Corruption detection (one flipped byte fails loudly with hash
    mismatch): PASS. Config fingerprint mismatch fails: PASS.
  - Cursor serialise/restore yields the identical next batch: PASS.
  - Wrap-around and next-token shift (x[:,1:] == y[:,:-1]): PASS.
- **Bugs found and fixed during the phase (module bugs):**
  1. np.save appends '.npy' to the atomic-write temp name — fixed.
  2. next_batch loop counted slices, not rows — fixed (row counter).
  3. manifest.json was not written to the cache dir — fixed.
  4. The pending encode batch was discarded when a limit stopped the
     stream — fixed (flush respects the token limit).
  5. _dropped_records uninitialised — fixed.
- **Blockers (carried forward):**
  - U-E1: gated dataset needs HF_TOKEN; real schema inspection (U-D1)
    pending; the streaming adapter is a small addition on the existing
    document-stream interface.
  - D-T1 consequence: a real vocabulary file is required from the user
    before any real-data run (the repository ships none).
  - U-D2: licence review required before sharing derived shards.
- **Validation criteria:** all met for the fixture path; token counts for
  the Development/Small tiers cannot be measured without the real dataset
  (blocked, carried). Phase 3 (fixture scope) complete.

## 2026-11-12 — Phase 4 — Research training and checkpoint recovery

- **Task:** Phase 4 of IMPLEMENTATION_PLAN.md.
- **Files added:** src/gpt2/research/__init__.py, config.py, train_research.py;
  tests/research/test_train_research.py, test_smoke_e2e.py;
  configs/{smoke,development}_{mha,mla}.json, configs/ablation_mla_no_pos.json.
- **Features (PRD §§11-12):** device from config (auto/cpu/cuda, no .cuda()
  calls); autocast bf16/fp16+scaler with fp32 CPU default; AdamW param groups
  (decay on matrix params only, zero on biases/norms/embeddings); linear
  warm-up 5% then decay to 10% of peak (D-L1); gradient accumulation;
  global-norm clip 1.0 with pre-clip norm logged; JSONL logging (TR-6
  fields); full-split token-weighted validation with exp(ppl) (avoids the
  baseline's mean-of-per-batch-ppl defect B-2); atomic checkpoints with RNG
  state (python/numpy/torch/cuda), data cursor, metrics, scaler, config +
  fingerprint; resume with loud fingerprint mismatch; non-finite loss
  checkpoints then stops; seeds for python/numpy/torch(+cuda).
- **Test evidence:** pytest tests/research -q → 9 passed. Full suite →
  **82 passed, 0 failed, 0 skipped**. compileall → PASS.
  - TR-2 accumulation equivalence: one batch of 4 vs 2x2 accumulation match
    to 7.2e-6; tolerance 1e-5 documented (fp32 summation order, D-TOL1).
  - T-R1 exact resume: 8 uninterrupted vs 3+resume(->8) parameters match to
    1e-7 on CPU. Required two design fixes:
    (1) `stop_after` — the partial run must follow the SAME schedule/config
        and merely stop early (a `steps=3` config would change the LR shape
        and diverge before the checkpoint);
    (2) `steps` excluded from the config fingerprint — extending a run is a
        normal resume; all other fields still verified loudly.
  - Atomic checkpoint write verified under simulated interruption (no
    partial file at the final path).
  - T-E2E smoke passes for BOTH arms on CPU (train 3 -> resume to 6,
    eval, JSONL fields, checkpoint payload schema).
- **Config seam for real data:** read_documents() reads a plain text
  document file; the HF streaming adapter plugs in there once U-E1 access
  is confirmed. Development-tier configs target the GPU machine (U-H1).
- **Blockers (unchanged):** U-E1 (HF token), real vocab file (D-T1), GPU
  for Development/Small tiers.
- **Validation criteria:** all met for CPU scope. Phase 4 complete.

## 2026-11-12 — Phase 5 — Autoregressive inference

- **Task:** Phase 5 of IMPLEMENTATION_PLAN.md.
- **Files added:** src/gpt2/research/generate.py, tests/research/test_generate.py.
- **Features (PRD §13):** prompt encoding with the repository tokenizer;
  prefill + incremental decoding with the KV cache for both arms; top-p
  nucleus sampling with a seeded generator (top_p=None is greedy argmax);
  stopping at </s> or max_new_tokens; left-truncation of over-long prompts
  with a warning, same rule for both arms; time-to-first-token and
  time-per-output-token timing.
- **Test evidence:** pytest tests/research/test_generate.py -q → 9 passed.
  Full suite → **91 passed, 0 failed, 0 skipped**. compileall → PASS.
  - T-K1 cache equivalence: incremental logits match the full forward
    pass at every position for 5 prompts, both arms, tolerance 1e-4
    (actual max diff ~1e-6 at these sizes).
  - Greedy decoding with cache equals full-recomputation argmax decoding
    for both arms (8 tokens).
  - Stopping at eos (forced one-hot eos logits) and at max_new_tokens:
    verified for mla.
  - Truncation warns and keeps the left-truncated suffix for both arms.
  - Measured cache counts match H1/PRD §8.5 exactly: mha 1536
    elements/token/layer (2 x 12 x 64), mla 288 (256 + 32), ratio 0.1875;
    per-token bytes at 12 layers fp16: 36,864 B vs 6,912 B; totals at
    context 1024: 37,748,736 B vs 7,077,888 B.
- **Blockers:** none. Phase 5 complete; T-K1 (critical) passes.

## 2026-11-12 — Phase 6 — BLOCKED (documented, not started)

- **Phase 6 (Small-tier runs, three seeds, ablations, benchmarks) cannot
  start.** Per IMPLEMENTATION_PLAN.md it requires Phases 4+5 (done) plus:
  - **U-H1:** a CUDA GPU (16 GB+). This machine is CPU-only; Development/
    Small configs are written and ready (configs/development_{mha,mla}.json).
  - **U-E1:** HF_TOKEN for nampdn-ai/tiny-webtext; the document-stream seam
    is ready (read_documents in train_research.py); schema inspection
    (U-D1) still required before the real loader.
  - **D-T1 consequence:** a real vocabulary file for the corpus (the
    repository ships none and the user chose the repository vocabulary).
  - LR sweep on the baseline (Phase 6 task 2) needs the GPU too.
- No experimental numbers were fabricated (R-9). The benchmark TOOL was
  built and verified instead (Phase 7 deliverable, see below).

## 2026-11-12 — Phase 7 (partial) — Benchmark tool

- **Files added:** src/gpt2/research/benchmark.py,
  tests/research/test_benchmark.py.
- **Features (AGENTS.md §6):** environment capture (python/torch/cuda
  versions, GPU name, driver, git commit, config fingerprint); explicit
  deterministic flags recorded; warm-up excluded; median/min/max over
  repeats (n recorded); peak memory reset per measurement (CUDA);
  separated forward, forward+backward, TTFT and per-output-token timings;
  structured JSON results written atomically.
- **Test evidence:** 5 passed, 1 skipped (skip reason: peak-memory test
  requires CUDA, machine is CPU-only — recorded per §5.3).
- **Tool verification run (NOT experimental results):** tiny 2-layer model
  on CPU, 10 repeats: mha forward 0.99 ms / fwd+bwd 2.33 ms; mla forward
  1.64 ms / fwd+bwd 3.84 ms. MLA is slower on the training path at this
  scale, consistent with H5's expectation (extra up-projections); this is
  a tool sanity check, not a comparison result. Output:
  experiments/results/benchmark_tool_check.json (not committed).
- **.gitignore updated:** experiments/runs/, experiments/results/, *.npy,
  venv-research/.
- **Remaining Phase 7 work:** final full-suite run, REPOSITORY_AUDIT.md
  update, final report — after Phase 6 runs (or on request without them).
