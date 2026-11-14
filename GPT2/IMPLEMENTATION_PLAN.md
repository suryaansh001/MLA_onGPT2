# Implementation Plan

This plan sequences the work described in `PRD.md` and governed by `AGENTS.md`. Each phase has an objective, tasks, affected files, dependencies, expected outputs, validation criteria and completion criteria. A phase is complete only when its completion criteria are satisfied and recorded in `experiments/LOG.md`. The Lead Orchestrator approves each phase (`AGENTS.md` §9.4).

**Phase order:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. Phases 1 and 3 may overlap as described in §9.2 of `AGENTS.md`. Phase 6 cannot start before Phases 4 and 5 are validated.

**Blockers that gate phases:**

| Blocker | Gates | Owner |
|---|---|---|
| Hardware and VRAM confirmed (U-H1) | Phase 4 (Small tier sizing) | User |
| Dataset access confirmed (U-E1) | Phase 3 | User |
| Tokenizer decision D-T1 | Phase 3 | User |
| Model size decision D-SZ1 | Phase 2 | User |
| Python and PyTorch versions (D-C1) | Phase 0 | User |

---

## Phase 0: Repository audit and setup

**Objective.** Establish a verified baseline: a working environment, the existing test suite's status, a reference capture of the unmodified model, and the research branch.

**Tasks.**

1. Create a branch `research/mla` from the audited commit. Confirm the working tree is clean.
2. Create a Python environment on a current interpreter. Record the Python version and the PyTorch version in `requirements-research.txt` with exact pins. Confirm CPU PyTorch is sufficient for smoke tests. Record the CUDA build and driver version separately if a GPU is present.
3. Run `python -m compileall src tests` on the unmodified repository. Record the result.
4. Run the existing test suite on the unmodified repository. Record the pass, fail and skip counts and any failure reason. Do not fix failures in the baseline without recording them as pre-existing.
5. Capture the baseline reference fixture (`AGENTS.md` §5.4): fixed seed, fixed input, outputs and parameter count of the unmodified `Transformer` with the default small configuration. Save it under `tests/fixtures/` with the commit hash.
6. Verify dataset access to `nampdn-ai/tiny-webtext` using `HF_TOKEN` from the environment. Stream one record and record its schema in `docs/dataset.md`. Do not download more than one record.
7. Check whether the hosted CI has passed recently and record its status.
8. Create `experiments/LOG.md`, `docs/decisions.md`, and `docs/architecture.md`, with the initial entries.

**Files affected.** `requirements-research.txt` (new), `tests/fixtures/` (new), `docs/` (new), `experiments/LOG.md` (new). No source files change.

**Dependencies.** User confirmation of Python and PyTorch versions (D-C1), hardware (U-H1), and dataset access (U-E1).

**Expected outputs.** Pinned requirements file. Test-suite status report. Baseline fixture. Dataset schema record. Initial log and decision register.

**Validation criteria.**

- `compileall` passes on the unmodified tree.
- The existing test suite has been run and its results are recorded, whether passing or failing.
- The baseline fixture reproduces when regenerated twice with the same seed on the same machine.
- One streamed record was read and its schema is recorded.

**Completion criteria.** All validation criteria are met. The unresolved items D-C1, U-H1 and U-E1 are either answered or explicitly carried forward as blockers. Phase 0 is logged and approved.

---

## Phase 1: Standalone MLA implementation

**Objective.** Implement MLA as a standalone module, independent of the GPT-2 model, and verify it numerically against the equations.

**Tasks.**

1. Write `src/gpt2/modeling/rotary.py`: construct RoPE cos and sin tables for dimension $d_R$ with base 10000 (configurable). Apply rotation to a tensor of shape `(..., seq, d_R)` using explicit position indices, so the same function serves prefill and incremental decoding.
2. Write `src/gpt2/modeling/latent_attention.py` with class `MultiHeadLatentAttention`:
   - Configuration: `dims`, `heads`, `head_dims` ($d_h$), `kv_latent_dims` ($d_c$), `query_latent_dims` ($d'_c$), `rope_head_dims` ($d_R$), `use_query_latent` (bool), `use_latent_norm` (bool), `dropout`.
   - Projections: $W^{DKV}$, $W^{UK}$, $W^{UV}$, $W^{DQ}$ (if enabled), $W^{UQ}$ (if enabled), $W^{QR}$, $W^{KR}$, $W^O$. Biases on all linear layers, matching the baseline.
   - RMSNorm on $c^{KV}$ and $c^Q$ when enabled, implemented directly (not via `torch.nn.RMSNorm`, which requires a newer PyTorch).
   - Dimension assertions at construction: $d = n_h d_h$ is not required by the formulation, but $d$ must be divisible by $n_h$ for the default rule; $d_R$ must be even.
   - `forward(h, past, mask)` returns `(output, present)` where `present = (c_kv_cache, k_rope_cache)`, matching the repository's `past` convention.
   - Reference path: explicit matmul attention with the repository's mask convention (True = masked), additive `-1e4` (matching `BaseAttention`), and dropout on probabilities.
3. Write unit tests in `tests/modeling/test_latent_attention.py`:
   - Output and cache shapes for prefill and incremental steps.
   - Per-equation tests: the shapes of $c^{KV}$, $k^C$, $v^C$, $k^R$ (shared across heads), $q^R$ (per head), concatenation, scale $\sqrt{d_h + d_R}$.
   - RoPE: rotating by position $p$ then by $q$ equals rotating by $p + q$ for inner products (relative-position property). This is a numerical test on random inputs.
   - Dimension errors: wrong input width, odd $d_R$, mismatched mask shape each raise a clear error.
   - Gradient flow: every projection receives a non-zero gradient (T-4).
4. Verify the scale and value-path choices by a test that replaces $v$ with a one-hot probe and checks the output.
5. Check whether `F.scaled_dot_product_attention` accepts query dimension 96 with value dimension 64 on the target device, and whether it matches the reference with the converted mask. Record the result in `docs/decisions.md` as D-S1. Do not wire SDPA into the module until T-S1 passes.

**Files affected.** `src/gpt2/modeling/rotary.py` (new), `src/gpt2/modeling/latent_attention.py` (new), `tests/modeling/test_latent_attention.py` (new), `docs/decisions.md`.

**Dependencies.** Phase 0 complete. Architecture Analyst's compatibility notes for the cache convention (`AGENTS.md` A-7).

**Expected outputs.** Standalone module passing its tests. SDPA compatibility finding recorded.

**Validation criteria.**

- All new unit tests pass. The count is recorded.
- The relative-position RoPE property holds to floating-point tolerance on random inputs.
- Gradient flow test passes for every projection, with the query latent enabled and disabled.
- Dimension errors fail with messages naming the tensor.
- Compilation passes.

**Completion criteria.** Validation criteria met. The SDPA question is answered with evidence, and the decision D-S1 is recorded. No change to any existing file. Phase 1 logged and approved.

---

## Phase 2: GPT-2 integration

**Objective.** Make the attention type selectable in the existing transformer, with the baseline path unchanged, and verify causal behaviour for both arms.

**Tasks.**

1. Add the `attention` argument (`'mha'` default, `'mla'`) to `TransformerLayer` and `Transformer` in `src/gpt2/modeling/transformer.py`. The `'mha'` branch constructs the existing `AttentionLayer` exactly as before. The `'mla'` branch constructs `MultiHeadLatentAttention`.
2. Add a configuration flag controlling the learned positional embedding (default on). When off, `Transformer` does not add the positional table. Test that the parameter count changes by exactly the table size.
3. Adapt the cache plumbing: `TransformerLayer` and `Transformer` pass the per-layer `past` through unchanged. Verify that the position offset is read correctly for MLA caches (A-7).
4. Ensure the mask passed to MLA follows the same convention as `AttentionLayer`. Both use the mask produced by `PadMasking` and `FutureMasking` with `bidirectional=False`.
5. Run the baseline-unchanged test (T-B1) against the fixture from Phase 0. Any difference is a defect in this phase.
6. Write causal invariance tests (T-C1) and padding invariance tests (T-C2) for both arms, in `tests/modeling/test_causal_invariance.py`.
7. Write the integration test (T-3): the full model with `attention='mla'`, forward and backward on a small CPU configuration, with gradient flow (T-4).
8. Confirm the parameter counts for the proposed configuration against PRD §8.5 (T-P1). Use the GPT-2 BPE vocabulary size (50,257) for the count.

**Files affected.** `src/gpt2/modeling/transformer.py` (additive change, Lead review required), `src/gpt2/modeling/__init__.py` (export), `tests/modeling/test_causal_invariance.py` (new), `tests/modeling/test_transformer_mla.py` (new).

**Dependencies.** Phase 1 complete. Decision D-SZ1 (model size) and D-P1 (positional embedding in primary arm).

**Expected outputs.** Selectable attention. Baseline unchanged. Causal and padding invariance verified for both arms. Parameter counts matching the analysis.

**Validation criteria.**

- T-B1 passes: baseline outputs and parameter count are identical to the fixture.
- T-C1 and T-C2 pass for both `mha` and `mla`.
- T-3 and T-4 pass for `mla`.
- T-P1 parameter counts match PRD §8.5 to the element: 115,904,640 for MLA and 123,849,984 for baseline at the proposed configuration with `seq_len` 256 and vocabulary 50,257.
- All existing tests still pass, with the count compared to Phase 0.
- Compilation passes.

**Completion criteria.** All validation criteria met. The Lead has reviewed the diff of `transformer.py`. Phase 2 logged and approved.

---

## Phase 3: Dataset pipeline

**Objective.** Build the streaming, bounded, cached, packed and reproducible data pipeline over Tiny WebText using the GPT-2 tokenizer.

**Tasks.**

1. Confirm decision D-T1 (tokenizer) and select the tokenizer library (U-D3). Pin it in `requirements-research.txt`.
2. Confirm the schema recorded in Phase 0 (U-D1). Identify the text field. If the schema is unexpected, stop and report.
3. Write `src/gpt2/data/token_corpus.py` (or the chosen research data module):
   - Streaming reader using `datasets.load_dataset(..., streaming=True)` with the token from `HF_TOKEN`. Configurable record limit and token limit, either of which stops the stream.
   - Batched GPT-2 BPE encoding, with the document separator `<|endoftext|>` (id 50256) appended to each document.
   - Deterministic split: assign each document to validation or training by a hash of its index with a configured seed and fraction. Assignment is computed before any shuffling.
   - Packing into fixed-length sequences of `seq_len + 1` tokens, concatenating documents with separators (D-D1 option A). The final partial sequence is dropped and the drop is counted.
   - Sharded caching: write `uint16` shards as `.npy` files, with a manifest JSON recording dataset identifier and revision, record count, token count, tokenizer identifier and vocabulary size, the configuration, the seed, the split fraction, and a hash of each shard.
   - Cache lookup verifies the manifest before use.
   - Cursor: a `(shard_index, offset)` pair that can be serialised into a checkpoint (PRD §12).
4. Write data tests (`tests/data/test_token_corpus.py`):
   - Tokeniser round-trip on sample text with the separator.
   - Two runs with the same configuration produce identical shard hashes (T-D1).
   - Training and validation document sets are disjoint on a sample of at least 1,000 documents (D-5).
   - Dropped-sample counts match a hand-computed value on a fixture.
   - Cursor serialisation and restoration yield the same next batch.
5. Measure the token count of the Development subset and the Small subset. Record the values. Use them to set the Small-tier step count in the configuration (PRD §9.5).

**Files affected.** `src/gpt2/data/token_corpus.py` or equivalent (new), `tests/data/test_token_corpus.py` (new), `configs/data_*.yaml` (new), `docs/dataset.md`, `requirements-research.txt`. `src/gpt2/data/corpus.py` is not changed.

**Dependencies.** Phase 0 (dataset access and schema). Decision D-T1 (User). Phase 2 is not required, so Phase 3 may overlap with Phase 1 or 2.

**Expected outputs.** Data module with tests. Cached Development shards and manifest, stored outside the repository. Token counts for each tier.

**Validation criteria.**

- All data tests pass, with counts recorded.
- Shard hashes are reproducible across two runs.
- Streaming with a bounded record limit does not download beyond the limit. Verified by counting records read.
- Token counts for each tier are recorded and consistent with the manifest.
- Compilation passes.

**Completion criteria.** Validation criteria met. Token counts recorded. The licence review (U-D2) is either complete or carried forward as a blocker before shard sharing. Phase 3 logged and approved.

---

## Phase 4: Training and checkpoint recovery

**Objective.** Implement the research training loop with device handling, mixed precision, clipping, accumulation, warm-up, logging, full-split evaluation and exact resume, and run both arms through the Smoke and Development tiers.

**Tasks.**

1. Write the research configuration schema (`src/gpt2/research/config.py`): typed fields for model, data, optimiser, schedule, precision, logging, checkpointing and seeds. Serialise the configuration into every checkpoint and compute a fingerprint.
2. Write `src/gpt2/research/train_research.py`:
   - Device from configuration. No `.cuda()` calls in new code.
   - `torch.autocast` with bfloat16 where supported, otherwise float16 with a gradient scaler.
   - Optimiser parameter groups per TR-9.
   - Warm-up for 5 % of steps, then decay to 10 % of peak (PRD §9.6, D-L1).
   - Gradient accumulation with equivalence test (TR-2).
   - Global-norm clipping at 1.0, logging the pre-clip norm.
   - JSON-lines logging at configured intervals.
   - Full-split validation at each evaluation interval.
   - Checkpoint every `save_steps` and at completion, atomic, with RNG state and data cursor (PRD §12).
   - Resume from checkpoint with fingerprint verification.
   - Non-finite loss handling: write a checkpoint, then stop.
3. Write tests (`tests/research/test_train_research.py`):
   - Accumulation equivalence (TR-2).
   - Exact resume (T-R1): train $N$ steps uninterrupted, and train $k$ steps, checkpoint, resume to $N$. Parameter hashes match within tolerance on CPU.
   - Fingerprint mismatch fails loudly.
   - Optimiser group sizes are logged and are as expected.
   - Checkpoint write is atomic under simulated interruption.
4. Write configurations for the Smoke and Development tiers for both arms, and an ablation configuration with positional embedding disabled (H4).
5. Run the end-to-end smoke test (T-E2E) on CPU for both arms.
6. Run the Development tier for both arms on the available GPU. Record hardware, precision and wall time.
7. Run a short learning-rate check on the baseline only to confirm the chosen peak learning rate does not diverge at Development scale. Record the result.

**Files affected.** `src/gpt2/research/` (new), `configs/` (new), `tests/research/` (new), `experiments/runs/` (outputs, not committed).

**Dependencies.** Phase 2 (model interface), Phase 3 (data interface).

**Expected outputs.** Training module with tests. Development-tier runs for both arms with logs. Smoke-test report.

**Validation criteria.**

- T-R1 (exact resume) passes on CPU.
- T-E2E passes on CPU for both arms.
- Accumulation equivalence passes.
- Development runs complete for both arms without non-finite losses.
- Logs contain every field required by TR-6 for every logged step.
- Checkpoint files load in the existing `evaluate` and `generate` commands for the baseline (backward compatibility, FR-15). Record the result; if not compatible, document why.

**Completion criteria.** Validation criteria met. Development-tier results for both arms are recorded with their configuration fingerprints. Phase 4 logged and approved. Critical failures (T-R1, T-E2E) must pass before approval.

---

## Phase 5: Autoregressive inference

**Objective.** Implement generation with the KV cache for both arms and verify cache equivalence.

**Tasks.**

1. Write `src/gpt2/research/generate.py`:
   - Prompt encoding with the GPT-2 tokenizer.
   - Prefill and incremental decoding with the cache.
   - Top-p sampling with a seeded generator.
   - Stopping at `<|endoftext|>` or the maximum length (PRD §13).
   - Prompt truncation from the left with a warning when the prompt exceeds the context.
   - Separate timing of time to first token and time per output token.
2. Write cache-equivalence tests (T-K1): for at least five prompts and both arms, incremental logits match full-pass logits at every position within tolerance.
3. Write a test that a multi-step generation with MLA produces the same tokens as a full-recomputation reference under greedy decoding for a fixed short prompt.
4. Verify that the stopping rule and truncation rule are the same for both arms.
5. Record the cache-size analysis: at 1,024 tokens, batch 1, fp16, measured cache bytes match the analytic values in PRD §8.5 for both arms.

**Files affected.** `src/gpt2/research/generate.py` (new), `tests/research/test_generate.py` (new). `src/gpt2/generate_sentences.py` is not changed.

**Dependencies.** Phase 2 (model interface). Phase 4 checkpoints are useful for realistic tests but are not strictly required; random-initialised models suffice for equivalence.

**Expected outputs.** Inference module, equivalence tests, cache-size verification.

**Validation criteria.**

- T-K1 passes for both arms on at least five prompts.
- Greedy-decoding equivalence holds.
- Measured cache bytes match the analytic values.
- Compilation passes.

**Completion criteria.** Validation criteria met. Phase 5 logged and approved. A failing T-K1 is critical and blocks Phase 6.

---

## Phase 6: Baseline and MLA evaluation

**Objective.** Run the Small tier with three seeds per arm, the ablations, and the benchmarks. Record results with full provenance.

**Tasks.**

1. Confirm the Small-tier configuration: model size (D-SZ1), token budget from Phase 3, learning rate from the baseline sweep.
2. Run the baseline learning-rate sweep on the baseline arm only. Record each run. Select the peak learning rate by the validation loss at the final step. Record the grid and the selection.
3. Run the Small tier for both arms, three seeds each, with identical data order per seed. Record run identifiers and configuration fingerprints.
4. Run the ablations on the Small tier, three seeds where feasible: no learned positional embedding (H4); no query latent; no latent RMSNorm; latent rank $d_c \in \{2, 4, 6\} d_h$ (RQ5).
5. Run the benchmark tool (Phase 7 deliverable, may be run here if the tool is ready): forward and backward latency, peak memory, KV-cache bytes at context lengths {256, 512, 1024} if the model supports them, training throughput, generation latency.
6. Compute, for each comparison: mean and standard deviation across seeds, the difference in validation perplexity, and a 95 % confidence interval. Evaluate H2 against the pre-registered margin of ±2 % relative.
7. If the parameter-matched arm D-M1 is run, record it separately as a secondary comparison.
8. Write results to `experiments/results/` in structured form. Write the analysis only after the numbers are final.

**Files affected.** `experiments/runs/`, `experiments/results/`, `experiments/analysis.md`, `configs/small_*.yaml`, `docs/decisions.md`.

**Dependencies.** Phases 4 and 5 validated. Phase 6 cannot start before them. Hardware confirmed (U-H1).

**Expected outputs.** Raw metric files for every run. Results tables. Confidence intervals. Benchmark results with environment records. Analysis with H1–H5 outcomes.

**Validation criteria.**

- Every run has a configuration fingerprint and a complete log.
- Three seeds per arm are complete for the primary comparison.
- Results files are generated by script from raw logs, not typed by hand (DOC-4).
- Confidence intervals are computed from the stated method.
- H1 is verified by counting. H2 is evaluated against the pre-registered margin. H3 is reported without reinterpretation. H4 and H5 are reported with their measured values.
- Limitations from PRD §10.2 are included in the analysis.

**Completion criteria.** All primary runs and the primary comparison are complete and reported. Ablations are reported, including any that were not run and why. Phase 6 logged and approved.

---

## Phase 7: Testing, benchmarking and documentation

**Objective.** Finalise the test suite, the benchmark tool, the documentation, and the final report, and confirm all acceptance criteria.

**Tasks.**

1. Complete the benchmark tool (`src/gpt2/research/benchmark.py`): warm-up, repeats with median and spread, environment capture, deterministic flags recorded, peak-memory reset, separated forward, backward and generation timings (`AGENTS.md` §6).
2. Add the benchmark tests: environment record is complete; repeat count is honoured; peak memory reset is effective.
3. Run the full test suite on the final tree. Record counts.
4. Run the end-to-end smoke test once more on the final tree, for both arms, on CPU.
5. Run the baseline-unchanged test (T-B1) on the final tree.
6. Check the acceptance criteria in `PRD.md` §22 one by one, with evidence for each.
7. Update `REPOSITORY_AUDIT.md`: correct any entries that changed, add the new files, and record the final status of each unknown in §12.
8. Update `docs/architecture.md` with the final design and the decision register.
9. Write the final report: what was built, what was measured, hypothesis outcomes with intervals, limitations, unresolved decisions, negative findings, and reproduction instructions.
10. Confirm that no credentials, large artefacts or dataset shards are committed. Check `.gitignore` covers `experiments/runs/`, `*.npy`, and token files.
11. Tag the final commit.

**Files affected.** `src/gpt2/research/benchmark.py` (new), `tests/research/test_benchmark.py` (new), `REPOSITORY_AUDIT.md`, `docs/`, `experiments/REPORT.md` (new), `.gitignore` (additive).

**Dependencies.** Phase 6 complete.

**Expected outputs.** Final test report. Benchmark tool. Updated documentation. Final report. Tagged commit.

**Validation criteria.**

- Full test suite passes, with counts recorded. Any skipped test has a stated reason.
- Compilation passes.
- Baseline-unchanged test passes on the final tree.
- Every acceptance criterion in PRD §22 has evidence attached.
- No secrets or large artefacts in the commit.

**Completion criteria.** All acceptance criteria in PRD §22 are satisfied, or the unmet ones are listed in the report with their cause. Lead Orchestrator approves the final report. Project complete under `AGENTS.md` §10.

---

## Summary of phase outputs

| Phase | Principal output | Critical test |
|---|---|---|
| 0 | Environment, baseline fixture, schema record | Baseline reproduces |
| 1 | Standalone MLA module | Relative-position RoPE property; gradient flow |
| 2 | Selectable attention in transformer | T-B1 baseline unchanged; T-C1 causal invariance |
| 3 | Streaming, cached, packed data pipeline | Shard hash reproducibility; split disjointness |
| 4 | Research training with exact resume | T-R1 exact resume; T-E2E smoke |
| 5 | Cached inference | T-K1 cache equivalence |
| 6 | Small-tier results with three seeds | Reproducible results from logs |
| 7 | Benchmark tool, final report | All PRD §22 acceptance criteria |
