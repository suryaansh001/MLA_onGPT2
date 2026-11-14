# AGENTS.md: Execution Contract for the Coding Agent

This file is binding on every agent that works in this repository. It is the source of truth for how work is done. The other source documents are `PRD.md` (what to build and why), `REPOSITORY_AUDIT.md` (what the repository contains and where the risks are), and `IMPLEMENTATION_PLAN.md` (in what order). If two documents conflict, the order of precedence is: this file, then `PRD.md`, then `IMPLEMENTATION_PLAN.md`, then `REPOSITORY_AUDIT.md`. Where the audit disagrees with the code itself, the code wins and the audit is corrected.

---

## 1. Project context

**What the project is.** A controlled research study of Multi-Head Latent Attention (MLA), from DeepSeek-V2, inside a small GPT-2-style causal language model. The repository is `affjljoo3581/gpt2`, audited at commit `7b159669f6225231ee75f5ea110c1be69f4ce8ef`.

**What architecture is modified.** The multi-head self-attention block inside each transformer layer. The rest of the model is preserved.

**What the research objective is.** To measure, under matched conditions, the effect of replacing standard attention with MLA on quality (validation perplexity), parameters, compute, memory, KV-cache size and generation latency. The outcome is not assumed. The PRD defines hypotheses H1–H5 and the pre-registered equivalence margin for H2.

**Which components must be preserved.** Token embeddings. Learned positional embeddings, in the primary arm (PRD D-P1). Pre-LN transformer block structure. Residual connections. LayerNorm. Feed-forward MLP. The output head tied to the token embedding. Causal masking. Autoregressive generation. The causal language-modelling objective. The repository's existing `train`, `evaluate`, `generate` and `visualize` commands.

**Not in scope.** DeepSeekMoE, FP8, KV quantisation, FlashAttention kernels, YaRN, absorption-based inference (may be recorded as future work), pretrained-weight conversion, multi-GPU training.

---

## 2. Repository rules

**R-1. Inspect before modifying.** Before editing any file, read it in full. Before editing any function, read its callers. Record the file read in the implementation log.

**R-2. Preserve existing functionality.** The baseline path must keep producing identical outputs. This is tested, not assumed (see §5.4).

**R-3. Do not unnecessarily restructure.** Add new modules beside existing ones. Do not rename, move or re-export existing public symbols. Do not reorganise directories.

**R-4. Do not overwrite existing implementations without understanding them.** In particular: `TokenizedCorpus`, `Trainer`, `Transformer`, `AttentionLayer` and `Vocab` are depended on by the existing commands and tests. Change them only additively, behind a default that preserves current behaviour.

**R-5. Additive attention selection.** `Transformer` and `TransformerLayer` gain an `attention` argument with default `'mha'`. Passing `'mla'` selects the new block. No existing call site changes.

**R-6. Do not change CI or the Python version matrix without approval.** Record in the log that CI targets Python 3.6 to 3.8 and that research code will not run there. Raise it to the orchestrator; do not edit `.github/workflows/build.yml` on your own.

**R-7. Dependencies.** New dependencies go only in `requirements-research.txt`, pinned. Do not modify `requirements.txt` in ways that change what current users install. Do not install packages not needed for the current phase.

**R-8. Do not commit or publish credentials.** Hugging Face tokens are read from the `HF_TOKEN` environment variable. Never write a token to a file, a log, a configuration, or a shell history that will be committed.

**R-9. Do not fabricate.** Do not invent dataset schema, benchmark numbers, test results or paper claims. If something has not been measured or retrieved, record it as unknown.

**R-10. File locations.** Model code under `src/gpt2/modeling/` or `src/gpt2/research/`. Data code under `src/gpt2/data/` or the research data module. Tests mirror the source tree under `tests/`. Experiment outputs under `experiments/runs/<run_id>/`, never committed if large. Configuration under `configs/`.

---

## 3. Architecture rules

**A-1. Implement actual MLA.** The implementation must follow DeepSeek-V2 Eqs. 9–19 as stated in `PRD.md` §8.1. Specifically: shared latent $c^{KV}$ across heads; separate content and RoPE pathways; RoPE on the shared key $k^R$ and the per-head query $q^R$; concatenated query and key; scale $\sqrt{d_h + d_R}$; values from the content pathway only.

**A-2. Do not conflate MLA with other low-rank methods.** Do not describe a projection of the sequence axis, a generic bottleneck, or GQA/MQA as MLA. Code comments and docstrings must use the terms in the PRD.

**A-3. Preserve standard GPT-2 attention as a baseline.** The `mha` path is kept and tested. It is not replaced by MLA in any file.

**A-4. Validate all projection dimensions.** Each module asserts its expected input and output dimensions at construction and at the boundary of `forward`. Dimension errors must fail at the boundary with a message naming the tensor, not later inside a matrix multiplication. `d_R` must be even. `d` must be divisible by `n_h`.

**A-5. Preserve causal masking.** Causal masking is applied to the MLA arm exactly as to the baseline. The repository convention is True = masked. Any call into `F.scaled_dot_product_attention` converts the mask at a single documented point. A causal invariance test is mandatory (§5.2, T-C1).

**A-6. Handle positional embeddings explicitly.** The learned positional embedding is controlled by an explicit configuration flag. The RoPE pathway is controlled separately. The configuration names which positional mechanisms are active, and the run log records it. Do not combine them implicitly.

**A-7. Preserve the cache convention.** The first element of each per-layer cache tuple has the sequence axis at dimension `-2`, as the repository's `Transformer` requires for position offsets. Test it after multi-step generation (§5.2, T-K1).

**A-8. Numerical reference first.** Implement the explicit matmul attention as the reference. A fused kernel (SDPA) is accepted only after it matches the reference within a documented tolerance on the test inputs, including the mask.

**A-9. Shape and initialisation.** Initialise MLA projections as the baseline linear layers are initialised (normal with standard deviation 0.02 where the baseline does, otherwise PyTorch default). Record the choice in `docs/decisions.md` (D-I1).

**A-10. No placeholders.** Do not create stub implementations, `NotImplementedError` bodies left in shipped code, or `TODO` functions that return fake values. A missing piece is reported as a blocker, not stubbed.

---

## 4. Dataset rules

**D-1. Never download the full dataset by default.** Default data access is streaming with a bounded `take`. Full materialisation requires an explicit configuration flag and an explicit user instruction.

**D-2. Support streaming and bounded subsets.** Record limit and token limit are both configurable. Either can stop the stream.

**D-3. Cache tokenised data.** Tokenised shards are written to disk as `uint16` arrays with a manifest. A cache hit must be verified against the manifest hash before use.

**D-4. Use reproducible preprocessing.** The same configuration and dataset revision must produce byte-identical shards. The stream order is deterministic. Any shuffling uses a fixed seed and is recorded.

**D-5. Never mix validation data into training.** The validation split is determined by a deterministic hash of the document index (or the document's identifier if the schema provides one), before any shuffling. A test asserts that the sets are disjoint on a sample of at least 1,000 documents.

**D-6. Respect the access gate.** Before the first run, confirm access to the gated dataset. If access is missing, stop and report a blocker. Do not work around the gate.

**D-7. Inspect before writing the loader.** Stream one record, record the field names and types, and write the schema to the log before writing any parsing code (PRD U-D1).

**D-8. Count and log every dropped sample.** Filtering, truncation or dropping must report the number of samples affected. Silent dropping is a defect.

**D-9. Use the GPT-2 tokenizer for the MLA data path.** The repository's custom WordPiece-style vocabulary is not used by the new path (PRD D-T1, recommended option). Record the tokenizer identifier and vocabulary size in the manifest.

**D-10. Separate artefact licensing from code licensing.** Tokenised shards derived from the dataset are not published or committed. Note the dataset terms in the manifest.

---

## 5. Training and testing rules

### 5.1 Training

**TR-1. Configurable training.** Every hyperparameter is set from a configuration file, with command-line override. No value is hardcoded in the training loop.

**TR-2. Gradient accumulation.** Supported. An accumulated update over $k$ micro-batches of size $b$ must equal a single update over one batch of size $kb$, within floating-point tolerance. Tested.

**TR-3. Mixed precision.** Use `torch.autocast`. Use bfloat16 where the device supports it; otherwise float16 with a gradient scaler. Record the precision in the run log. Do not use apex in new code.

**TR-4. Gradient clipping.** Global norm clipping at the configured value (default 1.0). Log the pre-clip norm.

**TR-5. Checkpoint saving and restoration.** Follow PRD §12 exactly, including RNG state, the data cursor, the configuration fingerprint and atomic writes.

**TR-6. Logging.** Log training loss, validation loss, perplexity, learning rate, gradient norm, tokens per second and elapsed time, to a JSON-lines file, at configured intervals.

**TR-7. Full-split validation.** Each evaluation covers the whole validation split. Sampled evaluation is not used for comparisons.

**TR-8. Device handling.** Device is a configuration value. Do not call `.cuda()` in new code. Smoke tests must run on CPU.

**TR-9. Optimiser groups.** Weight decay applies to matrix parameters only. Biases, normalisation gains and embeddings are excluded. Log the parameter group sizes at startup.

**TR-10. Stop conditions.** Training stops on the configured step count, on a non-finite loss (with a checkpoint written first), or on an explicit interrupt, in which case a checkpoint is written before exit.

### 5.2 Testing

**T-1. Python compilation.** `python -m compileall src tests` succeeds before any phase is marked complete.

**T-2. Unit tests.** Each new module has unit tests for shapes, dtype, and the specific equations it implements. Use small, hand-checkable inputs where possible.

**T-3. Integration tests.** The full model with `attention='mla'` runs forward and backward on a small configuration, on CPU.

**T-4. Forward and backward propagation.** Every trainable parameter receives a non-zero gradient in a test configuration with the RoPE pathway, the content pathway and the query latent all active. Parameters that are legitimately unused must be listed and justified.

**T-C1. Causal invariance.** For a model in eval mode, the logits at position $t$ are unchanged when tokens at positions $> t$ are altered. Tested for both `mha` and `mla`, at several positions, with padding present and absent.

**T-C2. Padding invariance.** Logits for non-pad positions are unchanged when the sequence is extended with padding on the right.

**T-K1. Cache equivalence.** For a fixed prompt and continuation, incremental decoding with cache produces the same logits as a full forward pass over the same prefix, position by position, within a documented tolerance. Tested for at least five prompts and both arms.

**T-R1. Checkpoint recovery.** Training for $N$ steps and training for $k$ steps, saving, then resuming to $N$, produce matching parameter hashes within the stated tolerance. Tested on CPU with a small configuration.

**T-B1. Baseline unchanged.** With `attention='mha'`, the outputs and parameter count of the baseline match the outputs and parameter count from the unmodified repository, for a fixed seed and input. Verified against a reference captured from the audited commit before any edit.

**T-P1. Parameter and cache counts.** Parameter and cache counts match the analytic values in PRD §8.5, to the element, for the proposed configuration.

**T-D1. Data.** Tokeniser round-trip on sample text. Deterministic shard hashes across two runs. Validation and training disjointness. Dropped-sample counts match the expected count on a fixture.

**T-E2E. End-to-end smoke test.** On CPU, with the smoke configuration, complete data preparation, training for both arms, checkpoint, resume, evaluation and generation. Record wall time and outputs.

**T-S1. SDPA equivalence.** If the fused path is used, its output matches the explicit reference on the test inputs, with the mask, within the stated tolerance.

### 5.3 Test execution rules

- Run the relevant tests after each change. Run the full suite before marking a phase complete.
- A test that is skipped must say why in the log. Do not mark a phase complete with skipped required tests.
- Do not weaken a tolerance to make a test pass. If a test fails for a numerical reason, diagnose it, record the cause, and either fix the code or record a documented, justified tolerance change in `docs/decisions.md`, approved by the orchestrator.

### 5.4 Baseline capture

Before any edit to the model code, capture a reference: outputs of the unmodified `Transformer` for a fixed seed and fixed input, saved as a fixture under `tests/fixtures/`, together with the parameter count and the commit hash. T-B1 compares against it.

---

## 6. Benchmarking rules

**B-1.** Warm up before timing. Report the median of at least 20 repeats, and the spread.

**B-2.** Record the environment: GPU model, driver, CUDA, PyTorch version, Python version, commit hash, and configuration fingerprint.

**B-3.** Set `torch.backends.cudnn.benchmark` and deterministic flags explicitly, and record them.

**B-4.** Measure peak memory with `torch.cuda.reset_peak_memory_stats()` before each measurement.

**B-5.** Separate forward, backward, and generation timings. Separate time to first token from time per output token.

**B-6.** Never compare timings across different hardware, precision or batch configurations without labelling the difference.

**B-7.** Results are written to a structured file. Plots are generated from that file, not typed by hand.

---

## 7. Documentation rules

**DOC-1. Implementation log.** Maintain `experiments/LOG.md` (or `docs/implementation_log.md`). Each entry records: date, phase, task, files read, files changed, commands run, test results (actual output summary), decisions, blockers. Entries are appended, never rewritten.

**DOC-2. Architectural decisions.** Maintain `docs/decisions.md`. Each decision has an identifier (matching the PRD register where applicable), the question, the options, the choice, and the reason. Changes to a decision are new entries that reference the old.

**DOC-3. Test results.** Record actual test output. Quote the summary line (number passed, failed, skipped) from the tool, not an estimate.

**DOC-4. Experimental results.** Written only after the run has completed. Include the run identifier, configuration fingerprint, seed, environment and the raw metric file path. Never write an expected result into the results before it is measured.

**DOC-5. Update architecture documentation.** When a module's behaviour changes, update the module docstring and the relevant section of the audit or PRD, and note the change in the log.

**DOC-6. Plain language for limitations.** Every results report includes the limitations in PRD §10.2 and any new ones discovered during the work.

---

## 8. Execution rules

**E-1. Work phase by phase.** Follow `IMPLEMENTATION_PLAN.md`. Do not start a phase until the previous phase's completion criteria are met or formally waived by the orchestrator.

**E-2. Validate each phase before proceeding.** Each phase ends with its validation criteria run and recorded.

**E-3. Create checkpoints after milestones.** A milestone is a git commit with a message naming the phase and the criteria met. Do not commit untested code. Do not commit large artefacts.

**E-4. Do not proceed past critical failures.** A failing T-C1, T-K1, T-R1, T-B1 or T-E2E test is critical. Work stops until it passes. A failing test is not "to be fixed later".

**E-5. Report blockers explicitly.** A blocker is written to the log with: what is blocked, what was tried, what is needed, and who must act. Then stop work on the blocked item and continue with any independent work.

**E-6. Keep changes small and reviewable.** One concern per commit. A diff that touches more than one module for more than one reason is split.

**E-7. Ask before deciding unresolved items.** Items in PRD §23 with owner "User" are not decided by an agent. Record the question and the recommended default, and wait.

---

## 9. Multi-agent workflow

The work is divided among eight roles. Each role has a single owner for each file. Roles can be performed by separate agent instances or sequentially by one agent, but the boundaries are the same in both cases.

### 9.1 Roles

**Role 1: Architecture Analyst.**
- *Responsibilities:* Understand the existing GPT-2 implementation. Check MLA compatibility with the transformer, cache and mask conventions. Review architectural decisions. Maintain `docs/decisions.md` with the Lead.
- *Inputs:* `REPOSITORY_AUDIT.md`, `PRD.md`, the repository source.
- *Outputs:* Compatibility notes, decision entries, a list of call sites affected by any signature change.
- *Owns:* `docs/decisions.md`, `docs/architecture.md`.
- *Depends on:* Nothing. Runs first.

**Role 2: MLA Implementation Agent.**
- *Responsibilities:* Implement latent query and KV compression, decoupled RoPE, attention, the cache, and the reference and fused paths. Handle tensor dimensions and attention correctness.
- *Inputs:* Architecture Analyst's compatibility notes, PRD §8.
- *Outputs:* `src/gpt2/modeling/latent_attention.py`, `src/gpt2/modeling/rotary.py`, the `attention` argument in `transformer.py` (additive), unit tests for these modules.
- *Owns:* the two new modeling modules and their tests; the additive change in `transformer.py`, coordinated with the Lead.
- *Depends on:* Role 1's notes.

**Role 3: Data Engineering Agent.**
- *Responsibilities:* Access the gated dataset. Inspect the schema. Implement streaming, GPT-2 BPE tokenisation, sharded caching, packing, deterministic splitting, the manifest, and the data tests.
- *Inputs:* PRD §10, `AGENTS.md` §4.
- *Outputs:* `src/gpt2/data/token_corpus.py`, data configuration, manifest format, data tests, `docs/dataset.md` with the recorded schema.
- *Owns:* new data modules and their tests. Does not modify `TokenizedCorpus`.
- *Depends on:* Access confirmation from the user (U-E1). Can start the schema inspection and manifest format in parallel with Role 2.

**Role 4: Training Engineer.**
- *Responsibilities:* Implement the research training loop, device handling, autocast, clipping, accumulation, warm-up, logging, full-split evaluation, checkpointing and exact resume, and the research configuration system.
- *Inputs:* PRD §§9, 11, 12; the data interface from Role 3; the model interface from Role 2.
- *Outputs:* `src/gpt2/research/` modules, configuration files, checkpoint tests.
- *Owns:* `src/gpt2/research/` and `configs/`.
- *Depends on:* Roles 2 and 3 for interfaces. Can start the configuration schema and the checkpoint format in parallel.

**Role 5: Testing and Validation Agent.**
- *Responsibilities:* Compilation, unit tests, integration tests, causal and padding invariance, cache equivalence, checkpoint recovery, baseline-unchanged test, parameter and cache count checks, numerical checks for fused kernels, and the end-to-end smoke test.
- *Inputs:* All other roles' outputs, `AGENTS.md` §5.
- *Outputs:* Test files under `tests/`, test reports in the log, a pass/fail summary per phase.
- *Owns:* `tests/`, the baseline fixture under `tests/fixtures/`.
- *Depends on:* The code under test. Writes tests from the specification in parallel with implementation where possible, and runs them when the code is available.

**Role 6: Research and Benchmarking Agent.**
- *Responsibilities:* Run the baseline and MLA arms for the three tiers and ablations. Measure the metrics in PRD §15. Compute confidence intervals. Record results with environment and configuration. Analyse performance and limitations.
- *Inputs:* Working training pipeline, benchmark tool, configurations.
- *Outputs:* Raw metric files, benchmark files, results tables, analysis section.
- *Owns:* `experiments/runs/`, `experiments/results/`, the analysis document.
- *Depends on:* Phases 4 and 5 complete and validated. Cannot start before them.

**Role 7: Documentation Agent.**
- *Responsibilities:* Maintain the implementation log, architecture and experiment documentation, and the dataset documentation. Keep docs consistent with code after each change.
- *Inputs:* Logs and outputs from all roles.
- *Outputs:* `experiments/LOG.md`, `docs/*.md`, updates to module docstrings where asked.
- *Owns:* `experiments/LOG.md`, `docs/`.
- *Depends on:* All roles, but writes continuously.

**Role 8: Lead Orchestrator and Reviewer.**
- *Responsibilities:* Coordinate dependencies between roles. Review every change to a shared file. Prevent conflicting modifications. Approve completion of each phase. Maintain the decision register with the Architecture Analyst. Escalate user-owned decisions.
- *Inputs:* Everything.
- *Outputs:* Phase approvals, review notes, the list of blockers, the final report.
- *Owns:* The merge decision on every shared file: `transformer.py`, `__main__.py`, `modeling/__init__.py`, `requirements-research.txt`, `AGENTS.md`, `IMPLEMENTATION_PLAN.md`, `PRD.md`.
- *Depends on:* Everything. Has authority to halt any role's work.

### 9.2 Parallelism

| Can run in parallel | Reason |
|---|---|
| Role 1 analysis with Role 3 schema inspection | Schema inspection does not depend on model code. |
| Role 2 MLA implementation with Role 3 tokeniser and manifest work | Separate modules, no shared files. |
| Role 4 configuration schema with Role 2 implementation | Configuration depends only on the interface, which Role 1 fixes first. |
| Role 5 writing tests with Roles 2 and 3 implementing | Tests written from the specification. |
| Role 7 documentation with any other role | Documentation follows, does not block. |

| Must run sequentially | Reason |
|---|---|
| Baseline capture (§5.4) before any edit to `transformer.py` | Otherwise the baseline is not provably unchanged. |
| Role 1 interface decisions before Role 2 writes code | The interface is the contract. |
| Role 2 and Role 3 interfaces before Role 4 training | Training depends on both. |
| Phase validation before the next phase | Enforced by the orchestrator. |
| Role 6 experiments after Phase 5 | Benchmarks and runs need a validated pipeline. |

### 9.3 Handling conflicting changes

1. **Ownership.** Each file has one owning role. Only the owner edits it. Other roles request changes through the Lead.
2. **Shared files** (`transformer.py`, `__main__.py`, `modeling/__init__.py`, `requirements-research.txt`, the top-level documents) are edited only by the Lead or with the Lead's explicit approval for a specific change.
3. **Interface changes.** A change to a function signature or a configuration key is recorded in `docs/decisions.md` before the change, with every affected caller listed.
4. **Concurrent edits.** Before editing a file, the owning role re-reads it. If the file changed since it was read, the role re-reads and re-applies its change. Edits are never made against a stale view.
5. **Disagreement.** If two roles disagree on an interface or a decision, both write their position to the decision log, and the Lead decides. The decision and its reason are recorded.
6. **Rollback.** A change that breaks a critical test is reverted in a commit, not patched forward in the same step, unless the fix is trivial and verified.

### 9.4 How the orchestrator validates deliverables

For each phase, the Lead checks, in order:

1. The phase's completion criteria in `IMPLEMENTATION_PLAN.md` are each marked with the actual test output or artefact that proves them.
2. `python -m compileall` passes on the current tree.
3. The full test suite passes, with the count of passed, failed and skipped tests recorded.
4. The baseline-unchanged test passes.
5. Every changed shared file has been reviewed by the Lead, with the diff in the log.
6. The implementation log has an entry for the phase, with the blockers, if any, listed.
7. Unresolved decisions relevant to the phase are either resolved or carried forward explicitly.

A phase is approved only when all seven checks are satisfied. A check that cannot be run is a failed check, and the reason is recorded.

---

## 10. Definition of done for the whole project

The project is complete when the acceptance criteria in `PRD.md` §22 are all satisfied, and the final report lists: what was built, what was measured, what hypotheses were supported or rejected with their intervals, the limitations, the unresolved decisions, and any negative findings. A project in which the MLA arm was not measured to be better is still complete if the report says so clearly.

---

## 11. Quick reference for the agent

- Read before you write. Record what you read.
- Preserve the baseline. Test that you did.
- Do not stub. Report blockers.
- Do not fabricate numbers. Measure them.
- Do not claim MLA is better. Measure whether it is.
- Do not download the full dataset. Stream and bound.
- Do not touch CI, the Python matrix, or the user-owned decisions without approval.
- Commit small, tested milestones.

## Appendix: Known baseline defects (carried, documented, not silently fixed)

- **B-1.** `batch_eval` is ignored on single-process training (audit R16).
- **B-2.** The aggregate perplexity reported by `evaluate` is a mean of per-batch perplexities, not exp of the token-weighted mean loss (audit §7.2).
- **B-3.** The final partial evaluation batch is dropped (audit R17).
- **B-4.** The Tiny WebText column schema is unverified (PRD U-D1).

Rule: the baseline path keeps these behaviours so its outputs stay identical (R-2). New research code computes its own metrics and records the definition it uses. Any change to baseline behaviour needs its own equivalence test and Lead Orchestrator approval.
