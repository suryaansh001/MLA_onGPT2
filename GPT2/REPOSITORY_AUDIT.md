# Repository Audit: affjljoo3581/gpt2

**Audited revision:** commit `7b159669f6225231ee75f5ea110c1be69f4ce8ef` (2024-07-04, "Update README.md"), shallow clone of `main`/`master` HEAD.
**Method:** static reading of every file under `src/`, `tests/`, `.github/`, `requirements.txt`, `README.md` and the two notebooks. No code was executed, no dependencies were installed, and the existing test suite was **not run** in this stage. The parameter arithmetic in PRD §8.5 was recomputed independently and matches (123,849,984 at `seq_len` 256).
**Licence:** Apache-2.0.

---

## 1. Identity and what this repository is

This is a self-contained PyTorch implementation of a GPT-2-style decoder-only language model, written from the OpenAI GPT-2 paper rather than from the Hugging Face or nanoGPT codebases. The README describes it as "PyTorch implementation of OpenAI GPT-2" with training, generation and metrics-visualisation commands.

Important consequences for the MLA project:

- It does **not** use `transformers`, `tiktoken`, or any Hugging Face model class. No code path loads OpenAI GPT-2 weights.
- It is **not** bit-compatible with OpenAI GPT-2 in several respects (activation function, tokenizer, bias and normalisation details). See Section 4.
- Its data pipeline consumes **pre-tokenized text files plus a vocabulary file**, not raw text. The vocabulary is a WordPiece-style list with `##` continuation markers, built externally (the README recommends the author's own tool, Expanda, which is not in this repository).

---

## 2. Repository structure and file responsibilities

| Path | Lines | Responsibility |
|---|---:|---|
| `src/gpt2/__main__.py` | 22 | CLI entry point. Dispatches `train`, `evaluate`, `generate`, `visualize` subcommands. |
| `src/gpt2/modeling/attention.py` | 116 | `BaseAttention` (scaled dot-product with additive mask), `MultiHeadAttention` (head split/merge), `AttentionLayer` (Q/K/V/output projections plus KV concatenation for `past`). |
| `src/gpt2/modeling/transformer.py` | 117 | `TransformerLayer` (pre-LN attention and FFN with residuals), `Transformer` (embeddings, layer stack, final LN, tied output head, mask construction, cache plumbing). |
| `src/gpt2/modeling/embedding.py` | 63 | `PositionalEmbedding` (learned absolute positions, with resize logic on state-dict load), `TokenEmbedding` (also used transposed as the output projection). |
| `src/gpt2/modeling/feedforward.py` | 36 | `Swish` activation and `PositionwiseFeedForward` (Linear → Swish → Dropout → Linear, expansion `rate`). |
| `src/gpt2/modeling/masking.py` | 45 | `PadMasking` and `FutureMasking`. Both return **True = masked** tensors. |
| `src/gpt2/modeling/__init__.py` | 6 | Re-exports. |
| `src/gpt2/utils/fusing.py` | 6 | Chooses apex `FusedAdam` / `FusedLayerNorm` if importable; otherwise `torch.optim.AdamW` / `torch.nn.LayerNorm`. |
| `src/gpt2/data/vocabulary.py` | 51 | `Vocab`: reads a whitespace-separated vocab file, prepends `<s> </s> <pad>`, pads reported size to a multiple of 8. |
| `src/gpt2/data/tokenization.py` | 81 | `Tokenizer`: greedy longest-match WordPiece-style subword tokenizer with Chinese character splitting. Used for generation prompts only. |
| `src/gpt2/data/corpus.py` | 65 | `TokenizedCorpus`: streams lines from a text file, maps tokens to indices, adds BOS/EOS, pads, drops sequences that do not fit, supports `repeat`, and exposes `where()`/`assign()` file-offset checkpointing. |
| `src/gpt2/data/dataset.py` | 16 | Abstract `Dataset` interface (`skip`, `fetch`, `where`, `assign`). |
| `src/gpt2/training/training.py` | 217 | `Trainer`: training loop, periodic eval, checkpointing, resume, optional apex AMP, optional DDP via `mp.spawn`. |
| `src/gpt2/training/configuration.py` | 29 | `TrainConfig` (batch sizes, steps, paths, `use_amp`, `gpus`). |
| `src/gpt2/training/specification.py` | 29 | Abstract `TrainingSpec`. |
| `src/gpt2/training/recording.py` | 29 | `Recorder`: accumulates per-batch metrics, averages them per eval interval, and formats the progress-bar string. |
| `src/gpt2/train_model.py` | 153 | `GPT2TrainingSpec` (model construction with `bidirectional=False`, optimizer, LR schedule, objectives) and the `train` CLI definition with all default hyperparameters. |
| `src/gpt2/generation/generation.py` | 67 | `Generator`: autoregressive loop using the `past` cache, top-p (nucleus) sampling, optional GPU with `.half()`. |
| `src/gpt2/generation/specification.py` | 21 | Abstract `GenerationSpec`. |
| `src/gpt2/generation/configuration.py` | 9 | `GenerateConfig`. |
| `src/gpt2/generate_sentences.py` | 83 | `GPT2GenerationSpec` and the interactive `generate` CLI. Note: `dropout=0`. |
| `src/gpt2/evaluation/evaluation.py` | 54 | `Evaluator`: iterates a non-repeating corpus, averages batch metrics. |
| `src/gpt2/evaluation/specification.py` | 19 | Abstract `EvaluationSpec`. |
| `src/gpt2/evaluation/configuration.py` | 18 | `EvaluateConfig` (`total_steps = -1` means run to exhaustion). |
| `src/gpt2/evaluate_model.py` | 93 | `GPT2EvaluationSpec` (CE with `reduction='none'`, masked mean loss, perplexity) and the `evaluate` CLI. |
| `src/gpt2/visualize_metrics.py` | 146 | Matplotlib plots of recorded training metrics from a saved model file. |
| `tests/` | ~700 | Pytest suite: shape and value tests for attention, masking, embedding, feed-forward, transformer; tests for corpus, tokenization, vocabulary, generation, configuration and recording. |
| `.github/workflows/build.yml` | 41 | CI: Python matrix 3.6, 3.7, 3.8; flake8 (hard-fail on syntax/undefined names); pytest with coverage upload to Codecov. |
| `requirements.txt` | 4 | `regex`, `tqdm`, `torch`, `numpy`, `matplotlib`. **No version pins.** |
| `README.md` | 190 | Usage, CLI reference, apex instructions, Colab links. |
| `GPT2_Interactive_Notebook.ipynb` | — | Colab: downloads model, vocab and eval corpus from a GCS bucket (placeholder path), runs generate and evaluate. Config: `seq_len=128, layers=24, heads=16, dims=1024, rate=4`. |
| `korean_gpt2_302M_demo.ipynb` | — | Downloads a release asset `gpt2-ko-302M-512L.pth` and `vocab-ko-302M.txt` from GitHub releases, with `seq_len=512, layers=24, heads=16, dims=1024, rate=4`. This is the author's own Korean model, not an OpenAI checkpoint. |
| `.gitignore`, `LICENSE`, `example-figure.png` | — | Housekeeping. |

Not present: any `setup.py`, `pyproject.toml`, package `__init__` version metadata, configuration files (YAML/JSON), a `Makefile`, Docker files, or a conversion script for external weights.

---

## 3. Architecture as implemented

### 3.1 Model graph (`modeling/transformer.py`)

The forward pass of `Transformer` is:

1. `offset = past[0][0].size(-2)` if a cache is supplied, otherwise 0.
2. Build a mask: `PadMasking(x, offset)`, plus `FutureMasking(x, offset)` when `bidirectional=False`. The two are added, so a position is masked if either applies.
3. `x = TokenEmbedding(x) + PositionalEmbedding(x, offset)`, followed by dropout.
4. For each `TransformerLayer`: `x = x + Attn(LN(x))`, then `x = x + FFN(LN(x))`. This is pre-LN.
5. `x = LN_head(x)`, then `logits = x @ W_emb^T`. The output projection is **tied** to the input token embedding.
6. Returns logits in training mode. In eval mode it returns `(logits, present)` where `present` is a list of per-layer `(k, v)` caches.

Default CLI hyperparameters (`train_model.py`): `seq_len=64, layers=12, heads=16, dims=1024, rate=4, dropout=0.1`. The author's shipped Korean model uses 24 layers, 16 heads, 1024 dims.

### 3.2 Attention (`modeling/attention.py`)

`AttentionLayer` applies four **biased** `nn.Linear(dims, dims)` layers: `proj_q`, `proj_k`, `proj_v` and an output `linear`. The head count is `heads`, so the per-head dimension is `dims / heads`. Keys and values from `past` are concatenated along the sequence axis before attention, so the cache is the full `(k, v)` pair with shape `(..., past_len, dims)`.

`BaseAttention` computes `softmax(q k^T / sqrt(d_k) + mask*(-1e4)) v`, followed by dropout on the probabilities. The scale uses `k.size(-1)`. The mask convention is **True = masked**, applied as an additive `-1e4`. Note this is the opposite of `torch.nn.functional.scaled_dot_product_attention`, where a boolean `attn_mask` marks positions that **may** attend.

### 3.3 Feed-forward (`modeling/feedforward.py`)

`Linear(d, 4d) → Swish → Dropout → Linear(4d, d)`, with Swish defined as `x * sigmoid(x)`. This is **SiLU**, not GELU. The original GPT-2 uses GELU.

### 3.4 Embeddings and normalisation

- Token embedding and positional embedding are `nn.Embedding` subclasses initialised with `N(0, 0.02)`.
- The positional table has exactly `seq_len` rows. `_load_from_state_dict` truncates or extends it when loading a checkpoint saved with a different `seq_len`, which is a deliberate convenience.
- LayerNorm is `gpt2.utils.fusing.LayerNorm`: `torch.nn.LayerNorm` unless apex is installed. It uses default epsilon and learnable affine parameters.

### 3.5 Causal masking and bidirectional flag

`Transformer(bidirectional=True)` is the constructor default, but **every spec in the repository passes `bidirectional=False`** (training, evaluation and generation). Causal behaviour therefore depends on the caller. Tests in `test_transformer.py` exercise `bidirectional=False`.

### 3.6 KV cache

`TransformerLayer` returns `(x, past)` where `past = (k, v)` in eval mode. The `Transformer` collects one tuple per layer. Generation (`generation/generation.py`) passes only the newest token after the first step and uses `past[0][0].size(-2)` as the position offset. Any MLA cache must preserve the convention that the **sequence length is `size(-2)` of the first element of each per-layer tuple**, or the offset logic breaks silently.

---

## 4. Deviations from OpenAI GPT-2 (must be documented in the PRD)

| Aspect | OpenAI GPT-2 | This repository | Impact on MLA project |
|---|---|---|---|
| FFN activation | GELU | Swish / SiLU | Baseline is not GPT-2-exact. Acceptable if documented; both arms must share it. |
| Tokenizer | Byte-level BPE, 50,257 tokens | Custom WordPiece-style vocab from a text file, padded to a multiple of 8 | Conflicts with the requirement "GPT-2 tokenizer compatibility". Decision required (PRD §10.4, D-T1). |
| Positional embedding | Learned, 1024 rows | Learned, `seq_len` rows (default 64) | Retained by default. Compatible with the PRD's stated preservation requirement. |
| Output head | Tied to token embedding | Tied | Matches. |
| Pre/post LN | Pre-LN | Pre-LN with final LN | Matches. |
| Linear biases | Yes | Yes (all projections) | Matches. |
| Weight init | N(0, 0.02) with residual scaling | N(0, 0.02) on embeddings; `nn.Linear` defaults elsewhere | No residual-path scaling. Must be kept identical across arms. |
| Attention dropout | Yes | Yes (on probabilities) | Matches. |

---

## 5. Data pipeline

### 5.1 Input format

`TokenizedCorpus` reads a **text file where each line is one document already tokenized into whitespace-separated subword strings**. Each token is looked up in the `Vocab` dictionary. The loader:

- Prepends `<s>` and appends `</s>`.
- Discards any line where `len(tokens) + 2 > seq_len`. This happens silently, with no counter or log.
- Pads with `<pad>` to `seq_len + 1`, then produces `input = indices[:-1]` and `output = indices[1:]`.
- Does **not** pack multiple documents into one sequence. Each sample is one document, padded.
- Wraps around to the start when `repeat=True` and stops with `StopIteration` when `repeat=False`.

### 5.2 Checkpointing of data position

`where()` returns the Python text-file `tell()` cookie. `assign()` seeks to it. This works for resumption on the same machine and Python build, but the value is opaque and not portable.

### 5.3 Tokenizer (generation only)

`Tokenizer` is used to encode prompts and decode output. It normalises whitespace and control characters, splits Chinese characters, lowercases, and runs greedy longest-match subword segmentation against the vocabulary. Training does not use it; the corpus must already be tokenized externally.

### 5.4 Compatibility with Hugging Face streaming

Incompatible with the current design without a new loader. The corpus loader is line-oriented and file-backed, and `where()`/`assign()` assume a seekable file. A Hugging Face streaming iterator has no seek position, so the resume contract must be redesigned (PRD §12, data cursor).

---

## 6. Training

### 6.1 Loop (`training/training.py`)

- Single-process by default. Multi-GPU uses `torch.multiprocessing.spawn` with NCCL on a hardcoded `tcp://127.0.0.1:8000` init method.
- Every step: `optimizer.zero_grad()`, fetch a batch, forward, `loss.backward()`, `optimizer.step()`, `scheduler.step()`. **No gradient accumulation, no gradient clipping, no gradient norm logging.**
- Device handling is **hardcoded to CUDA**: `model.cuda()`, `map_location='cuda'`, `v.cuda()` on every batch. There is no CPU path for training.
- Mixed precision uses NVIDIA apex `amp.initialize(opt_level='O2')` when `--use_amp` is set. The code does not use `torch.autocast`.
- Evaluation runs every `eval_steps` on the eval corpus with `model.eval()`, batch size `batch_eval`, under `torch.no_grad()`. Eval is on a single batch per interval, not a full pass. Its metric is therefore a noisy sample, not a validation loss.
- Recorder averages per-batch values between eval points. The eval point's `step` is stamped on the averaged value.
- **Defect: `batch_eval` is ignored in the single-process path.** `Trainer._fetch_from` calls `dataset.fetch(self.config.batch_train)` in its non-distributed branch and discards its `batch` argument, so one-GPU evaluation uses `batch_train`. The distributed branch honours `batch_eval`. Both arms must therefore set `batch_train` and `batch_eval` identically, and the MLA evaluator must read its own batch size explicitly (R16).

### 6.2 Optimiser and schedule (`train_model.py`)

- Optimiser: `fusing.Adam` (apex `FusedAdam` if installed, otherwise `torch.optim.AdamW`), `lr=base_lr` (default `1e-4`), `weight_decay=wd_rate` (default `1e-2`).
- **Weight decay is applied to all parameters**, including LayerNorm gains, biases and embeddings. There are no parameter groups.
- Schedule: `LambdaLR` with `lambda step: 1 - step / total_steps`. This is **linear decay to zero, with no warm-up**.
- Default `total_steps = 1,000,000`. A short experiment must set this explicitly, because the schedule depends on it.

### 6.3 Loss

`CrossEntropyLoss(ignore_index=pad_idx, reduction='mean')` on `logits.transpose(1, 2)` against `output`. Padding is ignored by the loss, not by attention masking alone.

### 6.4 Checkpoint contents

Periodic checkpoint (`--save_checkpoint_path`, every `save_steps`) saves: `step`, `recorder` (the full metric history), `model`, `optimizer`, `scheduler`, `train_dataset.where()`, `eval_dataset.where()`, and `amp` state if enabled.

Not saved: RNG state (Python, NumPy, torch CPU and CUDA), the dataloader's internal state beyond the file offset, and the `TrainConfig`. Resuming therefore does not reproduce the exact dropout masks or data order of an uninterrupted run.

Final artifact (`--save_model_path`): `{'model': state_dict, 'metrics': recorder.metrics}`. The model file does not record architecture hyperparameters. Loading requires the caller to supply `--layers --heads --dims --rate --seq_len` again. A mismatch fails at `load_state_dict` or, for `seq_len`, is silently truncated or extended by the positional-embedding loader.

### 6.5 Hyperparameter interface

Hyperparameters are CLI arguments only. There are no config files and no config serialisation into the checkpoint. This matters for reproducibility of the three experiment tiers.

---

## 7. Inference, generation and evaluation

### 7.1 Generation (`generation/generation.py`, `generate_sentences.py`)

- Encodes the prompt, prepends `<s>`, then loops until `len(words) == seq_len`. It **does not stop at `</s>`** inside the loop; `decode_tokens` truncates after the first EOS afterwards.
- Uses the KV cache: one forward pass per generated token after the first.
- Sampling: top-p. Probabilities are sorted, cumulative mass above `nucleus_prob` is zeroed, and `mask[0] = False` keeps at least one token. Sampling uses `multinomial`.
- `use_gpu` moves the model to CUDA and calls `.half()`. The CPU path stays in float32.
- Prompt longer than `seq_len` produces no output. There is no sliding window.

### 7.2 Evaluation (`evaluate_model.py`, `evaluation/evaluation.py`)

- Runs over a non-repeating corpus. Batches are averaged uniformly, so the final figure is a mean of batch means rather than a token-weighted mean. Batches with fewer valid tokens count equally.
- Loss is masked mean cross-entropy computed per batch. Per-batch perplexity is `exp(batch mean loss)`. The reported aggregate is an unweighted mean over batches of both loss and per-batch perplexity. The aggregate perplexity is therefore **not** `exp` of the corpus-level token-weighted mean loss. By Jensen's inequality it is at least as large. The MLA comparison must use a separately computed corpus-level perplexity (PRD §15).
- `use_gpu` converts the model to half precision, which slightly changes the reported numbers relative to a float32 run.
- `total_steps = -1` runs to exhaustion. `EvaluateConfig.iterate()` is the only place this is handled.

### 7.3 Metrics visualisation

`visualize_metrics.py` reads the `metrics` dictionary from a model file and plots training and evaluation curves with Matplotlib, including an interactive mode. This utility is independent of the model code and requires no changes.

---

## 8. Tests and CI

### 8.1 Existing tests

Listed for coverage of invariants that must not break:

| File | Verifies |
|---|---|
| `tests/modeling/test_attention.py` | Output shapes for `BaseAttention`, `MultiHeadAttention`, `AttentionLayer` with and without mask and `past`; numerical values under a small hand-computed mask. |
| `tests/modeling/test_masking.py` | Pad and future mask shapes, values, and offset behaviour. Confirms True = masked. |
| `tests/modeling/test_embedding.py` | Positional embedding shape, offset, and state-dict resize. |
| `tests/modeling/test_feedforward.py` | Feed-forward shape. |
| `tests/modeling/test_transformer.py` | Layer and full-model output shapes; `past` accumulation across calls; a generation loop of 10 single-token steps. |
| `tests/data/test_corpus.py`, `test_tokenization.py`, `test_vocabulary.py` | Corpus fetching, `where/assign`, tokenization rules, vocabulary indexing. |
| `tests/generation/test_generation.py` | Generation behaviour. |
| `tests/training/test_configuration.py`, `test_recording.py` | Config and recorder logic. |

**These tests have not been executed in this audit.** Their status on current PyTorch is unknown and must be established in Phase 0 of the implementation plan.

Notable absence: there is **no test that verifies a causal property end to end**, such as "output at position *t* is unchanged when tokens after *t* are changed". `test_masking.py` checks mask values, not the consequence on the model output. The MLA project must add this.

### 8.2 CI

- Matrix: Python 3.6, 3.7 and 3.8. These versions are long unsupported. Current PyTorch wheels do not install on 3.6 and generally do not on 3.7 or 3.8, so **CI as written probably fails today**. Unverified; the failure status should be checked in the hosted CI history.
- Lint: `flake8` hard-fails only on syntax errors and undefined names (`E9,F63,F7,F82`). Everything else is advisory.
- Tests: `pytest --cov=./src tests/*`, with Codecov upload set to `fail_ci_if_error: true`.

---

## 9. Integration risks

Ordered by severity for the MLA project.

| # | Risk | Severity | Evidence | Mitigation |
|---|---|---|---|---|
| R1 | **Tokenizer mismatch.** The repo uses a custom WordPiece-style vocabulary, not GPT-2 BPE. The PRD requires GPT-2 tokenizer compatibility. | High | `data/vocabulary.py`, `data/tokenization.py`, README | Decision D-T1 (PRD §10.4). Recommended: add a new token-ID data path using GPT-2 BPE (vocab 50,257), leave the existing path intact. |
| R2 | **Dataset is gated.** Tiny WebText requires Hugging Face login and terms acceptance. | High | Dataset page: "You need to agree to share your Hugging Face information" | Agents must use a token from the environment (`HF_TOKEN`), never a committed credential. Record the accepted-terms status in the implementation log. |
| R3 | **Dataset schema unverified.** The column names and text field were not retrieved in this stage. | High | Dataset page reachable; viewer and file listings not retrieved | Phase 3 must inspect one streamed record and record the schema before writing the loader. |
| R4 | **Resume contract incompatible with streaming.** `where()` relies on seekable files. | High | `data/corpus.py` | Redesign: cache tokenized shards to disk with an index, and checkpoint the shard index and in-shard offset. |
| R5 | **Hardcoded CUDA.** Training and generation fail on CPU. Smoke tests on a CPU-only machine are impossible without a device abstraction. | High | `training.py` `.cuda()`, `map_location='cuda'` | Add a device parameter in new code. Do not change existing behaviour in the baseline path, or if changed, prove equivalence with tests. |
| R6 | **Mask convention inversion.** The repo's True = masked is the opposite of `scaled_dot_product_attention`'s boolean convention. A silent inversion would make attention attend only to the masked positions. | High | `masking.py`, `attention.py` | Single conversion helper, with a test that a causal model's output is invariant to future tokens. |
| R7 | **Cache semantics.** `Transformer` derives the position offset from `past[0][0].size(-2)`. An MLA cache with a different first-element shape breaks positions silently. | Medium | `transformer.py` | MLA cache first element must be `(..., past_len, d_c)`. Test offset after multi-step generation against a full forward pass. |
| R8 | **Positional embedding interaction.** Learned absolute positions plus RoPE in the attention pathway duplicate positional information. | Medium | `embedding.py`, DeepSeek-V2 §2.1.3 | Decision D-P1 (PRD §8.6). Primary arm keeps `wpe`, ablation removes it. |
| R9 | **Silent data loss.** Lines longer than `seq_len - 2` are dropped with no count. | Medium | `corpus.py` `_fetch_one` | New loader must log and count dropped samples and truncate or pack instead. |
| R10 | **Missing gradient clipping and warm-up.** Baseline recipe is not stable enough for a fair baseline without them. | Medium | `train_model.py`, `training.py` | Add both as configurable options, defaulting to off for reproducing the repo, and enable identically in both arms. |
| R11 | **apex dependency.** `--use_amp` requires NVIDIA apex, which is hard to install. | Medium | README, `training.py` | New code uses `torch.autocast`. Keep apex path untouched. |
| R12 | **Unpinned dependencies and dead Python versions.** Reproducibility is weak and CI likely broken. | Medium | `requirements.txt`, `build.yml` | Pin versions in a new `requirements-research.txt`. Do not alter CI without approval. |
| R13 | **Eval noise.** In-training eval uses one batch per interval, so curves are noisy. | Low | `training.py` `_eval_step` | New evaluator uses a fixed full validation pass for comparisons. |
| R14 | **Checkpoint lacks architecture metadata.** Loading needs matching CLI flags. | Low | `training.py` save | New checkpoints store a config dictionary alongside weights. Old checkpoints stay loadable. |
| R15 | **Half-precision evaluation.** `.half()` changes reported perplexity slightly. | Low | `evaluate_model.py` | Benchmarks use an explicit dtype setting recorded in results. |
| R16 | **Eval batch size ignored in single-process training.** `_fetch_from` discards `batch` in the non-distributed branch, so `--batch_eval` has no effect on one GPU. | Medium | `training/training.py` `_fetch_from` | Preserve in the baseline path and document. The MLA research trainer reads its eval batch size explicitly. |
| R17 | **Final partial eval batch silently dropped.** With `repeat=False`, `fetch` raises `StopIteration` mid-batch and `_eval_step` discards the sequences already fetched into that batch. | Low for ranking, Medium for absolute perplexity | `evaluation/evaluation.py` `_eval_step`, `data/corpus.py` | The new corpus-level evaluator counts every sequence and reports the count. |
| R18 | **README and CLI disagree.** The README generation example uses `--top_p`, which the parser does not define (the flag is `--nucleus_prob`). `--model` works only through argparse prefix matching. | Low | README, `generate_sentences.py` | Document in the new runbook. Do not change README in the baseline path without approval. |

---

## 10. Recommended modifications (additive)

The principle is to add new code beside existing code and make the attention class selectable, with `mha` remaining the default.

1. **Attention factory.** Add an `attention` argument (`'mha'` | `'mla'`) to `Transformer` and `TransformerLayer`, default `'mha'`. Existing call sites are unchanged. The baseline path must produce identical outputs before and after the change, and this must be tested.
2. **New modules.** Add `src/gpt2/modeling/latent_attention.py` for `MultiHeadLatentAttention` and RoPE helpers, plus `src/gpt2/modeling/rotary.py` if preferred. Export through `modeling/__init__.py`.
3. **New data path.** Add `src/gpt2/data/token_corpus.py` for Hugging Face streaming, GPT-2 BPE encoding, sharded `.npy` token caching, fixed-length packing, and deterministic train/validation split. Do not modify `TokenizedCorpus`.
4. **New training entry point.** Add a new module, for example `src/gpt2/research/train_research.py`, with device selection, `torch.autocast`, gradient clipping, accumulation, warm-up, config-serialised checkpoints and exact-resume RNG state. The existing `train` subcommand stays as it is.
5. **New evaluation and benchmarking.** Add `src/gpt2/research/benchmark.py` for latency, memory and KV-cache measurement. Keep existing `evaluate` unchanged.
6. **CLI.** Add subcommands to `__main__.py` only through the existing `add_subparser` pattern.
7. **Tests.** Add tests under `tests/modeling/test_latent_attention.py`, `tests/research/`, and `tests/data/test_token_corpus.py`.
8. **Requirements.** Create `requirements-research.txt`. Do not change `requirements.txt` in ways that break existing users.

---

## 11. New files required (summary)

| File | Purpose |
|---|---|
| `src/gpt2/modeling/latent_attention.py` | MLA module with latent compression, decoupled RoPE, cache. |
| `src/gpt2/modeling/rotary.py` | RoPE cos/sin construction and application for `d_R` dims. |
| `src/gpt2/data/token_corpus.py` | Streaming, caching, packing and splitting loader. |
| `src/gpt2/research/__init__.py`, `config.py` | Typed research configuration, serialised into checkpoints. |
| `src/gpt2/research/train_research.py` | Device-aware, checkpoint-safe training. |
| `src/gpt2/research/benchmark.py` | Latency, memory, KV-cache, throughput measurement. |
| `src/gpt2/research/generate.py` | Autoregressive generation with cache-equivalence checks. |
| `tests/...` | As listed in Section 10. |
| `requirements-research.txt` | Pinned research dependencies. |
| `configs/*.yaml` or `*.json` | Smoke, development and small-scale experiment configs. |
| `PRD.md`, `AGENTS.md`, `REPOSITORY_AUDIT.md`, `IMPLEMENTATION_PLAN.md`, `experiments/LOG.md` | Documentation. |

---

## 12. Unknowns that remain

These are stated explicitly rather than assumed.

1. Whether the existing test suite passes on a current PyTorch version. Not executed.
2. Whether CI currently passes on hosted runners. Not checked.
3. The exact column schema and text field name of `nampdn-ai/tiny-webtext`. Not retrieved.
4. Whether the dataset's gated access is approved for the project's account. Not checked.
5. The user's local GPU model, VRAM and whether it supports bfloat16. Not provided.
6. Whether the user's local clone matches this commit exactly. This audit used a fresh clone of the public repository.
7. Whether the author's Korean checkpoint release assets are still available. Not checked.
8. Whether PyTorch SDPA accepts `q` and `k` head dimension `d_h + d_R` different from value dimension `d_h` on the target backend without falling back to the math path. Requires a test in Phase 1.
