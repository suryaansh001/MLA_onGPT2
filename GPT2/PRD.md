# PRD: GPT-2 with Multi-Head Latent Attention (MLA)

**Status:** Draft v1, specification only. No implementation has begun.
**Source repository:** `affjljoo3581/gpt2`, commit `7b159669f6225231ee75f5ea110c1be69f4ce8ef`.
**Companion documents:** `REPOSITORY_AUDIT.md`, `AGENTS.md`, `IMPLEMENTATION_PLAN.md`.

**Section map against the requested PRD contents.** 1 Title: §1. 2 Overview: §2. 3 Motivation: §3. 4 Problem: §4. 5 Objectives: §5. 6 Questions: §6.1. 7 Hypotheses: §6.2. 8 Existing architecture: §7. 9 Proposed architecture: §8. 10 Maths: §8.1. 11 Integration strategy: §8.2 to §8.9. 12 Dataset and limitations: §10.1 to §10.2. 13 Preprocessing: §10.3 to §10.5. 14 Training: §11. 15 Checkpoints: §12. 16 Inference: §13. 17 Baseline comparison: §14. 18 Metrics: §15. 19 Hardware: §16. 20 Functional requirements: §17. 21 Non-functional: §18. 22 Technical constraints: §19. 23 Risks: §20. 24 Deliverables: §21. 25 Acceptance: §22. Unresolved decisions: §23. Open questions: §24.

**Labelling convention used throughout.** Each substantive statement carries one of four tags:

- **[E] Established:** a fact from the source paper, the repository code, or a primary source, verified during this stage.
- **[P] Proposed:** a design decision made by this document, reasoned but not yet validated.
- **[H] Hypothesis:** an experimental claim to be tested. It must not be reported as a finding until tested.
- **[U] Unresolved:** a question the implementation team must settle, with a recommended default.

---

## 1. Project title

**Latent-Attention GPT-2: A Controlled Study of Multi-Head Latent Attention in a Small Causal Language Model**

## 2. Project overview [P]

This project takes the GPT-2 decoder implementation in `affjljoo3581/gpt2` and replaces its conventional multi-head self-attention with Multi-Head Latent Attention, the attention mechanism introduced in DeepSeek-V2. Every other component of the model is kept as close to the repository baseline as possible. The two architectures are then trained and evaluated under identical conditions.

The output is an experimental comparison, not a product. Success means a rigorous, reproducible answer to the research questions in Section 6, whichever direction the answer points.

## 3. Research motivation [E][P]

Autoregressive decoding is usually limited by the key-value (KV) cache. [E] DeepSeek-V2 states that during generation the KV cache of standard multi-head attention becomes the bottleneck for inference efficiency, and that MQA and GQA reduce it at a cost to quality. [E] MLA was introduced to reduce the cache while maintaining, and in its authors' experiments improving, quality relative to standard MHA.

[E] The authors' quality claim comes from their own experiments on 16B and 250B-parameter mixture-of-experts models trained on 1.33T and 420B tokens respectively (their Appendix D.2, Table 9). Those results are not evidence about small dense models trained on small corpora. This is the central gap the present project investigates.

[P] Small models are a useful testbed because they can be trained many times with multiple seeds, which is needed to separate an architectural effect from noise. A result at small scale is not a claim about DeepSeek-scale behaviour.

## 4. Problem statement [P]

Three questions are open for a small GPT-2-style model:

1. Can MLA be integrated into GPT-2's pre-LN decoder without changing anything except the attention block, and still train stably?
2. At matched training conditions, how do quality, parameter count, compute cost and memory compare with the standard attention baseline?
3. Does the cache reduction predicted from the architecture translate into measured memory and latency savings at the context lengths a small model can reach?

## 5. Research objectives [P]

- **O1.** Implement MLA as a standalone, tested module that follows the DeepSeek-V2 formulation, including decoupled RoPE.
- **O2.** Integrate MLA into the repository's GPT-2 transformer as a selectable attention type, leaving the baseline path unchanged.
- **O3.** Build a reproducible data pipeline over `nampdn-ai/tiny-webtext` that supports streaming, bounded subsets and token caching.
- **O4.** Train baseline and MLA models under identical conditions across three experiment tiers.
- **O5.** Measure parameters, latency, memory, KV-cache size, throughput, validation loss, perplexity and generation latency.
- **O6.** Report results honestly, including null results, with the limits of the small-scale setting stated.

## 6. Research questions and hypotheses

### 6.1 Research questions [P]

- **RQ1.** Does MLA train stably at small scale when integrated into GPT-2 without the DeepSeek-scale stabilisers?
- **RQ2.** At matched steps and tokens, how does validation perplexity under MLA compare with standard attention?
- **RQ3.** How much does decoupled RoPE and the retained learned positional embedding interact, and does removing the learned embedding change quality?
- **RQ4.** What are the measured KV-cache, memory and latency effects of MLA, and at what context length do they become material?
- **RQ5.** How sensitive is MLA to its latent dimensions (the compression rank)?

### 6.2 Hypotheses [H]

Each hypothesis states what would count as support. None is assumed true.

- **H1 (cache, deterministic).** The per-token, per-layer cache of MLA is `d_c + d_R^h` elements, against `2 n_h d_h` for MHA. At the proposed configuration (Section 9.4) this is 288 against 1536, which is 18.75 % of baseline. This follows from the architecture and is verified by counting, not by experiment.
- **H2 (quality, equivalence).** Validation perplexity of MLA at the Small tier is within a pre-registered relative margin of the baseline. Proposed margin: ±2 % relative to baseline, measured over three seeds. Support requires the difference to fall inside the margin with a 95 % confidence interval. Rejection requires the interval to lie entirely outside it. Results inside the interval are reported as "no meaningful difference at this scale", not as superiority.
- **H3 (quality, direction).** MLA is *not* expected to outperform baseline at this scale. [P] This is stated to prevent post-hoc reinterpretation. A finding that MLA is better would be reported with the same statistical rigour as a finding that it is worse.
- **H4 (positional ablation).** Removing the learned absolute positional embedding while keeping decoupled RoPE changes validation perplexity by a measurable amount. Direction is unknown.
- **H5 (latency).** At the proposed size, MLA's training forward pass is slower than baseline, because of the additional up-projections, unless fused. Generation latency per token benefits only when the cache is large enough to be memory-bound, which is unlikely at context lengths the Small tier reaches.

## 7. Existing GPT-2 architecture overview [E]

The repository implements a decoder-only transformer with these properties, verified by reading `src/gpt2/modeling/` at the audited commit.

- Token embedding and learned absolute positional embedding, summed at input. The positional table has `seq_len` rows.
- `L` pre-LN transformer blocks. Each block is `x ← x + Attn(LN(x))` then `x ← x + FFN(LN(x))`.
- Attention: four biased linear projections (`Q`, `K`, `V`, output), `h` heads, per-head dimension `d_h = d / h`, scaled dot product, dropout on attention probabilities.
- Feed-forward: `Linear(d, rate·d) → Swish → Dropout → Linear(rate·d, d)`. **Swish, not GELU.**
- Final LayerNorm, then output logits through the transposed token embedding (tied weights).
- Causal masking through a future mask combined with a padding mask.
- KV cache through a `past` list of per-layer tuples; position offset derived from the cache.

Deviations from OpenAI GPT-2 are listed in `REPOSITORY_AUDIT.md` §4. The most important are the Swish activation and the custom tokenizer. Both arms of this study share these deviations.

## 8. Proposed GPT-2 + MLA architecture

### 8.1 Mathematical formulation [E for equations, P for placement]

The equations below follow DeepSeek-V2 §2.1 (Eqs. 9–19) and Appendix C. Notation: `h_t ∈ ℝ^d` is the attention input at position `t`; `n_h` is the number of heads; `d_h` the per-head dimension; `d_c` the KV latent dimension; `d'_c` the query latent dimension; `d_R` the per-head dimension of the decoupled RoPE pathway.

**Query latent compression [E, Eqs. 12–13]**

$$c^Q_t = W^{DQ} h_t, \qquad q^C_t = W^{UQ} c^Q_t$$

with $W^{DQ} \in \mathbb{R}^{d'_c \times d}$ and $W^{UQ} \in \mathbb{R}^{n_h d_h \times d'_c}$.

**KV latent compression [E, Eqs. 9–11]**

$$c^{KV}_t = W^{DKV} h_t, \qquad k^C_t = W^{UK} c^{KV}_t, \qquad v^C_t = W^{UV} c^{KV}_t$$

with $W^{DKV} \in \mathbb{R}^{d_c \times d}$ and $W^{UK}, W^{UV} \in \mathbb{R}^{n_h d_h \times d_c}$. The latent $c^{KV}_t$ is shared across all heads. It is the only per-token quantity that must be cached for the content path.

**Decoupled RoPE [E, Eqs. 14–17]**

$$q^R_t = \mathrm{RoPE}(W^{QR} c^Q_t), \qquad k^R_t = \mathrm{RoPE}(W^{KR} h_t)$$

with $W^{QR} \in \mathbb{R}^{n_h d_R \times d'_c}$ and $W^{KR} \in \mathbb{R}^{d_R \times d}$. The query RoPE component is per head. The key RoPE component is **a single vector shared across heads**, so it costs $d_R$ elements per token rather than $n_h d_R$.

The per-head query and key are concatenations:

$$q_{t,i} = [q^C_{t,i};\, q^R_{t,i}], \qquad k_{t,i} = [k^C_{t,i};\, k^R_t]$$

**Attention [E, Eq. 18]**

$$o_{t,i} = \sum_{j \le t} \mathrm{Softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_R}}\right) v^C_{j,i}$$

Note two details. The scale is $\sqrt{d_h + d_R}$, the full query-key dimension, not $\sqrt{d_h}$. The values are the content values $v^C$ only. Rotary embedding is never applied to values.

**Output [E, Eq. 19]**

$$u_t = W^O [o_{t,1}; \ldots; o_{t,n_h}], \qquad W^O \in \mathbb{R}^{d \times n_h d_h}$$

**Cache [E, §2.1.3]**

At inference the cache holds $c^{KV}_j$ and $k^R_j$ for each past token $j$ and each layer. That is $(d_c + d_R)$ elements per token per layer.

**Absorption [E, §2.1.2 and Appendix C].** During inference $W^{UK}$ can be absorbed into $W^{UQ}$ and $W^{UV}$ into $W^O$, so keys and values never need to be materialised. [P] The absorption is an optimisation. It is not required for correctness and is **out of scope for the initial implementation**, which materialises keys and values for clarity. Absorption is recorded as a possible future benchmark (Section 16).

**Why RoPE is decoupled [E, §2.1.3].** Applying RoPE to $k^C$ would place a position-dependent rotation between $W^{UQ}$ and $W^{UK}$. Those two matrices could then no longer be merged for inference, and the keys for all prefix tokens would need to be recomputed. A separate, small RoPE pathway avoids this.

### 8.2 Components: essential versus optional [P, grounded in E]

| Component | Status | Reason |
|---|---|---|
| KV down-projection $W^{DKV}$ and shared latent $c^{KV}$ | **Essential** | Defines MLA's cache reduction. |
| Key and value up-projections $W^{UK}, W^{UV}$ | **Essential** | Reconstructs per-head content keys and values. |
| Decoupled RoPE pathway $W^{QR}, W^{KR}$ | **Essential** for positional encoding with RoPE | Without it, positional information would be lost or RoPE would break the compression. |
| Shared (single-head) RoPE key $k^R$ | **Essential** to the cache count in H1 | Caching one $d_R$ vector instead of $n_h d_R$ is what keeps the cache small. |
| Query latent compression $c^Q$, $W^{DQ}, W^{UQ}$ | **Optional** | [E] The paper uses it to reduce activation memory during training and states it does not reduce the inference cache. Included in the primary arm for fidelity; removal is an ablation. |
| RMSNorm on latents $c^{KV}$ and $c^Q$ | **Recommended** | [E, §3.1.2] The paper adds RMSNorm after compressed latents and scaling factors at the bottlenecks "to ensure stable training". The paper does not say whether these are needed at small scale. Included in the primary arm; ablation proposed. |
| Absorption of up-projections into $W^{UQ}$, $W^O$ | **Optional, out of scope** | Inference-time optimisation. |
| FP8 weights, KV quantisation, FlashAttention-2 kernels | **Out of scope** | Implementation-specific optimisations of the DeepSeek-V2 system. |
| Mixture-of-experts FFN, device-limited routing, load-balancing losses | **Out of scope** | DeepSeekMoE is a separate contribution. This project keeps the repository's dense FFN. |
| YaRN context extension | **Out of scope** | Not needed for the context lengths in this study. |

Conventional low-rank attention and MLA are different things. Projecting the sequence axis (as in Linformer-style methods) compresses across tokens. MLA compresses the channel axis of the KV representation into a per-token latent and reconstructs per-head keys and values. Calling a generic low-rank projection "MLA" is incorrect and must not appear in code comments or results.

### 8.3 Tensor dimensions at the proposed configuration [P]

Configuration: $B$ batch, $T$ query length, $S$ key length (= past + current), $d = 768$, $n_h = 12$, $d_h = 64$, $d_c = 256$, $d'_c = 256$, $d_R = 32$.

| Tensor | Shape |
|---|---|
| $h_t$ | $(B, T, 768)$ |
| $c^Q$ | $(B, T, 256)$ |
| $c^{KV}$ | $(B, T, 256)$ |
| $q^C$ | $(B, T, 12, 64)$ |
| $q^R$ | $(B, T, 12, 32)$ |
| $k^C$, $v^C$ | $(B, S, 12, 64)$ |
| $k^R$ | $(B, S, 1, 32)$, broadcast over heads |
| $q$ (after concat, transposed) | $(B, 12, T, 96)$ |
| $k$ (after concat, transposed) | $(B, 12, S, 96)$ |
| $v$ (transposed) | $(B, 12, S, 64)$ |
| Attention scores | $(B, 12, T, S)$ |
| Attention output before $W^O$ | $(B, T, 768)$ |
| Cache per layer (content, rope) | $(B, S, 256)$, $(B, S, 32)$ |

### 8.4 Proposed dimension derivation rules [P]

These are defaults, not validated optima. Each is a candidate for the rank sweep in RQ5.

- $d_h = d / n_h$. This keeps the baseline head dimension, so both arms have identical head geometry. With $d = 768$ and $n_h = 12$, $d_h = 64$.
- $d_R = d_h / 2$, which must be even for RoPE pairing. This is the ratio used in DeepSeek-V2 [E, Table 1 caption: $d_R^h = d_h/2$]. Gives $d_R = 32$.
- $d_c = 4 d_h$. This is the ratio used in DeepSeek-V2 [E, Table 1 caption: $d_c = 4 d_h$]. Gives $d_c = 256$.
- $d'_c \approx 0.3 d$, rounded to a multiple of $d_h$. DeepSeek-V2 uses $d'_c = 1536$ with $d = 5120$, a ratio of 0.3 [E]. For $d = 768$, $0.3 d \approx 230$, rounded to 256. The paper gives no rule; this rounding is [P].

[U] The ratio $d_c / d_h = 4$ was chosen by DeepSeek-V2 for a 128-head, 5120-dimensional model. Whether it is appropriate for a 12-head, 768-dimensional model is untested. The sweep in RQ5 covers $d_c \in \{2, 4, 6\} \times d_h$.

### 8.5 Parameter and cache accounting at the proposed configuration [P, verified arithmetic]

Per attention block, counting biases on all linear layers and two RMSNorm gain vectors:

| Quantity | Baseline MHA | MLA |
|---|---:|---:|
| Attention parameters per layer | 2,362,368 | 1,700,256 |
| FFN parameters per layer (unchanged) | 4,722,432 | 4,722,432 |
| Total parameters, 12 layers, vocab 50,257, `seq_len` 256 | 123,849,984 | 115,904,640 |
| Attention parameter ratio | 1.000 | 0.720 |
| Cache elements per token per layer | 1,536 | 288 |
| Cache ratio | 1.000 | 0.1875 (81.25 % reduction) |
| Cache bytes per token, all layers, fp16 | 36,864 B | 6,912 B |

Interpretation [P]: the cache reduction is large, but the total parameter reduction is only 6.4 % because the vocabulary embedding dominates parameter count at this scale. The report must state both. Comparing "parameters" without this context would mislead.

[U] Parameter-matched arm. MLA is smaller by about 7.9 M parameters. A fair alternative equalises parameter count by widening the FFN in the MLA arm. Recommended: report the matched-compute comparison as primary, and the parameter-matched comparison as a secondary ablation. Decision D-M1.

### 8.6 Positional embedding decision [P, with unresolved status]

- **Baseline:** learned absolute positional embedding added to the input. This is GPT-2's design.
- **MLA:** RoPE is applied inside attention to the decoupled pathway only.

GPT-2 retains learned positions by default, so the MLA arm would carry both absolute and rotary position information. This is not invalid, but the two sources overlap. The study must separate their effects.

- **Primary MLA arm (D-P1, recommended):** keep the learned positional embedding. This changes one thing relative to the baseline, which is the attention block.
- **Ablation arm:** remove the learned positional embedding. Rotary positions are the only positional signal. This tests RQ3 and H4.

Consequence for generalisation [P]: a learned positional table of `seq_len` rows cannot be used beyond `seq_len`. The ablation arm has no such table, so it could be evaluated at longer contexts. Whether RoPE extrapolates well at this scale is unknown, and this is not claimed.

### 8.7 KV cache semantics [P]

- The cache stores $c^{KV}$ (shape $(B, S, d_c)$) and $k^R$ (shape $(B, S, d_R)$) for every layer, after RoPE is applied to $k^R$. Storing rotated keys is correct because a cached token's absolute position never changes.
- The repository's `Transformer` reads the position offset from `past[0][0].size(-2)`. The MLA cache's first element therefore has shape $(B, S, d_c)$, so the sequence axis stays at `-2`.
- The first cache element is the content latent, not keys and values. This is a semantic change the repository code does not expect, so the offset logic must be tested explicitly.

### 8.8 Scaled dot-product attention [U]

[E] PyTorch's `F.scaled_dot_product_attention` accepts a custom `scale` and a boolean `attn_mask`. [U] Two things are not verified and must be tested in Phase 1: (a) whether the fused backends accept query and key dimension 96 with value dimension 64, and (b) whether the boolean mask convention is applied as expected. The repository's convention is True = masked, which is the opposite of SDPA's. Recommended default: an explicit matmul implementation (the repository's own style) as the reference, and SDPA as a checked optimisation. Decision D-S1.

### 8.9 Weight initialisation [P]

The repository initialises token and positional embeddings with $N(0, 0.02)$ and leaves other linear layers at PyTorch defaults. No residual-path scaling is applied. MLA projections will use the same convention as the baseline's linear layers. The DeepSeek-V2 standard deviation of 0.006 [E, §3.1.2] applies to a much larger model and is not adopted. Decision D-I1.

### 8.10 Reusable pretrained weights [E, P]

- [E] The repository ships no baseline checkpoint. Its notebooks download checkpoints from a placeholder GCS bucket and from release assets that use a custom Korean vocabulary.
- [E] The repository contains no code to load OpenAI GPT-2 weights. Loading them would require a converter, a GELU-to-Swish decision, and a vocabulary change.
- [P] Conclusion: **no pretrained weights are reused.** Both arms train from scratch under identical conditions. This removes a confound: an MLA model initialised from a standard-attention checkpoint would inherit learned structure that the study is meant to measure.
- [U] Optional warm-start experiment (not in scope for v1): initialise $W^{UK}$ and $W^{UV}$ from a truncated SVD of a trained baseline's key and value projections. Recorded as future work.

## 9. Experiment design and proposed hyperparameters

### 9.1 Matching principle [P]

Both arms must match on: tokenizer and vocabulary, dataset split and order seed, sequence length, batch size in tokens, number of optimiser steps, learning-rate peak and schedule, weight decay, dropout, gradient clipping, numeric precision, hardware, and random seed set. The only difference is the attention block, plus the positional-embedding ablation where explicitly labelled.

### 9.2 Baseline definition [P, D-B0]

The baseline is the repository's own GPT-2 implementation as audited, including its Swish activation, its tied embeddings and its pre-LN blocks. It is **not** an OpenAI-exact GPT-2. Justification: any deviation from the repository baseline would make the comparison a comparison of two codebases. The deviations are documented and shared by both arms.

### 9.3 Experiment tiers

All values are configuration parameters, not constants.

| Tier | Records | Sequence length | Steps | Purpose |
|---|---|---|---|---|
| Smoke | 50–100 | 64 | 50–100 | Checks that code runs end to end. Not for results. |
| Development | ~1,000 | 128 | 1,000–2,000 | Validates architecture and training stability. |
| Small | ~5,000–10,000 | 256 | Set from measured token budget (see §9.5) | Comparative evaluation with 3 seeds. |

The record counts are approximate and their token equivalent is unknown until the dataset is inspected (see §11, U-D1).

### 9.4 Proposed model size [P]

Primary size: $L = 12$, $d = 768$, $n_h = 12$, $d_h = 64$, rate 4. This matches GPT-2 small in depth, width and heads, with the repository's Swish activation and custom vocabulary. The total baseline parameter count with the GPT-2 BPE vocabulary (50,257) and `seq_len` 256 is about 123.8 M (123,849,984, §8.5). At the GPT-2 context length of 1024 it is about 124.4 M, because the learned positional table grows by 0.59 M parameters.

[U] The repository's default size (12 layers, 16 heads, 1,024 dimensions) is about 1.5 times larger. Using it would make the study heavier without obvious benefit. Decision D-SZ1: adopt GPT-2 small dimensions for the Small tier, and keep the repository default as an optional configuration.

### 9.5 Token budget [P]

Token budget is defined as $\text{steps} \times \text{batch size} \times \text{seq\_len}$, and is fixed across arms. The Small tier's step count is chosen so that total tokens are at least the dataset's usable token count times the number of epochs the study intends. This is computed after measuring the tokenized subset, not assumed. The number of epochs must be reported, because a small dataset repeated many times produces memorisation that masks architectural differences.

### 9.6 Optimiser and schedule [P, D-L1]

The repository's recipe (AdamW, linear decay to zero, no warm-up, no clipping) is not adequate as a stable recipe for both arms. Proposed for both arms:

- AdamW with $\beta = (0.9, 0.95)$ and weight decay 0.1 on matrix parameters only, zero on biases, norms and embeddings. [E] DeepSeek-V2 uses $\beta = (0.9, 0.95)$ and weight decay 0.1; the parameter-group split is [P].
- Peak learning rate chosen by a small sweep on the baseline only, then reused for MLA. Sweep grid to be defined in `AGENTS.md`.
- Linear warm-up over the first 5 % of steps, then cosine or linear decay to 10 % of peak. [P]
- Gradient clipping at global norm 1.0. [E] DeepSeek-V2 uses 1.0.

[U] Using the same LR for both arms is the matching rule. The LR may be favourable to one arm. The sweep is run on the baseline and a separate sweep on MLA is an optional robustness check.

### 9.7 Dropout [P]

Dropout 0.1 on residual, embedding and attention probabilities, matching the repository default. Evaluation is always in eval mode with dropout disabled.

## 10. Dataset description and limitations

### 10.1 Dataset [E]

Source: `https://huggingface.co/datasets/nampdn-ai/tiny-webtext`, identifier `nampdn-ai/tiny-webtext`.

- [E] The card reports 2.32 million rows, English text, total download size of 1.26 GB, parquet format, and an MIT licence.
- [E] The card states the data is "augmented using subset of Falcon-RefinedWeb", with additional text generated by `falcon-7b-instruct` "to improve the quality and diversity of the texts". The dataset therefore contains **model-generated text**.
- [E] The card states the goal is to help models "learn about perception on web text while neutralizing the bias of the source text using critical thinking methods". This is an unusual framing for a pretraining corpus, and its effect on the study is unknown.
- [E] **Access is gated.** The page states users "have to accept the conditions to access its files and content" and requires a Hugging Face login.
- [U] **Schema not verified.** The column names, text field name and per-record token statistics were not retrieved. U-D1: the Data Engineering agent must stream one record and record the schema in the implementation log before the loader is written.

### 10.2 Limitations [P]

- **Synthetic text.** Part of the corpus is generated by a 7B instruction model. The study therefore measures modelling of a mixture of web and synthetic text, which may not transfer to natural web text.
- **Small size.** At the Small tier the corpus is a few million tokens at most, far smaller than the data a 124 M-parameter model is designed for. Models will be data-limited, and comparative conclusions apply to this regime only.
- **Domain.** Content is general English web text. Results are not evidence about code, mathematics or multilingual behaviour.
- **Upstream provenance.** Terms for the RefinedWeb subset and the generated augmentation were not reviewed in this stage. Redistribution of derived artefacts (tokenised shards) must be checked against those terms before any artefact is shared. U-D2.
- **Splits.** The card does not describe a validation split. The project creates one deterministically (§10.3).

### 10.3 Data pipeline requirements [P]

Functional requirements, detailed in §17:

- Streaming through `datasets.load_dataset(..., streaming=True)` using an access token from the environment. Never a committed token.
- Bounded subsets by record count and by token count, both configurable.
- Tokenisation with the GPT-2 byte-level BPE (vocabulary 50,257). [U] See D-T1 below.
- Batched encoding.
- Caching of tokenised shards to disk as `uint16` arrays (`uint16` holds every GPT-2 token id), with a manifest recording dataset revision, record count, token count, tokenizer identifier and a hash of the configuration.
- Fixed-length sequence packing. Document separator is the `<|endoftext|>` token, id 50256. [P]
- Deterministic train and validation split by hashing the document index, so that no document appears in both.
- Reproducibility: the same configuration produces byte-identical shards.

### 10.4 Decision D-T1: tokenizer [U]

The repository tokenizer is a custom WordPiece-style vocabulary, and the PRD's requirement is GPT-2 tokenizer compatibility. These conflict.

- **Option A (recommended):** GPT-2 byte-level BPE, vocabulary 50,257, through a tokenizer library. The new data path uses it. The repository's custom path is left intact and is not used by the MLA experiments. Strongest fidelity to "GPT-2".
- **Option B:** Reuse the repository's custom vocabulary. Requires an external vocabulary build, which the repository does not provide.

Default: Option A. This requires a new dependency (`tiktoken` or `tokenizers`) and network access on first use. U-D3: the implementation team confirms this.

### 10.5 Decision D-D1: packing and document boundaries [U]

Packing concatenates documents into fixed-length sequences. Attention can then cross document boundaries.

- **Option A (recommended for v1):** Concatenate with `<|endoftext|>` separators and allow cross-document attention. This is how GPT-2 pretraining is usually done. Simple and standard.
- **Option B:** Mask attention across document boundaries. More correct, more code, and interacts with the cache.

Default: Option A. This is recorded as a limitation.

## 11. Training pipeline requirements

- Device-agnostic training: CPU for smoke tests, CUDA for real runs. The baseline path is not changed. Device selection is a configuration option.
- Mixed precision with `torch.autocast` (bfloat16 where supported, otherwise float16 with a gradient scaler). Apex is not required.
- Gradient accumulation to reach a target token batch on memory-limited hardware. Accumulated and non-accumulated runs must produce the same update for the same token batch, which is a test.
- Gradient clipping at global norm 1.0, logged as a metric.
- Warm-up and decay as in §9.6.
- Logging of training loss, validation loss, perplexity, learning rate, gradient norm and tokens per second, every configurable interval, to a structured file (JSON lines), not only the progress bar.
- Fixed validation: each evaluation covers the full validation split, not a single batch.
- Seeds set for Python, NumPy and torch, including CUDA.

## 12. Checkpoint and recovery requirements

- A checkpoint is written every `save_steps` and at the end of training.
- It contains: model state, optimiser state, scheduler state, step, RNG state for Python, NumPy, torch CPU and all CUDA devices, the data cursor (shard index and in-shard offset), the metric history, the scaler state if used, and a serialised configuration with a schema version.
- Writes are atomic: write to a temporary file and rename.
- **Exact resume:** a run interrupted at step *k* and resumed from its checkpoint must produce the same parameters at step *N* as an uninterrupted run, within floating-point nondeterminism. This is tested with a small configuration by comparing parameter hashes. [P] Where bit-exact equality is impossible on CUDA because of nondeterministic kernels, the test uses tolerance and records the source of nondeterminism.
- Loading a checkpoint verifies the configuration fingerprint and fails loudly on a mismatch, rather than silently truncating positional embeddings.
- Backward compatibility: the repository's existing checkpoint format remains loadable by the existing `evaluate` and `generate` commands.

## 13. Inference requirements

- Autoregressive generation with the KV cache for the MLA arm, including the cache semantics in §8.7.
- **Cache equivalence test:** for a fixed input, the logits from incremental decoding with cache must match the logits of a full forward pass over the same prefix, within a stated tolerance, for every position.
- Top-p sampling with configurable nucleus probability and a fixed random seed for reproducible samples.
- Generation stops at `<|endoftext|>` or at the maximum length, whichever comes first. This is a change from the repository, which runs to `seq_len`, and it is recorded as such.
- Prompt longer than the context window is handled explicitly: truncated from the left, with a warning. The baseline and MLA arms use the same rule.
- Generation latency is measured as time to first token and time per output token, separately.

## 14. Baseline comparison strategy

- Both arms are trained for the same token budget, data order, seeds and hardware.
- Three seeds per arm for the Small tier. [P] Three is the minimum for an interval estimate. More seeds are better if budget allows.
- Learning-rate sweep on the baseline only, with the chosen rate applied to both arms.
- The primary comparison is at matched tokens and steps.
- The secondary comparison is parameter-matched (D-M1), if run.
- Ablations, each run on the Small tier with three seeds where feasible: no learned positional embedding (H4); no query compression $c^Q$; no RMSNorm on latents; latent rank sweep ($d_c \in \{2, 4, 6\} d_h$).
- Reported statistics: mean and standard deviation across seeds, and a 95 % confidence interval on the difference in validation perplexity.

## 15. Evaluation metrics

| Category | Metric | Definition | Notes |
|---|---|---|---|
| Size | Total parameters | Sum of all trainable parameter elements | Report with and without embeddings. |
| Size | Attention parameters per layer | Parameters in the attention block only | Verified by count. |
| Memory | KV-cache elements per token | Cache elements per token, all layers | Counted analytically and verified by allocation. |
| Memory | KV-cache bytes at context *T* | Bytes at fp16 for *T* tokens, batch 1 | Measured across *T*. |
| Memory | Peak GPU memory, forward and backward | `torch.cuda.max_memory_allocated` | Reset between runs. |
| Latency | Forward-pass latency | Milliseconds per batch, median over repeats after warm-up | Excludes data loading. |
| Latency | Backward-pass latency | As forward, for the backward pass | |
| Throughput | Training tokens per second | Tokens per second over steady state | Steady state excludes the first steps. |
| Quality | Validation loss | Mean token cross-entropy on the full validation split | Token-weighted. |
| Quality | Validation perplexity | $\exp$(token-weighted mean validation NLL over the whole split), not a mean of per-batch perplexities | Same. |
| Generation | Time to first token | Prompt processing time | |
| Generation | Time per output token | Average over the generated tokens | Measured at several context lengths. |

## 16. Hardware assumptions [U]

- [U] Hardware is not specified in the request. U-H1: the user confirms the GPU model and VRAM.
- Working assumption: one NVIDIA GPU with at least 16 GB of memory for the Small tier, and CPU for smoke tests.
- Precision assumption: bfloat16 if the GPU supports it, otherwise float16 with loss scaling. Recorded per run.
- Multi-GPU training is out of scope. The repository's distributed path is not used.
- The benchmark records the GPU model, driver, CUDA and PyTorch versions for every result.

## 17. Functional requirements

Each requirement has an identifier for traceability.

- **FR-01.** Standalone MLA module implementing Eqs. 9–19 of DeepSeek-V2, with configurable $d$, $n_h$, $d_h$, $d_c$, $d'_c$, $d_R$, and options to disable $c^Q$ and the latent RMSNorm.
- **FR-02.** Decoupled RoPE implementation for the $d_R$ dimensions, with configurable base frequency (default 10000 [P]).
- **FR-03.** Attention type selectable as `mha` or `mla`, default `mha`, in both the transformer block and the full model.
- **FR-04.** Learned positional embedding optionally disabled, for the ablation.
- **FR-05.** Causal masking applied to both arms, verified by test.
- **FR-06.** KV cache for MLA with the semantics of §8.7, compatible with the repository's position-offset convention.
- **FR-07.** Streaming dataset loader with bounded subsets, token-count limits, batched GPT-2 BPE tokenisation, sharded `uint16` caching, packing and deterministic split.
- **FR-08.** Research training entry point with the features of §11.
- **FR-09.** Checkpointing and exact resume as in §12.
- **FR-10.** Full-split validation with perplexity.
- **FR-11.** Inference with KV cache, top-p sampling and the stopping rules in §13.
- **FR-12.** Benchmark tool measuring §15 metrics, with environment capture.
- **FR-13.** Configuration files for the three tiers, plus the ablation configurations.
- **FR-14.** Results written to structured files, with the configuration and dataset manifest hash for each run.
- **FR-15.** The original `train`, `evaluate`, `generate` and `visualize` commands continue to work unchanged.

## 18. Non-functional requirements

- **NFR-01. Correctness first.** The reference attention is the explicit implementation. Optimised kernels are accepted only after numerical agreement with the reference.
- **NFR-02. Reproducibility.** A result can be regenerated from the configuration, the dataset manifest, the seed and the recorded environment.
- **NFR-03. Honesty of reporting.** Results are reported as measured. Negative and null results are reported with the same prominence. No experimental outcome is written before it is measured.
- **NFR-04. Minimal disruption.** Existing modules are changed only additively. The baseline path produces identical outputs before and after changes, verified by test.
- **NFR-05. Resource bounds.** The smoke test completes in under five minutes on CPU. Development runs complete on a single GPU within a day. [P] These are targets, not measured values.
- **NFR-06. Security.** No credentials in the repository or in logs. The Hugging Face token is read from the environment.
- **NFR-07. Observability.** Every run writes a log with its configuration, environment and metric history.
- **NFR-08. Licensing.** The project's own code remains under the repository's Apache-2.0 licence. Dataset terms are respected (see §10.2).

## 19. Technical constraints

- Python and PyTorch versions are pinned in `requirements-research.txt`. [U] The pinned versions are chosen in Phase 0 and recorded.
- The repository's CI targets Python 3.6 to 3.8. The research code cannot run on those versions with modern PyTorch. [E, from audit] Decision D-C1: research code targets a current Python, and the CI matrix is updated only with approval.
- No apex dependency for the new code.
- No modification of the existing `TokenizedCorpus` class.
- Attention mask convention must be converted explicitly at the boundary to any SDPA call.
- The MLA cache's first element must keep the sequence axis at dimension `-2`.
- New dependencies: a tokenizer library and `datasets`, both listed in `requirements-research.txt`.

## 20. Risks and mitigation strategies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Tokenizer mismatch leaves the study unable to claim GPT-2 fidelity | High if unresolved | High | D-T1 resolved before Phase 2. |
| Gated dataset access blocked or revoked | Medium | High | Access verified in Phase 0. Smoke tests use a small public fallback only if approved. |
| Dataset schema differs from expectation | Medium | Medium | Inspect one record before writing the loader (U-D1). |
| Training is unstable without warm-up and clipping | Medium | Medium | Both added to both arms; stability reported. |
| Silent mask inversion produces attention to the wrong positions | Medium | High | Causal invariance test and mask helper with a single conversion point. |
| Cache offset error produces silently wrong generation | Medium | High | Cache-equivalence test against full forward pass. |
| Results dominated by data-limited regime | High | Medium | Stated as a limitation; token budget reported; epochs reported. |
| Seed variance hides a real difference, or creates a false one | Medium | High | Three seeds minimum; confidence intervals; pre-registered margin (H2). |
| Benchmark timing noise from thermal or shared hardware | Medium | Medium | Warm-up, repeats, median, environment capture. |
| Absorption or kernel optimisations introduce numerical drift | Low (not in scope) | Medium | Explicit reference path retained. |
| Synthetic text in the corpus biases conclusions | High | Medium | Stated as a limitation. Optional rerun on a natural-text subset if one is identified. |
| Overclaiming MLA's benefit | Medium | High | H3 is written in advance. Results language is restricted to what is measured. |

## 21. Expected deliverables

1. Standalone MLA module with unit tests (FR-01, FR-02).
2. Integrated transformer with selectable attention (FR-03 to FR-06).
3. Research data pipeline with cached shards and manifest (FR-07).
4. Research training, checkpointing and resume (FR-08, FR-09).
5. Inference with KV cache and cache-equivalence test (FR-11).
6. Benchmark tool and environment capture (FR-12).
7. Configurations for three tiers and ablations (FR-13).
8. Experimental results in structured form, with analysis.
9. Documentation: this PRD, `AGENTS.md`, `REPOSITORY_AUDIT.md`, `IMPLEMENTATION_PLAN.md`, the experiment log, and a short results report.
10. A written list of limitations, unresolved questions and negative findings.

## 22. Acceptance criteria

- **AC-01.** All unit tests pass, including the causal invariance test, the cache-equivalence test, and the baseline-unchanged test.
- **AC-02.** The smoke test completes end to end on CPU, covering data, training, checkpoint, resume, evaluation and generation, for both arms.
- **AC-03.** A Development run completes for both arms on a single GPU with logged metrics.
- **AC-04.** The Small tier produces three seeds per arm with the full metric set in §15.
- **AC-05.** Cache-equivalence holds within tolerance for every generated position in a test suite of at least five prompts.
- **AC-06.** Parameter counts and cache counts match the analytic values in §8.5, to the element.
- **AC-07.** Resume reproduces the uninterrupted run's parameters within the stated tolerance.
- **AC-08.** Every reported number traces to a log file with its configuration hash and environment.
- **AC-09.** The report states H1 through H5 with their outcomes and the confidence intervals, without claiming findings beyond the measured results.
- **AC-10.** All unresolved decisions in §23 are either resolved with a recorded rationale or explicitly carried forward as limitations.

## 23. Unresolved decisions register

| ID | Question | Recommended default | Owner | Needed by |
|---|---|---|---|---|
| D-T1 | Tokenizer: GPT-2 BPE or repository vocabulary | GPT-2 BPE | User | Phase 3 |
| D-P1 | Keep learned positional embedding in primary MLA arm | Keep | Lead | Phase 2 |
| D-D1 | Packing: cross-document attention allowed | Allowed, recorded | Lead | Phase 3 |
| D-S1 | Attention implementation: explicit or SDPA | Explicit reference, SDPA after check | MLA agent | Phase 1 |
| D-M1 | Parameter-matched comparison | Secondary | Lead | Phase 6 |
| D-SZ1 | Model size | GPT-2 small dimensions | User | Phase 2 |
| D-L1 | LR schedule and warm-up | Warm-up 5 %, decay to 10 % | Training agent | Phase 4 |
| D-I1 | Initialisation std for MLA projections | 0.02, matching baseline | MLA agent | Phase 2 |
| D-C1 | Python and PyTorch versions; CI matrix | Current Python; CI unchanged pending approval | User | Phase 0 |
| D-B0 | Baseline definition | Repository implementation | Lead | Phase 0 |
| D-H1 | Hardware | Single NVIDIA GPU, 16 GB or more | User | Phase 0 |
| D-E1 | Evaluation: full split or sampled | Full split | Research agent | Phase 4 |
| U-D1 | Dataset column schema | To be inspected | Data agent | Phase 3 |
| U-D2 | Upstream licence terms for derived artefacts | To be reviewed | User | Before sharing artefacts |
| U-D3 | Tokenizer library choice | `tiktoken` or `tokenizers` | Data agent | Phase 3 |
| U-H1 | GPU model and VRAM | Not provided | User | Phase 0 |
| U-E1 | Dataset access approval on the user's account | Not checked | User | Phase 0 |

## 24. Open research questions beyond v1

These are the questions the implementation team should weigh when reporting, because they determine whether the project is a meaningful study or only a code change.

- **Is the comparison informative at this scale?** A null result at small scale is weak evidence about large scale. The report must say so.
- **What is the right comparison?** Matched steps, matched tokens, matched parameters and matched compute give different answers. The primary choice is stated before the runs.
- **Does the cache reduction matter at the lengths reachable here?** At 256 tokens the cache is small in absolute terms, so the memory benefit is modest. The benchmark should show the length at which it becomes material, which is a more useful result than a single number.
- **Does the positional ablation expose a real effect?** H4 is the most direct test of the RoPE and absolute-position interaction the task asks about.
- **Can MLA be evaluated on longer contexts?** Only the ablation arm can be run beyond the training context. Whether RoPE extrapolates well here is unknown and must not be assumed.
- **Does the rank sweep show a plateau?** If quality is flat across $d_c$, the latent is over-provisioned at this scale. If it falls sharply below some rank, that is a finding about the minimum latent size.

## 25. Acceptance of this PRD

This PRD is a specification. It authorises Phase 0 of `IMPLEMENTATION_PLAN.md`. It does not authorise changes to model code. Changes to the decisions in §23 require a written update to this document before implementation proceeds past the phase that depends on them.
