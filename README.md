# GPT-2 with Multi-Head Latent Attention (MLA)

A PyTorch implementation of GPT-2 in which the standard multi-head self-attention can be
swapped, per model, for **Multi-Head Latent Attention (MLA)** in the DeepSeek-V2 style —
compressing the KV cache to **18.75%** of the baseline size while keeping the baseline
path bit-identical.

The base model is the [`affjljoo3581/GPT2`](https://github.com/affjljoo3581/GPT2) PyTorch
implementation at commit `7b159669f6225231ee75f5ea110c1be69f4ce8ef`. The multi-head
self-attention block inside each transformer layer is the only architecture modified;
everything else (token embeddings, learned positional embeddings, pre-LN blocks, tied
output head, causal masking) is preserved.

## What is implemented

| Area | Status |
|---|---|
| MLA module (`src/gpt2/modeling/latent_attention.py`) — shared KV latent `c^KV`, decoupled RoPE keys `k^R`, content-only values, scale `1/sqrt(d_h + d_R)` (DeepSeek-V2 Eqs. 9–19) | Done |
| Rotary position embeddings, GPT-NeoX-style (`src/gpt2/modeling/rotary.py`) | Done |
| Additive attention selection: `Transformer(..., attention='mha' | 'mla')`, default `'mha'` | Done |
| Optional positional-embedding flag (for the no-PE ablation) | Done |
| Research training: autocast, gradient accumulation, warmup/decay LR, grad clipping, JSONL logging, atomic checkpoints with **exact resume**, non-finite-loss stop | Done |
| Data pipeline: repository vocab, sharded `uint16` cache + manifest with SHA-256 verification, deterministic splits, resumable cursor | Done |
| Cached generation with top-p sampling; cache vs full-recompute equivalence verified | Done |
| Benchmark tool: environment capture, median/min/max repeats, peak memory (CUDA), TTFT vs per-token | Done |

**Verified properties** (test suite: **96 passed, 1 skipped** — the skip is a CUDA-only
peak-memory test on a CPU-only dev machine):

- Baseline unchanged: with `attention='mha'`, outputs and parameter count are bit-identical
  to the unmodified repository, verified against a captured reference fixture (T-B1).
- Causal and padding invariance for both attention arms (T-C1, T-C2).
- KV-cache equivalence for incremental decoding, both arms, 5 prompts (T-K1).
- Exact checkpoint resume (T-R1) and gradient-accumulation equivalence (TR-2).
- Parameter counts match the PRD analytic values to the element (T-P1):
  baseline **123,849,984** vs MLA **115,904,640** parameters (GPT-2 small dims, vocab 50,257, seq_len 256).
- KV cache per token per layer: **288** elements (MLA: 256 latent + 32 RoPE) vs **1536**
  (MHA: 2 × 12 heads × 64 dims) = **0.1875×**.

## Repository structure

```
GPT2/
├── src/gpt2/
│   ├── modeling/        # transformer, attention (mha), latent_attention (mla), rotary
│   ├── data/            # vocab, tokenizer, corpus, tokenised corpus pipeline
│   ├── research/        # research config, training, generation, benchmarking
│   └── ...              # baseline train / evaluate / generate / visualize commands
├── tests/               # unit + integration tests, baseline reference fixture
├── configs/             # smoke / development run configs for both arms + no-PE ablation
├── docs/                # decision register, architecture notes, dataset docs
├── experiments/LOG.md   # append-only implementation log (per phase)
├── scripts/             # baseline fixture capture tool
├── PRD.md               # product requirements (hypotheses H1–H5, specs)
├── IMPLEMENTATION_PLAN.md
├── AGENTS.md            # execution contract
└── REPOSITORY_AUDIT.md  # audit of the base repository
```

## Getting started

```bash
python -m venv venv-research
./venv-research/bin/pip install -r requirements-research.txt

# run the test suite (from the GPT2/ directory)
PYTHONPATH=src ./venv-research/bin/python -m pytest tests -q
```

Requirements: Python 3.12, PyTorch 2.15 (CPU build pinned in `requirements-research.txt`;
CUDA builds of the same version work — the device is chosen by configuration, not import).

## Pending work

The implementation, tests and tooling are complete. What remains is **Phase 6 — the
experimental runs** (Development/Small tiers, 3 seeds, LR sweep, no-PE ablation) and the
final report. Configs are ready in `configs/`. Blocked on:

1. **CUDA GPU (16 GB+)** — the current development machine is CPU-only; the tier
   configurations need a GPU for reasonable run times. All code is device-agnostic
   (CPU/GPU auto-selected from configuration).
2. **`HF_TOKEN`** — access to the gated dataset `nampdn-ai/tiny-webtext`, plus the
   one-record schema inspection required before wiring the real data loader
   (the streaming seam is already in place).
3. **A vocabulary file** — the repository ships no vocab file; a real one is needed
   before training on real data.

Benchmarks and experimental numbers are intentionally **not** in this README yet —
per the project's no-fabrication rule (AGENTS.md R-9), results are added only after
they are measured on the target hardware. See `experiments/LOG.md` for the full
phase-by-phase history and `docs/decisions.md` for every design decision.

## License

Apache-2.0 (inherited from the base implementation). See [LICENSE](LICENSE).
