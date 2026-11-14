# Architecture notes (GPT-2 baseline + MLA)

## Baseline (unmodified, audited commit 7b15966)

- `Transformer` (src/gpt2/modeling/transformer.py): token embedding +
  learned absolute positional embedding (table of `seq_len` rows), L pre-LN
  blocks, final LayerNorm, tied output projection through the token
  embedding matrix transpose.
- `TransformerLayer`: `x ← x + Attn(LN(x))`, then `x ← x + FFN(LN(x))`.
- `AttentionLayer` (attention.py): four biased linear projections
  (q, k, v, output), `MultiHeadAttention` with scale `1/sqrt(d_h)` and an
  additive `-1e4` on masked positions; mask convention **True = masked**.
- Cache: per-layer tuple `(k, v)`, each `(batch, heads, kv_len, head_dim)`;
  keys/values are concatenated with the past at `-2`. The position offset is
  read as `past[0][0].size(-2)`.
- Masking (masking.py): `PadMasking` + `FutureMasking` (causal), shapes
  `(batch, q_len, kv_len)` broadcast over the head axis inside attention
  (`unsqueeze(-3)`).
- Feed-forward: `Linear(d, 4d) → Swish → Dropout → Linear(4d, d)` (Swish,
  not GELU — deviation from OpenAI GPT-2 shared by both arms).
- Positional table is resized on load (`_load_from_state_dict`), so shorter
  checkpoints load into longer models by extending the table.

## MLA cache integration contract (Phase 2)

- `Transformer` reads the offset from `past[0][0].size(-2)`. For MLA the
  per-layer cache tuple must therefore be `(c_kv, k_rope)` with shapes
  `(batch, kv_len, d_c)` and `(batch, kv_len, d_R)`: the first element keeps
  the sequence axis at `-2` (A-7). This is a semantic change of the tuple
  contents only, not of the offset logic.
- MLA `forward(h, past, mask)` returns `(output, present)` like
  `AttentionLayer`; the layer's `past` flows through `TransformerLayer`
  unchanged.
- The mask produced by `PadMasking`/`FutureMasking` applies unchanged; MLA
  uses the same True = masked convention and the same additive `-1e4`.

## MLA module design (Phase 1)

- Eqs. 9–19 of DeepSeek-V2 (PRD §8.1). Projections: W^DKV (d→d_c), W^UK and
  W^UV (d_c→d), W^DQ (d→d'_c, optional), W^UQ (d'_c→d, optional), W^QR
  (d'_c→h·d_R), W^KR (d→d_R), W^O (d→d). Biases on all linears, matching the
  baseline. RMSNorm on c^KV and c^Q when enabled.
- Decoupled RoPE: cos/sin tables for d_R dims (base 10000), applied to
  per-head q^R and the single shared k^R. Scale of attention scores is
  `1/sqrt(d_h + d_R)`. Values are content values only.
- Cache after RoPE on k^R (a cached token's absolute position is fixed, so
  storing rotated keys is correct).
- Reference path is explicit matmul (NFR-01). SDPA only after T-S1.
