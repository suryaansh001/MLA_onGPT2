import math
import torch
import torch.nn as nn
from gpt2.modeling.attention import Past
from gpt2.modeling.rotary import apply_rotary_embedding
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root-mean-square normalisation with a learnable gain, used on the
    compressed latents (c^KV and c^Q) as in DeepSeek-V2 §3.1.2.

    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x * scale)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2 (§2.1, Eqs. 9-19).

    The key and value content of every head is compressed into a single
    shared per-token latent c^KV (channel-axis compression of the KV
    representation), from which per-head content keys and values are
    reconstructed with up-projections. Positional information travels
    through a decoupled RoPE pathway: per-head queries q^R and a single
    shared key k^R, both rotated. Attention scores use the full concatenated
    query-key dimension (scale 1/sqrt(d_h + d_R)) and attend to the content
    values only; RoPE is never applied to values.

    At inference the cache stores (c^KV, k^R_rotated) per layer; the first
    element keeps the sequence axis at dimension -2, matching the
    repository's position-offset convention in `Transformer`.

    Tensor          Type            Shape
    ===========================================================================
    h               float           (..., seq_len, dims)
    past (*)        float           (..., past_len, kv_latent_dims),
                                    (..., past_len, rope_head_dims)
    mask            bool            (..., seq_len, past_len + seq_len),
                                    True = masked
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., past_len + seq_len, kv_latent_dims),
                                    (..., past_len + seq_len, rope_head_dims)
    ===========================================================================
    """
    def __init__(self,
                 heads: int,
                 dims: int,
                 dropout: float = 0.1,
                 head_dims: Optional[int] = None,
                 kv_latent_dims: int = 256,
                 query_latent_dims: int = 256,
                 rope_head_dims: int = 32,
                 use_query_latent: bool = True,
                 use_latent_norm: bool = True):
        super().__init__()
        if dims % heads != 0:
            raise ValueError(
                f'dims: must be divisible by heads '
                f'(dims={dims}, heads={heads})')

        self.heads = heads
        self.dims = dims
        self.head_dims = head_dims if head_dims is not None else dims // heads
        self.kv_latent_dims = kv_latent_dims
        self.query_latent_dims = query_latent_dims
        self.rope_head_dims = rope_head_dims
        self.use_query_latent = use_query_latent
        self.use_latent_norm = use_latent_norm

        if rope_head_dims % 2 != 0:
            raise ValueError(
                f'rope_head_dims: RoPE key dimension must be even '
                f'(multiple of 2), got {rope_head_dims}')
        if rope_head_dims <= 0:
            raise ValueError(
                f'rope_head_dims: must be positive, got {rope_head_dims}')
        if kv_latent_dims <= 0:
            raise ValueError(
                f'kv_latent_dims: must be positive, got {kv_latent_dims}')
        if use_query_latent and query_latent_dims <= 0:
            raise ValueError(
                f'query_latent_dims: must be positive when the query latent '
                f'is enabled, got {query_latent_dims}')

        # KV compression: shared latent c^KV and per-head reconstruction.
        self.proj_kv_down = nn.Linear(dims, kv_latent_dims)          # W^DKV
        self.proj_k_up = nn.Linear(kv_latent_dims, heads * self.head_dims)   # W^UK
        self.proj_v_up = nn.Linear(kv_latent_dims, heads * self.head_dims)   # W^UV

        # Decoupled RoPE pathway: per-head queries and one shared key.
        self.proj_rope_k = nn.Linear(dims, rope_head_dims)           # W^KR
        self.proj_rope_q = nn.Linear(
            query_latent_dims if use_query_latent else dims,
            heads * rope_head_dims)                                  # W^QR

        # Query content path: latent compression (W^DQ, W^UQ), or a direct
        # projection W^Q when the query latent is disabled.
        if use_query_latent:
            self.proj_query_down = nn.Linear(dims, query_latent_dims)         # W^DQ
            self.proj_query_up = nn.Linear(
                query_latent_dims, heads * self.head_dims)                    # W^UQ
        else:
            self.proj_query = nn.Linear(dims, heads * self.head_dims)         # W^Q

        self.linear = nn.Linear(heads * self.head_dims, dims)            # W^O

        if use_latent_norm:
            self.norm_kv = RMSNorm(kv_latent_dims)
            if use_query_latent:
                self.norm_query = RMSNorm(query_latent_dims)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, head_dims: int) -> torch.Tensor:
        # (..., seq_len, heads * head_dims) -> (..., heads, seq_len, head_dims)
        x = x.view(x.shape[:-1] + (self.heads, head_dims))
        return x.transpose(-3, -2)

    def forward(self,
                h: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Past]:
        if h.size(-1) != self.dims:
            raise ValueError(
                f'h: expected the last dimension to be {self.dims}, '
                f'got {h.size(-1)}')

        seq_len = h.size(-2)
        past_len = 0
        if past is not None:
            c_past, rope_past = past
            past_len = c_past.size(-2)
            if c_past.size(-1) != self.kv_latent_dims:
                raise ValueError(
                    f'past[0] (cached c^KV): expected the last dimension to '
                    f'be {self.kv_latent_dims}, got {c_past.size(-1)}')
            if c_past.shape[:-2] != h.shape[:-2]:
                raise ValueError(
                    f'past[0] (cached c^KV): expected batch shape '
                    f'{tuple(h.shape[:-2])}, got {tuple(c_past.shape[:-2])}')
            if rope_past.size(-1) != self.rope_head_dims:
                raise ValueError(
                    f'past[1] (cached k^R): expected the last dimension to '
                    f'be {self.rope_head_dims}, got {rope_past.size(-1)}')
            if rope_past.size(-2) != past_len:
                raise ValueError(
                    f'past[1] (cached k^R): expected the sequence length '
                    f'{past_len} to match past[0], got {rope_past.size(-2)}')
            if rope_past.shape[:-2] != h.shape[:-2]:
                raise ValueError(
                    f'past[1] (cached k^R): expected batch shape '
                    f'{tuple(h.shape[:-2])}, got {tuple(rope_past.shape[:-2])}')

        kv_len = past_len + seq_len
        if mask is not None and (mask.size(-2) != seq_len or
                                 mask.size(-1) != kv_len):
            raise ValueError(
                f'mask: expected shape (..., {seq_len}, {kv_len}) '
                f'(True = masked), got {tuple(mask.shape[-2:])}')

        # Explicit absolute positions of the current tokens, derived from
        # the cache length (a cached token's position never changes, so the
        # rotated k^R can be stored in the cache).
        positions = torch.arange(past_len, kv_len, device=h.device)

        # KV latent compression and reconstruction (Eqs. 10-11).
        c_kv = self.proj_kv_down(h)                      # (..., T, d_c)
        if self.use_latent_norm:
            c_kv = self.norm_kv(c_kv)

        # Decoupled RoPE key (Eqs. 15, 17): a single vector shared by all
        # heads, rotated once and cached in its rotated form.
        rope_k = self.proj_rope_k(h)                     # (..., T, d_R)
        rope_k = apply_rotary_embedding(rope_k, positions)

        # Query content and RoPE pathways (Eqs. 12-14, 16).
        if self.use_query_latent:
            c_q = self.proj_query_down(h)                # (..., T, d'_c)
            if self.use_latent_norm:
                c_q = self.norm_query(c_q)
            query = self.proj_query_up(c_q)              # (..., T, n_h d_h)
            rope_q = self.proj_rope_q(c_q)               # (..., T, n_h d_R)
        else:
            query = self.proj_query(h)                   # (..., T, n_h d_h)
            rope_q = self.proj_rope_q(h)                 # (..., T, n_h d_R)
        rope_q = apply_rotary_embedding(rope_q, positions)

        # Concatenate content and RoPE parts (Eq. 13: q = [q^C; q^R]).
        query = torch.cat((self._split_heads(query, self.head_dims),
                           self._split_heads(rope_q, self.rope_head_dims)),
                          dim=-1)                        # (..., n_h, T, d_h + d_R)

        # Full cache: the latent is the only per-token content quantity.
        if past is not None:
            c_all = torch.cat((c_past, c_kv), dim=-2)    # (..., S, d_c)
            rope_all = torch.cat((rope_past, rope_k), dim=-2)  # (..., S, d_R)
        else:
            c_all, rope_all = c_kv, rope_k

        keys = self.proj_k_up(c_all)                     # (..., S, n_h d_h)
        values = self.proj_v_up(c_all)                   # (..., S, n_h d_h)

        keys = self._split_heads(keys, self.head_dims)   # (..., n_h, S, d_h)
        values = self._split_heads(values, self.head_dims)  # (..., n_h, S, d_h)

        # The shared RoPE key k^R is broadcast over the heads before the
        # concatenation (Eq. 13: k = [k^C; k^R], with k^R shared).
        rope_keys = rope_all.unsqueeze(-3).expand(
            keys.shape[:-1] + (self.rope_head_dims,))    # (..., n_h, S, d_R)
        keys = torch.cat((keys, rope_keys), dim=-1)      # (..., n_h, S, d_h + d_R)

        # Attention (Eq. 18): scale over the full query-key dimension,
        # values from the content pathway only, repository mask convention
        # (True = masked) with the same additive -1e4 as BaseAttention.
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(
            self.head_dims + self.rope_head_dims)
        if mask is not None:
            scores = scores + mask.unsqueeze(-3).type_as(scores) \
                * scores.new_tensor(-1e4)
        probs = self.dropout(scores.softmax(-1))
        output = torch.matmul(probs, values)             # (..., n_h, T, d_h)

        # Merge the heads and project (Eq. 19).
        output = output.transpose(-3, -2).contiguous().view(
            output.shape[:-3] + (seq_len, self.heads * self.head_dims))
        output = self.linear(output)                     # (..., T, dims)

        return output, (c_all, rope_all)
