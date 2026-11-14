import torch
from typing import Optional


def rotary_tables(positions: torch.Tensor,
                  head_dims: int,
                  base: float = 10000.0):
    """
    Build the RoPE cosine and sine tables for a given set of absolute
    positions.

    Tensor          Type            Shape
    ===========================================================================
    positions       long            (...)
    ---------------------------------------------------------------------------
    cos             float           (..., head_dims // 2)
    sin             float           (..., head_dims // 2)
    ===========================================================================
    """
    if head_dims % 2 != 0:
        raise ValueError(
            f'head_dims: RoPE requires an even dimension '
            f'(multiple of 2), got {head_dims}')

    # theta_i = base^(-2i / head_dims), i in [0, head_dims / 2).
    theta = base ** (-torch.arange(0, head_dims, 2, dtype=torch.float32,
                                   device=positions.device) / head_dims)

    # angles[..., p, i] = p * theta_i.
    angles = positions.to(torch.float32).unsqueeze(-1) * theta

    return angles.cos(), angles.sin()


def apply_rotary_embedding(x: torch.Tensor,
                           positions: torch.Tensor,
                           base: float = 10000.0) -> torch.Tensor:
    """
    Apply rotary position embedding to the last dimension of `x` at the
    explicit absolute positions given in `positions`. The same call serves
    prefill (positions = offset .. offset + seq) and incremental decoding
    (a single position per step).

    The rotation is the pair-wise rotation of GPT-NeoX style: the last
    dimension is split into two halves, and each pair (x_i, x_{i + d/2}) is
    rotated by the angle p * theta_i.

    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, head_dims)
    positions       long            (..., seq_len) (broadcastable to x)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, head_dims)
    ===========================================================================
    """
    cos, sin = rotary_tables(positions, x.size(-1), base)

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
