import math
import pytest
import torch

from gpt2.modeling.latent_attention import MultiHeadLatentAttention, RMSNorm
from gpt2.modeling.rotary import apply_rotary_embedding, rotary_tables


def build_module(**overrides):
    kwargs = dict(heads=2, dims=8, kv_latent_dims=8, query_latent_dims=8,
                  rope_head_dims=4, use_query_latent=True,
                  use_latent_norm=True, dropout=0.0)
    kwargs.update(overrides)
    torch.manual_seed(7)
    return MultiHeadLatentAttention(**kwargs).eval()


def set_identity_weights(module, names):
    # out[i] = in[i] for i < min(out, in), rest zero: per-head splitting
    # maps input coordinate blocks to the matching head blocks unchanged.
    with torch.no_grad():
        for name in names:
            linear = getattr(module, name)
            linear.weight.zero_()
            linear.weight.copy_(torch.eye(linear.weight.size(0),
                                          linear.weight.size(1)))
            if linear.bias is not None:
                linear.bias.zero_()


def set_zero_weights(module, names):
    with torch.no_grad():
        for name in names:
            linear = getattr(module, name)
            linear.weight.zero_()
            if linear.bias is not None:
                linear.bias.zero_()


def test_output_and_cache_shapes_prefill_and_incremental():
    module = build_module()

    x = torch.randn(2, 5, 8)
    output, (c, rope) = module(x)
    assert output.shape == (2, 5, 8)
    assert c.shape == (2, 5, 8)          # (..., S, d_c)
    assert rope.shape == (2, 5, 4)       # (..., S, d_R), shared over heads

    # Incremental step: the cache grows by one token per step.
    output, (c, rope) = module(torch.randn(2, 1, 8), past=(c, rope))
    assert output.shape == (2, 1, 8)
    assert c.shape == (2, 6, 8)
    assert rope.shape == (2, 6, 4)

    # The first cache element keeps the sequence axis at dimension -2
    # (repository position-offset convention, A-7).
    assert c.size(-2) == rope.size(-2) == 6


def test_higher_rank_batch_shapes():
    module = build_module()
    x = torch.randn(3, 2, 5, 8)
    output, (c, rope) = module(x)
    assert output.shape == (3, 2, 5, 8)
    assert c.shape == (3, 2, 5, 8)
    assert rope.shape == (3, 2, 5, 4)

    output, (c, rope) = module(torch.randn(3, 2, 1, 8), past=(c, rope),
                               mask=torch.zeros((3, 2, 1, 6)).bool())
    assert c.shape == (3, 2, 6, 8)


def test_latent_equation_shapes():
    # c^KV has kv_latent_dims; k^C, v^C are per-head with head_dims;
    # k^R is a single shared vector (not per-head); q^R is per-head.
    module = build_module(kv_latent_dims=6, query_latent_dims=6,
                          rope_head_dims=4, heads=2, dims=8)
    x = torch.randn(2, 5, 8)

    c_kv = module.norm_kv(module.proj_kv_down(x))
    assert c_kv.shape == (2, 5, 6)

    keys = module._split_heads(module.proj_k_up(c_kv), module.head_dims)
    values = module._split_heads(module.proj_v_up(c_kv), module.head_dims)
    assert keys.shape == (2, 2, 5, 4)    # (B, n_h, S, d_h)
    assert values.shape == (2, 2, 5, 4)

    rope_k = module.proj_rope_k(x)
    assert rope_k.shape == (2, 5, 4)     # (B, S, d_R): shared, not n_h*d_R

    rope_q = module.proj_rope_q(c_kv)
    assert rope_q.shape == (2, 5, 8)     # (B, T, n_h * d_R): per head


def test_scale_uses_full_query_key_dimension():
    # With the RoPE projections zeroed, q and k have trailing zero
    # components, so the dot products are unaffected but the scale is
    # 1/sqrt(d_h + d_R) and NOT 1/sqrt(d_h). Identity up/down projections
    # make c^KV = k^C = v^C = h exactly.
    module = build_module(kv_latent_dims=8, query_latent_dims=8,
                          rope_head_dims=2, use_latent_norm=False,
                          use_query_latent=False)
    set_identity_weights(module, ['proj_kv_down', 'proj_k_up', 'proj_v_up',
                                  'proj_query', 'linear'])
    set_zero_weights(module, ['proj_rope_k', 'proj_rope_q'])

    torch.manual_seed(11)
    x = torch.randn(1, 4, 8)
    output, _ = module(x)

    heads = x.view(1, 4, 2, 4).transpose(-3, -2)      # (B, n_h, T, d_h)
    scores = heads @ heads.transpose(-2, -1) / math.sqrt(4 + 2)
    expected = (scores.softmax(-1) @ heads).transpose(-3, -2).reshape(1, 4, 8)

    assert torch.allclose(output, expected, atol=1e-5)

    wrong_scale = heads @ heads.transpose(-2, -1) / math.sqrt(4)
    wrong = (wrong_scale.softmax(-1) @ heads).transpose(-3, -2).reshape(1, 4, 8)
    assert (output - wrong).abs().max() > 1e-3


def test_values_are_content_only_one_hot_probe():
    # With identity projections and one-hot inputs, the output rows are
    # convex combinations of one-hot value rows: they must stay one-hot-like
    # (sum 1, in [0, 1]) and a masked position must receive zero attention.
    module = build_module(kv_latent_dims=8, rope_head_dims=2,
                          use_latent_norm=False, use_query_latent=False)
    set_identity_weights(module, ['proj_kv_down', 'proj_k_up', 'proj_v_up',
                                  'proj_query', 'linear'])
    set_zero_weights(module, ['proj_rope_k', 'proj_rope_q'])

    x = torch.eye(4, 8).unsqueeze(0)                 # one-hot rows, T=4
    mask = torch.zeros(1, 4, 4).bool()
    mask[0, :, 1:3] = True                           # positions 1, 2 masked

    output, _ = module(x, mask=mask)
    assert torch.allclose(output.sum(-1), torch.ones(1, 4), atol=1e-5)
    assert output.min() >= -1e-6 and output.max() <= 1 + 1e-5
    # No value mass may flow from the masked positions.
    assert torch.allclose(output[0, :, 1:3], torch.zeros(1, 4, 2),
                          atol=1e-4)


def test_masked_positions_excluded():
    module = build_module(use_latent_norm=False, use_query_latent=False)
    set_identity_weights(module, ['proj_kv_down', 'proj_k_up', 'proj_v_up',
                                  'proj_query', 'linear'])
    set_zero_weights(module, ['proj_rope_k', 'proj_rope_q'])

    x = torch.randn(1, 4, 8)

    mask_all = torch.zeros(1, 4, 4).bool()
    output_all, _ = module(x, mask=mask_all)

    # With only the diagonal unmasked (True = masked elsewhere), every
    # token attends to itself (softmax of {0, -1e4, ...} is one-hot up to
    # 1e-4-scale leakage).
    mask_self = (~torch.eye(4).bool()).unsqueeze(0)
    output_self, _ = module(x, mask=mask_self)
    assert torch.allclose(output_self, x, atol=1e-3)
    assert not torch.allclose(output_self, output_all, atol=1e-3)


def test_rope_relative_position_property():
    # Rotating x by position p and y by position q: their inner product
    # depends only on the difference q - p (relative-position property).
    torch.manual_seed(3)
    head_dims = 8
    x = torch.randn(1, 16, head_dims)
    y = torch.randn(1, 16, head_dims)

    p, q = 5, 13
    x_rot = apply_rotary_embedding(x, torch.tensor([p]))
    y_rot = apply_rotary_embedding(y, torch.tensor([q]))

    y_rot_shifted = apply_rotary_embedding(y, torch.tensor([q - p]))
    x_zero = apply_rotary_embedding(x, torch.tensor([0]))

    inner = (x_rot * y_rot).sum(-1)
    expected = (x_zero * y_rot_shifted).sum(-1)
    assert torch.allclose(inner, expected, atol=1e-5)


def test_rope_rotation_is_invertible_by_composition():
    # Rotating by position p then by -p restores the input exactly
    # (R(p)^T R(p) = I), the core of the relative-position property.
    torch.manual_seed(5)
    x = torch.randn(1, 10, 6)
    rotated = apply_rotary_embedding(x, torch.tensor([17]))
    restored = apply_rotary_embedding(rotated, torch.tensor([-17]))
    assert torch.allclose(restored, x, atol=1e-5)


def test_rotary_tables_shapes_and_odd_dim_error():
    cos, sin = rotary_tables(torch.arange(4), 6)
    assert cos.shape == (4, 3) and sin.shape == (4, 3)
    assert torch.allclose(cos[0], torch.ones(3))
    assert torch.allclose(sin[0], torch.zeros(3))

    # The rotation preserves the norm of each rotated coordinate pair
    # (coords k and k + d/2 are rotated together).
    torch.manual_seed(9)
    x = torch.randn(2, 7, 6)
    rotated = apply_rotary_embedding(x, torch.arange(7))
    half = x.size(-1) // 2
    pair_norm = torch.stack((x[..., :half], x[..., half:]), -1).norm(dim=-1)
    rotated_norm = torch.stack((rotated[..., :half],
                                rotated[..., half:]), -1).norm(dim=-1)
    assert torch.allclose(pair_norm, rotated_norm, atol=1e-5)

    with pytest.raises(ValueError, match='head_dims'):
        rotary_tables(torch.arange(4), 5)


def test_dimension_errors_name_the_tensor():
    module = build_module()

    # Wrong input width names the tensor 'h'.
    with pytest.raises(ValueError, match='h:'):
        module(torch.randn(2, 5, 7))

    # Mismatched mask shape names the tensor 'mask'.
    x = torch.randn(2, 5, 8)
    with pytest.raises(ValueError, match='mask:'):
        module(x, mask=torch.zeros(2, 5, 9).bool())

    # Wrong cached latent width names 'past[0]'.
    c, rope = module(x)[1]
    with pytest.raises(ValueError, match='past\\[0\\]'):
        module(torch.randn(2, 1, 8), past=(torch.randn(2, 5, 7), rope))

    # Wrong cached RoPE width names 'past[1]'.
    with pytest.raises(ValueError, match='past\\[1\\]'):
        module(torch.randn(2, 1, 8), past=(c, torch.randn(2, 5, 3)))


def test_construction_errors():
    with pytest.raises(ValueError, match='rope_head_dims'):
        MultiHeadLatentAttention(heads=2, dims=8, rope_head_dims=3)
    with pytest.raises(ValueError, match='heads'):
        MultiHeadLatentAttention(heads=3, dims=8)
    with pytest.raises(ValueError, match='kv_latent_dims'):
        MultiHeadLatentAttention(heads=2, dims=8, kv_latent_dims=0)


@pytest.mark.parametrize('use_query_latent', [True, False])
@pytest.mark.parametrize('use_latent_norm', [True, False])
def test_gradient_flows_to_every_projection(use_query_latent,
                                            use_latent_norm):
    module = build_module(use_query_latent=use_query_latent,
                          use_latent_norm=use_latent_norm)
    module.train()

    x = torch.randn(2, 6, 8, requires_grad=True)
    output, _ = module(x)
    output.sum().backward()

    unused = [name for name, param in module.named_parameters()
              if param.grad is None or param.grad.abs().sum() == 0]
    assert unused == [], f'parameters without gradient: {unused}'


def test_rmsnorm_normalises_and_scales():
    norm = RMSNorm(4)
    x = torch.randn(3, 4)
    out = norm(x)
    assert out.shape == x.shape
    # Without the gain the RMS of the output is 1 (up to eps).
    plain = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    assert torch.allclose(out / norm.weight, plain, atol=1e-6)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    assert torch.allclose(norm(x) / 2.0, plain, atol=1e-6)
