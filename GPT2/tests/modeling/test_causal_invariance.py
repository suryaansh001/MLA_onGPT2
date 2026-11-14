import hashlib
import json
import torch

from gpt2.modeling import Transformer, MultiHeadLatentAttention

FIXTURE = 'tests/fixtures/baseline_reference.npz'
META = 'tests/fixtures/baseline_reference.json'


def test_baseline_unchanged_against_fixture():
    # T-B1: with attention='mha' (the default), outputs and parameter count
    # must match the reference captured from the unmodified repository.
    meta = json.load(open(META))
    assert meta['commit'] == '7b159669f6225231ee75f5ea110c1be69f4ce8ef'

    torch.manual_seed(meta['seed_model'])
    config = meta['config']
    model = Transformer(layers=config['layers'], pad_idx=config['pad_idx'],
                        words=config['words'], seq_len=config['seq_len'],
                        heads=config['heads'], dims=config['dims'],
                        rate=config['rate'],
                        bidirectional=config['bidirectional']).eval()
    assert sum(p.numel() for p in model.parameters()) == meta['param_count']

    data = torch.load(FIXTURE, weights_only=False)
    torch.manual_seed(meta['seed_input'])
    x = torch.randint(config['words'], (1, 8))
    x_next = torch.randint(config['words'], (1, 1))
    assert torch.equal(x, data['input'])

    with torch.no_grad():
        logits, past = model(x)
        logits_next, _ = model(x_next, past=past)

    assert torch.equal(logits, data['logits'])
    assert torch.equal(logits_next, data['logits_next'])

    digest = hashlib.sha256(
        logits.numpy().tobytes() + logits_next.numpy().tobytes()).hexdigest()
    assert digest == meta['logits_sha256']


def test_baseline_unchanged_with_explicit_attention_argument():
    # Passing attention='mha' explicitly must produce identical results to
    # the default construction.
    torch.manual_seed(1)
    a = Transformer(layers=2, pad_idx=0, words=80, seq_len=32, heads=2,
                    dims=16, rate=4, bidirectional=False).eval()
    torch.manual_seed(1)
    b = Transformer(layers=2, pad_idx=0, words=80, seq_len=32, heads=2,
                    dims=16, rate=4, bidirectional=False,
                    attention='mha').eval()

    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)

    x = torch.randint(1, 80, (1, 8))
    logits_a = a(x)[0] if isinstance(a(x), tuple) else a(x)
    logits_b = b(x)[0] if isinstance(b(x), tuple) else b(x)
    assert torch.equal(logits_a, logits_b)


def test_unknown_attention_type_rejected():
    try:
        Transformer(layers=1, pad_idx=0, words=80, seq_len=32, heads=2,
                    dims=16, attention='gqa')
    except ValueError as error:
        assert "expected 'mha' or 'mla'" in str(error)
    else:
        raise AssertionError('unknown attention type accepted')


def test_mla_forward_backward_integration():
    # T-3/T-4: the full model with attention='mla' runs forward and backward
    # on CPU and every parameter receives a non-zero gradient.
    model = Transformer(layers=2, pad_idx=0, words=80, seq_len=32, heads=2,
                        dims=16, rate=4, bidirectional=False,
                        attention='mla')
    model.train()

    x = torch.randint(1, 80, (2, 6))
    targets = torch.randint(1, 80, (2, 6))
    logits = model(x)                    # training mode returns logits only
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 80),
                                             targets.view(-1))
    loss.backward()

    unused = [name for name, param in model.named_parameters()
              if param.grad is None or param.grad.abs().sum() == 0]
    assert unused == [], f'parameters without gradient: {unused}'


def test_mla_positional_embedding_flag_changes_count_exactly():
    # Disabling the learned positional embedding removes exactly
    # seq_len * dims parameters (H4 ablation, FR-04).
    with_pos = Transformer(layers=1, pad_idx=0, words=80, seq_len=32,
                           heads=2, dims=16, attention='mla')
    without_pos = Transformer(layers=1, pad_idx=0, words=80, seq_len=32,
                              heads=2, dims=16, attention='mla',
                              positional_embedding=False)

    count_pos = sum(p.numel() for p in with_pos.parameters())
    count_no = sum(p.numel() for p in without_pos.parameters())
    assert count_pos - count_no == 32 * 16
    assert not any('positional_embedding' in name
                   for name, _ in without_pos.named_parameters())


def test_mla_positional_embedding_disabled_runs():
    model = Transformer(layers=1, pad_idx=0, words=80, seq_len=32, heads=2,
                        dims=16, attention='mla',
                        positional_embedding=False).eval()
    x = torch.randint(1, 80, (1, 8))
    logits, past = model(x)
    assert logits.shape == (1, 8, 80)
    assert past[0][0].size(-2) == 8

    logits, past = model(torch.randint(1, 80, (1, 1)), past=past)
    assert past[0][0].size(-2) == 9


def test_parameter_counts_match_prd():
    # T-P1: analytic values from PRD 8.5 at the proposed configuration
    # (L=12, d=768, n_h=12, d_h=64, rate=4, vocab 50,257, seq_len 256).
    words, seq_len, dims, heads, layers = 50257, 256, 768, 12, 12

    baseline = Transformer(layers=layers, pad_idx=50256, words=words,
                           seq_len=seq_len, heads=heads, dims=dims, rate=4,
                           bidirectional=False).eval()
    assert sum(p.numel() for p in baseline.parameters()) == 123849984

    mla = Transformer(layers=layers, pad_idx=50256, words=words,
                      seq_len=seq_len, heads=heads, dims=dims, rate=4,
                      bidirectional=False, attention='mla',
                      kv_latent_dims=256, query_latent_dims=256,
                      rope_head_dims=32).eval()
    assert sum(p.numel() for p in mla.parameters()) == 115904640

    # Attention parameters per layer: count only the attention submodule.
    mha_attn = sum(p.numel() for p in baseline.transformers[0].attn.parameters())
    mla_attn = sum(p.numel() for p in mla.transformers[0].attn.parameters())
    assert mha_attn == 2362368
    assert mla_attn == 1700256
