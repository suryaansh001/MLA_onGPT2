import torch

from gpt2.modeling import Transformer


def build_model(attention, layers=2, words=80, seq_len=32, heads=2, dims=16,
                rate=4, bidirectional=False, positional_embedding=True):
    torch.manual_seed(42)
    return Transformer(layers=layers, pad_idx=0, words=words,
                       seq_len=seq_len, heads=heads, dims=dims, rate=rate,
                       bidirectional=bidirectional,
                       attention=attention,
                       positional_embedding=positional_embedding).eval()


def run_model(model, x, past=None):
    with torch.no_grad():
        return model(x, past=past) if past is not None else model(x)


def test_causal_invariance_mha():
    model = build_model('mha')
    x = torch.randint(1, 80, (1, 10))

    logits_full, _ = run_model(model, x)

    # Altering tokens strictly after position t must not change the logits
    # at position t (and before).
    for t in (0, 3, 9):
        x_altered = x.clone()
        x_altered[:, t + 1:] = torch.randint(1, 80, (1, 10 - t - 1))
        logits_altered, _ = run_model(model, x_altered)
        assert torch.allclose(logits_full[:, :t + 1], logits_altered[:, :t + 1],
                              atol=1e-5), f'position {t}'


def test_causal_invariance_mla():
    model = build_model('mla')
    x = torch.randint(1, 80, (1, 10))

    logits_full, _ = run_model(model, x)

    for t in (0, 3, 9):
        x_altered = x.clone()
        x_altered[:, t + 1:] = torch.randint(1, 80, (1, 10 - t - 1))
        logits_altered, _ = run_model(model, x_altered)
        assert torch.allclose(logits_full[:, :t + 1], logits_altered[:, :t + 1],
                              atol=1e-5), f'position {t}'


def test_causal_invariance_with_padding_both_arms():
    for attention in ('mha', 'mla'):
        model = build_model(attention)
        x = torch.randint(1, 80, (1, 10))
        x[0, 7:] = 0                                  # padding
        logits_full, _ = run_model(model, x)

        for t in (0, 4, 8):
            x_altered = x.clone()
            x_altered[:, t + 1:] = torch.randint(1, 80, (1, 10 - t - 1))
            x_altered[0, 7:] = 0
            logits_altered, _ = run_model(model, x_altered)
            assert torch.allclose(logits_full[:, :t + 1],
                                  logits_altered[:, :t + 1], atol=1e-5), \
                f'{attention}, position {t}'


def test_padding_invariance_both_arms():
    # Extending a sequence with padding on the right must not change the
    # logits of the non-padding positions.
    for attention in ('mha', 'mla'):
        model = build_model(attention)
        x = torch.randint(1, 80, (1, 8))
        x_padded = torch.cat((x, torch.zeros(1, 4, dtype=torch.long)), dim=-1)

        logits, _ = run_model(model, x)
        logits_padded, _ = run_model(model, x_padded)

        assert torch.allclose(logits, logits_padded[:, :8], atol=1e-5), \
            attention
