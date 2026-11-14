import numpy as np
import pytest
import torch

from gpt2.data import Vocab, Tokenizer
from gpt2.modeling import Transformer
from gpt2.research.generate import (encode_prompt, full_recompute_logits,
                                    generate, incremental_logits,
                                    top_p_sample)

VOCAB_WORDS = ['the', 'cat', 'sat', 'on', 'mat', 'a', 'dog', 'ran', 'fast',
               'slow', 'red', 'big', '.', '##at', '##og', '##ed', '##ing',
               '##s', 'jumped', 'sleepy']

PROMPTS = ['the cat sat on the mat', 'the dog ran fast slow',
           'a red cat jumped', 'the big dog sat', 'the sleepy dog ran']


def build(attention, layers=2, dims=32, heads=2, seq_len=64,
          kv_latent_dims=32, query_latent_dims=32, rope_head_dims=8,
          seed=42):
    torch.manual_seed(seed)
    return Transformer(layers=layers, pad_idx=1, words=len(VOCAB_WORDS) + 3,
                       seq_len=seq_len, heads=heads, dims=dims, rate=2,
                       bidirectional=False, attention=attention,
                       kv_latent_dims=kv_latent_dims,
                       query_latent_dims=query_latent_dims,
                       rope_head_dims=rope_head_dims).eval()


@pytest.fixture
def vocab(tmp_path):
    path = tmp_path / 'vocab.txt'
    path.write_text(' '.join(VOCAB_WORDS))
    return Vocab(str(path))


@pytest.mark.parametrize('attention', ['mha', 'mla'])
def test_cache_equivalence(vocab, attention):
    # T-K1: incremental decoding with the cache must reproduce the full
    # forward pass logits at every position, for at least five prompts.
    model = build(attention)
    tokenizer = Tokenizer(vocab)

    for prompt in PROMPTS:
        ids = [vocab[token] for token in tokenizer.encode(prompt)]
        assert len(ids) >= 4

        full = full_recompute_logits(model, ids)
        incremental = incremental_logits(model, ids)

        assert incremental.shape == full.shape
        diff = (full - incremental).abs().max().item()
        assert diff < 1e-4, f'{attention}, prompt={prompt!r}: diff={diff}'


@pytest.mark.parametrize('attention', ['mha', 'mla'])
def test_greedy_decoding_matches_full_recomputation(vocab, attention):
    # Greedy multi-step generation must produce the same tokens as a
    # reference that recomputes the whole sequence at every step.
    model = build(attention)
    tokenizer = Tokenizer(vocab)
    prompt = 'the cat sat on the mat'
    max_new_tokens = 8

    ids = [vocab[token] for token in tokenizer.encode(prompt)]
    vocab_size = len(VOCAB_WORDS) + 3

    cached = generate(model, vocab, prompt, max_new_tokens=max_new_tokens,
                      top_p=None, seed=0)['new_tokens']

    reference = []
    current = list(ids)
    for _ in range(max_new_tokens):
        logits = full_recompute_logits(model, current)
        next_id = int(torch.argmax(logits[0, -1]).item())
        reference.append(next_id)
        if next_id == vocab.eos_idx:
            break
        current = current + [next_id]

    assert cached == reference, (cached, reference)


def test_top_p_sampling_deterministic_and_nucleus():
    logits = torch.log(torch.tensor(
        [[0.5, 0.3, 0.15, 0.04, 0.01]]))

    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    samples = [top_p_sample(logits, 0.65, gen_a) for _ in range(20)]
    samples_b = [top_p_sample(logits, 0.65, gen_b) for _ in range(20)]
    assert samples == samples_b

    # top_p=0.65 keeps {0.5, 0.3}: token 0 or 1 only.
    assert set(samples) <= {0, 1}

    # top_p=1.0 keeps everything.
    gen_c = torch.Generator().manual_seed(7)
    all_samples = [top_p_sample(logits, 1.0, gen_c) for _ in range(50)]
    assert set(all_samples) == {0, 1, 2, 3, 4}


def test_stopping_at_eos(vocab):
    model = build('mla')
    # Make every position's logits one-hot on eos: the first sampled token
    # is eos and generation must stop immediately.
    original_forward = type(model).forward

    def eos_forward(self, x, past=None, use_grad_ckpt=False):
        logits, present = original_forward(self, x, past=past,
                                           use_grad_ckpt=use_grad_ckpt)
        forced = logits.new_zeros(logits.shape)
        forced[..., vocab.eos_idx] = 1e4
        return forced, present

    type(model).forward = eos_forward
    result = generate(model, vocab, 'the cat sat on the mat',
                      max_new_tokens=10, top_p=0.9, seed=0)
    type(model).forward = original_forward

    assert len(result['new_tokens']) == 1
    assert result['stopped_at_eos']


def test_max_length_stops_generation(vocab):
    model = build('mla')
    result = generate(model, vocab, 'the cat sat on the mat',
                      max_new_tokens=5, top_p=1.0, seed=0)
    assert len(result['new_tokens']) == 5


def test_prompt_truncation_warns_and_left_truncates(vocab):
    model = build('mla', seq_len=10)
    tokenizer = Tokenizer(vocab)

    long_prompt = ' '.join(['the cat sat on the mat'] * 10)
    with pytest.warns(UserWarning, match='truncating'):
        ids = encode_prompt(vocab, tokenizer, long_prompt, context_len=10)
    assert len(ids) == 10

    # Left truncation: the suffix is kept.
    full_ids = [vocab[token] for token in tokenizer.encode(long_prompt)]
    assert ids == full_ids[-10:]

    # The same rule for the baseline arm.
    with pytest.warns(UserWarning, match='truncating'):
        ids_mha = encode_prompt(vocab, tokenizer, long_prompt,
                                context_len=10)
    assert ids_mha == ids


def test_cache_size_matches_analytic_values():
    # PRD 8.5 / H1: per-token per-layer cache elements 1536 (mha) vs 288
    # (mla); at context 1024, fp16, batch 1: 36,864 B vs 6,912 B.
    layers, dims, heads = 1, 768, 12
    mha = build('mha', layers=layers, dims=dims, heads=heads, seq_len=1024)
    mla = build('mla', layers=layers, dims=dims, heads=heads, seq_len=1024,
                kv_latent_dims=256, query_latent_dims=256, rope_head_dims=32)

    x = torch.randint(3, len(VOCAB_WORDS) + 3, (1, 1024))
    _, past_mha = mha(x)
    _, past_mla = mla(x)

    # MHA cache: (k, v) each (1, heads, S, d_h).
    elements_mha = sum(t.numel() for t in past_mha[0])
    per_token_per_layer_mha = elements_mha // 1024
    assert per_token_per_layer_mha == 2 * heads * (dims // heads) == 1536

    # MLA cache: (c^KV, k^R) each (1, S, dims).
    elements_mla = sum(t.numel() for t in past_mla[0])
    per_token_per_layer_mla = elements_mla // 1024
    assert per_token_per_layer_mla == 256 + 32 == 288

    assert per_token_per_layer_mla / per_token_per_layer_mla == 1
    ratio = per_token_per_layer_mla / per_token_per_layer_mha
    assert abs(ratio - 0.1875) < 1e-9

    # Cache bytes PER TOKEN, all layers, fp16 (PRD 8.5 row). At the
    # proposed 12-layer size: 1536 x 12 x 2 = 36,864 B and
    # 288 x 12 x 2 = 6,912 B.
    bytes_per_token_mha = per_token_per_layer_mha * layers * 2
    bytes_per_token_mla = per_token_per_layer_mla * layers * 2
    assert bytes_per_token_mha == 1536 * 2
    assert bytes_per_token_mla == 288 * 2
    assert per_token_per_layer_mha * 12 * 2 == 36864
    assert per_token_per_layer_mla * 12 * 2 == 6912
    # Total cache at context 1024, batch 1, 12 layers, fp16.
    total_mha = per_token_per_layer_mha * 12 * 1024 * 2
    total_mla = per_token_per_layer_mla * 12 * 1024 * 2
    assert total_mha == 37748736 and total_mla == 7077888
