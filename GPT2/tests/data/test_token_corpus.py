import json
import pytest
import os
import torch

from gpt2.data import Vocab
from gpt2.data.token_corpus import TokenCorpus, TokenCorpusConfig
from gpt2.research.config import (ResearchConfig, ModelConfig, DataConfig,
                                  OptimiserConfig)
from gpt2.research.train_research import (build_model, build_optimizer,
                                          lr_at, load_checkpoint,
                                          save_checkpoint, train)

VOCAB_WORDS = ['the', 'cat', 'sat', 'on', 'mat', 'a', 'dog', 'ran', 'fast',
               'slow', 'red', 'big', '.', '##at', '##og', '##ed', '##ing',
               '##s', 'jumped', 'sleepy']


def make_config(tmp_path, steps=12, attention='mla', **data_overrides):
    data_kwargs = dict(
        vocab_path=str(tmp_path / 'vocab.txt'),
        cache_dir=str(tmp_path / 'cache'),
        val_fraction=0.25, data_seed=5, shard_size=2,
        batch_size=2, grad_accumulation=1)
    data_kwargs.update(data_overrides)
    return ResearchConfig(
        model=ModelConfig(layers=1, dims=16, heads=2, seq_len=8, rate=2,
                          dropout=0.0, attention=attention),
        data=DataConfig(**data_kwargs),
        optimiser=OptimiserConfig(peak_lr=1e-3, weight_decay=0.1,
                                  clip_norm=1.0),
        steps=steps, seed=42, device='cpu', precision='fp32',
        log_interval=5, eval_interval=100, save_steps=100,
        run_name='test')


@pytest.fixture
def env(tmp_path):
    (tmp_path / 'vocab.txt').write_text(' '.join(VOCAB_WORDS))
    with open(tmp_path / 'vocab.txt.documents', 'w') as fp:
        for _ in range(40):
            fp.write('the cat sat on the mat.\n')
            fp.write('the big dog ran fast slow.\n')
            fp.write('a red cat jumped fast.\n')
    return tmp_path


def build_corpus_for(config):
    corpus_config = TokenCorpusConfig(seq_len=config.model.seq_len,
                                      val_fraction=config.data.val_fraction,
                                      seed=config.data.data_seed,
                                      shard_size=config.data.shard_size)
    vocab = Vocab(config.data.vocab_path)
    with open(config.data.vocab_path + '.documents') as fp:
        documents = [line.strip() for line in fp if line.strip()]
    return TokenCorpus.build(corpus_config, vocab, documents,
                             cache_dir=config.data.cache_dir)


def param_hash(model):
    return torch.sum(torch.stack([
        p.detach().double().sum() for p in model.parameters()])).item()


def full_param_state(model):
    return [p.detach().clone() for p in model.parameters()]


def test_accumulation_equivalence(env):
    # TR-2: an accumulated update over k micro-batches of size b must equal
    # a single update over one batch of size k*b, within tolerance.
    corpus = build_corpus_for(make_config(env))

    torch.manual_seed(42)
    model_single = build_model(make_config(env), words=len(VOCAB_WORDS) + 3,
                               pad_idx=1)
    torch.manual_seed(42)
    model_accum = build_model(make_config(env), words=len(VOCAB_WORDS) + 3,
                              pad_idx=1)

    opt_single = build_optimizer(make_config(env), model_single)
    opt_accum = build_optimizer(make_config(env), model_accum)

    x_np, y_np = corpus.next_batch('train', batch_size=4)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    # Single update over one batch of 4.
    logits = model_single(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    opt_single.step()

    # Accumulated update over 2 micro-batches of 2.
    opt_accum.zero_grad(set_to_none=True)
    for micro in range(2):
        xm = x[micro * 2:(micro + 1) * 2]
        ym = y[micro * 2:(micro + 1) * 2]
        logits = model_accum(xm)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ym.reshape(-1)) / 2
        loss.backward()
    opt_accum.step()

    for pa, pb in zip(model_single.parameters(), model_accum.parameters()):
        # Tolerance 1e-5: fp32 summation order differs between one batch of
        # 4 and two batches of 2 (documented in docs/decisions.md).
        assert torch.allclose(pa, pb, atol=1e-5), \
            (pa - pb).abs().max().item()


def test_exact_resume(env):
    # T-R1: training N steps uninterrupted vs k steps, checkpoint, resume
    # to N: matching parameter states (CPU, fp32 -> bit-comparable).
    config = make_config(env, steps=8)
    corpus_a = build_corpus_for(config)
    torch.manual_seed(config.seed)
    model_a = build_model(config, words=len(VOCAB_WORDS) + 3, pad_idx=1)
    result_a = train(config, corpus=corpus_a, model=model_a,
                     log_path=str(env / 'a.jsonl'),
                     checkpoint_path=str(env / 'a.pt'))
    assert result_a['final_step'] == 8
    state_a = full_param_state(model_a)

    corpus_b = build_corpus_for(config)
    torch.manual_seed(config.seed)
    model_b = build_model(config, words=len(VOCAB_WORDS) + 3, pad_idx=1)
    # The partial run uses the SAME configuration and schedule; it merely
    # stops early at step 3 (stop_after), which is the T-R1 scenario.
    train(config, corpus=corpus_b, model=model_b, stop_after=3,
          log_path=str(env / 'b.jsonl'), checkpoint_path=str(env / 'b.pt'))

    # Resume from the 3-step checkpoint to 8 steps with the full config.
    result_d = train(config, corpus=corpus_b, model=model_b,
                     resume_from=str(env / 'b.pt'),
                     log_path=str(env / 'd.jsonl'),
                     checkpoint_path=str(env / 'd.pt'))
    assert result_d['final_step'] == 8
    state_d = full_param_state(model_b)

    for pa, pd in zip(state_a, state_d):
        assert torch.allclose(pa, pd, atol=1e-7), \
            f'max diff {(pa - pd).abs().max().item()}'


def test_fingerprint_mismatch_fails_loudly(env):
    config = make_config(env, steps=3)
    corpus = build_corpus_for(config)
    torch.manual_seed(config.seed)
    model = build_model(config, words=len(VOCAB_WORDS) + 3, pad_idx=1)
    train(config, corpus=corpus, model=model,
          log_path=str(env / 'a.jsonl'), checkpoint_path=str(env / 'a.pt'))

    other = make_config(env, steps=3)
    other.optimiser.peak_lr = 5e-3      # different fingerprint

    with pytest.raises(ValueError, match='fingerprint'):
        load_checkpoint(str(env / 'a.pt'), other, model)


def test_optimizer_group_sizes(env):
    config = make_config(env)
    torch.manual_seed(0)
    model = build_model(config, words=len(VOCAB_WORDS) + 3, pad_idx=1)
    optimizer = build_optimizer(config, model)

    decay_params = sum(p.numel() for p in optimizer.param_groups[0]['params'])
    no_decay_params = sum(p.numel()
                          for p in optimizer.param_groups[1]['params'])
    total = sum(p.numel() for p in model.parameters())
    assert decay_params + no_decay_params == total

    names_no_decay = {n for n, _ in model.named_parameters()
                      if 'bias' in n or 'norm' in n or 'embedding' in n}
    grouped_no_decay = sum(1 for p in optimizer.param_groups[1]['params'])
    assert grouped_no_decay == len([p for n, p in model.named_parameters()
                                    if n in names_no_decay])

    # Embeddings must not be decayed.
    for group in optimizer.param_groups:
        for p in group['params']:
            if p is model.token_embedding.weight:
                assert group['weight_decay'] == 0.0


def test_checkpoint_write_is_atomic(env, monkeypatch):
    # A crash between the temp write and the rename must leave no partial
    # checkpoint at the final path.
    config = make_config(env, steps=2)
    corpus = build_corpus_for(config)
    torch.manual_seed(config.seed)
    model = build_model(config, words=len(VOCAB_WORDS) + 3, pad_idx=1)
    optimizer = build_optimizer(config, model)

    final_path = str(env / 'crash.pt')
    real_replace = os.replace

    def crashing_replace(src, dst):
        if dst == final_path:
            raise OSError('simulated interruption')
        return real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', crashing_replace)
    try:
        save_checkpoint(final_path, config, model, optimizer, 1, corpus, [],
                        None)
    except OSError:
        pass
    finally:
        monkeypatch.undo()

    assert not os.path.exists(final_path)
    assert os.path.exists(final_path + '.tmp')

    # A clean save afterwards produces a loadable checkpoint.
    save_checkpoint(final_path, config, model, optimizer, 2, corpus, [], None)
    payload = load_checkpoint(final_path, config, model, optimizer)
    assert payload['step'] == 2


def test_lr_schedule_warmup_and_decay(env):
    config = make_config(env, steps=100)
    peak = config.optimiser.peak_lr
    # Warm-up: first 5% of steps ramp to peak.
    assert lr_at(config, 0) < peak
    assert abs(lr_at(config, 4) - peak) < 1e-12
    # Decay to 10% of peak at the final step.
    assert abs(lr_at(config, 99) - peak * 0.1) < peak * 1e-3
    # Monotone decay after warm-up.
    assert lr_at(config, 50) < lr_at(config, 10) <= peak


def test_non_finite_loss_stops_with_checkpoint(env, monkeypatch):
    config = make_config(env, steps=4)
    corpus = build_corpus_for(config)
    torch.manual_seed(config.seed)
    model = build_model(config, words=len(VOCAB_WORDS) + 3, pad_idx=1)

    # Force a non-finite training loss. In training mode the model returns
    # logits only; eval mode returns (logits, past).
    original_forward = type(model).forward

    def poisoned(self, x, past=None, use_grad_ckpt=False):
        if self.training:
            return original_forward(self, x) * float('nan')
        return original_forward(self, x, past, use_grad_ckpt)

    monkeypatch.setattr(type(model), 'forward', poisoned)
    result = train(config, corpus=corpus, model=model,
                   log_path=str(env / 'nf.jsonl'),
                   checkpoint_path=str(env / 'nf.pt'))
    monkeypatch.undo()

    assert result['steps_completed'] < 4
    assert os.path.exists(str(env / 'nf.pt'))
    with open(str(env / 'nf.jsonl')) as fp:
        events = [json.loads(line) for line in fp]
    assert any(e.get('event') == 'non_finite_loss' for e in events)
