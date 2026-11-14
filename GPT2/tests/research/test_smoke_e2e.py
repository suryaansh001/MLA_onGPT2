import json
import os
import torch

from gpt2.data import Vocab
from gpt2.data.token_corpus import TokenCorpus, TokenCorpusConfig
from gpt2.research.config import (ResearchConfig, ModelConfig, DataConfig,
                                  OptimiserConfig)
from gpt2.research.train_research import train, build_model

import pytest

VOCAB_WORDS = ['the', 'cat', 'sat', 'on', 'mat', 'a', 'dog', 'ran', 'fast',
               'slow', 'red', 'big', '.', '##at', '##og', '##ed', '##ing',
               '##s', 'jumped', 'sleepy']


@pytest.mark.parametrize('attention', ['mha', 'mla'])
def test_end_to_end_smoke(tmp_path, attention):
    # T-E2E: data preparation, training, checkpoint, resume, evaluation —
    # the complete CPU pipeline for one arm.
    vocab_path = tmp_path / 'vocab.txt'
    vocab_path.write_text(' '.join(VOCAB_WORDS))
    documents_path = tmp_path / 'vocab.txt.documents'
    with open(documents_path, 'w') as fp:
        for _ in range(30):
            fp.write('the cat sat on the mat.\n')
            fp.write('the big dog ran fast slow.\n')
            fp.write('a red cat jumped fast.\n')

    config = ResearchConfig(
        model=ModelConfig(layers=2, dims=32, heads=2, seq_len=16, rate=2,
                          attention=attention,
                          kv_latent_dims=32, query_latent_dims=32,
                          rope_head_dims=8),
        data=DataConfig(vocab_path=str(vocab_path),
                        cache_dir=str(tmp_path / 'cache'),
                        val_fraction=0.2, data_seed=7, shard_size=4,
                        batch_size=2),
        optimiser=OptimiserConfig(peak_lr=1e-3),
        steps=6, seed=42, device='cpu', precision='fp32',
        log_interval=2, eval_interval=3, save_steps=3,
        run_name=f'smoke_{attention}')

    corpus = TokenCorpus.build(
        TokenCorpusConfig(seq_len=config.model.seq_len,
                          val_fraction=config.data.val_fraction,
                          seed=config.data.data_seed,
                          shard_size=config.data.shard_size),
        Vocab(str(vocab_path)),
        [line.strip() for line in open(documents_path) if line.strip()],
        cache_dir=str(tmp_path / 'cache'))

    torch.manual_seed(config.seed)
    model = build_model(config, words=len(VOCAB_WORDS) + 3, pad_idx=1)

    log_path = tmp_path / f'{attention}.jsonl'
    ckpt_path = tmp_path / f'{attention}.pt'

    # Train 3 steps, checkpoint.
    train(config, corpus=corpus, model=model, stop_after=3,
          log_path=str(log_path), checkpoint_path=str(ckpt_path))
    assert os.path.exists(str(ckpt_path))

    # Resume to 6 steps.
    result = train(config, corpus=corpus, model=model,
                   resume_from=str(ckpt_path),
                   log_path=str(log_path), checkpoint_path=str(ckpt_path))
    assert result['final_step'] == 6
    assert 'val_loss' in result and 'val_ppl' in result

    # The log is JSON lines with the required fields (TR-6).
    events = [json.loads(line) for line in open(str(log_path))]
    training_records = [e for e in events if 'step' in e
                        and 'loss' in e]
    assert training_records, 'no training records logged'
    for record in training_records:
        assert {'step', 'loss', 'lr', 'grad_norm', 'grad_norm_pre_clip',
                'tokens_per_second'} <= set(record)
    assert any(e.get('event') == 'start' for e in events)
    assert any(e.get('event') == 'eval' for e in events)

    # The checkpoint stores the full resume payload (PRD §12).
    payload = torch.load(str(ckpt_path), map_location='cpu',
                         weights_only=False)
    assert {'schema_version', 'config', 'config_fingerprint', 'step',
            'model', 'optimizer', 'rng', 'data_cursor', 'metrics'} \
        <= set(payload)
    assert payload['step'] == 6

    # The run artefacts live under the caller-provided tmp_path, not the
    # repository's experiments/runs/ directory.
    assert 'experiments' not in str(tmp_path)
