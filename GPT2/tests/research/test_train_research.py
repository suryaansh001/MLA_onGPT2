import json
import os

import numpy as np
import pytest

from gpt2.data import Vocab, Tokenizer
from gpt2.data.token_corpus import (TokenCorpus, TokenCorpusConfig,
                                    split_assignment)

VOCAB_WORDS = ['the', 'cat', 'sat', 'on', 'mat', 'a', 'dog', 'ran', 'fast',
               'slow', 'red', 'big', '.', '##at', '##og', '##ed', '##ing',
               '##s', 'jumped', 'sleepy']


@pytest.fixture
def vocab_path(tmp_path):
    path = tmp_path / 'vocab.txt'
    path.write_text(' '.join(VOCAB_WORDS))
    return str(path)


@pytest.fixture
def vocab(vocab_path):
    return Vocab(vocab_path)


def test_tokenizer_round_trip_with_separator(vocab):
    tokenizer = Tokenizer(vocab)
    text = 'the cat sat on the mat.'
    ids = [vocab[token] for token in tokenizer.encode(text)]

    decoded = tokenizer.decode([vocab[i] for i in ids])
    reencoded = [vocab[token] for token in tokenizer.encode(decoded)]
    assert reencoded == ids


def test_split_assignment_disjoint_and_deterministic():
    assignments = [split_assignment(i, seed=17, val_fraction=0.2)
                   for i in range(2000)]
    train = {i for i, s in enumerate(assignments) if s == 'train'}
    val = {i for i, s in enumerate(assignments) if s == 'val'}

    assert len(train) + len(val) == 2000
    assert not train & val
    assert 300 < len(val) < 500          # ~20% with slack

    again = [split_assignment(i, seed=17, val_fraction=0.2)
             for i in range(2000)]
    assert again == assignments


def test_build_and_pack_documents(vocab, tmp_path):
    documents = ['the cat sat on the mat.'] * 20
    config = TokenCorpusConfig(seq_len=8, val_fraction=0.5, seed=3,
                               shard_size=4)
    corpus = TokenCorpus.build(config, vocab, documents,
                               cache_dir=str(tmp_path / 'cache'))

    # Each document encodes to 7 ids (6 words + '.') + 1 separator = 8
    # tokens; 20 documents produce 160 tokens; 160 // 9 = 17 packed
    # sequences, 7 tokens dropped.
    total = corpus.num_sequences('train') + corpus.num_sequences('val')
    assert total == 17
    dropped = corpus.manifest['counts']['dropped_final_sequences']
    assert dropped['train'] + dropped['val'] == 7

    manifest_docs = corpus.manifest['counts']['documents']
    assert manifest_docs['train'] + manifest_docs['val'] == 20

    # Sequence content: every row is uint16 of length seq_len + 1.
    seq = corpus.sequence('train', 0)
    assert seq.dtype == np.uint16
    assert seq.shape == (9,)


def test_two_runs_produce_identical_shards(vocab, tmp_path):
    documents = [f'the {w} dog ran fast.' for w in
                 ('red', 'big', 'sleepy', 'slow', 'fast')]
    config = TokenCorpusConfig(seq_len=6, val_fraction=0.25, seed=11,
                               shard_size=3)

    corpus_a = TokenCorpus.build(config, vocab, documents,
                                 cache_dir=str(tmp_path / 'a'))
    corpus_b = TokenCorpus.build(config, vocab, documents,
                                 cache_dir=str(tmp_path / 'b'))

    hashes_a = {(s, e['file']): e['sha256']
                for s in ('train', 'val')
                for e in corpus_a.manifest['splits'][s]['shards']}
    hashes_b = {(s, e['file']): e['sha256']
                for s in ('train', 'val')
                for e in corpus_b.manifest['splits'][s]['shards']}
    assert hashes_a and hashes_a == hashes_b

    for (split, file), digest in hashes_a.items():
        with open(os.path.join(str(tmp_path / 'a'), file), 'rb') as fp:
            assert len(fp.read()) > 0


def test_record_limit_drops_and_counts(vocab, tmp_path):
    documents = ['the cat sat on the mat.'] * 30
    config = TokenCorpusConfig(seq_len=8, val_fraction=0.0, seed=5)
    corpus = TokenCorpus.build(config, vocab, documents,
                               cache_dir=str(tmp_path / 'cache'),
                               record_limit=10)

    counts = corpus.manifest['counts']
    assert counts['records_read'] == 10
    assert counts['records_dropped'] == 20
    assert counts['documents']['train'] == 10


def test_token_limit_drops_and_counts(vocab, tmp_path):
    documents = ['the cat sat on the mat.'] * 10   # 7 tokens each
    config = TokenCorpusConfig(seq_len=8, val_fraction=0.0, seed=5)
    corpus = TokenCorpus.build(config, vocab, documents,
                               cache_dir=str(tmp_path / 'cache'),
                               token_limit=21)

    counts = corpus.manifest['counts']
    assert counts['tokens_encoded'] <= 21
    assert counts['records_dropped'] >= 1
    # Two documents fit (16 tokens); 16 // 9 = 1 full sequence, 7 dropped.
    total = corpus.num_sequences('train') + corpus.num_sequences('val')
    assert total == 1


def test_load_verifies_manifest_and_detects_corruption(vocab, tmp_path):
    documents = ['the cat sat on the mat.'] * 10
    config = TokenCorpusConfig(seq_len=8, val_fraction=0.0, seed=5,
                               shard_size=2)
    cache = str(tmp_path / 'cache')
    TokenCorpus.build(config, vocab, documents, cache_dir=cache)

    # Clean load succeeds.
    corpus = TokenCorpus.load(config, cache)
    assert corpus.num_sequences('train') > 0

    # Corrupt a shard byte: the manifest hash check must fail loudly.
    shard_file = corpus.manifest['splits']['train']['shards'][0]['file']
    with open(os.path.join(cache, shard_file), 'r+b') as fp:
        fp.seek(64)
        fp.write(b'\xff')

    with pytest.raises(ValueError, match='sha256|hash'):
        TokenCorpus.load(config, cache)


def test_load_rejects_config_mismatch(vocab, tmp_path):
    documents = ['the cat sat on the mat.'] * 10
    TokenCorpus.build(TokenCorpusConfig(seq_len=8), vocab, documents,
                      cache_dir=str(tmp_path / 'cache'))

    with pytest.raises(ValueError, match='config'):
        TokenCorpus.load(TokenCorpusConfig(seq_len=16),
                         str(tmp_path / 'cache'))


def test_cursor_serialisation_restores_next_batch(vocab, tmp_path):
    documents = ['the cat sat on the mat.'] * 10
    config = TokenCorpusConfig(seq_len=8, val_fraction=0.0, seed=5,
                               shard_size=2)
    cache = str(tmp_path / 'cache')
    corpus = TokenCorpus.build(config, vocab, documents, cache_dir=cache)

    # Snapshot the cursor BEFORE the first batch: restoring it must yield
    # the same next batch (checkpoint semantics, PRD 12).
    cursor = corpus.get_cursor('train')
    x1, y1 = corpus.next_batch('train', batch_size=2)

    # Serialise the cursor into a dict (checkpoint payload, PRD 12).
    state = {'train_cursor': list(cursor)}
    with open(os.path.join(cache, 'cursor.json'), 'w') as fp:
        json.dump(state, fp)

    # Advance further, then restore the saved cursor.
    corpus.next_batch('train', batch_size=3)
    restored = tuple(json.load(open(os.path.join(cache, 'cursor.json')))
                     ['train_cursor'])
    corpus.set_cursor('train', restored)
    assert corpus.get_cursor('train') == restored

    x2, y2 = corpus.next_batch('train', batch_size=2)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_next_batch_shapes_and_wrapping(vocab, tmp_path):
    documents = ['the cat sat on the mat.'] * 10
    config = TokenCorpusConfig(seq_len=8, val_fraction=0.0, seed=5,
                               shard_size=2)
    corpus = TokenCorpus.build(config, vocab, documents,
                               cache_dir=str(tmp_path / 'cache'))

    total = corpus.num_sequences('train')
    x, y = corpus.next_batch('train', batch_size=total)
    assert x.shape == y.shape == (total, 8)
    # Targets are the inputs shifted by one.
    assert np.array_equal(x[:, 1:], y[:, :-1])

    # Wrapping: requesting more batches keeps producing data from the start.
    x_wrapped, _ = corpus.next_batch('train', batch_size=2)
    assert np.array_equal(x_wrapped, x[:2])


def test_separator_present_between_documents(vocab, tmp_path):
    # Six text ids + one separator = 7 tokens per document; 3 documents
    # give 21 tokens packed as 3 rows of 7 with no final drop, so all three
    # separators are kept in the packed stream.
    documents = ['the cat sat on the mat', 'the big dog ran fast slow',
                 'the sleepy red dog sat fast']
    config = TokenCorpusConfig(seq_len=6, val_fraction=0.0, seed=5)
    corpus = TokenCorpus.build(config, vocab, documents,
                               cache_dir=str(tmp_path / 'cache'))
    eos = corpus.manifest['tokenizer']['eos_idx']
    tokens = np.concatenate([corpus.sequence('train', i)
                             for i in range(corpus.num_sequences('train'))])
    assert corpus.manifest['counts']['dropped_final_sequences']['train'] == 0
    assert int((tokens == eos).sum()) == 3
