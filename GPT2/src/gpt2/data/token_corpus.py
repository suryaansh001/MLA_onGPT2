import hashlib
import json
import os
import shutil
import tempfile
from gpt2.data import Vocab, Tokenizer
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


MANIFEST_SCHEMA_VERSION = 1
ENCODE_BATCH_SIZE = 64


def split_assignment(index: int,
                     seed: int,
                     val_fraction: float) -> str:
    """Deterministically assign a document index to 'train' or 'val'.

    The assignment depends only on the document index, the seed and the
    validation fraction. It is computed before any shuffling, so no document
    can appear in both splits (AGENTS.md D-5).
    """
    digest = hashlib.md5(f'{seed}:{index}'.encode()).hexdigest()
    uniform = int(digest[:8], 16) / float(0xffffffff)
    return 'val' if uniform < val_fraction else 'train'


class TokenCorpusConfig:
    """Configuration for the research tokenised corpus."""

    def __init__(self,
                 seq_len: int,
                 val_fraction: float = 0.02,
                 seed: int = 1234,
                 shard_size: int = 1000):
        if seq_len < 1:
            raise ValueError(f'seq_len: must be positive, got {seq_len}')
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(
                f'val_fraction: must be in [0, 1), got {val_fraction}')
        if shard_size < 1:
            raise ValueError(f'shard_size: must be positive, got {shard_size}')

        self.seq_len = seq_len
        self.val_fraction = val_fraction
        self.seed = seed
        self.shard_size = shard_size

    def fingerprint(self) -> Dict:
        return {'seq_len': self.seq_len,
                'val_fraction': self.val_fraction,
                'seed': self.seed,
                'shard_size': self.shard_size}


class TokenCorpus:
    """
    Packed, sharded, tokenised corpus for the research pipeline.

    Documents are encoded with the repository tokenizer (D-T1: repository
    vocabulary), separated by the vocabulary's end-of-sentence token,
    deterministically split by document index, packed into fixed-length
    sequences of `seq_len + 1` tokens (D-D1 option A: cross-document
    attention allowed), and cached as uint16 `.npy` shards with a JSON
    manifest (PRD FR-07).
    """

    def __init__(self, config: TokenCorpusConfig):
        self.config = config
        self.manifest: Optional[Dict] = None
        self._cache_dir: Optional[str] = None
        self._owned_cache_dir = False
        self._shards: Dict[str, List[np.ndarray]] = {'train': [], 'val': []}
        self._cursors: Dict[str, Tuple[int, int]] = {'train': (0, 0),
                                                     'val': (0, 0)}
        self._dropped_records = 0

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build(cls,
              config: TokenCorpusConfig,
              vocab: Vocab,
              documents: Iterable[str],
              cache_dir: Optional[str] = None,
              dataset_identifier: str = 'fixture',
              dataset_revision: str = 'unavailable',
              record_limit: Optional[int] = None,
              token_limit: Optional[int] = None) -> 'TokenCorpus':
        """Encode, split, pack and cache a stream of documents.

        `documents` is an iterable of raw text. `record_limit` bounds the
        number of documents read; `token_limit` bounds the number of encoded
        tokens (including separators). Whichever limit stops the stream
        first, the remaining documents are dropped and counted (D-8).
        """
        corpus = cls(config)
        tokenizer = Tokenizer(vocab)
        eos_idx = vocab.eos_idx

        streams: Dict[str, List[int]] = {'train': [], 'val': []}
        doc_counts: Dict[str, int] = {'train': 0, 'val': 0}
        records_read = 0
        records_dropped = 0
        tokens_encoded = 0
        stopped = False

        batch: List[Tuple[int, str]] = []
        for index, document in enumerate(documents):
            if stopped:
                records_dropped += 1
                continue
            if record_limit is not None and records_read >= record_limit:
                records_dropped += 1
                stopped = True
                continue
            records_read += 1
            batch.append((index, document))
            if len(batch) < ENCODE_BATCH_SIZE:
                continue

            tokens_in_batch = corpus._encode_batch(
                batch, tokenizer, vocab, eos_idx, streams, doc_counts,
                tokens_encoded, token_limit)
            tokens_encoded += tokens_in_batch
            if token_limit is not None and tokens_encoded >= token_limit:
                stopped = True
            batch = []

        if batch:
            # Flush the pending batch even when a limit stopped the stream:
            # the documents already read still belong in the corpus.
            tokens_encoded += corpus._encode_batch(
                batch, tokenizer, vocab, eos_idx, streams, doc_counts,
                tokens_encoded, token_limit)

        corpus._pack_and_cache(streams, doc_counts, cache_dir,
                               dataset_identifier, dataset_revision, vocab,
                               records_read,
                               records_dropped + getattr(corpus,
                                                         '_dropped_records',
                                                         0),
                               tokens_encoded)
        return corpus

    def _encode_batch(self,
                      batch: List[Tuple[int, str]],
                      tokenizer: Tokenizer,
                      vocab: Vocab,
                      eos_idx: int,
                      streams: Dict[str, List[int]],
                      doc_counts: Dict[str, int],
                      tokens_so_far: int,
                      token_limit: Optional[int]) -> int:
        """Encode a batch of (index, document) pairs into the split streams.

        Returns the number of tokens added. Documents whose encoding would
        push the running total past `token_limit` are dropped (counted as
        records dropped in the manifest) and the batch stops adding.
        """
        encoded = []
        for index, document in batch:
            ids = [vocab[token] for token in tokenizer.encode(document)]
            ids.append(eos_idx)
            encoded.append((index, ids))

        added = 0
        for index, ids in encoded:
            if token_limit is not None and \
                    tokens_so_far + added + len(ids) > token_limit:
                self._dropped_records += 1
                continue
            split = split_assignment(index, self.config.seed,
                                     self.config.val_fraction)
            streams[split].extend(ids)
            doc_counts[split] += 1
            added += len(ids)
        return added

    # ------------------------------------------------------------------
    # Packing and caching
    # ------------------------------------------------------------------

    def _pack_and_cache(self,
                        streams: Dict[str, List[int]],
                        doc_counts: Dict[str, int],
                        cache_dir: Optional[str],
                        dataset_identifier: str,
                        dataset_revision: str,
                        vocab: Vocab,
                        records_read: int,
                        records_dropped: int,
                        tokens_encoded: int) -> None:
        seq_len_plus_one = self.config.seq_len + 1
        shard_arrays: Dict[str, List[np.ndarray]] = {'train': [], 'val': []}
        dropped_sequences: Dict[str, int] = {'train': 0, 'val': 0}

        self._owned_cache_dir = cache_dir is None
        if self._owned_cache_dir:
            cache_dir = tempfile.mkdtemp(prefix='token_corpus_')

        for split in ('train', 'val'):
            tokens = streams[split]
            usable = (len(tokens) // seq_len_plus_one) * seq_len_plus_one
            # The final partial sequence is dropped and counted (PRD FR-07).
            dropped_sequences[split] = len(tokens) - usable

            sequences = np.asarray(tokens[:usable], dtype=np.uint16) \
                .reshape(-1, seq_len_plus_one)
            for shard_index in range(0, len(sequences),
                                     self.config.shard_size):
                shard_arrays[split].append(
                    sequences[shard_index:shard_index + self.config.shard_size])

        manifest = {
            'schema_version': MANIFEST_SCHEMA_VERSION,
            'dataset': {'identifier': dataset_identifier,
                        'revision': dataset_revision},
            'tokenizer': {
                'identifier': 'gpt2-research-vocab',
                'vocab_size': len(vocab),
                'eos_idx': vocab.eos_idx,
                'pad_idx': vocab.pad_idx,
            },
            'config': self.config.fingerprint(),
            'counts': {
                'records_read': records_read,
                'records_dropped': records_dropped,
                'tokens_encoded': tokens_encoded,
                'dropped_final_sequences': dropped_sequences,
                'documents': dict(doc_counts),
            },
            'splits': {},
        }

        self._write_shards(manifest, shard_arrays, cache_dir)
        with open(os.path.join(cache_dir, 'manifest.json'), 'w') as fp:
            json.dump(manifest, fp, indent=2)
        self.manifest = manifest
        self._cache_dir = cache_dir
        for split in ('train', 'val'):
            self._shards[split] = [
                np.load(os.path.join(cache_dir, shard['file']))
                for shard in manifest['splits'][split]['shards']]

    def _write_shards(self,
                      manifest: Dict,
                      shard_arrays: Dict[str, List[np.ndarray]],
                      cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        for split in ('train', 'val'):
            entries = []
            for shard_index, shard in enumerate(shard_arrays[split]):
                filename = f'{split}-{shard_index:04d}.npy'
                path = os.path.join(cache_dir, filename)

                # Atomic write: temporary file, then rename. np.save
                # appends '.npy' when missing, so the temp name keeps it.
                tmp_path = path + '.tmp.npy'
                np.save(tmp_path, shard)
                os.replace(tmp_path, path)

                with open(path, 'rb') as fp:
                    digest = hashlib.sha256(fp.read()).hexdigest()
                entries.append({'file': filename,
                                'sequences': int(shard.shape[0]),
                                'sha256': digest})
            manifest['splits'][split] = {'shards': entries}

    # ------------------------------------------------------------------
    # Loading from cache
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config: TokenCorpusConfig, cache_dir: str) -> 'TokenCorpus':
        """Load a cached corpus, verifying the manifest and every shard
        hash before use (AGENTS.md D-3)."""
        with open(os.path.join(cache_dir, 'manifest.json')) as fp:
            manifest = json.load(fp)

        if manifest['schema_version'] != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f'manifest: unsupported schema version '
                f'{manifest["schema_version"]}')
        if manifest['config'] != config.fingerprint():
            raise ValueError(
                f'config: cached corpus fingerprint {manifest["config"]} '
                f'does not match the requested configuration '
                f'{config.fingerprint()}')

        corpus = cls(config)
        for split in ('train', 'val'):
            shards = []
            for shard in manifest['splits'][split]['shards']:
                path = os.path.join(cache_dir, shard['file'])
                with open(path, 'rb') as fp:
                    digest = hashlib.sha256(fp.read()).hexdigest()
                if digest != shard['sha256']:
                    raise ValueError(
                        f'shard {shard["file"]}: content hash {digest} does '
                        f'not match the manifest hash {shard["sha256"]}')
                shards.append(np.load(path))
            corpus._shards[split] = shards

        corpus.manifest = manifest
        corpus._cache_dir = cache_dir
        corpus._owned_cache_dir = False
        return corpus

    def persist_manifest(self, path: str) -> None:
        with open(path, 'w') as fp:
            json.dump(self.manifest, fp, indent=2)

    def __del__(self):
        if self._owned_cache_dir and self._cache_dir and \
                os.path.isdir(self._cache_dir):
            shutil.rmtree(self._cache_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Access and iteration
    # ------------------------------------------------------------------

    def num_sequences(self, split: str) -> int:
        return sum(shard.shape[0] for shard in self._shards[split])

    def sequence(self, split: str, index: int) -> np.ndarray:
        total = self.num_sequences(split)
        if not 0 <= index < total:
            raise IndexError(
                f'index: {index} out of range for split {split} with '
                f'{total} sequences')
        shard_index, offset = divmod(index, self.config.shard_size)
        return self._shards[split][shard_index][offset]

    def get_cursor(self, split: str) -> Tuple[int, int]:
        return self._cursors[split]

    def set_cursor(self, split: str, cursor: Tuple[int, int]) -> None:
        shard_index, offset = cursor
        if not 0 <= shard_index < len(self._shards[split]):
            raise ValueError(
                f'cursor: shard index {shard_index} out of range for '
                f'split {split}')
        if not 0 <= offset < self.config.shard_size:
            raise ValueError(
                f'cursor: offset {offset} out of range')
        self._cursors[split] = (shard_index, offset)

    def next_batch(self, split: str,
                   batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the next (inputs, targets) batch, advancing the cursor.

        Inputs are sequences[:, :-1] and targets sequences[:, 1:] over
        `seq_len + 1` packed tokens. The cursor wraps to the start of the
        corpus at the end.
        """
        batch_x, batch_y = [], []
        rows = 0
        while rows < batch_size:
            shard_index, offset = self._cursors[split]
            if shard_index >= len(self._shards[split]):
                shard_index, offset = 0, 0
            shard = self._shards[split][shard_index]
            take = min(batch_size - rows, shard.shape[0] - offset)
            batch_x.append(shard[offset:offset + take, :-1])
            batch_y.append(shard[offset:offset + take, 1:])
            rows += take

            offset += take
            if offset >= shard.shape[0]:
                shard_index, offset = shard_index + 1, 0
            self._cursors[split] = (shard_index, offset)

        return (np.concatenate(batch_x, axis=0).astype(np.int64),
                np.concatenate(batch_y, axis=0).astype(np.int64))
