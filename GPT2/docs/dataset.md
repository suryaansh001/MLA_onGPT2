# Dataset documentation

## Status

- **Real dataset (nampdn-ai/tiny-webtext): blocked on access (U-E1).** The
  dataset card states access is gated behind a Hugging Face login. No
  `HF_TOKEN` is set in this environment, so the real schema inspection
  (PRD U-D1) has NOT been performed. Per AGENTS.md D-6/D-7 the loader is not
  written against a guessed schema; the pipeline is written against the
  document-stream interface and verified with a local fixture.
- **Tokenizer (D-T1, user decision): repository vocabulary.** The
  repository ships the `Vocab`/`Tokenizer` classes but no vocabulary file.
  A real vocabulary file for the corpus is required from the user before
  any real-data run. Tests use a small fixture vocabulary.

## Pipeline (src/gpt2/data/token_corpus.py)

- **Input interface:** an iterable of raw text documents. A streaming HF
  datasets adapter will be plugged in here once access is confirmed; the
  adapter is a separate concern from the corpus logic.
- **Encoding:** batched (64 documents per batch) with the repository
  `Tokenizer`; every document is terminated with the vocabulary's `</s>`
  (eos) token, whose id is recorded in the manifest. Dropped-document
  counts are recorded in `manifest['counts']` (D-8).
- **Limits:** `record_limit` and `token_limit` both stop the stream; the
  pending batch is flushed so already-read documents are kept.
- **Split:** `split_assignment(index, seed, val_fraction)` — MD5 hash of
  the document index, computed before any shuffling; train/val sets are
  disjoint by construction (verified to 2,000 documents in tests).
- **Packing (D-D1 option A):** documents are concatenated with separators
  into rows of `seq_len + 1` tokens; cross-document attention is allowed.
  The final partial row is dropped and counted
  (`dropped_final_sequences`).
- **Cache:** uint16 `.npy` shards (`shard_size` sequences each), written
  atomically (temp file + rename), plus `manifest.json` recording: schema
  version, dataset identifier/revision, tokenizer identifier and vocab
  size, the config fingerprint, counts (records read/dropped, tokens
  encoded, dropped final sequences, documents per split) and a SHA-256 per
  shard. `TokenCorpus.load` verifies the config fingerprint and every shard
  hash before use; a corrupted byte fails loudly.
- **Cursor:** `(shard_index, offset_in_shard)` per split, serialisable for
  checkpointing (PRD §12); `next_batch` advances it and wraps at the end;
  inputs/targets are the standard next-token shift over the packed rows.

## Manifest example

```json
{
  "schema_version": 1,
  "dataset": {"identifier": "fixture", "revision": "unavailable"},
  "tokenizer": {"identifier": "gpt2-research-vocab", "vocab_size": 24,
                "eos_idx": 1, "pad_idx": 2},
  "config": {"seq_len": 8, "val_fraction": 0.5, "seed": 3, "shard_size": 4},
  "counts": {"records_read": 20, "records_dropped": 0, "tokens_encoded": 160,
             "dropped_final_sequences": {"train": 3, "val": 4},
             "documents": {"train": 10, "val": 10}},
  "splits": {"train": {"shards": [{"file": "train-0000.npy",
                                   "sequences": 4, "sha256": "..."}]}, ...}
}
```

## Open items

- U-D1: stream one real record and record its schema here before any
  real loader work.
- U-D2: licence review before sharing any derived shard.
- Real vocabulary file for the corpus (user decision D-T1 consequence).
