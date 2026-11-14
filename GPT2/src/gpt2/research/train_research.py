import json
import math
import os
import random
import time
from gpt2.data import Vocab
from gpt2.data.token_corpus import TokenCorpus, TokenCorpusConfig
from gpt2.modeling import Transformer
from gpt2.research.config import ResearchConfig
from typing import Dict, Optional, Tuple

import numpy as np
import torch

CHECKPOINT_SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# Construction helpers
# ----------------------------------------------------------------------

def resolve_device(config: ResearchConfig) -> torch.device:
    if config.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(config.device)


def resolve_precision(config: ResearchConfig,
                      device: torch.device) -> Tuple[str, bool]:
    """Return (autocast dtype or None, use_grad_scaler).

    bfloat16 where supported; float16 with a gradient scaler otherwise;
    float32 stays plain (TR-3)."""
    precision = config.precision
    if precision == 'auto':
        if device.type == 'cuda':
            precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
        else:
            precision = 'fp32'
    if precision == 'bf16':
        return torch.bfloat16, False
    if precision == 'fp16':
        return torch.float16, True
    return None, False


def build_corpus(config: ResearchConfig) -> TokenCorpus:
    corpus_config = TokenCorpusConfig(
        seq_len=config.model.seq_len,
        val_fraction=config.data.val_fraction,
        seed=config.data.data_seed,
        shard_size=config.data.shard_size)

    cache_dir = config.data.cache_dir
    if cache_dir and os.path.exists(
            os.path.join(cache_dir, 'manifest.json')):
        try:
            return TokenCorpus.load(corpus_config, cache_dir)
        except ValueError:
            pass    # fall through and rebuild

    vocab = Vocab(config.data.vocab_path)
    documents = read_documents(config)
    return TokenCorpus.build(corpus_config, vocab, documents,
                             cache_dir=cache_dir,
                             dataset_identifier='documents-stream',
                             record_limit=config.data.record_limit,
                             token_limit=config.data.token_limit)


def read_documents(config: ResearchConfig):
    """Yield documents from a plain text source file (one document per
    line). Real dataset streaming is added at this seam once gated access
    is confirmed (PRD U-E1)."""
    with open(config.data.vocab_path + '.documents', 'r',
              encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield line


def build_model(config: ResearchConfig,
                words: int,
                pad_idx: int) -> Transformer:
    model_config = config.model
    return Transformer(
        layers=model_config.layers,
        pad_idx=pad_idx,
        words=words,
        seq_len=model_config.seq_len,
        heads=model_config.heads,
        dims=model_config.dims,
        rate=model_config.rate,
        dropout=model_config.dropout,
        bidirectional=False,
        attention=model_config.attention,
        positional_embedding=model_config.positional_embedding,
        kv_latent_dims=model_config.kv_latent_dims,
        query_latent_dims=model_config.query_latent_dims,
        rope_head_dims=model_config.rope_head_dims,
        use_query_latent=model_config.use_query_latent,
        use_latent_norm=model_config.use_latent_norm)


def build_optimizer(config: ResearchConfig,
                    model: torch.nn.Module) -> torch.optim.AdamW:
    """AdamW with weight decay on matrix parameters only (TR-9): biases,
    LayerNorm gains and embeddings are excluded."""
    no_decay_names = ('bias', 'norm', 'embedding')
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name.lower() for token in no_decay_names):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [{'params': decay, 'weight_decay': config.optimiser.weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    return torch.optim.AdamW(groups, lr=config.optimiser.peak_lr,
                             betas=config.optimiser.betas)


def lr_at(config: ResearchConfig, step: int) -> float:
    """Linear warm-up over the first 5% of steps, then linear decay to 10%
    of peak (D-L1). Step is 0-based; the update at step s uses lr_at(s)."""
    peak = config.optimiser.peak_lr
    warmup_steps = max(1, int(config.optimiser.warmup_fraction * config.steps))
    total_decay_steps = max(1, config.steps - 1 - warmup_steps)
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    fraction = min(1.0, (step - warmup_steps) / total_decay_steps)
    return peak * (1.0 - fraction * (1.0 - config.optimiser.final_fraction))


# ----------------------------------------------------------------------
# Checkpointing (PRD §12)
# ----------------------------------------------------------------------

def capture_rng() -> Dict:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path: str,
                    config: ResearchConfig,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    step: int,
                    corpus: TokenCorpus,
                    metrics: list,
                    scaler: Optional[torch.amp.GradScaler]) -> None:
    payload = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'config': config.to_dict(),
        'config_fingerprint': config.fingerprint(),
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'rng': capture_rng(),
        'data_cursor': {split: list(corpus.get_cursor(split))
                        for split in ('train', 'val')},
        'metrics': metrics,
    }
    # Atomic write: temporary file, then rename (PRD §12).
    tmp_path = path + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str,
                    config: ResearchConfig,
                    model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional[torch.amp.GradScaler] = None
                    ) -> Dict:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f'checkpoint: unsupported schema version '
            f'{payload.get("schema_version")}')
    if payload.get('config_fingerprint') != config.fingerprint():
        raise ValueError(
            f'checkpoint: configuration fingerprint '
            f'{payload.get("config_fingerprint")} does not match the '
            f'current configuration {config.fingerprint()}')

    model.load_state_dict(payload['model'])
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if scaler is not None and payload.get('scaler') is not None:
        scaler.load_state_dict(payload['scaler'])
    restore_rng(payload['rng'])
    return payload


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(config: ResearchConfig,
             model: torch.nn.Module,
             corpus: TokenCorpus,
             device: torch.device,
             dtype=None,
             batch_size: int = 16) -> Dict:
    """Full-split validation (TR-7): token-weighted mean NLL over every
    sequence of the split; perplexity is exp of that, not a mean of
    per-batch perplexities (AGENTS.md Appendix B-2 avoided)."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, corpus.num_sequences('val'), batch_size):
        count = min(batch_size, corpus.num_sequences('val') - start)
        xs = np.stack([corpus.sequence('val', start + i)
                       for i in range(count)])
        x = torch.from_numpy(xs[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(xs[:, 1:].astype(np.int64)).to(device)
        if dtype is None:
            output = model(x)
        else:
            with torch.autocast(device_type=device.type, dtype=dtype):
                output = model(x)
        # Eval mode returns (logits, past).
        logits = output[0] if isinstance(output, tuple) else output
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            reduction='sum')
        total_nll += nll.item()
        total_tokens += y.numel()
    model.train()

    mean_nll = total_nll / max(total_tokens, 1)
    return {'val_loss': mean_nll,
            'val_ppl': math.exp(min(mean_nll, 20.0)),
            'val_tokens': total_tokens}


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train(config: ResearchConfig,
          corpus: Optional[TokenCorpus] = None,
          model: Optional[Transformer] = None,
          resume_from: Optional[str] = None,
          log_path: Optional[str] = None,
          checkpoint_path: Optional[str] = None,
          stop_after: Optional[int] = None) -> Dict:
    """Run the training loop.

    `stop_after` stops early at that step while keeping the schedule and
    configuration of the full run — used for the exact-resume test (T-R1),
    where the partial run must follow the same LR schedule as the resumed
    continuation.
    """
    device = resolve_device(config)
    dtype, use_scaler = resolve_precision(config, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) \
        if use_scaler else None

    # Seeds for Python, NumPy and torch, including CUDA (PRD §11).
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if corpus is None:
        corpus = build_corpus(config)

    if model is None:
        vocab = Vocab(config.data.vocab_path)
        model = build_model(config, words=len(vocab), pad_idx=vocab.pad_idx)
    model.to(device)
    model.train()

    optimizer = build_optimizer(config, model)

    start_step = 0
    metrics: list = []
    if resume_from is not None:
        payload = load_checkpoint(resume_from, config, model, optimizer,
                                  scaler)
        start_step = payload['step']
        metrics = payload.get('metrics', [])
        cursor = payload['data_cursor']['train']
        corpus.set_cursor('train', tuple(cursor))

    autocast_enabled = dtype is not None
    accum = config.data.grad_accumulation
    micro_batches_per_step = accum
    tokens_per_step = config.data.batch_size * accum * config.model.seq_len

    log_path = log_path or os.path.abspath(
        f'experiments/runs/{config.run_name}/log.jsonl')
    checkpoint_path = checkpoint_path or os.path.abspath(
        f'experiments/runs/{config.run_name}/checkpoint.pt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    log_file = open(log_path, 'a')

    def log(record: Dict) -> None:
        record['time'] = time.time()
        log_file.write(json.dumps(record) + '\n')
        log_file.flush()

    log({'event': 'start', 'config_fingerprint': config.fingerprint(),
         'device': str(device), 'precision': config.precision,
         'steps': config.steps,
         'param_groups': [len(g['params'])
                          for g in optimizer.param_groups],
         'tokens_per_step': tokens_per_step})

    batch_x = config.data.batch_size
    if corpus.num_sequences('train') < batch_x:
        batch_x = corpus.num_sequences('train')

    effective_steps = stop_after if stop_after is not None else config.steps
    step = start_step
    while step < effective_steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_grad_norm = None
        step_start = time.time()

        for micro in range(micro_batches_per_step):
            x_np, y_np = corpus.next_batch('train', batch_x)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)

            if autocast_enabled:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1)) / micro_batches_per_step
            else:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)) / micro_batches_per_step

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += loss.item() * micro_batches_per_step

        if scaler is not None:
            scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters()
                 if p.grad is not None]
        step_grad_norm = torch.norm(torch.stack(
            [g.norm() for g in grads])).item() if grads else 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimiser.clip_norm)

        for group in optimizer.param_groups:
            group['lr'] = lr_at(config, step)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        loss_value = step_loss / micro_batches_per_step
        step += 1

        if not math.isfinite(loss_value):
            # Non-finite loss: write a checkpoint, then stop (TR-10).
            log({'event': 'non_finite_loss', 'step': step,
                 'loss': loss_value})
            save_checkpoint(checkpoint_path, config, model, optimizer,
                            step, corpus, metrics, scaler)
            break

        record = {'step': step, 'loss': loss_value,
                  'lr': lr_at(config, step),
                  'grad_norm': grad_norm.item(),
                  'grad_norm_pre_clip': step_grad_norm,
                  'tokens_per_second': tokens_per_step /
                  max(time.time() - step_start, 1e-9)}
        metrics.append(record)

        if step % config.log_interval == 0:
            elapsed = time.time() - step_start
            record['elapsed'] = elapsed
            log(record)

        if step % config.eval_interval == 0 or step == effective_steps:
            val = evaluate(config, model, corpus, device, dtype)
            record.update(val)
            log({'event': 'eval', 'step': step, **val})
            metrics.append({'step': step, **val})

        if step % config.save_steps == 0 or step == effective_steps:
            save_checkpoint(checkpoint_path, config, model, optimizer,
                            step, corpus, metrics, scaler)

    log_file.close()

    final_val = evaluate(config, model, corpus, device, dtype)
    return {'steps_completed': step - start_step,
            'final_step': step,
            'config_fingerprint': config.fingerprint(),
            **final_val}
