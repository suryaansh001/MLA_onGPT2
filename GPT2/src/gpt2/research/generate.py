import time
import warnings
from gpt2.data import Vocab, Tokenizer
from typing import Dict, List, Optional, Tuple

import torch


@torch.no_grad()
def encode_prompt(vocab: Vocab,
                  tokenizer: Tokenizer,
                  prompt: str,
                  context_len: int) -> List[int]:
    """Encode a prompt and left-truncate it when it exceeds the context
    window, with a warning (PRD §13). The same rule applies to both arms."""
    ids = [vocab[token] for token in tokenizer.encode(prompt)]
    if len(ids) > context_len:
        warnings.warn(
            f'prompt length {len(ids)} exceeds the context window '
            f'{context_len}; truncating from the left')
        ids = ids[-context_len:]
    return ids


def top_p_sample(logits: torch.Tensor,
                 top_p: Optional[float],
                 generator: Optional[torch.Generator]) -> int:
    """Nucleus sampling: keep the smallest prefix of the sorted
    distribution with cumulative probability >= top_p, renormalise and
    sample (PRD FR-11). `top_p=None` is greedy argmax."""
    if top_p is None:
        return int(torch.argmax(logits, dim=-1).item())

    probs = logits.softmax(-1)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cumulative = sorted_probs.cumsum(-1)

    # Positions whose cumulative mass up to and including them is needed:
    # keep the first index where the cumulative sum reaches top_p and
    # everything before it.
    keep = cumulative - sorted_probs < top_p
    keep[..., 0] = True
    filtered = sorted_probs * keep
    filtered = filtered / filtered.sum(-1, keepdim=True)

    choice = torch.multinomial(filtered, 1, generator=generator)
    return sorted_idx.gather(-1, choice).item()


@torch.no_grad()
def generate(model: torch.nn.Module,
             vocab: Vocab,
             prompt: str,
             max_new_tokens: int,
             top_p: Optional[float] = 0.9,
             seed: int = 0,
             device: Optional[torch.device] = None) -> Dict:
    """Autoregressive generation with the KV cache for either arm.

    Prefill on the prompt, then incremental decoding. Stops at the
    vocabulary's end-of-sentence token or at max_new_tokens, whichever
    comes first (PRD §13). Returns the ids, the sampling seed, and timing
    (time to first token, time per output token).
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    context_len = getattr(getattr(model, 'positional_embedding', None),
                          'num_embeddings', None)
    if context_len is None:
        context_len = 10 ** 9     # no learned table: bounded by max tokens

    tokenizer = Tokenizer(vocab)
    ids = encode_prompt(vocab, tokenizer, prompt, context_len)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    timings: Dict[str, float] = {}
    per_token_times: List[float] = []
    new_tokens: List[int] = []

    start = time.time()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    output = model(x)
    logits, past = output
    timings['time_to_first_token'] = time.time() - start

    for _ in range(max_new_tokens):
        start = time.time()
        next_id = top_p_sample(logits[:, -1, :], top_p, generator)
        per_token_times.append(time.time() - start)
        new_tokens.append(next_id)

        if next_id == vocab.eos_idx:
            break

        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        output = model(x, past=past)
        logits, past = output

    if was_training:
        model.train()

    timings['time_per_output_token'] = (
        sum(per_token_times) / len(per_token_times)
        if per_token_times else 0.0)

    return {'ids': ids + new_tokens,
            'new_tokens': new_tokens,
            'stopped_at_eos': bool(new_tokens and
                                   new_tokens[-1] == vocab.eos_idx),
            'seed': seed,
            'timings': timings}


@torch.no_grad()
def full_recompute_logits(model: torch.nn.Module,
                          ids: List[int],
                          device: Optional[torch.device] = None
                          ) -> torch.Tensor:
    """Reference: logits for a full forward pass over the given ids."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    output = model(x)
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def incremental_logits(model: torch.nn.Module,
                       ids: List[int],
                       device: Optional[torch.device] = None
                       ) -> torch.Tensor:
    """Reference: logits from feeding the ids one token at a time with the
    KV cache (T-K1). Returns (1, len(ids), vocab)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    past = None
    outputs = []
    for position, token in enumerate(ids):
        x = torch.tensor([[token]], dtype=torch.long, device=device)
        output = model(x, past=past) if past is not None else model(x)
        logits, past = output if isinstance(output, tuple) else (output,
                                                                 None)
        outputs.append(logits[:, -1:, :])
    return torch.cat(outputs, dim=1)
