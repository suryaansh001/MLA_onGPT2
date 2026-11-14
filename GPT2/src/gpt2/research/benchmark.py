import json
import os
import platform
import statistics
import subprocess
import time
from typing import Dict, List, Optional

import torch

import gpt2.research


def capture_environment(config_fingerprint: Optional[str] = None) -> Dict:
    """Record the environment for every benchmark (AGENTS.md B-2)."""
    env: Dict = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_version': torch.backends.cudnn.version()
        if torch.cuda.is_available() else None,
        'gpt2_research_version': '1',
        'config_fingerprint': config_fingerprint,
    }
    if torch.cuda.is_available():
        env['gpu_name'] = torch.cuda.get_device_name(0)
        env['gpu_count'] = torch.cuda.device_count()
        env['driver'] = _nvidia_driver_version()
    else:
        env['gpu_name'] = None
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True).strip()
        env['commit'] = commit
    except Exception:
        env['commit'] = 'unknown'
    return env


def _nvidia_driver_version() -> Optional[str]:
    try:
        output = subprocess.check_output(['nvidia-smi'], text=True)
        for line in output.splitlines():
            if 'Driver Version' in line:
                return line.split('Driver Version:')[1].split()[0].rstrip(',')
    except Exception:
        pass
    return None


def set_deterministic_flags(deterministic: bool = True) -> Dict:
    """Set and record the benchmark flags explicitly (AGENTS.md B-3)."""
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(deterministic,
                                               warn_only=True)
        except Exception:
            pass
    return {'cudnn_benchmark': not deterministic,
            'cudnn_deterministic': deterministic,
            'torch_deterministic_algorithms': deterministic}


def _median_spread(samples: List[float]) -> Dict[str, float]:
    return {'median': statistics.median(samples),
            'min': min(samples),
            'max': max(samples),
            'n': len(samples)}


def benchmark_model(model: torch.nn.Module,
                    seq_len: int,
                    batch_size: int = 1,
                    vocab_size: int = 100,
                    warmup: int = 5,
                    repeats: int = 20,
                    device: Optional[torch.device] = None) -> Dict:
    """Measure forward/backward latency and peak memory (AGENTS.md §6).

    Warm-up precedes timing; each statistic is the median over `repeats`
    with min/max spread; peak memory is reset before each measurement
    (B-1, B-4)."""
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    x = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(3, vocab_size, (batch_size, seq_len),
                            device=device)

    # Warm-up (excluded from statistics).
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    model.train()
    for _ in range(warmup):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    forward_times = []
    for _ in range(repeats):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1e3)

    backward_times = []
    peak_memory = []
    for _ in range(repeats):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1e3)
        if device.type == 'cuda':
            peak_memory.append(torch.cuda.max_memory_allocated())

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    result = {
        'seq_len': seq_len,
        'batch_size': batch_size,
        'forward_ms': _median_spread(forward_times),
        'forward_backward_ms': _median_spread(backward_times),
        'parameters': sum(p.numel() for p in model.parameters()),
    }
    if peak_memory:
        result['peak_memory_bytes'] = {
            'median': statistics.median(peak_memory),
            'max': max(peak_memory)}
    return result


@torch.no_grad()
def benchmark_generation(model: torch.nn.Module,
                         seq_len: int,
                         batch_size: int = 1,
                         vocab_size: int = 100,
                         steps: int = 32,
                         device: Optional[torch.device] = None) -> Dict:
    """Separate time to first token from time per output token (B-5)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    x = torch.randint(3, vocab_size, (batch_size, 1), device=device)

    for _ in range(3):                     # warm-up
        output = model(x)
        logits, past = output

    samples = []
    for _ in range(5):
        start = time.perf_counter()
        output = model(x)
        logits, past = output
        prefill_ms = (time.perf_counter() - start) * 1e3

        token_times = []
        token = int(logits[:, -1].argmax())
        for _ in range(steps):
            start = time.perf_counter()
            output = model(torch.tensor([[token]], device=device),
                           past=past)
            logits, past = output
            token = int(logits[:, -1].argmax())
            token_times.append((time.perf_counter() - start) * 1e3)
        samples.append((prefill_ms, token_times[-steps:]))

    return {
        'time_to_first_token_ms': _median_spread([s[0] for s in samples]),
        'time_per_output_token_ms': _median_spread(
            [statistics.median(s[1]) for s in samples]),
        'decoded_steps': steps,
    }


def write_results(results: Dict, path: str) -> None:
    """Structured results file; plots are generated from it, never typed
    (B-7, DOC-4)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fp:
        json.dump(results, fp, indent=2)
    os.replace(tmp, path)
