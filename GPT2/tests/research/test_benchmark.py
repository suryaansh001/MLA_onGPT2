import json
import os

import pytest
import torch

from gpt2.modeling import Transformer
from gpt2.research.benchmark import (benchmark_generation, benchmark_model,
                                     capture_environment,
                                     set_deterministic_flags, write_results)


def build_model(attention):
    torch.manual_seed(0)
    return Transformer(layers=1, pad_idx=0, words=32, seq_len=16, heads=2,
                       dims=16, rate=2, bidirectional=False,
                       attention=attention).eval()


@pytest.mark.parametrize('attention', ['mha', 'mla'])
def test_benchmark_structure_and_repeats(attention):
    model = build_model(attention)
    result = benchmark_model(model, seq_len=16, batch_size=1, vocab_size=32,
                             warmup=2, repeats=5)

    assert result['parameters'] == sum(p.numel() for p in model.parameters())
    for timing in ('forward_ms', 'forward_backward_ms'):
        stats = result[timing]
        assert stats['n'] == 5
        assert stats['min'] <= stats['median'] <= stats['max']
        assert stats['median'] > 0

    generation = benchmark_generation(model, seq_len=16, vocab_size=32,
                                      steps=4)
    assert generation['decoded_steps'] == 4
    for timing in ('time_to_first_token_ms', 'time_per_output_token_ms'):
        stats = generation[timing]
        assert stats['n'] == 5 and stats['median'] > 0


def test_environment_record_is_complete():
    env = capture_environment(config_fingerprint='abc123')
    assert {'python', 'torch', 'cuda_available', 'commit',
            'config_fingerprint', 'gpu_name'} <= set(env)
    assert env['config_fingerprint'] == 'abc123'
    assert env['commit']


def test_deterministic_flags_recorded():
    flags = set_deterministic_flags(deterministic=True)
    assert flags['cudnn_deterministic'] is True
    assert flags['cudnn_benchmark'] is False
    assert flags['torch_deterministic_algorithms'] is True

    set_deterministic_flags(deterministic=False)


def test_peak_memory_reset_effective():
    if not torch.cuda.is_available():
        pytest.skip('peak-memory reset requires CUDA; CPU has no peak stats')
    model = build_model('mha')
    result = benchmark_model(model, seq_len=16, vocab_size=32, repeats=2)
    assert 'peak_memory_bytes' in result
    assert result['peak_memory_bytes']['median'] > 0


def test_write_results_atomic(tmp_path):
    path = tmp_path / 'results.json'
    write_results({'key': 'value'}, str(path))
    assert json.load(open(str(path))) == {'key': 'value'}

    with open(str(path), 'rb') as fp:
        assert len(fp.read()) > 0
    assert not os.path.exists(str(path) + '.tmp')
