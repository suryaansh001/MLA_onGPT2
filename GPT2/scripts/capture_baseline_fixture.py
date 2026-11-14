import hashlib
import json
import subprocess
import sys

import torch

from gpt2.modeling import Transformer

FIXTURE_PATH = 'tests/fixtures/baseline_reference.npz'
META_PATH = 'tests/fixtures/baseline_reference.json'


def commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def build_model():
    torch.manual_seed(1234)
    model = Transformer(layers=2, pad_idx=0, words=80, seq_len=100,
                        heads=2, dims=16, rate=4, bidirectional=False).eval()
    return model


def run_fixture():
    model = build_model()
    param_count = sum(p.numel() for p in model.parameters())

    torch.manual_seed(4321)
    x = torch.randint(80, (1, 8))
    with torch.no_grad():
        logits, past = model(x)

    # Incremental decoding step with a length-1 query (batch 1). A 1-D input
    # tensor is not supported by the baseline's head splitting when a 2-D
    # past is present (pre-existing baseline behaviour), so use shape (1, 1).
    with torch.no_grad():
        logits_next, past_next = model(torch.randint(80, (1, 1)), past=past)

    torch.save({
        'input': x,
        'logits': logits,
        'past_k': past[0][0],
        'past_v': past[0][1],
        'logits_next': logits_next,
        'past_next_k': past_next[0][0],
        'past_next_v': past_next[0][1],
    }, FIXTURE_PATH)

    digest = hashlib.sha256(
        logits.numpy().tobytes() + logits_next.numpy().tobytes()).hexdigest()
    meta = {
        'commit': commit_hash(),
        'seed_model': 1234,
        'seed_input': 4321,
        'param_count': param_count,
        'config': {'layers': 2, 'pad_idx': 0, 'words': 80, 'seq_len': 100,
                   'heads': 2, 'dims': 16, 'rate': 4, 'bidirectional': False},
        'logits_sha256': digest,
        'logits_shape': list(logits.shape),
    }
    with open(META_PATH, 'w') as fp:
        json.dump(meta, fp, indent=2)
    print(json.dumps(meta, indent=2))


def verify_fixture():
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == \
        json.load(open(META_PATH))['param_count'], 'param count mismatch'

    data = torch.load(FIXTURE_PATH, weights_only=False)
    torch.manual_seed(4321)
    x = torch.randint(80, (1, 8))
    assert torch.equal(x, data['input']), 'input mismatch'

    with torch.no_grad():
        logits, past = model(x)
        logits_next, _ = model(torch.randint(80, (1, 1)), past=past)

    assert torch.equal(logits, data['logits']), 'logits mismatch'
    assert torch.equal(logits_next, data['logits_next']), 'next logits mismatch'
    print('fixture verified: outputs identical')


if __name__ == '__main__':
    if '--verify' in sys.argv:
        verify_fixture()
    else:
        run_fixture()
        verify_fixture()
