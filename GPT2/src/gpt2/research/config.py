import hashlib
import json
import os
from typing import Dict, Optional


class ModelConfig:
    def __init__(self,
                 layers: int,
                 dims: int,
                 heads: int,
                 seq_len: int,
                 rate: int = 4,
                 dropout: float = 0.1,
                 attention: str = 'mha',
                 positional_embedding: bool = True,
                 kv_latent_dims: int = 256,
                 query_latent_dims: int = 256,
                 rope_head_dims: int = 32,
                 use_query_latent: bool = True,
                 use_latent_norm: bool = True):
        if attention not in ('mha', 'mla'):
            raise ValueError(
                f"attention: expected 'mha' or 'mla', got {attention}")
        self.layers = layers
        self.dims = dims
        self.heads = heads
        self.seq_len = seq_len
        self.rate = rate
        self.dropout = dropout
        self.attention = attention
        self.positional_embedding = positional_embedding
        self.kv_latent_dims = kv_latent_dims
        self.query_latent_dims = query_latent_dims
        self.rope_head_dims = rope_head_dims
        self.use_query_latent = use_query_latent
        self.use_latent_norm = use_latent_norm


class DataConfig:
    def __init__(self,
                 vocab_path: str,
                 cache_dir: Optional[str] = None,
                 record_limit: Optional[int] = None,
                 token_limit: Optional[int] = None,
                 val_fraction: float = 0.02,
                 data_seed: int = 1234,
                 shard_size: int = 1000,
                 batch_size: int = 8,
                 grad_accumulation: int = 1):
        if batch_size < 1:
            raise ValueError(f'batch_size: must be positive, got {batch_size}')
        if grad_accumulation < 1:
            raise ValueError(
                f'grad_accumulation: must be positive, got {grad_accumulation}')
        self.vocab_path = vocab_path
        self.cache_dir = cache_dir
        self.record_limit = record_limit
        self.token_limit = token_limit
        self.val_fraction = val_fraction
        self.data_seed = data_seed
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation


class OptimiserConfig:
    def __init__(self,
                 peak_lr: float = 2.5e-4,
                 weight_decay: float = 0.1,
                 betas=(0.9, 0.95),
                 clip_norm: float = 1.0,
                 warmup_fraction: float = 0.05,
                 final_fraction: float = 0.1):
        self.peak_lr = peak_lr
        self.weight_decay = weight_decay
        self.betas = tuple(betas)
        self.clip_norm = clip_norm
        self.warmup_fraction = warmup_fraction
        self.final_fraction = final_fraction


class ResearchConfig:
    """Typed configuration for a research run (PRD FR-13, TR-1).

    Every hyperparameter lives here; nothing is hardcoded in the training
    loop. The configuration is serialised into every checkpoint and hashed
    into a fingerprint that resume verifies (PRD §12).
    """

    def __init__(self,
                 model: ModelConfig,
                 data: DataConfig,
                 optimiser: OptimiserConfig,
                 steps: int,
                 seed: int = 42,
                 device: str = 'auto',
                 precision: str = 'auto',
                 log_interval: int = 10,
                 eval_interval: int = 50,
                 save_steps: int = 100,
                 run_name: str = 'run'):
        if steps < 1:
            raise ValueError(f'steps: must be positive, got {steps}')
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError(f"device: expected auto/cpu/cuda, got {device}")
        if precision not in ('auto', 'fp32', 'bf16', 'fp16'):
            raise ValueError(
                f"precision: expected auto/fp32/bf16/fp16, got {precision}")
        self.model = model
        self.data = data
        self.optimiser = optimiser
        self.steps = steps
        self.seed = seed
        self.device = device
        self.precision = precision
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_steps = save_steps
        self.run_name = run_name

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            'schema_version': 1,
            'model': vars(self.model),
            'data': vars(self.data),
            'optimiser': {**vars(self.optimiser),
                          'betas': list(self.optimiser.betas)},
            'steps': self.steps,
            'seed': self.seed,
            'device': self.device,
            'precision': self.precision,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval,
            'save_steps': self.save_steps,
            'run_name': self.run_name,
        }

    def fingerprint(self) -> str:
        # `steps` is excluded: resuming a k-step checkpoint to a larger N
        # with otherwise identical configuration is a normal operation
        # (T-R1), and the step count does not change the computation.
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        payload = json.loads(canonical)
        payload.pop('steps', None)
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def from_dict(cls, payload: Dict) -> 'ResearchConfig':
        if payload.get('schema_version') != 1:
            raise ValueError(
                f"config: unsupported schema version "
                f"{payload.get('schema_version')}")
        model = ModelConfig(**payload['model'])
        data = DataConfig(**payload['data'])
        optimiser = OptimiserConfig(**payload['optimiser'])
        return cls(model, data, optimiser,
                   steps=payload['steps'], seed=payload['seed'],
                   device=payload['device'], precision=payload['precision'],
                   log_interval=payload['log_interval'],
                   eval_interval=payload['eval_interval'],
                   save_steps=payload['save_steps'],
                   run_name=payload['run_name'])

    def save(self, path: str) -> None:
        tmp = path + '.tmp'
        with open(tmp, 'w') as fp:
            json.dump(self.to_dict(), fp, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> 'ResearchConfig':
        with open(path) as fp:
            return cls.from_dict(json.load(fp))
