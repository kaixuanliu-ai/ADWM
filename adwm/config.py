"""Default hyperparameters of the world model, per environment.

These are the values used in the paper. Every field can be overridden from the
adwm.train command line, and the values a model was trained with are stored in
its checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DataConfig:
    data_dir: str
    models: list[str] = field(default_factory=list)   # behavior-policy subdirs of data_dir
    prompts: list[str] = field(default_factory=lambda: ["p0", "p1"])
    seq_len: int = 10
    stride: int = 1
    success_upsample: int = 1


@dataclass
class ModelConfig:
    d_word: int = 128
    d_z: int = 64
    d_a: int = 32
    d_h: int = 128
    d_hidden: int = 256
    d_llm: int = 256
    n_soft_tokens: int = 8
    n_heads: int = 4
    n_layers: int = 4
    history_max_len: int = 64
    T: int = 1000
    p_uncond: float = 0.1


@dataclass
class LossConfig:
    lambda_idm: float = 0.1
    lambda_bc: float = 0.1
    lambda_div: float = 0.05
    lambda_reward: float = 10.0


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler_eta_min_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    out: str = "checkpoints/wm.pt"
    resume: str | None = None
    log_interval: int = 50


@dataclass
class EnvConfig:
    name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fixed_n_actions: int | None = None


ENV_CONFIGS: dict[str, EnvConfig] = {
    "alfworld": EnvConfig(
        name="alfworld",
        data=DataConfig(data_dir="data/alfworld"),
        train=TrainConfig(out="checkpoints/alfworld/wm.pt"),
    ),
    "scienceworld": EnvConfig(
        name="scienceworld",
        data=DataConfig(data_dir="data/scienceworld"),
        train=TrainConfig(out="checkpoints/scienceworld/wm.pt"),
    ),
    "hotpotqa": EnvConfig(
        name="hotpotqa",
        data=DataConfig(data_dir="data/hotpotqa", seq_len=6),
        train=TrainConfig(out="checkpoints/hotpotqa/wm.pt"),
        fixed_n_actions=4,  # <unk>, search, lookup, finish
    ),
}


def available_envs() -> list[str]:
    return sorted(ENV_CONFIGS)


def get_env_config(env: str) -> EnvConfig:
    """Return a copy of the default config for `env`."""
    if env not in ENV_CONFIGS:
        raise ValueError(f"Unknown env: {env}. Available: {', '.join(available_envs())}")
    return deepcopy(ENV_CONFIGS[env])


def config_to_dict(config: EnvConfig) -> dict[str, Any]:
    return asdict(config)


def model_kwargs(config: EnvConfig) -> dict[str, Any]:
    """Keyword arguments for adwm.models.world_model.build_world_model."""
    values = asdict(config.model)
    values.update(asdict(config.loss))
    return values
