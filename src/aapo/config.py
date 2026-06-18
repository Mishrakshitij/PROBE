"""Typed config loader using OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


# ── Config schemas ─────────────────────────────────────────────────────

@dataclass
class DBConfig:
    path: str = "data/flights.db"
    num_airlines: int = 8
    num_airports: int = 35
    num_routes: int = 120
    num_restricted_routes: int = 6
    num_flights: int = 600
    fare_classes_per_flight: int = 3


@dataclass
class TargetAgentConfig:
    # Model config now lives in models.yaml; these are non-model settings
    max_tool_rounds: int = 5
    system_prompt_path: str = "configs/skyassist_prompt.txt"


@dataclass
class RewardWeights:
    alpha_progress: float = 0.3
    beta_novelty: float = 0.2
    gamma_plausibility: float = 0.1
    normalize: bool = True
    softplus_floor: bool = True  # for GFlowNet R>0


@dataclass
class RewardSchedule:
    alpha_start: float = 0.5
    alpha_end: float = 0.1
    beta_start: float = 0.1
    beta_end: float = 0.4
    gamma: float = 0.1  # constant
    warmup_steps: int = 100


@dataclass
class CurriculumConfig:
    k_max_stages: list[int] = field(default_factory=lambda: [4, 8, 12, 14])
    advance_threshold: float = 0.15  # min violation yield to advance
    stall_patience: int = 50         # episodes with no new policy → advance


@dataclass
class GRPOConfig:
    group_size: int = 8
    lr: float = 1e-5
    kl_coeff: float = 0.05
    clip_range: float = 0.2
    normalize_advantages: bool = True


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 5e-6
    top_k: int = 2
    bottom_k: int = 2


@dataclass
class GFlowNetConfig:
    lr_logZ: float = 1e-3
    lr_policy: float = 1e-5
    reward_floor: float = 0.01  # softplus additive floor


@dataclass
class TrainingConfig:
    method: str = "grpo"  # grpo | dpo | gflownet
    auditor_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    batch_size: int = 4
    num_steps: int = 2000
    eval_every: int = 50
    checkpoint_every: int = 200
    ema_decay: float = 0.995
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    gflownet: GFlowNetConfig = field(default_factory=GFlowNetConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    reward_schedule: RewardSchedule = field(default_factory=RewardSchedule)


@dataclass
class ReplayConfig:
    max_size: int = 10000
    dedup_threshold: float = 0.92
    dedup_cadence: int = 100
    priority_recency_weight: float = 0.3
    priority_rarity_weight: float = 0.4
    priority_novelty_weight: float = 0.3
    embedding_model: str = "all-MiniLM-L6-v2"
    min_cluster_size: int = 3


@dataclass
class EvalConfig:
    num_episodes: int = 200
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    bootstrap_samples: int = 1000
    human_sample_size: int = 50


@dataclass
class AAPOConfig:
    db: DBConfig = field(default_factory=DBConfig)
    target: TargetAgentConfig = field(default_factory=TargetAgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str | Path | None = None) -> AAPOConfig:
    """Load config from YAML, merging with defaults."""
    schema = OmegaConf.structured(AAPOConfig)
    if path and Path(path).exists():
        user = OmegaConf.load(path)
        cfg = OmegaConf.merge(schema, user)
    else:
        cfg = schema
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]
