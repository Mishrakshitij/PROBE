"""Shared data types for AAPO — all frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────────────────────

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def weight(self) -> float:
        return {"critical": 4.0, "high": 2.0, "medium": 1.0, "low": 0.5}[self.value]


class PolicyCategory(str, Enum):
    BUSINESS = "business"
    SAFETY = "safety"
    PRIVACY = "privacy"
    OPERATIONAL = "operational"


class BookingStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    HELD = "held"
    PENDING = "pending"


class FlightStatus(str, Enum):
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    FULL = "full"


# ── Core data types ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class Turn:
    role: Role
    content: str
    turn_index: int
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    name: str
    category: PolicyCategory
    severity: Severity
    description: str
    is_deterministic: bool = True


@dataclass(frozen=True)
class ViolationLabel:
    policy_id: str
    severity: Severity
    turn_index: int
    evidence: str
    confidence: float = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    violation: float = 0.0
    progress: float = 0.0
    novelty: float = 0.0
    plausibility: float = 0.0
    proximity: float = 0.0
    total: float = 0.0


@dataclass
class Episode:
    episode_id: str
    turns: list[Turn] = field(default_factory=list)
    violations: list[ViolationLabel] = field(default_factory=list)
    rewards: list[RewardBreakdown] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return sum(r.total for r in self.rewards)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def transcript(self) -> str:
        lines = []
        for t in self.turns:
            lines.append(f"[{t.role.value}] {t.content}")
            for tc in t.tool_calls:
                lines.append(f"  -> {tc.tool_name}({tc.arguments}) = {tc.result}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CandidateGroup:
    """G candidates sampled from the same prefix state."""
    prefix_turns: tuple[Turn, ...]
    turn_index: int
    candidates: tuple[str, ...]           # G candidate utterances
    continuations: tuple[Episode, ...]     # G continued episodes
    rewards: tuple[float, ...]             # G reward scores
    log_probs: tuple[float, ...] = ()     # G log-probs under current policy


@dataclass(frozen=True)
class ClusterInfo:
    cluster_id: int
    centroid_episode_id: str
    policy_ids: tuple[str, ...]
    size: int
    mean_reward: float
    exemplar_transcript: str = ""
