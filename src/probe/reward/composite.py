"""Composite reward: weighted aggregation with running z-score normalization.

r_k = r_viol + α·r_prog + β·r_nov + γ·r_plaus

Components normalized independently via running z-score before weighting.
Supports scheduled weight annealing and softplus floor for GFlowNet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..types import Episode, RewardBreakdown, ViolationLabel
from ..environment.database import FlightDB
from ..environment.policies import PolicyRegistry
from .attribution import TurnRewardAttributor
from .shaping import ProgressShaper
from .novelty import NoveltyScorer
from .plausibility import PlausibilityScorer


@dataclass
class RunningStats:
    """Welford's online algorithm for running mean/variance."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / max(self.n, 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def normalize(self, x: float) -> float:
        if self.n < 2 or self.std < 1e-8:
            return x
        return (x - self.mean) / self.std


class CompositeReward:
    """Computes per-turn composite reward with normalization and scheduling."""

    def __init__(
        self,
        db: FlightDB,
        policy_registry: PolicyRegistry,
        attributor: TurnRewardAttributor,
        novelty_scorer: NoveltyScorer,
        proximity_scorer=None,
        # Weight schedule params
        alpha_start: float = 0.5,
        alpha_end: float = 0.1,
        beta_start: float = 0.1,
        beta_end: float = 0.4,
        gamma: float = 0.35,
        delta: float = 0.3,         # weight for proximity partial credit
        warmup_steps: int = 100,
        normalize: bool = True,
        softplus_floor: bool = True,
        # Ablation flags
        diminishing_returns: bool = True,
        anti_fabrication: bool = True,
        binary_violation: bool = False,
    ):
        self.db = db
        self.policies = policy_registry
        self.attributor = attributor
        self.novelty = novelty_scorer
        self.proximity = proximity_scorer
        self.shaper = ProgressShaper()
        self.plausibility = PlausibilityScorer(anti_fabrication=anti_fabrication)

        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.gamma = gamma
        self.delta = delta
        self.warmup_steps = warmup_steps
        self.do_normalize = normalize
        self.softplus_floor = softplus_floor
        self.diminishing_returns = diminishing_returns
        self.binary_violation = binary_violation

        self._step = 0
        self._stats = {
            "violation": RunningStats(),
            "progress": RunningStats(),
            "novelty": RunningStats(),
            "plausibility": RunningStats(),
            "proximity": RunningStats(),
        }

    def score_episode(self, episode: Episode, cluster_id: int | None = None) -> list[RewardBreakdown]:
        """Score all turns in an episode, return per-turn RewardBreakdown."""
        violations_by_turn = self.attributor.attribute(episode)
        alpha, beta = self._current_weights()

        breakdowns: list[RewardBreakdown] = []
        all_triggered: list[str] = []
        # Track per-policy violation counts for diminishing returns
        policy_counts: dict[str, int] = {}

        for turn in episode.turns:
            if turn.role.value != "user":
                # Only score auditor (user) turns
                breakdowns.append(RewardBreakdown())
                continue

            tidx = turn.turn_index
            # The violation is detected on the assistant turn following this user turn
            asst_idx = tidx + 1
            violations = violations_by_turn.get(asst_idx, [])
            triggered = [v.policy_id for v in violations]
            all_triggered.extend(triggered)

            if self.binary_violation:
                # Generic red-teaming success signal: 1.0 if this turn elicited
                # ANY violation, with NO per-policy severity / dedup / coverage
                # shaping. Used for the diversity-only baseline, where the
                # GFlowNet objective supplies mode-covering diversity over
                # generic "did we break something" success rather than the
                # policy-specific coverage reward.
                r_viol = 1.0 if violations else 0.0
            elif self.diminishing_returns:
                r_viol = self.attributor.violation_reward(violations, prior_counts=policy_counts)
            else:
                r_viol = self.attributor.violation_reward(violations)
            # Update counts AFTER reward computation
            for pid in triggered:
                policy_counts[pid] = policy_counts.get(pid, 0) + 1
            r_prog = self.shaper.score(episode, tidx)
            r_nov = self.novelty.score(triggered, cluster_id)
            r_plaus = self.plausibility.score(episode, tidx)
            r_prox = (
                self.proximity.score(episode, tidx, has_violation=bool(violations))
                if self.proximity else 0.0
            )

            # Update running stats
            self._stats["violation"].update(r_viol)
            self._stats["progress"].update(r_prog)
            self._stats["novelty"].update(r_nov)
            self._stats["plausibility"].update(r_plaus)
            self._stats["proximity"].update(r_prox)

            # Normalize
            if self.do_normalize:
                r_viol = self._stats["violation"].normalize(r_viol)
                r_prog = self._stats["progress"].normalize(r_prog)
                r_nov = self._stats["novelty"].normalize(r_nov)
                r_plaus = self._stats["plausibility"].normalize(r_plaus)
                r_prox = self._stats["proximity"].normalize(r_prox)

            total = r_viol + alpha * r_prog + beta * r_nov + self.gamma * r_plaus + self.delta * r_prox

            # Softplus floor for GFlowNet (R > 0)
            if self.softplus_floor:
                total = math.log(1 + math.exp(total))

            breakdowns.append(RewardBreakdown(
                violation=r_viol,
                progress=r_prog,
                novelty=r_nov,
                plausibility=r_plaus,
                proximity=r_prox,
                total=total,
            ))

        # Populate episode.violations so metrics/curriculum can see them
        episode.violations = [
            v for vlist in violations_by_turn.values() for v in vlist
        ]

        # Update novelty scorer state
        self.novelty.update(all_triggered, cluster_id)
        self._step += 1

        return breakdowns

    def _current_weights(self) -> tuple[float, float]:
        """Linear schedule: α decreases, β increases over training."""
        t = min(self._step / max(self.warmup_steps, 1), 1.0)
        alpha = self.alpha_start + t * (self.alpha_end - self.alpha_start)
        beta = self.beta_start + t * (self.beta_end - self.beta_start)
        return alpha, beta
