"""Curriculum scheduler: horizon growth + turn-index sampling.

Phase schedule:
  - Start with short K_max (4), grow to 8, 12, 14
  - Advance when violation yield > threshold for current K
  - Within each K_max, sample training turn indices weighted toward early turns initially
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any


class CurriculumScheduler:
    """Manages horizon stages and turn-index distribution."""

    def __init__(
        self,
        k_max_stages: list[int] | None = None,
        advance_threshold: float = 0.15,
        stall_patience: int = 50,
        turn_bias_decay: float = 0.98,
    ):
        self.stages = k_max_stages or [4, 8, 12, 14]
        self.advance_threshold = advance_threshold
        self.stall_patience = stall_patience
        self.turn_bias_decay = turn_bias_decay

        self._stage_idx = 0
        self._step = 0
        self._recent_yields: deque[float] = deque(maxlen=stall_patience)
        self._new_policies_seen: deque[bool] = deque(maxlen=stall_patience)

    @property
    def k_max(self) -> int:
        return self.stages[min(self._stage_idx, len(self.stages) - 1)]

    @property
    def stage(self) -> int:
        return self._stage_idx

    def sample_turn_index(self, rng: random.Random | None = None) -> int:
        """Sample a turn index for training, biased toward earlier turns initially."""
        rng = rng or random.Random()
        max_t = self.k_max

        # Bias decays over training: initially favor early turns, later uniform
        bias = self.turn_bias_decay ** self._step
        weights = [(1.0 - bias) + bias * (1.0 / (1 + t)) for t in range(max_t)]
        total = sum(weights)
        weights = [w / total for w in weights]

        return rng.choices(range(max_t), weights=weights, k=1)[0]

    def report(self, violation_yield: float, new_policy_triggered: bool):
        """Report metrics from an episode for advancement decisions."""
        self._step += 1
        self._recent_yields.append(violation_yield)
        self._new_policies_seen.append(new_policy_triggered)

    def should_advance(self) -> bool:
        """Check if we should advance to the next curriculum stage."""
        if self._stage_idx >= len(self.stages) - 1:
            return False

        if len(self._recent_yields) < self.stall_patience:
            return False

        avg_yield = sum(self._recent_yields) / len(self._recent_yields)
        any_new = any(self._new_policies_seen)

        # Advance if yield is high enough AND no new policies in recent window
        if avg_yield >= self.advance_threshold and not any_new:
            return True

        return False

    def advance(self):
        """Move to next stage."""
        if self._stage_idx < len(self.stages) - 1:
            self._stage_idx += 1
            self._recent_yields.clear()
            self._new_policies_seen.clear()

    def convergence_check(self, discovery_rate: float, slope_threshold: float = 0.001) -> bool:
        """Check if training should stop (convergence).

        Stop if: no new policies for patience window AND discovery rate slope < threshold
        """
        if len(self._recent_yields) < self.stall_patience:
            return False
        no_new = not any(self._new_policies_seen)
        return no_new and discovery_rate < slope_threshold
