"""Prioritized replay buffer.

Priority ∝ recency × rarity × novelty.
Supports prefix sampling for shared-prefix training.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from ..types import Episode, Turn


class ReplayBuffer:
    """Fixed-size replay buffer with priority-weighted sampling."""

    def __init__(
        self,
        max_size: int = 10000,
        recency_weight: float = 0.3,
        rarity_weight: float = 0.4,
        novelty_weight: float = 0.3,
    ):
        self.max_size = max_size
        self.recency_weight = recency_weight
        self.rarity_weight = rarity_weight
        self.novelty_weight = novelty_weight

        self._episodes: deque[Episode] = deque(maxlen=max_size)
        self._priorities: deque[float] = deque(maxlen=max_size)
        self._policy_counts: dict[str, int] = {}
        self._insert_order: int = 0

    def add(self, episode: Episode, novelty_score: float = 0.0):
        """Add an episode to the buffer."""
        # Compute priority
        recency = 1.0  # newest always gets max recency
        rarity = self._compute_rarity(episode)
        priority = (self.recency_weight * recency +
                    self.rarity_weight * rarity +
                    self.novelty_weight * novelty_score)

        self._episodes.append(episode)
        self._priorities.append(priority)
        self._insert_order += 1

        # Update policy counts
        for v in episode.violations:
            self._policy_counts[v.policy_id] = self._policy_counts.get(v.policy_id, 0) + 1

        # Decay recency of older episodes
        decay = 0.999
        for i in range(len(self._priorities) - 1):
            self._priorities[i] *= decay

    def sample(self, n: int = 1, rng: random.Random | None = None) -> list[Episode]:
        """Sample n episodes weighted by priority."""
        rng = rng or random.Random()
        if not self._episodes:
            return []
        n = min(n, len(self._episodes))
        priorities = list(self._priorities)
        total = sum(priorities)
        if total <= 0:
            return list(rng.sample(list(self._episodes), n))
        weights = [p / total for p in priorities]
        return rng.choices(list(self._episodes), weights=weights, k=n)

    def sample_prefix(self, target_turn: int, rng: random.Random | None = None) -> list[Turn] | None:
        """Sample a prefix of length 2*target_turn from the buffer."""
        rng = rng or random.Random()
        eligible = [ep for ep in self._episodes if ep.num_turns >= 2 * target_turn]
        if not eligible:
            return None
        ep = rng.choice(eligible)
        return ep.turns[:2 * target_turn]

    def _compute_rarity(self, episode: Episode) -> float:
        if not episode.violations:
            return 0.5  # base rarity for no violations
        total = sum(self._policy_counts.values()) or 1
        rarity = 0.0
        for v in episode.violations:
            freq = self._policy_counts.get(v.policy_id, 0) / total
            rarity += 1.0 / (1.0 + freq * 10)
        return rarity / len(episode.violations)

    @property
    def size(self) -> int:
        return len(self._episodes)

    def all_episodes(self) -> list[Episode]:
        return list(self._episodes)

    def clear(self):
        self._episodes.clear()
        self._priorities.clear()
        self._policy_counts.clear()
