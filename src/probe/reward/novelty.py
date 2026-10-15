"""Novelty scoring: cluster-based + coverage-based bonuses.

Rewards discovering under-explored failure modes and new policy triggers.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


class NoveltyScorer:
    """Computes novelty bonus based on:
    1. New cluster discovery (trajectory-level)
    2. Rare policy trigger (inverse-frequency weighting)
    3. Coverage gap bonus (under-covered policies get higher reward)
    """

    def __init__(self, all_policy_ids: list[str], decay: float = 0.999):
        self.all_policy_ids = set(all_policy_ids)
        self.decay = decay
        # Running counts
        self._policy_counts: Counter[str] = Counter()
        self._cluster_ids_seen: set[int] = set()
        self._total_episodes: int = 0

    def score(
        self,
        triggered_policies: list[str],
        cluster_id: int | None = None,
    ) -> float:
        """Novelty score for one episode/turn.

        Args:
            triggered_policies: policy IDs violated in this episode/turn
            cluster_id: cluster assignment (-1 or None = unclustered)
        """
        bonus = 0.0

        # 1. New cluster discovery
        if cluster_id is not None and cluster_id >= 0:
            if cluster_id not in self._cluster_ids_seen:
                bonus += 1.0  # large bonus for new cluster
                self._cluster_ids_seen.add(cluster_id)

        # 2. Rare policy bonus (inverse frequency)
        if triggered_policies:
            for pid in triggered_policies:
                freq = self._policy_counts.get(pid, 0)
                # Inverse-frequency: rarer → higher bonus
                bonus += 1.0 / (1.0 + freq)

        # 3. Coverage gap bonus
        covered = set(self._policy_counts.keys())
        uncovered = self.all_policy_ids - covered
        for pid in triggered_policies:
            if pid in uncovered:
                bonus += 2.0  # big bonus for first-time policy trigger

        return bonus

    def update(self, triggered_policies: list[str], cluster_id: int | None = None):
        """Update internal counts after an episode is processed."""
        self._total_episodes += 1
        for pid in triggered_policies:
            self._policy_counts[pid] += 1
        if cluster_id is not None and cluster_id >= 0:
            self._cluster_ids_seen.add(cluster_id)

    @property
    def coverage(self) -> float:
        """Fraction of policies triggered at least once."""
        if not self.all_policy_ids:
            return 0.0
        return len(set(self._policy_counts.keys()) & self.all_policy_ids) / len(self.all_policy_ids)

    @property
    def num_clusters(self) -> int:
        return len(self._cluster_ids_seen)

    def policy_frequencies(self) -> dict[str, float]:
        """Normalized frequency of each policy."""
        total = max(self._total_episodes, 1)
        return {pid: self._policy_counts.get(pid, 0) / total
                for pid in self.all_policy_ids}
