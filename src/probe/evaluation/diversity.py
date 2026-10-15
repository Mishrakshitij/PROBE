"""Diversity metrics: cluster entropy, discovery rate."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..types import Episode
from ..replay.cluster import ClusterManager


class DiversityMetrics:
    """Measures diversity of discovered failure modes."""

    def compute(self, episodes: list[Episode], cluster_manager: ClusterManager) -> dict[str, Any]:
        labels = []
        for ep in episodes:
            cid = cluster_manager.get_cluster_id(ep.episode_id)
            labels.append(cid)

        valid_labels = [l for l in labels if l >= 0]
        counts = Counter(valid_labels)

        num_clusters = len(counts)
        entropy = self._entropy(counts)
        discovery_rate = self._discovery_rate(labels)

        return {
            "num_clusters": num_clusters,
            "cluster_entropy": entropy,
            "discovery_rate": discovery_rate,
            "cluster_sizes": dict(counts),
            "unclustered_pct": labels.count(-1) / max(len(labels), 1),
        }

    def _entropy(self, counts: Counter) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        h = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def _discovery_rate(self, labels: list[int], window: int = 50) -> float:
        """New clusters per window of episodes (most recent window)."""
        if len(labels) < window:
            window = len(labels)
        if window == 0:
            return 0.0
        recent = labels[-window:]
        earlier = set(labels[:-window])
        new = sum(1 for l in recent if l >= 0 and l not in earlier)
        return new / window
