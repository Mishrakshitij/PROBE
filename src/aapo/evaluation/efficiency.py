"""Efficiency metrics: cost per new cluster/policy."""

from __future__ import annotations

from typing import Any

from ..types import Episode
from ..replay.cluster import ClusterManager


class EfficiencyMetrics:
    """Compute cost-efficiency of auditing."""

    def compute(
        self,
        episodes: list[Episode],
        cluster_manager: ClusterManager,
        total_target_calls: int = 0,
        total_judge_calls: int = 0,
    ) -> dict[str, Any]:
        n = len(episodes)
        n_clusters = cluster_manager.num_clusters
        n_policies = len({v.policy_id for ep in episodes for v in ep.violations})

        calls_per_cluster = total_target_calls / max(n_clusters, 1)
        calls_per_policy = total_target_calls / max(n_policies, 1)
        episodes_per_cluster = n / max(n_clusters, 1)
        judge_per_cluster = total_judge_calls / max(n_clusters, 1)

        return {
            "episodes_per_cluster": episodes_per_cluster,
            "target_calls_per_cluster": calls_per_cluster,
            "target_calls_per_policy": calls_per_policy,
            "judge_calls_per_cluster": judge_per_cluster,
            "total_target_calls": total_target_calls,
            "total_judge_calls": total_judge_calls,
        }
