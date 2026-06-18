"""Policy coverage metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..types import Episode


class CoverageMetrics:
    """Compute policy coverage statistics across episodes."""

    def __init__(self, all_policy_ids: list[str]):
        self.all_policies = set(all_policy_ids)

    def compute(self, episodes: list[Episode]) -> dict[str, Any]:
        triggered: set[str] = set()
        policy_counts: Counter[str] = Counter()
        combo_counts: Counter[tuple[str, ...]] = Counter()

        for ep in episodes:
            ep_policies = set()
            for v in ep.violations:
                triggered.add(v.policy_id)
                policy_counts[v.policy_id] += 1
                ep_policies.add(v.policy_id)
            if ep_policies:
                combo = tuple(sorted(ep_policies))
                combo_counts[combo] += 1

        coverage_pct = len(triggered & self.all_policies) / max(len(self.all_policies), 1)
        return {
            "coverage_pct": coverage_pct,
            "policies_triggered": sorted(triggered),
            "policies_missing": sorted(self.all_policies - triggered),
            "num_unique_combos": len(combo_counts),
            "per_policy_counts": dict(policy_counts),
            "top_combos": combo_counts.most_common(10),
        }

    def coverage_curve(self, episodes: list[Episode]) -> list[float]:
        """Coverage as a function of episodes seen (for plotting)."""
        seen: set[str] = set()
        curve = []
        for ep in episodes:
            for v in ep.violations:
                seen.add(v.policy_id)
            curve.append(len(seen & self.all_policies) / max(len(self.all_policies), 1))
        return curve
