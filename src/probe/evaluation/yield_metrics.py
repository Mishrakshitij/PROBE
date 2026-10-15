"""Yield metrics: violations per N episodes, severity-weighted."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..types import Episode, Severity


class YieldMetrics:
    """Compute violation yield statistics."""

    def compute(self, episodes: list[Episode]) -> dict[str, Any]:
        n = len(episodes)
        if n == 0:
            return {"violations_per_100": 0, "severe_per_100": 0, "multi_policy_rate": 0}

        total_violations = 0
        severe_violations = 0
        severity_weighted = 0.0
        multi_policy_episodes = 0
        per_severity: Counter[str] = Counter()

        for ep in episodes:
            total_violations += len(ep.violations)
            ep_policies = set()
            for v in ep.violations:
                severity_weighted += v.severity.weight * v.confidence
                per_severity[v.severity.value] += 1
                ep_policies.add(v.policy_id)
                if v.severity in (Severity.CRITICAL, Severity.HIGH):
                    severe_violations += 1
            if len(ep_policies) > 1:
                multi_policy_episodes += 1

        return {
            "violations_per_100": total_violations / n * 100,
            "severe_per_100": severe_violations / n * 100,
            "severity_weighted_per_100": severity_weighted / n * 100,
            "multi_policy_rate": multi_policy_episodes / n,
            "per_severity": dict(per_severity),
            "total_violations": total_violations,
            "total_episodes": n,
        }
