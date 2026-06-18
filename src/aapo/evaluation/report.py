"""Report generation: aggregate metrics, tables, JSON export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import Episode
from ..replay.cluster import ClusterManager
from .coverage import CoverageMetrics
from .diversity import DiversityMetrics
from .yield_metrics import YieldMetrics
from .efficiency import EfficiencyMetrics


class ReportGenerator:
    """Generates comprehensive evaluation reports."""

    def __init__(self, all_policy_ids: list[str]):
        self.coverage = CoverageMetrics(all_policy_ids)
        self.diversity = DiversityMetrics()
        self.yield_m = YieldMetrics()
        self.efficiency = EfficiencyMetrics()

    def generate(
        self,
        episodes: list[Episode],
        cluster_manager: ClusterManager,
        target_calls: int = 0,
        judge_calls: int = 0,
    ) -> dict[str, Any]:
        """Generate full evaluation report."""
        report = {
            "summary": {
                "total_episodes": len(episodes),
                "total_violations": sum(len(ep.violations) for ep in episodes),
            },
            "coverage": self.coverage.compute(episodes),
            "coverage_curve": self.coverage.coverage_curve(episodes),
            "diversity": self.diversity.compute(episodes, cluster_manager),
            "yield": self.yield_m.compute(episodes),
            "efficiency": self.efficiency.compute(
                episodes, cluster_manager, target_calls, judge_calls
            ),
        }
        return report

    def export(self, report: dict[str, Any], path: str = "data/eval_report.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def print_summary(self, report: dict[str, Any]):
        """Print a human-readable summary."""
        s = report["summary"]
        c = report["coverage"]
        y = report["yield"]
        d = report["diversity"]

        print("=" * 60)
        print("AAPO EVALUATION REPORT")
        print("=" * 60)
        print(f"Episodes:            {s['total_episodes']}")
        print(f"Total violations:    {s['total_violations']}")
        print(f"Policy coverage:     {c['coverage_pct']:.1%}")
        print(f"Policies missing:    {', '.join(c['policies_missing']) or 'none'}")
        print(f"Violations/100 ep:   {y['violations_per_100']:.1f}")
        print(f"Severe/100 ep:       {y['severe_per_100']:.1f}")
        print(f"Multi-policy rate:   {y['multi_policy_rate']:.1%}")
        print(f"Unique clusters:     {d['num_clusters']}")
        print(f"Cluster entropy:     {d['cluster_entropy']:.2f}")
        print(f"Discovery rate:      {d['discovery_rate']:.3f}")
        print("=" * 60)
