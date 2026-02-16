"""Failure catalog: structured export of discovered violations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import ClusterInfo, Episode
from .cluster import ClusterManager


class FailureCatalog:
    """Generates a structured failure catalog from clustered episodes.

    Each cluster entry:
      - cluster_id, size
      - violated policies (set)
      - severity (max across policies)
      - minimal reproduction transcript
      - tool logs
      - responsible turns
      - mean reward
    """

    def __init__(self, cluster_manager: ClusterManager):
        self.cm = cluster_manager

    def build(self, episodes: list[Episode]) -> list[dict[str, Any]]:
        """Build catalog entries for all clusters."""
        # Group episodes by cluster
        cluster_episodes: dict[int, list[Episode]] = {}
        for ep in episodes:
            cid = self.cm.get_cluster_id(ep.episode_id)
            if cid >= 0:
                cluster_episodes.setdefault(cid, []).append(ep)

        catalog = []
        for cid, eps in sorted(cluster_episodes.items()):
            # Find exemplar: highest reward episode
            exemplar = max(eps, key=lambda e: e.total_reward)
            all_policies = set()
            all_severities = set()
            responsible_turns = set()

            for ep in eps:
                for v in ep.violations:
                    all_policies.add(v.policy_id)
                    all_severities.add(v.severity.value)
                    responsible_turns.add(v.turn_index)

            max_severity = "low"
            for s in ["critical", "high", "medium", "low"]:
                if s in all_severities:
                    max_severity = s
                    break

            catalog.append({
                "cluster_id": cid,
                "size": len(eps),
                "violated_policies": sorted(all_policies),
                "max_severity": max_severity,
                "responsible_turns": sorted(responsible_turns),
                "mean_reward": sum(e.total_reward for e in eps) / len(eps),
                "exemplar_id": exemplar.episode_id,
                "exemplar_transcript": exemplar.transcript()[:2000],
                "tool_calls": self._extract_tool_calls(exemplar),
            })

        return catalog

    def export(self, episodes: list[Episode], path: str = "data/failure_catalog.json"):
        """Build and save catalog to JSON."""
        catalog = self.build(episodes)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"clusters": catalog, "total_clusters": len(catalog)}, f, indent=2)
        return catalog

    def _extract_tool_calls(self, episode: Episode) -> list[dict[str, Any]]:
        calls = []
        for t in episode.turns:
            for tc in t.tool_calls:
                calls.append({
                    "turn": t.turn_index,
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "result_preview": str(tc.result)[:200] if tc.result else None,
                    "error": tc.error,
                })
        return calls
