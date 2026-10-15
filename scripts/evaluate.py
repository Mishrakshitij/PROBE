#!/usr/bin/env python3
"""Run the full evaluation suite on collected episodes.

Usage:
    python scripts/evaluate.py [--log-dir data/logs] [--output data/eval_report.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from probe.environment.database import FlightDB
from probe.environment.policies import PolicyRegistry
from probe.environment.logger import InteractionLogger
from probe.replay.embedder import TrajectoryEmbedder
from probe.replay.cluster import ClusterManager
from probe.evaluation.report import ReportGenerator
from probe.types import Episode, Turn, Role, ViolationLabel, RewardBreakdown, Severity, ToolCall


def load_episodes_from_logs(log_dir: str) -> list[Episode]:
    """Reconstruct Episode objects from JSONL logs."""
    logger = InteractionLogger(log_dir)
    raw = logger.load_episodes()
    episodes = []
    for r in raw:
        turns = []
        for td in r.get("turns", []):
            tool_calls = tuple(
                ToolCall(tc["tool_name"], tc["arguments"], tc.get("result"), tc.get("error"))
                for tc in td.get("tool_calls", [])
            )
            turns.append(Turn(
                role=Role(td["role"]),
                content=td["content"],
                turn_index=td["turn_index"],
                tool_calls=tool_calls,
            ))
        violations = [
            ViolationLabel(
                policy_id=vd["policy_id"],
                severity=Severity(vd["severity"]),
                turn_index=vd["turn_index"],
                evidence=vd["evidence"],
                confidence=vd.get("confidence", 1.0),
            )
            for vd in r.get("violations", [])
        ]
        rewards = [
            RewardBreakdown(**rd)
            for rd in r.get("rewards", [])
        ]
        ep = Episode(
            episode_id=r["episode_id"],
            turns=turns,
            violations=violations,
            rewards=rewards,
            metadata=r.get("metadata", {}),
        )
        episodes.append(ep)
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Evaluate PROBE episodes")
    parser.add_argument("--log-dir", type=str, default="data/logs")
    parser.add_argument("--output", type=str, default="data/eval_report.json")
    args = parser.parse_args()

    policies = PolicyRegistry()
    episodes = load_episodes_from_logs(args.log_dir)
    print(f"Loaded {len(episodes)} episodes from {args.log_dir}")

    if not episodes:
        print("No episodes found. Run training or baselines first.")
        return

    embedder = TrajectoryEmbedder()
    cluster_mgr = ClusterManager(embedder)
    cluster_mgr.add_and_cluster(episodes)

    reporter = ReportGenerator(policies.policy_ids)
    report = reporter.generate(episodes, cluster_mgr)
    reporter.print_summary(report)
    reporter.export(report, args.output)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
