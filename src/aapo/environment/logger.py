"""Append-only interaction logger for episodes."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import Episode, Turn, ViolationLabel


class InteractionLogger:
    """Logs episodes to JSONL files for replay and analysis."""

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_file = self.log_dir / "episodes.jsonl"
        self._violation_file = self.log_dir / "violations.jsonl"

    def log_episode(self, episode: Episode):
        """Append a complete episode to the log."""
        record = {
            "episode_id": episode.episode_id,
            "num_turns": episode.num_turns,
            "total_reward": episode.total_reward,
            "num_violations": len(episode.violations),
            "violated_policies": list({v.policy_id for v in episode.violations}),
            "timestamp": datetime.now().isoformat(),
            "turns": [_turn_to_dict(t) for t in episode.turns],
            "violations": [_violation_to_dict(v) for v in episode.violations],
            "rewards": [
                {"violation": r.violation, "progress": r.progress,
                 "novelty": r.novelty, "plausibility": r.plausibility,
                 "total": r.total}
                for r in episode.rewards
            ],
            "metadata": episode.metadata,
        }
        with open(self._episode_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_violation(self, episode_id: str, violation: ViolationLabel):
        record = {
            "episode_id": episode_id,
            "policy_id": violation.policy_id,
            "severity": violation.severity.value,
            "turn_index": violation.turn_index,
            "evidence": violation.evidence,
            "confidence": violation.confidence,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self._violation_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def update_episode(self, episode: Episode):
        """Rewrite an episode in the log with updated violations/rewards."""
        if not self._episode_file.exists():
            return
        lines = self._episode_file.read_text().splitlines()
        with open(self._episode_file, "w") as f:
            for line in lines:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["episode_id"] == episode.episode_id:
                    rec["violations"] = [_violation_to_dict(v) for v in episode.violations]
                    rec["num_violations"] = len(episode.violations)
                    rec["violated_policies"] = list({v.policy_id for v in episode.violations})
                    rec["rewards"] = [
                        {"violation": r.violation, "progress": r.progress,
                         "novelty": r.novelty, "plausibility": r.plausibility,
                         "total": r.total}
                        for r in episode.rewards
                    ]
                    rec["total_reward"] = episode.total_reward
                f.write(json.dumps(rec) + "\n")

    def load_episodes(self) -> list[dict[str, Any]]:
        if not self._episode_file.exists():
            return []
        records = []
        with open(self._episode_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records


def _turn_to_dict(t: Turn) -> dict[str, Any]:
    return {
        "role": t.role.value,
        "content": t.content,
        "turn_index": t.turn_index,
        "tool_calls": [
            {"tool_name": tc.tool_name, "arguments": tc.arguments,
             "result": tc.result, "error": tc.error}
            for tc in t.tool_calls
        ],
    }


def _violation_to_dict(v: ViolationLabel) -> dict[str, Any]:
    return {
        "policy_id": v.policy_id,
        "severity": v.severity.value,
        "turn_index": v.turn_index,
        "evidence": v.evidence,
        "confidence": v.confidence,
    }
