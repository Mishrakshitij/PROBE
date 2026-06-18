"""Tests for episode runner (unit-level with mock auditor/target)."""

import pytest

from aapo.types import Episode, Turn, Role, ToolCall
from aapo.trajectory.episode_runner import EpisodeRunner


class MockAuditor:
    """Simple auditor that generates canned messages."""

    def __init__(self, messages: list[str]):
        self._messages = messages
        self._idx = 0

    def generate(self, history: list[Turn]) -> str:
        if self._idx >= len(self._messages):
            return ""
        msg = self._messages[self._idx]
        self._idx += 1
        return msg


class MockTarget:
    """Mock target that echoes back."""

    def generate(self, history: list[Turn]) -> Turn:
        last = history[-1].content if history else ""
        return Turn(
            role=Role.ASSISTANT,
            content=f"Echo: {last}",
            turn_index=len(history),
        )


class TestEpisodeRunner:
    def test_basic_episode(self):
        auditor = MockAuditor(["Hello", "Book a flight", "Thanks"])
        target = MockTarget()
        runner = EpisodeRunner(target, max_turns=5)
        ep = runner.run(auditor)
        assert ep.num_turns >= 2  # at least 1 user + 1 assistant
        assert ep.turns[0].role == Role.USER
        assert ep.turns[1].role == Role.ASSISTANT

    def test_respects_max_turns(self):
        auditor = MockAuditor(["msg"] * 20)
        target = MockTarget()
        runner = EpisodeRunner(target, max_turns=3)
        ep = runner.run(auditor)
        assert ep.num_turns <= 6  # 3 user + 3 assistant

    def test_stops_on_empty_auditor(self):
        auditor = MockAuditor(["Hello", ""])
        target = MockTarget()
        runner = EpisodeRunner(target, max_turns=10)
        ep = runner.run(auditor)
        assert ep.num_turns == 2  # only first exchange

    def test_from_prefix(self):
        prefix = [
            Turn(Role.USER, "Hi", 0),
            Turn(Role.ASSISTANT, "Hello!", 1),
        ]
        auditor = MockAuditor(["Continue", "End"])
        target = MockTarget()
        runner = EpisodeRunner(target, max_turns=10)
        ep = runner.run_from_prefix(auditor, prefix, additional_turns=2)
        assert ep.num_turns >= 4  # prefix + continuation
