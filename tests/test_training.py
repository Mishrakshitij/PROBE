"""Tests for training components: curriculum, replay buffer."""

import random

import pytest

from probe.types import Episode, Turn, Role, ViolationLabel, Severity, RewardBreakdown
from probe.training.curriculum import CurriculumScheduler
from probe.replay.buffer import ReplayBuffer


class TestCurriculumScheduler:
    def test_initial_k_max(self):
        cs = CurriculumScheduler(k_max_stages=[6, 10, 16])
        assert cs.k_max == 6
        assert cs.stage == 0

    def test_advance(self):
        cs = CurriculumScheduler(k_max_stages=[6, 10], stall_patience=5)
        for _ in range(5):
            cs.report(violation_yield=0.2, new_policy_triggered=False)
        assert cs.should_advance()
        cs.advance()
        assert cs.k_max == 10

    def test_no_advance_with_new_policies(self):
        cs = CurriculumScheduler(k_max_stages=[6, 10], stall_patience=5)
        for _ in range(5):
            cs.report(violation_yield=0.2, new_policy_triggered=True)
        assert not cs.should_advance()

    def test_turn_index_sampling(self):
        cs = CurriculumScheduler(k_max_stages=[10])
        rng = random.Random(42)
        indices = [cs.sample_turn_index(rng) for _ in range(100)]
        assert all(0 <= i < 10 for i in indices)
        # Early bias: lower indices should be more common initially
        avg = sum(indices) / len(indices)
        assert avg < 5  # biased toward beginning


class TestReplayBuffer:
    def _make_episode(self, eid: str, policies: list[str] = None) -> Episode:
        ep = Episode(episode_id=eid, turns=[
            Turn(Role.USER, "test", 0),
            Turn(Role.ASSISTANT, "response", 1),
        ])
        if policies:
            for pid in policies:
                ep.violations.append(ViolationLabel(
                    policy_id=pid, severity=Severity.MEDIUM,
                    turn_index=1, evidence="test"
                ))
        ep.rewards = [RewardBreakdown(total=1.0)]
        return ep

    def test_add_and_size(self):
        buf = ReplayBuffer(max_size=100)
        buf.add(self._make_episode("ep1"))
        assert buf.size == 1

    def test_max_size(self):
        buf = ReplayBuffer(max_size=5)
        for i in range(10):
            buf.add(self._make_episode(f"ep{i}"))
        assert buf.size == 5

    def test_sample(self):
        buf = ReplayBuffer(max_size=100)
        for i in range(10):
            buf.add(self._make_episode(f"ep{i}"))
        samples = buf.sample(3)
        assert len(samples) == 3

    def test_sample_prefix(self):
        buf = ReplayBuffer(max_size=100)
        ep = Episode(episode_id="long", turns=[
            Turn(Role.USER, "u0", 0), Turn(Role.ASSISTANT, "a0", 1),
            Turn(Role.USER, "u1", 2), Turn(Role.ASSISTANT, "a1", 3),
            Turn(Role.USER, "u2", 4), Turn(Role.ASSISTANT, "a2", 5),
        ])
        buf.add(ep)
        prefix = buf.sample_prefix(target_turn=1)
        assert prefix is not None
        assert len(prefix) == 2
