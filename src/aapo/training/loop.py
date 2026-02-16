"""Main training orchestrator.

Ties together: auditor, target, reward, replay, curriculum, updaters.
Supports GRPO → DPO → GFlowNet phasing.
"""

from __future__ import annotations

import random
from typing import Any

from ..types import CandidateGroup, Episode
from ..auditor.model import AuditorPolicy
from ..auditor.ema import EMAPolicy
from ..auditor.sampler import CandidateSampler
from ..environment.target_agent import TargetAgent
from ..environment.database import FlightDB
from ..environment.policies import PolicyRegistry
from ..environment.logger import InteractionLogger
from ..reward.composite import CompositeReward
from ..reward.attribution import TurnRewardAttributor
from ..reward.novelty import NoveltyScorer
from ..trajectory.episode_runner import EpisodeRunner
from ..trajectory.prefix_runner import SharedPrefixRunner
from ..replay.buffer import ReplayBuffer
from .grpo import GRPOUpdater
from .dpo import DPOUpdater
from .gflownet import GFlowNetUpdater
from .curriculum import CurriculumScheduler


class TrainingLoop:
    """Main training loop orchestrating all components.

    One step:
      1. Curriculum → K_max, turn index t
      2. Replay buffer → prefix S_t (or fresh rollout)
      3. CandidateSampler → G candidates {u_t^i}
      4. SharedPrefixRunner → G continuations
      5. CompositeReward → scored CandidateGroup
      6. PolicyUpdater → update π_train
      7. EMA → update π_old
      8. ReplayBuffer → store episodes
      9. Periodic evaluation
    """

    def __init__(
        self,
        auditor: AuditorPolicy,
        target: TargetAgent,
        db: FlightDB,
        policy_registry: PolicyRegistry,
        reward: CompositeReward,
        replay: ReplayBuffer,
        curriculum: CurriculumScheduler,
        logger: InteractionLogger | None = None,
        method: str = "grpo",
        group_size: int = 8,
        batch_size: int = 4,
        ema_decay: float = 0.995,
        eval_every: int = 50,
        seed: int = 42,
    ):
        self.auditor = auditor
        self.target = target
        self.db = db
        self.policies = policy_registry
        self.reward = reward
        self.replay = replay
        self.curriculum = curriculum
        self.logger = logger
        self.method = method
        self.group_size = group_size
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.rng = random.Random(seed)

        # EMA policy
        self.ema = EMAPolicy(auditor, decay=ema_decay)
        self.sampler = CandidateSampler(auditor, self.ema)

        # Episode + prefix runners
        self.episode_runner = EpisodeRunner(target, logger)
        self.prefix_runner = SharedPrefixRunner(target)

        # Updaters (initialized lazily based on method)
        self._updater: Any = None

    def _get_updater(self):
        if self._updater is not None:
            return self._updater
        if self.method == "grpo":
            self._updater = GRPOUpdater(self.auditor)
        elif self.method == "dpo":
            self._updater = DPOUpdater(self.auditor, self.ema.policy)
        elif self.method == "gflownet":
            self._updater = GFlowNetUpdater(self.auditor)
        return self._updater

    def train(self, num_steps: int = 1000, callback: Any = None) -> list[dict[str, Any]]:
        """Run the full training loop.

        Args:
            num_steps: total training steps
            callback: optional function(step, metrics) called each step

        Returns:
            list of per-step metric dicts
        """
        metrics_history: list[dict[str, Any]] = []
        updater = self._get_updater()

        for step in range(num_steps):
            step_metrics: dict[str, Any] = {"step": step, "k_max": self.curriculum.k_max}

            # 1. Collect a batch of candidate groups
            batch: list[CandidateGroup] = []
            for _ in range(self.batch_size):
                group = self._collect_one_group()
                if group:
                    batch.append(group)

            # 2. Update policy
            if batch:
                if self.method == "gflownet":
                    episodes = [ep for g in batch for ep in g.continuations]
                    update_metrics = updater.update(episodes)
                elif self.method == "dpo":
                    update_metrics = updater.update(batch)
                else:
                    update_metrics = updater.update(batch)
                step_metrics.update(update_metrics)

            # 3. Update EMA
            self.ema.update(self.auditor)

            # 4. Curriculum management
            avg_yield = self._compute_yield(batch)
            new_policy = self._any_new_policy(batch)
            self.curriculum.report(avg_yield, new_policy)
            if self.curriculum.should_advance():
                self.curriculum.advance()
                step_metrics["stage_advanced"] = True

            step_metrics["violation_yield"] = avg_yield
            step_metrics["stage"] = self.curriculum.stage

            metrics_history.append(step_metrics)

            if callback:
                callback(step, step_metrics)

        return metrics_history

    def _collect_one_group(self) -> CandidateGroup | None:
        """Generate one candidate group: prefix → G candidates → score."""
        t = self.curriculum.sample_turn_index(self.rng)

        # Get a prefix: either from replay buffer or fresh episode
        prefix = self.replay.sample_prefix(t)
        if prefix is None:
            # Run a fresh episode to get a prefix
            ep = self.episode_runner.run(self.sampler)
            self.replay.add(ep)
            if len(ep.turns) > 2 * t:
                prefix = ep.turns[:2 * t]
            else:
                return None

        # Generate G candidates and run continuations
        group = self.prefix_runner.run(self.sampler, prefix, self.group_size)

        # Score each continuation
        rewards: list[float] = []
        for continuation in group.continuations:
            breakdowns = self.reward.score_episode(continuation)
            ep_reward = sum(b.total for b in breakdowns)
            rewards.append(ep_reward)
            continuation.rewards = breakdowns

        # Store in replay
        for ep in group.continuations:
            self.replay.add(ep)

        return CandidateGroup(
            prefix_turns=group.prefix_turns,
            turn_index=group.turn_index,
            candidates=group.candidates,
            continuations=group.continuations,
            rewards=tuple(rewards),
            log_probs=group.log_probs,
        )

    def _compute_yield(self, batch: list[CandidateGroup]) -> float:
        if not batch:
            return 0.0
        total_violations = 0
        total_episodes = 0
        for g in batch:
            for ep in g.continuations:
                total_episodes += 1
                total_violations += len(ep.violations)
        return total_violations / max(total_episodes, 1)

    def _any_new_policy(self, batch: list[CandidateGroup]) -> bool:
        for g in batch:
            for ep in g.continuations:
                for v in ep.violations:
                    if v.policy_id not in self.reward.novelty._policy_counts:
                        return True
        return False
