"""Main training orchestrator.

Ties together: auditor, target, reward, replay, curriculum, updaters.
Supports GRPO, DPO, GFlowNet phasing.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..types import CandidateGroup, Episode, RewardBreakdown
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
from ..auditor.scenarios import ScenarioPool
from ..trajectory.episode_runner import EpisodeRunner
from ..trajectory.prefix_runner import SharedPrefixRunner
from ..replay.buffer import ReplayBuffer
from .grpo import GRPOUpdater
from .dpo import DPOUpdater
from .gflownet import GFlowNetUpdater
from .curriculum import CurriculumScheduler
from .logger import TrainingLogger


class TrainingLoop:
    """Main training loop orchestrating all components.

    One step:
      1. Curriculum -> K_max, turn index t
      2. Replay buffer -> prefix S_t (or fresh rollout)
      3. CandidateSampler -> G candidates {u_t^i}
      4. SharedPrefixRunner -> G continuations
      5. CompositeReward -> scored CandidateGroup
      6. PolicyUpdater -> update pi_train
      7. EMA -> update pi_old
      8. ReplayBuffer -> store episodes
      9. Periodic evaluation
    """

    def __init__(
        self,
        auditor: AuditorPolicy,
        target: TargetAgent,
        db: FlightDB | None = None,
        policy_registry: PolicyRegistry | None = None,
        reward: CompositeReward = None,
        replay: ReplayBuffer = None,
        curriculum: CurriculumScheduler = None,
        logger: InteractionLogger | None = None,
        training_logger: TrainingLogger | None = None,
        method: str = "grpo",
        group_size: int = 8,
        batch_size: int = 4,
        continuation_turns: int = 1,
        ema_decay: float = 0.995,
        eval_every: int = 50,
        seed: int = 42,
        episode_runner: EpisodeRunner | None = None,
        scenario_pool: ScenarioPool | None = None,
    ):
        self.auditor = auditor
        self.target = target
        self.db = db
        self.policies = policy_registry
        self.reward = reward
        self.replay = replay
        self.curriculum = curriculum
        self.logger = logger
        self.training_logger = training_logger
        self.method = method
        self.group_size = group_size
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.ema_decay = ema_decay
        self.rng = random.Random(seed)

        # Scenario pool: caller may inject one (e.g. TauBench), else use DB or static
        if scenario_pool is not None:
            self.scenario_pool = scenario_pool
        elif db is not None:
            try:
                self.scenario_pool = ScenarioPool.from_db(str(db.db_path), seed=seed)
            except Exception:
                self.scenario_pool = ScenarioPool(seed=seed)
        else:
            self.scenario_pool = ScenarioPool(seed=seed)

        # EMA policy
        self.ema = EMAPolicy(auditor, decay=ema_decay)
        self.sampler = CandidateSampler(auditor, self.ema)

        # Episode + prefix runners: caller may inject custom runners (e.g. TauBenchEpisodeRunner)
        self.episode_runner = episode_runner if episode_runner is not None else EpisodeRunner(target, logger, max_turns=14)
        self.prefix_runner = SharedPrefixRunner(target, continuation_turns=continuation_turns)

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

    def train(self, num_steps: int = 1000, callback: Any = None, start_step: int = 0) -> list[dict[str, Any]]:
        """Run the full training loop.

        Args:
            num_steps: total training steps to run from start_step
            callback: optional function(step, metrics) called each step
            start_step: step index to start from (for resuming)

        Returns:
            list of per-step metric dicts
        """
        metrics_history: list[dict[str, Any]] = []
        updater = self._get_updater()
        tlog = self.training_logger

        for step in range(start_step, start_step + num_steps):
            if tlog:
                tlog.begin_step(step)

            step_metrics: dict[str, Any] = {"step": step, "k_max": self.curriculum.k_max}

            # 1. Collect a batch of candidate groups (parallel vLLM calls)
            with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
                results = list(pool.map(lambda _: self._collect_one_group(), range(self.batch_size)))
            batch: list[CandidateGroup] = [g for g in results if g is not None]

            # 2. Gather reward breakdowns across the batch for logging
            all_rewards, all_breakdowns, all_completion_lens, all_turn_counts, all_violations = (
                self._collect_batch_stats(batch)
            )

            # 3. Update policy
            update_metrics: dict[str, Any] = {}
            if batch:
                if self.method == "gflownet":
                    episodes = [ep for g in batch for ep in g.continuations]
                    update_metrics = updater.update(episodes)
                elif self.method == "dpo":
                    update_metrics = updater.update(batch)
                else:
                    update_metrics = updater.update(batch)
                step_metrics.update(update_metrics)

            # 4. Update EMA
            self.ema.update(self.auditor)

            # 5. Curriculum management
            avg_yield = self._compute_yield(batch)
            new_policy = self._any_new_policy(batch)
            self.curriculum.report(avg_yield, new_policy)
            if self.curriculum.should_advance():
                self.curriculum.advance()
                step_metrics["stage_advanced"] = True

            step_metrics["violation_yield"] = avg_yield
            step_metrics["stage"] = self.curriculum.stage

            # 6. Log comprehensive metrics
            if tlog:
                self._log_step(
                    tlog, step_metrics, update_metrics,
                    all_rewards, all_breakdowns, all_completion_lens,
                    all_turn_counts, all_violations, batch,
                )

            metrics_history.append(step_metrics)

            if callback:
                callback(step, step_metrics)

        if tlog:
            tlog.close()

        return metrics_history

    def _collect_batch_stats(
        self, batch: list[CandidateGroup]
    ) -> tuple[list[float], list[dict[str, float]], list[int], list[int], list[str]]:
        """Extract reward breakdowns, completion lengths, turn counts, and violations."""
        all_rewards: list[float] = []
        all_breakdowns: list[dict[str, float]] = []
        all_completion_lens: list[int] = []
        all_turn_counts: list[int] = []
        all_violations: list[str] = []

        for g in batch:
            all_rewards.extend(g.rewards)
            for ep in g.continuations:
                # Aggregate reward breakdowns
                for rb in ep.rewards:
                    all_breakdowns.append({
                        "violation": rb.violation,
                        "progress": rb.progress,
                        "novelty": rb.novelty,
                        "plausibility": rb.plausibility,
                        "total": rb.total,
                    })
                # Completion length (word count per user turn)
                for turn in ep.turns:
                    if turn.role.value == "user":
                        all_completion_lens.append(len(turn.content.split()))
                # Per-episode turn count
                all_turn_counts.append(len(ep.turns))
                # Violations
                for v in ep.violations:
                    all_violations.append(v.policy_id)

        return all_rewards, all_breakdowns, all_completion_lens, all_turn_counts, all_violations

    def _log_step(
        self,
        tlog: TrainingLogger,
        step_metrics: dict[str, Any],
        update_metrics: dict[str, Any],
        all_rewards: list[float],
        all_breakdowns: list[dict[str, float]],
        all_completion_lens: list[int],
        all_turn_counts: list[int],
        all_violations: list[str],
        batch: list[CandidateGroup],
    ):
        """Push all metrics into the TrainingLogger for this step."""
        # Core scalars from updater
        tlog.log(
            loss=update_metrics.get("loss", 0.0),
            pg_loss=update_metrics.get("pg_loss", 0.0),
            kl_loss=update_metrics.get("kl_loss", 0.0),
            kl_divergence=update_metrics.get("kl_divergence", 0.0),
            dpo_loss=update_metrics.get("dpo_loss", 0.0),
            gflownet_loss=update_metrics.get("gflownet_loss", 0.0),
            log_Z=update_metrics.get("log_Z", 0.0),
            grad_norm=update_metrics.get("grad_norm", 0.0),
            clip_fraction=update_metrics.get("clip_fraction", 0.0),
            learning_rate=update_metrics.get("learning_rate", 0.0),
            n_groups=update_metrics.get("n_groups", 0),
            n_pairs=update_metrics.get("n_pairs", 0),
            n_episodes=update_metrics.get("n_episodes", 0),
        )

        # Reward stats
        tlog.log_reward_stats(all_rewards, all_breakdowns)

        # Advantage/log-prob/ratio stats (from underscore-prefixed raw lists)
        if "_advantages" in update_metrics:
            tlog.log_advantage_stats(update_metrics["_advantages"])
        if "_log_probs" in update_metrics:
            tlog.log_log_prob_stats(update_metrics["_log_probs"])
        if "_ratios" in update_metrics:
            tlog.log_ratio_stats(update_metrics["_ratios"])

        # DPO-specific stats
        if "_margins" in update_metrics:
            tlog.log_dpo_stats(
                update_metrics["_margins"],
                update_metrics.get("_accuracies", []),
            )

        # GFlowNet-specific stats
        if "_residuals" in update_metrics:
            tlog.log_gflownet_stats(
                update_metrics["_residuals"],
                update_metrics.get("_log_r"),
            )

        # Completion lengths
        tlog.log_completion_stats(all_completion_lens)

        # Per-episode turn counts
        tlog.log_turn_count_stats(all_turn_counts)

        # Violations
        unique_policies = list(set(all_violations))
        tlog.log(
            violation_yield=step_metrics.get("violation_yield", 0.0),
            n_violations=len(all_violations),
            n_unique_policies=len(unique_policies),
            policies_triggered=unique_policies,
        )

        # Curriculum state
        tlog.log(
            curriculum_stage=self.curriculum.stage,
            k_max=self.curriculum.k_max,
            stage_advanced=step_metrics.get("stage_advanced", False),
        )

        # Reward schedule weights
        alpha, beta = self.reward._current_weights()
        tlog.log(alpha=alpha, beta=beta, gamma=self.reward.gamma)

        # Replay buffer size
        tlog.log(replay_buffer_size=len(self.replay))

        # EMA decay
        tlog.log(ema_decay=self.ema_decay)

        # Batch info
        tlog.log(batch_size=len(batch))

        tlog.end_step()

    def _collect_one_group(self) -> CandidateGroup | None:
        """Generate one candidate group: prefix -> G candidates -> score."""
        t = self.curriculum.sample_turn_index(self.rng)

        # Inject a random scenario for this group
        scenario = self.scenario_pool.sample()
        self.auditor.set_scenario(scenario.description)
        if self.ema and self.ema.policy:
            self.ema.policy.set_scenario(scenario.description)

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
