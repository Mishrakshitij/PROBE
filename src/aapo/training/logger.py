"""Comprehensive training logger for RL experiments.

Logs every metric at every step to JSONL (and optionally wandb).
Designed for research publication: captures all information needed
for post-hoc analysis and plotting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StepMetrics:
    """All metrics captured at a single training step."""

    # ── Identity ─────────────────────────────────────────────────────
    step: int = 0
    method: str = ""
    wallclock_s: float = 0.0
    timestamp: float = 0.0

    # ── Loss ─────────────────────────────────────────────────────────
    loss: float = 0.0
    pg_loss: float = 0.0
    kl_loss: float = 0.0
    dpo_loss: float = 0.0
    gflownet_loss: float = 0.0
    log_Z: float = 0.0

    # ── Gradient ─────────────────────────────────────────────────────
    grad_norm: float = 0.0

    # ── Rewards (across batch) ───────────────────────────────────────
    reward_total_min: float = 0.0
    reward_total_mean: float = 0.0
    reward_total_max: float = 0.0
    reward_total_std: float = 0.0
    reward_total_count: int = 0

    reward_violation_min: float = 0.0
    reward_violation_mean: float = 0.0
    reward_violation_max: float = 0.0
    reward_violation_std: float = 0.0

    reward_progress_min: float = 0.0
    reward_progress_mean: float = 0.0
    reward_progress_max: float = 0.0
    reward_progress_std: float = 0.0

    reward_novelty_min: float = 0.0
    reward_novelty_mean: float = 0.0
    reward_novelty_max: float = 0.0
    reward_novelty_std: float = 0.0

    reward_plausibility_min: float = 0.0
    reward_plausibility_mean: float = 0.0
    reward_plausibility_max: float = 0.0
    reward_plausibility_std: float = 0.0

    # ── Advantage stats (GRPO) ───────────────────────────────────────
    advantage_min: float = 0.0
    advantage_mean: float = 0.0
    advantage_max: float = 0.0
    advantage_std: float = 0.0

    # ── Log-prob stats ───────────────────────────────────────────────
    log_prob_min: float = 0.0
    log_prob_mean: float = 0.0
    log_prob_max: float = 0.0
    log_prob_std: float = 0.0

    # ── Ratio stats (GRPO) ──────────────────────────────────────────
    ratio_min: float = 0.0
    ratio_mean: float = 0.0
    ratio_max: float = 0.0
    ratio_std: float = 0.0
    clip_fraction: float = 0.0

    # ── KL divergence ────────────────────────────────────────────────
    kl_divergence: float = 0.0

    # ── DPO-specific ─────────────────────────────────────────────────
    implicit_accuracy: float = 0.0
    margin_min: float = 0.0
    margin_mean: float = 0.0
    margin_max: float = 0.0
    margin_std: float = 0.0

    # ── GFlowNet-specific ────────────────────────────────────────────
    residual_min: float = 0.0
    residual_mean: float = 0.0
    residual_max: float = 0.0
    residual_std: float = 0.0
    log_r_mean: float = 0.0

    # ── Completion / generation ──────────────────────────────────────
    completion_len_min: int = 0
    completion_len_mean: float = 0.0
    completion_len_max: int = 0
    completion_len_std: float = 0.0
    num_turns_min: int = 0
    num_turns_mean: float = 0.0
    num_turns_max: int = 0

    # ── Batch info ───────────────────────────────────────────────────
    n_groups: int = 0
    n_pairs: int = 0
    n_episodes: int = 0
    batch_size: int = 0

    # ── Violation / yield ────────────────────────────────────────────
    violation_yield: float = 0.0
    n_violations: int = 0
    n_unique_policies: int = 0
    policies_triggered: list[str] = field(default_factory=list)

    # ── Curriculum ───────────────────────────────────────────────────
    curriculum_stage: int = 0
    k_max: int = 6
    stage_advanced: bool = False

    # ── Reward schedule weights ──────────────────────────────────────
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    # ── Replay buffer ────────────────────────────────────────────────
    replay_buffer_size: int = 0

    # ── EMA ──────────────────────────────────────────────────────────
    ema_decay: float = 0.0

    # ── Learning rate ────────────────────────────────────────────────
    learning_rate: float = 0.0


def _safe_float(v: Any) -> float:
    """Convert to float, handling tensors and special values."""
    if hasattr(v, "item"):
        v = v.item()
    v = float(v)
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return v


class TrainingLogger:
    """Logs comprehensive per-step metrics to JSONL and optionally wandb.

    Usage:
        logger = TrainingLogger(log_dir="data/logs/run_01", use_wandb=True)
        logger.begin_step(step=0)
        logger.log(loss=0.5, grad_norm=1.2)
        logger.log_reward_stats(rewards=[...], breakdowns=[...])
        logger.end_step()
        logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path = "data/logs",
        run_name: str | None = None,
        method: str = "grpo",
        use_wandb: bool = False,
        wandb_project: str = "aapo",
        wandb_config: dict[str, Any] | None = None,
        print_every: int = 1,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.print_every = print_every

        # JSONL log file
        self._jsonl_path = self.log_dir / "training_metrics.jsonl"
        self._jsonl_file = open(self._jsonl_path, "a")

        # In-memory history for plotting
        self._history: list[dict[str, Any]] = []

        # Current step accumulator
        self._current: StepMetrics | None = None
        self._step_start: float = 0.0

        # wandb
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=run_name or f"aapo-{method}",
                    config=wandb_config or {},
                )
                self._wandb = wandb
            except ImportError:
                print("WARNING: wandb not installed, skipping wandb logging")

    def begin_step(self, step: int):
        """Start timing and create a fresh StepMetrics for this step."""
        self._current = StepMetrics(
            step=step,
            method=self.method,
            timestamp=time.time(),
        )
        self._step_start = time.time()

    def log(self, **kwargs: Any):
        """Log arbitrary key-value pairs into the current step."""
        if self._current is None:
            return
        for k, v in kwargs.items():
            if hasattr(self._current, k):
                setattr(self._current, k, v)

    def log_reward_stats(
        self,
        rewards: list[float],
        breakdowns: list[dict[str, float]] | None = None,
    ):
        """Compute and log min/mean/max/std for reward and its components.

        Args:
            rewards: list of total reward values across the batch
            breakdowns: list of dicts with keys {violation, progress, novelty, plausibility, total}
        """
        if self._current is None or not rewards:
            return

        arr = np.array(rewards)
        self._current.reward_total_min = _safe_float(arr.min())
        self._current.reward_total_mean = _safe_float(arr.mean())
        self._current.reward_total_max = _safe_float(arr.max())
        self._current.reward_total_std = _safe_float(arr.std())
        self._current.reward_total_count = len(rewards)

        if breakdowns:
            for component in ("violation", "progress", "novelty", "plausibility"):
                vals = np.array([b.get(component, 0.0) for b in breakdowns])
                if len(vals) > 0:
                    setattr(self._current, f"reward_{component}_min", _safe_float(vals.min()))
                    setattr(self._current, f"reward_{component}_mean", _safe_float(vals.mean()))
                    setattr(self._current, f"reward_{component}_max", _safe_float(vals.max()))
                    setattr(self._current, f"reward_{component}_std", _safe_float(vals.std()))

    def log_advantage_stats(self, advantages: list[float]):
        """Log advantage statistics (GRPO)."""
        if self._current is None or not advantages:
            return
        arr = np.array(advantages)
        self._current.advantage_min = _safe_float(arr.min())
        self._current.advantage_mean = _safe_float(arr.mean())
        self._current.advantage_max = _safe_float(arr.max())
        self._current.advantage_std = _safe_float(arr.std())

    def log_log_prob_stats(self, log_probs: list[float]):
        """Log log-probability statistics."""
        if self._current is None or not log_probs:
            return
        arr = np.array(log_probs)
        self._current.log_prob_min = _safe_float(arr.min())
        self._current.log_prob_mean = _safe_float(arr.mean())
        self._current.log_prob_max = _safe_float(arr.max())
        self._current.log_prob_std = _safe_float(arr.std())

    def log_ratio_stats(self, ratios: list[float], clip_range: float = 0.2):
        """Log probability ratio statistics and clip fraction (GRPO)."""
        if self._current is None or not ratios:
            return
        arr = np.array(ratios)
        self._current.ratio_min = _safe_float(arr.min())
        self._current.ratio_mean = _safe_float(arr.mean())
        self._current.ratio_max = _safe_float(arr.max())
        self._current.ratio_std = _safe_float(arr.std())
        clipped = np.sum((arr < 1.0 - clip_range) | (arr > 1.0 + clip_range))
        self._current.clip_fraction = _safe_float(clipped / max(len(arr), 1))

    def log_completion_stats(self, lengths: list[int]):
        """Log completion (generation) length statistics."""
        if self._current is None or not lengths:
            return
        arr = np.array(lengths)
        self._current.completion_len_min = int(arr.min())
        self._current.completion_len_mean = _safe_float(arr.mean())
        self._current.completion_len_max = int(arr.max())
        self._current.completion_len_std = _safe_float(arr.std())

    def log_turn_count_stats(self, turn_counts: list[int]):
        """Log per-episode turn count statistics."""
        if self._current is None or not turn_counts:
            return
        arr = np.array(turn_counts)
        self._current.num_turns_min = int(arr.min())
        self._current.num_turns_mean = _safe_float(arr.mean())
        self._current.num_turns_max = int(arr.max())

    def log_dpo_stats(self, margins: list[float], accuracies: list[float]):
        """Log DPO-specific statistics (margins and implicit accuracy)."""
        if self._current is None:
            return
        if margins:
            arr = np.array(margins)
            self._current.margin_min = _safe_float(arr.min())
            self._current.margin_mean = _safe_float(arr.mean())
            self._current.margin_max = _safe_float(arr.max())
            self._current.margin_std = _safe_float(arr.std())
        if accuracies:
            self._current.implicit_accuracy = _safe_float(
                np.mean(accuracies)
            )

    def log_gflownet_stats(self, residuals: list[float], log_rs: list[float] | None = None):
        """Log GFlowNet-specific statistics (TB residuals)."""
        if self._current is None:
            return
        if residuals:
            arr = np.array(residuals)
            self._current.residual_min = _safe_float(arr.min())
            self._current.residual_mean = _safe_float(arr.mean())
            self._current.residual_max = _safe_float(arr.max())
            self._current.residual_std = _safe_float(arr.std())
        if log_rs:
            self._current.log_r_mean = _safe_float(np.mean(log_rs))

    def end_step(self):
        """Finalize the step: compute wallclock, write to JSONL, optionally wandb."""
        if self._current is None:
            return

        self._current.wallclock_s = time.time() - self._step_start
        record = asdict(self._current)

        # Write JSONL
        self._jsonl_file.write(json.dumps(record, default=str) + "\n")
        self._jsonl_file.flush()

        # In-memory history
        self._history.append(record)

        # wandb
        if self._wandb is not None:
            # Flatten for wandb (remove list fields that wandb can't handle well)
            wb_record = {k: v for k, v in record.items() if not isinstance(v, list)}
            self._wandb.log(wb_record, step=self._current.step)

        # Console print
        if self._current.step % self.print_every == 0:
            self._print_step(self._current)

        self._current = None

    def _print_step(self, m: StepMetrics):
        """Print a concise but informative step summary to console."""
        parts = [
            f"Step {m.step:5d}",
            f"loss={m.loss:.4f}",
        ]
        if m.method == "grpo":
            parts.append(f"pg={m.pg_loss:.4f}")
            parts.append(f"kl={m.kl_loss:.4f}")
            parts.append(f"clip={m.clip_fraction:.3f}")
            parts.append(f"ratio=[{m.ratio_min:.3f},{m.ratio_max:.3f}]")
        elif m.method == "dpo":
            parts.append(f"acc={m.implicit_accuracy:.3f}")
            parts.append(f"margin={m.margin_mean:+.3f}[{m.margin_min:+.3f},{m.margin_max:+.3f}]")
        elif m.method == "gflownet":
            parts.append(f"logZ={m.log_Z:.3f}")
            parts.append(f"resid={m.residual_mean:+.3f}±{m.residual_std:.3f}")
        parts.extend([
            f"R={m.reward_total_mean:+.3f}[{m.reward_total_min:+.3f},{m.reward_total_max:+.3f}]",
            f"grad={m.grad_norm:.4f}",
            f"lp={m.log_prob_mean:.2f}[{m.log_prob_min:.2f},{m.log_prob_max:.2f}]",
            f"len={m.completion_len_mean:.0f}",
            f"yield={m.violation_yield:.3f}",
            f"stg={m.curriculum_stage}(k={m.k_max})",
            f"buf={m.replay_buffer_size}",
            f"lr={m.learning_rate:.1e}",
            f"t={m.wallclock_s:.1f}s",
        ])
        print(" | ".join(parts))

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return the in-memory history of all logged steps."""
        return self._history

    def load_history(self, path: str | Path | None = None) -> list[dict[str, Any]]:
        """Load training history from a JSONL file."""
        path = Path(path) if path else self._jsonl_path
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def close(self):
        """Flush and close all logging handles."""
        if self._jsonl_file and not self._jsonl_file.closed:
            self._jsonl_file.flush()
            self._jsonl_file.close()
        if self._wandb is not None:
            self._wandb.finish()
