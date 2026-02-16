"""Candidate sampler: wraps AuditorPolicy or EMAPolicy for shared-prefix sampling."""

from __future__ import annotations

from typing import Any

from ..types import Turn
from .model import AuditorPolicy
from .ema import EMAPolicy


class CandidateSampler:
    """Samples candidate messages using either the training or EMA policy.

    When EMA policy is available, samples from it (more stable) while
    computing log-probs under the training policy (for importance weighting).
    """

    def __init__(
        self,
        train_policy: AuditorPolicy,
        ema_policy: EMAPolicy | None = None,
    ):
        self.train_policy = train_policy
        self.ema_policy = ema_policy

    @property
    def sampling_policy(self) -> AuditorPolicy:
        """Policy used for generating candidates."""
        if self.ema_policy:
            return self.ema_policy.policy
        return self.train_policy

    def generate_candidates(self, history: list[Turn], n: int) -> list[str]:
        """Generate n candidates from the sampling policy."""
        return self.sampling_policy.generate_candidates(history, n)

    def log_prob(self, history: list[Turn], text: str) -> float:
        """Log-prob under the TRAINING policy (for gradient computation)."""
        return self.train_policy.log_prob(history, text)

    def log_prob_sampling(self, history: list[Turn], text: str) -> float:
        """Log-prob under the SAMPLING policy (for importance weighting)."""
        return self.sampling_policy.log_prob(history, text)
