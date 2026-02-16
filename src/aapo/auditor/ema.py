"""Exponential Moving Average (EMA) policy for stable candidate sampling."""

from __future__ import annotations

import copy

import torch

from .model import AuditorPolicy


class EMAPolicy:
    """Maintains an EMA copy of the auditor policy weights.

    Used for candidate sampling — more stable than the actively-training policy.
    Updated after each training step: θ_ema ← decay * θ_ema + (1 - decay) * θ_train
    """

    def __init__(self, source_policy: AuditorPolicy, decay: float = 0.995):
        self.decay = decay
        self.policy = copy.deepcopy(source_policy)
        self.policy.model.eval()
        # Freeze EMA model
        for p in self.policy.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, source_policy: AuditorPolicy):
        """Update EMA weights from the training policy."""
        for ema_param, src_param in zip(
            self.policy.model.parameters(),
            source_policy.model.parameters(),
        ):
            ema_param.data.mul_(self.decay).add_(src_param.data, alpha=1.0 - self.decay)
