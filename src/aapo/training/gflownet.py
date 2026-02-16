"""GFlowNet trajectory-balance loss for mode-covering sampling.

Target: p*(H) = R_total(H) / Z
TB loss: L = E_H [ (log Z + log p_θ(H) - log R_total(H))^2 ]

Applied on replay data after initial GRPO stabilization.
Softplus floor ensures R > 0.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..types import CandidateGroup, Episode
from ..auditor.model import AuditorPolicy


class GFlowNetUpdater:
    """Trajectory-balance GFlowNet objective for reward-proportional sampling."""

    def __init__(
        self,
        policy: AuditorPolicy,
        lr_logZ: float = 1e-3,
        lr_policy: float = 1e-5,
        reward_floor: float = 0.01,
    ):
        self.policy = policy
        self.reward_floor = reward_floor

        # log Z is a learnable scalar
        self.log_Z = torch.nn.Parameter(torch.tensor(0.0))

        self.optimizer = torch.optim.AdamW([
            {"params": policy.model.parameters(), "lr": lr_policy},
            {"params": [self.log_Z], "lr": lr_logZ},
        ], weight_decay=0.01)

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """One GFlowNet TB update on a batch of complete episodes.

        Each episode must have rewards computed (episode.total_reward).
        """
        self.policy.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        n = 0

        for ep in episodes:
            if not ep.turns:
                continue

            # Compute log p_θ(H) = sum of log-probs of auditor turns
            log_p = 0.0
            for i, turn in enumerate(ep.turns):
                if turn.role.value != "user":
                    continue
                prefix = ep.turns[:i]
                lp = self.policy.log_prob(prefix, turn.content)
                log_p += lp

            # R_total with softplus floor
            r = ep.total_reward
            r_safe = math.log(1 + math.exp(r)) + self.reward_floor
            log_r = math.log(r_safe)

            # TB loss: (log Z + log p_θ(H) - log R)^2
            residual = self.log_Z.item() + log_p - log_r
            loss = residual ** 2
            total_loss += loss
            n += 1

        if n > 0:
            # Recompute with gradient tracking
            batch_loss = self._compute_loss(episodes)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
            self.optimizer.step()

        self.policy.model.eval()
        return {
            "loss": total_loss / max(n, 1),
            "log_Z": self.log_Z.item(),
            "n_episodes": n,
        }

    def _compute_loss(self, episodes: list[Episode]) -> torch.Tensor:
        total = torch.tensor(0.0, requires_grad=True)
        count = 0

        for ep in episodes:
            if not ep.turns:
                continue
            log_p = torch.tensor(0.0, requires_grad=True)
            for i, turn in enumerate(ep.turns):
                if turn.role.value != "user":
                    continue
                prefix = ep.turns[:i]
                lp = self.policy.log_prob(prefix, turn.content)
                log_p = log_p + lp

            r = ep.total_reward
            r_safe = math.log(1 + math.exp(r)) + self.reward_floor
            log_r = math.log(r_safe)

            residual = self.log_Z + log_p - log_r
            total = total + residual ** 2
            count += 1

        return total / max(count, 1)
