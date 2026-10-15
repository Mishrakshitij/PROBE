"""GFlowNet trajectory-balance loss for mode-covering sampling.

Target: p*(H) = R_total(H) / Z
TB loss: L = E_H [ (log Z + log p_theta(H) - log R_total(H))^2 ]

Applied on replay data after initial GRPO stabilization.
Softplus floor ensures R > 0.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..types import Episode
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

        # log Z is a learnable scalar (must be on same device as policy)
        self.log_Z = torch.nn.Parameter(
            torch.tensor(0.0, device=policy.device)
        )

        self.optimizer = torch.optim.AdamW([
            {"params": policy.model.parameters(), "lr": lr_policy},
            {"params": [self.log_Z], "lr": lr_logZ},
        ], weight_decay=0.01)

    def update(self, episodes: list[Episode]) -> dict[str, Any]:
        """One GFlowNet TB update on a batch of complete episodes.

        Each episode must have rewards computed (episode.total_reward).
        Uses per-episode gradient accumulation to keep peak memory manageable.
        """
        self.policy.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        n = 0

        # Accumulators for detailed stats
        all_log_probs: list[float] = []
        all_rewards: list[float] = []
        all_residuals: list[float] = []
        all_log_r: list[float] = []

        n_episodes = sum(1 for ep in episodes if ep.turns)

        for ep in episodes:
            if not ep.turns:
                continue

            # Compute log p_theta(H) = sum of log-probs of auditor turns
            # Use differentiable log_prob so gradients flow to policy
            log_p = torch.tensor(0.0, device=self.policy.device)
            for i, turn in enumerate(ep.turns):
                if turn.role.value != "user":
                    continue
                prefix = ep.turns[:i]
                lp = self.policy.log_prob_differentiable(prefix, turn.content)
                log_p = log_p + lp

            log_p_val = log_p.item()
            all_log_probs.append(log_p_val)

            # R_total with softplus floor
            r = ep.total_reward
            all_rewards.append(r)
            r_safe = math.log(1 + math.exp(r)) + self.reward_floor
            log_r = math.log(r_safe)
            all_log_r.append(log_r)

            # TB loss: (log Z + log p_theta(H) - log R)^2
            residual = self.log_Z + log_p - log_r
            residual_val = residual.item()
            all_residuals.append(residual_val)

            loss_i = (residual ** 2) / max(n_episodes, 1)
            loss_i.backward()  # accumulate gradients, free activations

            total_loss += residual_val ** 2
            n += 1

        grad_norm = 0.0
        if n > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(), 1.0
            ).item()
            self.optimizer.step()

        self.policy.model.eval()

        metrics: dict[str, Any] = {
            "loss": total_loss / max(n, 1),
            "gflownet_loss": total_loss / max(n, 1),
            "log_Z": self.log_Z.item(),
            "n_episodes": n,
            "grad_norm": grad_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            # Detailed stats (raw lists for logger to compute full stats)
            "_log_probs": all_log_probs,
            "_rewards": all_rewards,
            "_residuals": all_residuals,
            "_log_r": all_log_r,
        }
        return metrics
