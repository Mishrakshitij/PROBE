"""GRPO: Group-based Relative Policy Optimization.

At state S, sample G candidates {u_i}, compute rewards {R_i}.
Group baseline b = mean(R_i), advantage A_i = R_i - b.
Update: ∇θ E_i [ A_i · log π_θ(u_i | S) ] + KL regularization.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..types import CandidateGroup
from ..auditor.model import AuditorPolicy


class GRPOUpdater:
    """Group-based policy optimization — no critic needed."""

    def __init__(
        self,
        policy: AuditorPolicy,
        lr: float = 1e-5,
        kl_coeff: float = 0.05,
        clip_range: float = 0.2,
        normalize_advantages: bool = True,
    ):
        self.policy = policy
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range
        self.normalize_advantages = normalize_advantages
        self.optimizer = torch.optim.AdamW(
            policy.model.parameters(), lr=lr, weight_decay=0.01
        )

    def update(self, batch: list[CandidateGroup], ref_policy: AuditorPolicy | None = None) -> dict[str, float]:
        """Perform one GRPO update step on a batch of candidate groups.

        Returns dict of training metrics.
        """
        self.policy.model.train()
        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl_loss = 0.0
        n_groups = 0

        for group in batch:
            if not group.rewards:
                continue

            rewards = torch.tensor(group.rewards, dtype=torch.float32)
            baseline = rewards.mean()
            advantages = rewards - baseline
            if self.normalize_advantages and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute log-probs under current policy
            log_probs = []
            for cand in group.candidates:
                lp = self.policy.log_prob(list(group.prefix_turns), cand)
                log_probs.append(lp)
            log_probs_t = torch.tensor(log_probs, dtype=torch.float32, requires_grad=True)

            # Policy gradient loss (with clipping)
            old_log_probs = torch.tensor(group.log_probs, dtype=torch.float32) if group.log_probs else log_probs_t.detach()
            ratio = torch.exp(log_probs_t - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # KL penalty against reference
            kl_loss = torch.tensor(0.0)
            if ref_policy is not None:
                ref_lps = []
                for cand in group.candidates:
                    rlp = ref_policy.log_prob(list(group.prefix_turns), cand)
                    ref_lps.append(rlp)
                ref_lps_t = torch.tensor(ref_lps, dtype=torch.float32)
                kl_loss = (log_probs_t - ref_lps_t).mean()

            loss = pg_loss + self.kl_coeff * kl_loss
            total_loss += loss.item()
            total_pg_loss += pg_loss.item()
            total_kl_loss += kl_loss.item()
            n_groups += 1

        if n_groups > 0:
            # Aggregate and backprop
            # Re-compute with gradients enabled
            self.optimizer.zero_grad()
            agg_loss = self._compute_batch_loss(batch, ref_policy)
            agg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
            self.optimizer.step()

        self.policy.model.eval()
        return {
            "loss": total_loss / max(n_groups, 1),
            "pg_loss": total_pg_loss / max(n_groups, 1),
            "kl_loss": total_kl_loss / max(n_groups, 1),
            "n_groups": n_groups,
        }

    def _compute_batch_loss(self, batch: list[CandidateGroup], ref_policy: AuditorPolicy | None) -> torch.Tensor:
        """Re-compute loss with gradient tracking for backprop."""
        total = torch.tensor(0.0, requires_grad=True, device=self.policy.device)

        for group in batch:
            if not group.rewards:
                continue
            rewards = torch.tensor(group.rewards, dtype=torch.float32)
            advantages = rewards - rewards.mean()
            if self.normalize_advantages and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for i, cand in enumerate(group.candidates):
                lp = self.policy.log_prob(list(group.prefix_turns), cand)
                lp_t = torch.tensor(lp, requires_grad=True)
                total = total - advantages[i] * lp_t

        return total / max(len(batch), 1)
