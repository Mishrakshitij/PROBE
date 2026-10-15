"""GRPO: Group-based Relative Policy Optimization.

At state S, sample G candidates {u_i}, compute rewards {R_i}.
Group baseline b = mean(R_i), advantage A_i = R_i - b.
Update: nabla_theta E_i [ A_i * log pi_theta(u_i | S) ] + KL regularization.
"""

from __future__ import annotations

from typing import Any

import torch

from ..types import CandidateGroup
from ..auditor.model import AuditorPolicy


class GRPOUpdater:
    """Group-based policy optimization -- no critic needed."""

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

    def update(self, batch: list[CandidateGroup], ref_policy: AuditorPolicy | None = None) -> dict[str, Any]:
        """Perform one GRPO update step on a batch of candidate groups.

        Uses per-candidate gradient accumulation to keep peak memory at ~1
        forward pass worth of activations.

        Returns dict of training metrics with comprehensive statistics.
        """
        self.policy.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl_loss = 0.0
        n_groups = 0

        # Accumulators for detailed stats
        all_advantages: list[float] = []
        all_log_probs: list[float] = []
        all_ratios: list[float] = []
        all_rewards: list[float] = []

        batch_size = max(len(batch), 1)

        for group in batch:
            if not group.rewards:
                continue

            rewards = torch.tensor(group.rewards, dtype=torch.float32)
            all_rewards.extend(group.rewards)
            baseline = rewards.mean()
            advantages = rewards - baseline
            if self.normalize_advantages and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            all_advantages.extend(advantages.tolist())

            old_log_probs = list(group.log_probs) if group.log_probs else None
            prefix = list(group.prefix_turns)

            # Batched differentiable forward pass: 1 fwd pass for all G candidates
            lps = self.policy.log_prob_differentiable_batch(prefix, list(group.candidates))

            # Batched ref policy log-probs (non-differentiable)
            ref_lps: list[float] | None = None
            if ref_policy is not None:
                ref_lps = ref_policy.log_prob_batch(prefix, list(group.candidates))

            # Accumulate losses for single backward (shared computation graph)
            group_loss = torch.tensor(0.0, device=self.policy.device)

            for i, (cand, lp) in enumerate(zip(group.candidates, lps)):
                lp_val = lp.item()
                all_log_probs.append(lp_val)

                # Ratio for clipping
                if old_log_probs is not None:
                    old_lp = old_log_probs[i]
                else:
                    old_lp = lp_val
                ratio_val = torch.exp(lp.detach() - old_lp).item()
                all_ratios.append(ratio_val)

                # Policy gradient loss (with clipping)
                ratio = torch.exp(lp - old_lp)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                adv = advantages[i].item()
                pg_loss_i = -torch.min(ratio * adv, clipped_ratio * adv)

                # KL penalty against reference
                kl_loss_i = 0.0
                if ref_lps is not None:
                    kl_loss_i = lp_val - ref_lps[i]

                group_loss = group_loss + (pg_loss_i + self.kl_coeff * kl_loss_i) / batch_size

                total_pg_loss += pg_loss_i.item()
                total_kl_loss += kl_loss_i if isinstance(kl_loss_i, float) else kl_loss_i.item()

            group_loss.backward()  # single backward for shared computation graph
            n_groups += 1

        total_loss = total_pg_loss + self.kl_coeff * total_kl_loss

        grad_norm = 0.0
        if n_groups > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(), 1.0
            ).item()
            self.optimizer.step()

        self.policy.model.eval()

        # Compute clip fraction
        clip_fraction = 0.0
        if all_ratios:
            r = torch.tensor(all_ratios)
            clipped = ((r < 1.0 - self.clip_range) | (r > 1.0 + self.clip_range)).float()
            clip_fraction = clipped.mean().item()

        metrics: dict[str, Any] = {
            "loss": total_loss / max(n_groups, 1),
            "pg_loss": total_pg_loss / max(n_groups, 1),
            "kl_loss": total_kl_loss / max(n_groups, 1),
            "kl_divergence": total_kl_loss / max(n_groups, 1),
            "n_groups": n_groups,
            "grad_norm": grad_norm,
            "clip_fraction": clip_fraction,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            # Detailed stats as lists (for the logger to compute min/mean/max/std)
            "_advantages": all_advantages,
            "_log_probs": all_log_probs,
            "_ratios": all_ratios,
            "_rewards": all_rewards,
        }
        return metrics
