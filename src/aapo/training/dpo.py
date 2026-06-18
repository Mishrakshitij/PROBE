"""DPO: Direct Preference Optimization from candidate pairs.

From each CandidateGroup, pick top-k winners and bottom-k losers.
DPO loss: maximize log sigma( beta (log pi(u+|S) - log pi(u-|S)) - beta (log pi_ref(u+|S) - log pi_ref(u-|S)) )
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..types import CandidateGroup
from ..auditor.model import AuditorPolicy


class DPOUpdater:
    """Direct Preference Optimization from candidate group rankings."""

    def __init__(
        self,
        policy: AuditorPolicy,
        ref_policy: AuditorPolicy,
        beta: float = 0.1,
        lr: float = 5e-6,
        top_k: int = 1,
        bottom_k: int = 1,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.beta = beta
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.optimizer = torch.optim.AdamW(
            policy.model.parameters(), lr=lr, weight_decay=0.01
        )

    def update(self, batch: list[CandidateGroup]) -> dict[str, Any]:
        """One DPO update step.

        Uses per-pair gradient accumulation to keep peak memory at ~2
        forward passes worth of activations (winner + loser).
        """
        self.policy.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        n_pairs = 0

        # Accumulators for detailed stats
        all_log_probs: list[float] = []
        all_rewards: list[float] = []
        all_margins: list[float] = []
        all_accuracies: list[float] = []

        # Count total pairs for loss normalization
        total_pair_count = 0
        for group in batch:
            if not group.rewards or len(group.rewards) < self.top_k + self.bottom_k:
                continue
            total_pair_count += self.top_k * self.bottom_k

        for group in batch:
            if not group.rewards or len(group.rewards) < self.top_k + self.bottom_k:
                continue

            all_rewards.extend(group.rewards)

            # Rank candidates by reward
            indexed = sorted(enumerate(group.rewards), key=lambda x: x[1], reverse=True)
            winners = [indexed[i][0] for i in range(self.top_k)]
            losers = [indexed[-(i + 1)][0] for i in range(self.bottom_k)]

            prefix = list(group.prefix_turns)

            # Collect unique texts for batched forward passes
            pair_indices: list[tuple[int, int]] = []
            for w_idx in winners:
                for l_idx in losers:
                    pair_indices.append((w_idx, l_idx))

            unique_idxs = sorted(set(idx for pair in pair_indices for idx in pair))
            unique_texts = [group.candidates[idx] for idx in unique_idxs]
            idx_to_pos = {idx: pos for pos, idx in enumerate(unique_idxs)}

            # Batched differentiable log-probs (1 fwd pass for all unique texts)
            diff_lps = self.policy.log_prob_differentiable_batch(prefix, unique_texts)
            # Batched ref policy log-probs (1 fwd pass, no gradients)
            ref_lps = self.ref_policy.log_prob_batch(prefix, unique_texts)

            # Accumulate losses for single backward (shared computation graph)
            group_loss = torch.tensor(0.0, device=self.policy.device)

            for w_idx, l_idx in pair_indices:
                lp_w = diff_lps[idx_to_pos[w_idx]]
                lp_l = diff_lps[idx_to_pos[l_idx]]
                all_log_probs.extend([lp_w.item(), lp_l.item()])

                ref_lp_w = ref_lps[idx_to_pos[w_idx]]
                ref_lp_l = ref_lps[idx_to_pos[l_idx]]

                logit = self.beta * ((lp_w - lp_l) - (ref_lp_w - ref_lp_l))
                group_loss = group_loss + (-F.logsigmoid(logit) / max(total_pair_count, 1))

                logit_val = logit.item()
                total_loss += (-F.logsigmoid(torch.tensor(logit_val))).item()
                all_margins.append(logit_val)
                all_accuracies.append(1.0 if logit_val > 0 else 0.0)
                n_pairs += 1

            group_loss.backward()  # single backward for shared computation graph

        grad_norm = 0.0
        if n_pairs > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(), 1.0
            ).item()
            self.optimizer.step()

        self.policy.model.eval()

        metrics: dict[str, Any] = {
            "loss": total_loss / max(n_pairs, 1),
            "dpo_loss": total_loss / max(n_pairs, 1),
            "n_pairs": n_pairs,
            "grad_norm": grad_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "implicit_accuracy": sum(all_accuracies) / max(len(all_accuracies), 1),
            "margin_mean": sum(all_margins) / max(len(all_margins), 1) if all_margins else 0.0,
            # Detailed stats (raw lists for logger to compute full stats)
            "_log_probs": all_log_probs,
            "_rewards": all_rewards,
            "_margins": all_margins,
            "_accuracies": all_accuracies,
        }
        return metrics
