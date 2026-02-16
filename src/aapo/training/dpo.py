"""DPO: Direct Preference Optimization from candidate pairs.

From each CandidateGroup, pick top-k winners and bottom-k losers.
DPO loss: maximize log σ( β (log π(u+|S) - log π(u-|S)) - β (log π_ref(u+|S) - log π_ref(u-|S)) )
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
        top_k: int = 2,
        bottom_k: int = 2,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.beta = beta
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.optimizer = torch.optim.AdamW(
            policy.model.parameters(), lr=lr, weight_decay=0.01
        )

    def update(self, batch: list[CandidateGroup]) -> dict[str, float]:
        """One DPO update step."""
        self.policy.model.train()
        total_loss = 0.0
        n_pairs = 0

        self.optimizer.zero_grad()

        for group in batch:
            if not group.rewards or len(group.rewards) < self.top_k + self.bottom_k:
                continue

            # Rank candidates by reward
            indexed = sorted(enumerate(group.rewards), key=lambda x: x[1], reverse=True)
            winners = [indexed[i][0] for i in range(self.top_k)]
            losers = [indexed[-(i + 1)][0] for i in range(self.bottom_k)]

            prefix = list(group.prefix_turns)

            for w_idx in winners:
                for l_idx in losers:
                    w_text = group.candidates[w_idx]
                    l_text = group.candidates[l_idx]

                    # Current policy log-probs
                    lp_w = self.policy.log_prob(prefix, w_text)
                    lp_l = self.policy.log_prob(prefix, l_text)

                    # Reference policy log-probs
                    with torch.no_grad():
                        ref_lp_w = self.ref_policy.log_prob(prefix, w_text)
                        ref_lp_l = self.ref_policy.log_prob(prefix, l_text)

                    # DPO loss
                    logit = self.beta * ((lp_w - lp_l) - (ref_lp_w - ref_lp_l))
                    loss = -F.logsigmoid(torch.tensor(logit))

                    total_loss += loss.item()
                    n_pairs += 1

        if n_pairs > 0:
            # Re-run with gradients for backprop
            agg_loss = self._compute_loss(batch)
            agg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
            self.optimizer.step()

        self.policy.model.eval()
        return {
            "loss": total_loss / max(n_pairs, 1),
            "n_pairs": n_pairs,
        }

    def _compute_loss(self, batch: list[CandidateGroup]) -> torch.Tensor:
        total = torch.tensor(0.0, requires_grad=True, device=self.policy.device)
        count = 0

        for group in batch:
            if not group.rewards or len(group.rewards) < self.top_k + self.bottom_k:
                continue
            indexed = sorted(enumerate(group.rewards), key=lambda x: x[1], reverse=True)
            winners = [indexed[i][0] for i in range(self.top_k)]
            losers = [indexed[-(i + 1)][0] for i in range(self.bottom_k)]
            prefix = list(group.prefix_turns)

            for w_idx in winners:
                for l_idx in losers:
                    lp_w = self.policy.log_prob(prefix, group.candidates[w_idx])
                    lp_l = self.policy.log_prob(prefix, group.candidates[l_idx])
                    ref_lp_w = self.ref_policy.log_prob(prefix, group.candidates[w_idx])
                    ref_lp_l = self.ref_policy.log_prob(prefix, group.candidates[l_idx])
                    logit = self.beta * ((lp_w - lp_l) - (ref_lp_w - ref_lp_l))
                    total = total - F.logsigmoid(torch.tensor(logit, requires_grad=True))
                    count += 1

        return total / max(count, 1)
