"""Shared-prefix candidate sampling: fan out G candidates from one prefix.

Core efficiency trick: generate G different auditor messages from the same
conversation prefix S_t, run short continuations for each, then score them
as a group for GRPO/DPO/GFlowNet updates.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from ..types import CandidateGroup, Episode, Role, Turn
from ..environment.target_agent import TargetAgent


class CandidateAuditorProtocol(Protocol):
    """Auditor that can generate multiple candidates."""
    def generate_candidates(self, history: list[Turn], n: int) -> list[str]: ...
    def log_prob(self, history: list[Turn], text: str) -> float: ...


class SharedPrefixRunner:
    """Generate G candidates from a shared prefix, run continuations, score them.

    Pipeline:
      prefix S_t → G auditor messages {u_t^i} → G short continuations → G rewards
    """

    def __init__(
        self,
        target: TargetAgent,
        continuation_turns: int = 2,
    ):
        self.target = target
        self.continuation_turns = continuation_turns

    def run(
        self,
        auditor: CandidateAuditorProtocol,
        prefix: list[Turn],
        group_size: int = 8,
    ) -> CandidateGroup:
        """Generate and evaluate a group of candidates from a shared prefix.

        Args:
            auditor: model that generates candidate messages and computes log-probs
            prefix: conversation history up to the branching point
            group_size: number of candidates G to generate

        Returns:
            CandidateGroup with candidates, continuations, rewards (rewards filled later)
        """
        turn_index = len(prefix)

        # Generate G candidate auditor messages
        candidates = auditor.generate_candidates(prefix, group_size)

        # For each candidate, run a short continuation
        continuations: list[Episode] = []
        log_probs: list[float] = []

        for i, candidate_text in enumerate(candidates):
            # Create user turn from candidate
            user_turn = Turn(
                role=Role.USER,
                content=candidate_text,
                turn_index=turn_index,
            )
            history = list(prefix) + [user_turn]

            # Run continuation: target responds, then optionally more turns
            episode = Episode(episode_id=f"cand_{uuid.uuid4().hex[:8]}")
            episode.turns = list(history)

            for step in range(self.continuation_turns):
                # Target response
                asst_turn = self.target.generate(episode.turns)
                asst_turn = Turn(
                    role=asst_turn.role,
                    content=asst_turn.content,
                    turn_index=turn_index + 1 + 2 * step,
                    tool_calls=asst_turn.tool_calls,
                )
                episode.turns.append(asst_turn)

                # If more continuation turns needed, generate a simple follow-up
                if step < self.continuation_turns - 1:
                    follow_up = auditor.generate_candidates(episode.turns, 1)
                    if follow_up:
                        fu_turn = Turn(
                            role=Role.USER,
                            content=follow_up[0],
                            turn_index=turn_index + 2 + 2 * step,
                        )
                        episode.turns.append(fu_turn)

            continuations.append(episode)
            # Compute log-prob of the candidate under current policy
            lp = auditor.log_prob(prefix, candidate_text)
            log_probs.append(lp)

        return CandidateGroup(
            prefix_turns=tuple(prefix),
            turn_index=turn_index,
            candidates=tuple(candidates),
            continuations=tuple(continuations),
            rewards=(),  # filled by reward layer
            log_probs=tuple(log_probs),
        )
