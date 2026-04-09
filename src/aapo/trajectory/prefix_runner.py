"""Shared-prefix candidate sampling: fan out G candidates from one prefix.

Core efficiency trick: generate G different auditor messages from the same
conversation prefix S_t, run short continuations for each, then score them
as a group for GRPO/DPO/GFlowNet updates.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from ..types import CandidateGroup, Episode, Role, Turn
from ..environment.target_agent import TargetAgent


class CandidateAuditorProtocol(Protocol):
    """Auditor that can generate multiple candidates."""
    def generate_candidates(self, history: list[Turn], n: int) -> list[str]: ...
    def log_prob(self, history: list[Turn], text: str) -> float: ...
    def log_prob_batch(self, history: list[Turn], texts: list[str]) -> list[float]: ...


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

        # For each candidate, run a single target response (continuation_turns=1).
        # These are independent vLLM calls — run them in parallel.
        def _run_continuation(candidate_text: str) -> Episode:
            user_turn = Turn(
                role=Role.USER,
                content=candidate_text,
                turn_index=turn_index,
            )
            history = list(prefix) + [user_turn]

            episode = Episode(episode_id=f"cand_{uuid.uuid4().hex[:8]}")
            episode.turns = list(history)

            asst_turn = self.target.generate(episode.turns)
            asst_turn = Turn(
                role=asst_turn.role,
                content=asst_turn.content,
                turn_index=turn_index + 1,
                tool_calls=asst_turn.tool_calls,
            )
            episode.turns.append(asst_turn)
            return episode

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            continuations = list(pool.map(_run_continuation, candidates))

        # Batch log-prob: one forward pass for all G candidates
        log_probs = auditor.log_prob_batch(prefix, candidates)

        return CandidateGroup(
            prefix_turns=tuple(prefix),
            turn_index=turn_index,
            candidates=tuple(candidates),
            continuations=tuple(continuations),
            rewards=(),  # filled by reward layer
            log_probs=tuple(log_probs),
        )
