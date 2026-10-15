"""Full multi-turn episode orchestration: auditor ↔ target ↔ tools."""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from ..types import Episode, Role, Turn
from ..environment.target_agent import TargetAgent
from ..environment.logger import InteractionLogger


class AuditorProtocol(Protocol):
    """Minimal interface for any auditor (learned or baseline)."""
    def generate(self, history: list[Turn]) -> str: ...


class EpisodeRunner:
    """Runs a full multi-turn episode between auditor and target agent.

    Flow per turn:
      1. Auditor generates user message u_k given history
      2. Target agent generates response c_k (possibly with tool calls)
      3. Both turns appended to episode
      4. Repeat until K_max or early stop
    """

    def __init__(
        self,
        target: TargetAgent,
        logger: InteractionLogger | None = None,
        max_turns: int = 14,
        loop_detection: bool = True,
    ):
        self.target = target
        self.logger = logger
        self.max_turns = max_turns
        self.loop_detection = loop_detection

    def run(
        self,
        auditor: AuditorProtocol,
        initial_context: str | None = None,
        episode_id: str | None = None,
    ) -> Episode:
        """Run a complete episode.

        Args:
            auditor: generates user messages
            initial_context: optional scenario setup for the auditor
            episode_id: optional ID, auto-generated if None
        """
        eid = episode_id or str(uuid.uuid4())[:12]
        episode = Episode(episode_id=eid)

        if initial_context:
            episode.metadata["initial_context"] = initial_context

        for k in range(self.max_turns):
            # Auditor generates user message
            user_text = auditor.generate(episode.turns)
            if not user_text or user_text.strip() == "":
                break

            user_turn = Turn(
                role=Role.USER,
                content=user_text,
                turn_index=2 * k,
            )
            episode.turns.append(user_turn)

            # Target agent responds (may include tool calls)
            asst_turn = self.target.generate(episode.turns)
            # Re-index to match our scheme
            asst_turn = Turn(
                role=asst_turn.role,
                content=asst_turn.content,
                turn_index=2 * k + 1,
                tool_calls=asst_turn.tool_calls,
            )
            episode.turns.append(asst_turn)

            # Early stop: if target signals end of conversation or loop detected
            if self._should_stop(asst_turn, episode):
                break

        if self.logger:
            self.logger.log_episode(episode)

        return episode

    def run_from_prefix(
        self,
        auditor: AuditorProtocol,
        prefix: list[Turn],
        additional_turns: int = 3,
        episode_id: str | None = None,
    ) -> Episode:
        """Continue an episode from a given prefix for N more turns."""
        eid = episode_id or str(uuid.uuid4())[:12]
        episode = Episode(episode_id=eid, turns=list(prefix))
        start_k = len(prefix) // 2

        for k in range(start_k, start_k + additional_turns):
            user_text = auditor.generate(episode.turns)
            if not user_text:
                break

            user_turn = Turn(role=Role.USER, content=user_text, turn_index=2 * k)
            episode.turns.append(user_turn)

            asst_turn = self.target.generate(episode.turns)
            asst_turn = Turn(
                role=asst_turn.role,
                content=asst_turn.content,
                turn_index=2 * k + 1,
                tool_calls=asst_turn.tool_calls,
            )
            episode.turns.append(asst_turn)

            if self._should_stop(asst_turn, episode):
                break

        return episode

    def _should_stop(self, turn: Turn, episode: Episode | None = None) -> bool:
        """Heuristic early stop conditions."""
        content = turn.content.lower()
        stop_signals = ["goodbye", "have a nice", "anything else i can help",
                        "is there anything else"]
        if any(s in content for s in stop_signals):
            return True

        # Loop detection: if last 3 user messages are >50% similar, stop
        if self.loop_detection and episode is not None:
            recent_user = [t for t in episode.turns if t.role.value == "user"][-3:]
            if len(recent_user) >= 3:
                words = [set(t.content.lower().split()) for t in recent_user]
                overlaps = []
                for a, b in zip(words, words[1:]):
                    overlaps.append(len(a & b) / max(len(a), 1))
                if all(o > 0.5 for o in overlaps):
                    return True

        return False
