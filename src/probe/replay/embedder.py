"""Sentence-transformer encoding for episodes/turns.

Used by clustering, dedup, and novelty scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import Episode


class TrajectoryEmbedder:
    """Embeds episodes using sentence-transformers on violation-triggering turns + context."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_episode(self, episode: Episode) -> np.ndarray:
        """Embed an episode into a single vector.

        Strategy: concatenate violation-triggering turns with surrounding context,
        then encode. If no violations, encode the full transcript.
        """
        text = self._extract_key_text(episode)
        return self.model.encode(text, normalize_embeddings=True)

    def embed_episodes(self, episodes: list[Episode]) -> np.ndarray:
        """Batch-embed multiple episodes."""
        texts = [self._extract_key_text(ep) for ep in episodes]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def _extract_key_text(self, episode: Episode) -> str:
        """Extract the most relevant text for embedding.

        Focus on violation-triggering turns + 1 turn of context on each side.
        """
        if not episode.violations:
            # No violations: use full transcript (truncated)
            return episode.transcript()[:512]

        violation_indices = {v.turn_index for v in episode.violations}
        relevant_indices = set()
        for idx in violation_indices:
            relevant_indices.update([idx - 1, idx, idx + 1])

        parts = []
        for t in episode.turns:
            if t.turn_index in relevant_indices:
                parts.append(f"[{t.role.value}] {t.content}")
                for tc in t.tool_calls:
                    parts.append(f"  tool:{tc.tool_name}")

        text = "\n".join(parts)
        return text[:512]
