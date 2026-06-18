"""Similarity threshold deduplication for the replay buffer."""

from __future__ import annotations

import numpy as np

from ..types import Episode
from .embedder import TrajectoryEmbedder


class Deduplicator:
    """Remove near-duplicate episodes based on embedding cosine similarity."""

    def __init__(self, embedder: TrajectoryEmbedder, threshold: float = 0.92):
        self.embedder = embedder
        self.threshold = threshold

    def deduplicate(self, episodes: list[Episode]) -> list[Episode]:
        """Return deduplicated subset, keeping higher-reward episodes."""
        if len(episodes) <= 1:
            return episodes

        embeddings = self.embedder.embed_episodes(episodes)
        # Sort by reward descending — keep higher-reward ones
        indexed = sorted(enumerate(episodes), key=lambda x: x[1].total_reward, reverse=True)

        keep_mask = [True] * len(episodes)
        for i in range(len(indexed)):
            if not keep_mask[indexed[i][0]]:
                continue
            for j in range(i + 1, len(indexed)):
                if not keep_mask[indexed[j][0]]:
                    continue
                sim = np.dot(embeddings[indexed[i][0]], embeddings[indexed[j][0]])
                if sim > self.threshold:
                    keep_mask[indexed[j][0]] = False  # remove lower-reward duplicate

        return [ep for ep, keep in zip(episodes, keep_mask) if keep]

    def is_duplicate(self, episode: Episode, existing_embeddings: np.ndarray) -> bool:
        """Check if a single episode is a near-duplicate of any existing."""
        if existing_embeddings.size == 0:
            return False
        emb = self.embedder.embed_episode(episode)
        sims = np.dot(existing_embeddings, emb)
        return bool(np.any(sims > self.threshold))
