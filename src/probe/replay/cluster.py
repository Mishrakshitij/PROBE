"""Incremental HDBSCAN clustering for failure mode discovery."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import ClusterInfo, Episode
from .embedder import TrajectoryEmbedder


class ClusterManager:
    """Manages clustering of violation episodes into failure mode groups."""

    def __init__(
        self,
        embedder: TrajectoryEmbedder,
        min_cluster_size: int = 3,
        min_samples: int = 2,
    ):
        self.embedder = embedder
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        self._embeddings: list[np.ndarray] = []
        self._episode_ids: list[str] = []
        self._labels: np.ndarray = np.array([], dtype=int)
        self._clusters: dict[int, ClusterInfo] = {}

    def add_and_cluster(self, episodes: list[Episode]) -> dict[int, ClusterInfo]:
        """Add new episodes, re-cluster, return updated cluster info."""
        if not episodes:
            return self._clusters

        new_embeddings = self.embedder.embed_episodes(episodes)
        for i, ep in enumerate(episodes):
            self._embeddings.append(new_embeddings[i])
            self._episode_ids.append(ep.episode_id)

        self._recluster()
        return self._clusters

    def get_cluster_id(self, episode_id: str) -> int:
        """Get cluster assignment for an episode. -1 = noise/unclustered."""
        if episode_id not in self._episode_ids:
            return -1
        idx = self._episode_ids.index(episode_id)
        if idx < len(self._labels):
            return int(self._labels[idx])
        return -1

    def _recluster(self):
        """Run HDBSCAN on all accumulated embeddings."""
        if len(self._embeddings) < self.min_cluster_size:
            self._labels = np.full(len(self._embeddings), -1, dtype=int)
            return

        import hdbscan
        X = np.array(self._embeddings)
        # Auto-lower for small datasets to avoid all-noise
        n = len(X)
        mcs = min(self.min_cluster_size, max(2, n // 5))
        ms = min(self.min_samples, mcs)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
        )
        self._labels = clusterer.fit_predict(X)

        # Build cluster info
        self._clusters.clear()
        unique_labels = set(self._labels)
        for label in unique_labels:
            if label == -1:
                continue
            mask = self._labels == label
            indices = np.where(mask)[0]
            cluster_ep_ids = [self._episode_ids[i] for i in indices]

            self._clusters[int(label)] = ClusterInfo(
                cluster_id=int(label),
                centroid_episode_id=cluster_ep_ids[0],  # approximate centroid
                policy_ids=(),  # filled externally
                size=len(indices),
                mean_reward=0.0,
            )

    @property
    def num_clusters(self) -> int:
        return len(self._clusters)

    @property
    def all_clusters(self) -> dict[int, ClusterInfo]:
        return dict(self._clusters)
