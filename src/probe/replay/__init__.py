"""Replay buffer, embedding, clustering, dedup, catalog."""

__all__ = ["ReplayBuffer", "ClusterManager", "FailureCatalog"]


def __getattr__(name):
    if name == "ReplayBuffer":
        from .buffer import ReplayBuffer
        return ReplayBuffer
    if name == "ClusterManager":
        from .cluster import ClusterManager
        return ClusterManager
    if name == "FailureCatalog":
        from .catalog import FailureCatalog
        return FailureCatalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
