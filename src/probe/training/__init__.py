"""Training layer: GRPO, DPO, GFlowNet updaters + curriculum + loop."""

__all__ = ["TrainingLoop"]


def __getattr__(name):
    if name == "TrainingLoop":
        from .loop import TrainingLoop
        return TrainingLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
