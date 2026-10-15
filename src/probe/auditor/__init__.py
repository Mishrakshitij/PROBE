"""Auditor: learned policy that plays the user role."""

__all__ = ["AuditorPolicy", "CandidateSampler", "EMAPolicy"]


def __getattr__(name):
    if name == "AuditorPolicy":
        from .model import AuditorPolicy
        return AuditorPolicy
    if name == "CandidateSampler":
        from .sampler import CandidateSampler
        return CandidateSampler
    if name == "EMAPolicy":
        from .ema import EMAPolicy
        return EMAPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
