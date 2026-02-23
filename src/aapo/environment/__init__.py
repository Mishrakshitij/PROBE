"""Flight-booking environment: DB, tools, policies, target agent.

Lazy imports for TargetAgent to avoid pulling in inference deps at import time.
"""

from .database import FlightDB
from .tools import ToolRegistry
from .policies import PolicyRegistry
from .logger import InteractionLogger


def __getattr__(name):
    if name == "TargetAgent":
        from .target_agent import TargetAgent
        return TargetAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FlightDB", "ToolRegistry", "PolicyRegistry", "TargetAgent", "InteractionLogger"]
