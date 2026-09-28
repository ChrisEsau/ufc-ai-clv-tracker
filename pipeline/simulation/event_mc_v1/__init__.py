"""Generic deterministic continuous-time event simulation kernel."""

from .config import FightConfig
from .engine import SimulationEngine, SimulationResult
from .rng import RNGManager, RNGStream
from .scheduler import EventRate, ExponentialScheduler, probability_to_rate
from .state import ActionAvailabilityState, FightState, Phase, StateDelta

__all__ = [
    "ActionAvailabilityState",
    "EventRate",
    "ExponentialScheduler",
    "FightConfig",
    "FightState",
    "Phase",
    "RNGManager",
    "RNGStream",
    "SimulationEngine",
    "SimulationResult",
    "StateDelta",
    "probability_to_rate",
]
