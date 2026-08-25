"""Clean mechanics boundary for causal Event Clock V2 action attempts."""

from .config import FighterMechanics, MechanicsInputs, StructuralMVPPlaceholders
from .resolution import (
    ActionOutcome,
    ActionResolution,
    FightTerminationRequest,
    StrikeConsequence,
    TransitionKind,
    TransitionRequest,
)
from .resolver import resolve_action

__all__ = [
    "ActionOutcome",
    "ActionResolution",
    "FightTerminationRequest",
    "FighterMechanics",
    "MechanicsInputs",
    "StrikeConsequence",
    "StructuralMVPPlaceholders",
    "TransitionKind",
    "TransitionRequest",
    "resolve_action",
]
