"""Clean mechanics boundary for causal Event Clock V2 action attempts."""

from .config import (
    FighterMechanics,
    KOKDArchitecture,
    MechanicsInputs,
    StructuralMVPPlaceholders,
)
from .resolution import (
    ActionOutcome,
    ActionResolution,
    FinishMethod,
    FightTerminationRequest,
    StrikeConsequence,
    TransitionKind,
    TransitionRequest,
)
from .resolver import resolve_action
from .physiology import advance_physiology, apply_action_consequence, recover_round

__all__ = [
    "ActionOutcome",
    "ActionResolution",
    "FinishMethod",
    "FightTerminationRequest",
    "FighterMechanics",
    "KOKDArchitecture",
    "MechanicsInputs",
    "StrikeConsequence",
    "StructuralMVPPlaceholders",
    "TransitionKind",
    "TransitionRequest",
    "resolve_action",
    "advance_physiology",
    "apply_action_consequence",
    "recover_round",
]
