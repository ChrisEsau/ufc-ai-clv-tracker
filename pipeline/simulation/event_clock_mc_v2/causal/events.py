"""Typed fighter-initiated action attempts for causal Event Clock V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from .state import Phase, Side


class ActionFamily(str, Enum):
    """The authoritative MVP action vocabulary exposed to the future brain."""

    STAND_ATTACK = "stand_attack"
    STAND_COUNTER = "stand_counter"
    PRESSURE = "pressure"
    RESET_RANGE = "reset_range"
    CLINCH_ENTRY = "clinch_entry"
    TAKEDOWN_ENTRY = "takedown_entry"

    CLINCH_STRIKE = "clinch_strike"
    CLINCH_CONTROL = "clinch_control"
    CLINCH_TAKEDOWN = "clinch_takedown"
    BREAK_CLINCH = "break_clinch"

    GROUND_STRIKE = "ground_strike"
    ADVANCE_POSITION = "advance_position"
    SUBMISSION_ATTACK = "submission_attack"
    CONTROL = "control"
    DISENGAGE = "disengage"

    ESCAPE_STAND = "escape_stand"
    IMPROVE_POSITION = "improve_position"
    REVERSAL = "reversal"
    BOTTOM_STRIKE = "bottom_strike"


@dataclass(frozen=True)
class ActionEvent:
    """A fighter's attempted action, before any mechanics resolution."""

    timestamp_seconds: float
    actor: Side
    action_family: ActionFamily
    source_phase: Phase

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_seconds, bool)
            or not isinstance(self.timestamp_seconds, Real)
            or not math.isfinite(self.timestamp_seconds)
            or self.timestamp_seconds < 0.0
        ):
            raise ValueError("timestamp_seconds must be a finite non-negative number")
        if not isinstance(self.actor, Side):
            raise ValueError("actor must be a Side value")
        if not isinstance(self.action_family, ActionFamily):
            raise ValueError("action_family must be an ActionFamily value")
        if not isinstance(self.source_phase, Phase):
            raise ValueError("source_phase must be a Phase value")
