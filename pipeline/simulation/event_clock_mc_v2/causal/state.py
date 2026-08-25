"""Authoritative physical fight state for the causal Event Clock V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    """Persistent physical phases represented by the causal timeline."""

    STANDING = "standing"
    CLINCH = "clinch"
    GROUND = "ground"


class Side(str, Enum):
    """A fighter's stable side in a simulated bout."""

    RED = "red"
    BLUE = "blue"

    @property
    def opponent(self) -> Side:
        return Side.BLUE if self is Side.RED else Side.RED


@dataclass(frozen=True)
class FightState:
    """Minimal Stage 1 state, validated at every construction boundary."""

    fight_time_seconds: float = 0.0
    round_number: int = 1
    phase: Phase = Phase.STANDING
    phase_started_at: float = 0.0
    clinch_controller: Side | None = None
    ground_controller: Side | None = None
    finished: bool = False
    winner: Side | None = None
    finish_method: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, Phase):
            raise ValueError("phase must be a Phase value")
        for field_name, controller in (
            ("clinch_controller", self.clinch_controller),
            ("ground_controller", self.ground_controller),
        ):
            if controller is not None and not isinstance(controller, Side):
                raise ValueError(f"{field_name} must be a Side value or None")
        if self.fight_time_seconds < 0.0:
            raise ValueError("fight_time_seconds cannot be negative")
        if self.round_number < 1:
            raise ValueError("round_number must be at least 1")
        if not 0.0 <= self.phase_started_at <= self.fight_time_seconds:
            raise ValueError("phase_started_at must be within elapsed fight time")

        if self.phase is Phase.STANDING:
            if self.clinch_controller is not None or self.ground_controller is not None:
                raise ValueError("standing cannot carry a controller")
        elif self.phase is Phase.CLINCH:
            if self.clinch_controller is None:
                raise ValueError("clinch requires a clinch controller")
            if self.ground_controller is not None:
                raise ValueError("clinch cannot carry a ground controller")
        elif self.phase is Phase.GROUND:
            if self.ground_controller is None:
                raise ValueError("ground requires a ground controller")
            if self.clinch_controller is not None:
                raise ValueError("ground cannot carry a clinch controller")

        if not self.finished and (self.winner is not None or self.finish_method is not None):
            raise ValueError("an unfinished fight cannot have a winner or finish method")
