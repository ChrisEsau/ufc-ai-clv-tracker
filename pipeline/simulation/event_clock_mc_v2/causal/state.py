"""Authoritative physical fight state for the causal Event Clock V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


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
class FighterMemory:
    """Bounded causal evidence accumulated within this simulated path only."""

    striking_edge: float = 0.0
    td_success_recent: float = 0.0
    td_failure_recent: float = 0.0
    td_defense_success_recent: float = 0.0
    control_success_recent: float = 0.0
    score_state: float = 0.0

    def __post_init__(self) -> None:
        for name in ("striking_edge", "score_state"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
        for name in ("td_success_recent", "td_failure_recent", "td_defense_success_recent", "control_success_recent"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class FightMemory:
    red: FighterMemory = FighterMemory()
    blue: FighterMemory = FighterMemory()
    updated_at_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.red, FighterMemory) or not isinstance(self.blue, FighterMemory):
            raise ValueError("fight memory sides must be FighterMemory")
        if not isinstance(self.updated_at_seconds, (int, float)) or not math.isfinite(self.updated_at_seconds) or self.updated_at_seconds < 0:
            raise ValueError("updated_at_seconds must be finite and non-negative")

    def fighter(self, side: Side) -> FighterMemory:
        if not isinstance(side, Side):
            raise ValueError("side must be a Side")
        return self.red if side is Side.RED else self.blue


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
    memory: FightMemory = FightMemory()

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
        if not isinstance(self.memory, FightMemory):
            raise ValueError("memory must be FightMemory")
        if self.memory.updated_at_seconds > self.fight_time_seconds:
            raise ValueError("memory cannot be updated beyond fight time")
