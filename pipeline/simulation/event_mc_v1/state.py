"""Authoritative mutable state and immutable state changes."""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class Phase(str, Enum):
    DISTANCE = "distance"
    CLINCH = "clinch"
    GROUND = "ground"


@dataclass(frozen=True)
class ActionAvailabilityState:
    """Inactive-by-default extension point for future semi-Markov actions."""

    busy_until_seconds: float | None = None
    cooldown_until_seconds: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cooldown_until_seconds",
            MappingProxyType(dict(self.cooldown_until_seconds)),
        )

    def is_available(self, action_family: str, at_seconds: float) -> bool:
        return (
            (self.busy_until_seconds is None or at_seconds >= self.busy_until_seconds)
            and at_seconds >= self.cooldown_until_seconds.get(action_family, 0.0)
        )


@dataclass
class FightState:
    fight_time_seconds: float = 0.0
    phase: Phase = Phase.DISTANCE
    ground_controller: str | None = None
    clinch_controller: str | None = None
    finished: bool = False
    finish_reason: str | None = None
    winner: str | None = None
    finish_method: str | None = None
    red_stamina: float = 1.0
    blue_stamina: float = 1.0
    red_cumulative_trauma: float = 0.0
    blue_cumulative_trauma: float = 0.0
    red_acute_vulnerability: float = 0.0
    blue_acute_vulnerability: float = 0.0
    action_availability: ActionAvailabilityState = field(
        default_factory=ActionAvailabilityState
    )


@dataclass(frozen=True)
class StateDelta:
    """An immutable request that only the engine may apply to ``FightState``."""

    phase: Phase | None = None
    ground_controller: str | None = None
    set_ground_controller: bool = False
    clinch_controller: str | None = None
    set_clinch_controller: bool = False
    finished: bool | None = None
    finish_reason: str | None = None
    winner: str | None = None
    finish_method: str | None = None
    red_stamina: float | None = None
    blue_stamina: float | None = None
    red_cumulative_trauma: float | None = None
    blue_cumulative_trauma: float | None = None
    red_acute_vulnerability: float | None = None
    blue_acute_vulnerability: float | None = None
    action_availability: ActionAvailabilityState | None = None
