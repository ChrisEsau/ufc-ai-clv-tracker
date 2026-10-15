"""Protocols and immutable results used to compose the generic kernel."""

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .config import FightConfig
from .events import ConsequenceEvent
from .rng import RNGStream
from .scheduler import EventRate
from .state import FightState, StateDelta


@dataclass(frozen=True)
class FightContext:
    config: FightConfig
    fight_time_seconds: float
    round_number: int


@dataclass(frozen=True)
class Resolution:
    delta: StateDelta = field(default_factory=StateDelta)
    consequence_events: tuple[ConsequenceEvent, ...] = ()
    payload: object = None


class ScheduledCandidate(Protocol):
    @property
    def candidate_id(self) -> str: ...

    @property
    def rng_stream(self) -> RNGStream: ...

    def resolve(
        self, state: FightState, context: FightContext, rng: np.random.Generator
    ) -> Resolution: ...


class RateProvider(Protocol):
    def candidates(
        self, state: FightState, context: FightContext
    ) -> Sequence[EventRate[ScheduledCandidate]]: ...


class TimeAdvanceModel(Protocol):
    def advance(
        self, state: FightState, context: FightContext, dt_seconds: float
    ) -> StateDelta: ...


class NoOpTimeAdvanceModel:
    def advance(
        self, state: FightState, context: FightContext, dt_seconds: float
    ) -> StateDelta:
        return StateDelta()
