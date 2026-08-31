"""Typed generic primary and lifecycle events."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KernelEvent:
    timestamp_seconds: float


@dataclass(frozen=True)
class PrimaryEvent(KernelEvent):
    candidate_id: str
    payload: Any = None


@dataclass(frozen=True)
class ConsequenceEvent(KernelEvent):
    event_type: str
    payload: Any = None


@dataclass(frozen=True)
class RoundStarted(KernelEvent):
    round_number: int


@dataclass(frozen=True)
class RoundEnded(KernelEvent):
    round_number: int


@dataclass(frozen=True)
class FightFinished(KernelEvent):
    reason: str
