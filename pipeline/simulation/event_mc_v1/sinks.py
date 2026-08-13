"""Write-only observers that cannot affect engine physics or RNG use."""

from dataclasses import dataclass, field
from typing import Protocol

from .events import KernelEvent
from .state import FightState


@dataclass(frozen=True)
class StateSnapshot:
    fight_time_seconds: float
    phase: str
    ground_controller: str | None
    clinch_controller: str | None
    finished: bool
    finish_reason: str | None
    winner: str | None
    finish_method: str | None
    red_stamina: float
    blue_stamina: float
    red_cumulative_trauma: float
    blue_cumulative_trauma: float
    red_acute_vulnerability: float
    blue_acute_vulnerability: float

    @classmethod
    def from_state(cls, state: FightState) -> "StateSnapshot":
        return cls(
            state.fight_time_seconds,
            state.phase.value,
            state.ground_controller,
            state.clinch_controller,
            state.finished,
            state.finish_reason,
            state.winner,
            state.finish_method,
            state.red_stamina,
            state.blue_stamina,
            state.red_cumulative_trauma,
            state.blue_cumulative_trauma,
            state.red_acute_vulnerability,
            state.blue_acute_vulnerability,
        )


class EventSink(Protocol):
    def on_time_advance(
        self, dt_seconds: float, before: StateSnapshot, after: StateSnapshot
    ) -> None: ...

    def on_event(
        self, event: KernelEvent, before: StateSnapshot, after: StateSnapshot
    ) -> None: ...

    def finalize(self) -> object: ...


class NullEventSink:
    def on_time_advance(self, dt_seconds, before, after) -> None:
        return None

    def on_event(self, event, before, after) -> None:
        return None

    def finalize(self) -> None:
        return None


@dataclass
class StatsEventSink:
    event_counts: dict[str, int] = field(default_factory=dict)
    time_advanced_seconds: float = 0.0

    def on_time_advance(self, dt_seconds, before, after) -> None:
        self.time_advanced_seconds += dt_seconds

    def on_event(self, event, before, after) -> None:
        name = type(event).__name__
        self.event_counts[name] = self.event_counts.get(name, 0) + 1

    def finalize(self) -> dict[str, object]:
        return {
            "event_counts": dict(self.event_counts),
            "time_advanced_seconds": self.time_advanced_seconds,
        }


@dataclass(frozen=True)
class TraceEntry:
    kind: str
    timestamp_seconds: float
    payload: object
    before: StateSnapshot
    after: StateSnapshot


@dataclass
class FullTraceEventSink:
    entries: list[TraceEntry] = field(default_factory=list)

    def on_time_advance(self, dt_seconds, before, after) -> None:
        self.entries.append(
            TraceEntry("time_advance", after.fight_time_seconds, dt_seconds, before, after)
        )

    def on_event(self, event, before, after) -> None:
        self.entries.append(
            TraceEntry("event", event.timestamp_seconds, event, before, after)
        )

    def finalize(self) -> tuple[TraceEntry, ...]:
        return tuple(self.entries)
