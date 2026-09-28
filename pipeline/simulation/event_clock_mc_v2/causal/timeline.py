"""Chronological, contiguous phase exposure for causal Event Clock V2."""

from __future__ import annotations

from dataclasses import dataclass

from .state import FightState, Phase, Side


@dataclass(frozen=True)
class PhaseSegment:
    start_time: float
    end_time: float
    phase: Phase
    controller: Side | None
    entry_reason: str
    exit_reason: str

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError("phase segment duration cannot be negative")
        _validate_controller(self.phase, self.controller)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class ActivePhase:
    start_time: float
    phase: Phase
    controller: Side | None
    entry_reason: str

    def __post_init__(self) -> None:
        if self.start_time < 0.0:
            raise ValueError("active phase start time cannot be negative")
        _validate_controller(self.phase, self.controller)


class PhaseTimeline:
    """Own completed exposure plus exactly one active phase segment."""

    def __init__(self, active: ActivePhase) -> None:
        self._segments: list[PhaseSegment] = []
        self._active: ActivePhase | None = active

    @classmethod
    def from_state(cls, state: FightState, *, entry_reason: str = "round_start") -> PhaseTimeline:
        return cls(
            ActivePhase(
                start_time=state.phase_started_at,
                phase=state.phase,
                controller=_controller_for(state),
                entry_reason=entry_reason,
            )
        )

    @property
    def segments(self) -> tuple[PhaseSegment, ...]:
        return tuple(self._segments)

    @property
    def active(self) -> ActivePhase | None:
        return self._active

    def transition(
        self,
        *,
        timestamp: float,
        phase: Phase,
        controller: Side | None,
        exit_reason: str,
        entry_reason: str,
    ) -> None:
        """Atomically close the active segment and open its explicit successor."""
        self._close(timestamp, exit_reason)
        self._active = ActivePhase(timestamp, phase, controller, entry_reason)

    def _close(self, timestamp: float, exit_reason: str) -> None:
        active = self._active
        if active is None:
            raise ValueError("timeline has no active phase to close")
        if timestamp < active.start_time:
            raise ValueError("time cannot move backward")
        if self._segments and self._segments[-1].end_time != active.start_time:
            raise ValueError("phase timeline must be contiguous")
        self._segments.append(
            PhaseSegment(
                start_time=active.start_time,
                end_time=timestamp,
                phase=active.phase,
                controller=active.controller,
                entry_reason=active.entry_reason,
                exit_reason=exit_reason,
            )
        )
        self._active = None

    def validate(self) -> None:
        """Fail loudly if completed segments overlap, gap, or run backward."""
        for index, segment in enumerate(self._segments):
            if segment.end_time < segment.start_time:
                raise ValueError("phase segment duration cannot be negative")
            if index and self._segments[index - 1].end_time != segment.start_time:
                raise ValueError("phase timeline must be chronological and contiguous")
        if self._active is not None and self._segments:
            if self._segments[-1].end_time != self._active.start_time:
                raise ValueError("active phase must be contiguous with completed timeline")

    def exposure_seconds(self) -> dict[Phase, float]:
        exposure = {phase: 0.0 for phase in Phase}
        for segment in self._segments:
            exposure[segment.phase] += segment.duration
        return exposure

    def segments_through(self, timestamp: float) -> tuple[PhaseSegment, ...]:
        """Return a non-destructive exposure snapshot through ``timestamp``."""
        active = self._active
        if active is None:
            raise ValueError("timeline has no active phase to snapshot")
        if timestamp < active.start_time:
            raise ValueError("snapshot time cannot precede the active phase")
        return self.segments + (
            PhaseSegment(
                start_time=active.start_time,
                end_time=timestamp,
                phase=active.phase,
                controller=active.controller,
                entry_reason=active.entry_reason,
                exit_reason="reporting_horizon",
            ),
        )

    def exposure_seconds_through(self, timestamp: float) -> dict[Phase, float]:
        """Calculate exposure through a horizon without changing the timeline."""
        exposure = {phase: 0.0 for phase in Phase}
        for segment in self.segments_through(timestamp):
            exposure[segment.phase] += segment.duration
        return exposure


def _controller_for(state: FightState) -> Side | None:
    if state.phase is Phase.CLINCH:
        return state.clinch_controller
    if state.phase is Phase.GROUND:
        return state.ground_controller
    return None


def _validate_controller(phase: Phase, controller: Side | None) -> None:
    """Apply the same authoritative controller contract to every timeline record."""
    if not isinstance(phase, Phase):
        raise ValueError("timeline phase must be a Phase value")
    if controller is not None and not isinstance(controller, Side):
        raise ValueError("timeline controller must be a Side value or None")
    if phase is Phase.STANDING and controller is not None:
        raise ValueError("standing phase cannot carry a controller")
    if phase is Phase.CLINCH and controller is None:
        raise ValueError("clinch phase requires a controller")
    if phase is Phase.GROUND and controller is None:
        raise ValueError("ground phase requires a controller")
