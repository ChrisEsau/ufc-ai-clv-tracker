"""Compact observer for Phase 2A action attempts and outcomes."""

from dataclasses import dataclass, field

from .components.actions import ActionAttempt, ActionOutcome
from .events import ConsequenceEvent, PrimaryEvent


@dataclass
class DistanceStatsSink:
    attempts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "red": {"strike": 0, "takedown": 0, "clinch_entry": 0},
            "blue": {"strike": 0, "takedown": 0, "clinch_entry": 0},
        }
    )
    outcomes: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"red": {}, "blue": {}}
    )
    transition_timestamps_seconds: list[float] = field(default_factory=list)

    def on_time_advance(self, dt_seconds, before, after) -> None:
        return None

    def on_event(self, event, before, after) -> None:
        if isinstance(event, PrimaryEvent) and isinstance(event.payload, ActionAttempt):
            attempt = event.payload
            self.attempts[attempt.side.value][attempt.action_family] += 1
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, ActionOutcome):
            outcome = event.payload
            key = f"{outcome.action_family}_{outcome.outcome}"
            side_outcomes = self.outcomes[outcome.side.value]
            side_outcomes[key] = side_outcomes.get(key, 0) + 1
            if outcome.action_family in {"takedown", "clinch_entry"} and outcome.outcome in {"landed", "entered"}:
                self.transition_timestamps_seconds.append(event.timestamp_seconds)

    def finalize(self) -> dict[str, object]:
        return {
            "attempts": {side: dict(counts) for side, counts in self.attempts.items()},
            "outcomes": {side: dict(counts) for side, counts in self.outcomes.items()},
            "transition_timestamps_seconds": tuple(self.transition_timestamps_seconds),
        }
