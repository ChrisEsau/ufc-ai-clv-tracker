"""Observer-only Phase 3 action, transition, residence, and control ledger."""

from dataclasses import dataclass, field

from .components.actions import ActionAttempt, ActionOutcome
from .events import ConsequenceEvent, PrimaryEvent
from .physiology import PhysiologyOutcome
from .finishes import FinishOutcome
from .submission_finishes import SubmissionFinishOutcome


@dataclass
class FlowStatsSink:
    attempts: dict[str, dict[str, int]] = field(default_factory=lambda: {"red": {}, "blue": {}})
    outcomes: dict[str, dict[str, int]] = field(default_factory=lambda: {"red": {}, "blue": {}})
    phase_seconds: dict[str, float] = field(default_factory=lambda: {"standing": 0.0, "ground": 0.0})
    clinch_control_seconds: dict[str, float] = field(default_factory=lambda: {"red": 0.0, "blue": 0.0})
    ground_control_seconds: dict[str, float] = field(default_factory=lambda: {"red": 0.0, "blue": 0.0})
    transitions: list[dict[str, object]] = field(default_factory=list)
    stamina_round_entries: list[dict[str, float]] = field(default_factory=list)
    physiology: list[PhysiologyOutcome] = field(default_factory=list)
    finishes: list[FinishOutcome] = field(default_factory=list)
    submission_checks: list[SubmissionFinishOutcome] = field(default_factory=list)

    def on_time_advance(self, dt_seconds, before, after) -> None:
        self.phase_seconds[before.phase] += dt_seconds
        if before.phase == "clinch" and before.clinch_controller:
            self.clinch_control_seconds[before.clinch_controller] += dt_seconds
        if before.phase == "ground" and before.ground_controller:
            self.ground_control_seconds[before.ground_controller] += dt_seconds

    def on_event(self, event, before, after) -> None:
        if type(event).__name__ == "RoundStarted":
            self.stamina_round_entries.append({"round": event.round_number, "red": after.red_stamina, "blue": after.blue_stamina})
        if isinstance(event, PrimaryEvent) and isinstance(event.payload, ActionAttempt):
            side, family = event.payload.side.value, event.payload.action_family
            self.attempts[side][family] = self.attempts[side].get(family, 0) + 1
            if before.phase != after.phase or before.ground_controller != after.ground_controller:
                self.transitions.append({"timestamp_seconds": event.timestamp_seconds, "from_phase": before.phase, "to_phase": after.phase, "from_controller": before.ground_controller or before.clinch_controller, "to_controller": after.ground_controller or after.clinch_controller})
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, ActionOutcome):
            side, family, outcome = event.payload.side.value, event.payload.action_family, event.payload.outcome
            key = f"{family}_{outcome}"
            self.outcomes[side][key] = self.outcomes[side].get(key, 0) + 1
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, PhysiologyOutcome):
            self.physiology.append(event.payload)
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, FinishOutcome):
            self.finishes.append(event.payload)
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, SubmissionFinishOutcome):
            self.submission_checks.append(event.payload)

    def finalize(self) -> dict[str, object]:
        return {
            "attempts": {side: dict(values) for side, values in self.attempts.items()},
            "outcomes": {side: dict(values) for side, values in self.outcomes.items()},
            "phase_seconds": dict(self.phase_seconds),
            "clinch_control_seconds": dict(self.clinch_control_seconds),
            "ground_control_seconds": dict(self.ground_control_seconds),
            "transitions": tuple(self.transitions),
            "stamina_round_entries": tuple(self.stamina_round_entries),
            "physiology": tuple(self.physiology),
            "finishes": tuple(self.finishes),
            "submission_checks": tuple(self.submission_checks),
        }
