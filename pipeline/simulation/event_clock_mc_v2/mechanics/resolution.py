"""Typed, immutable outputs from causal action mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.simulation.event_clock_mc_v2.causal.events import ActionEvent
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side


class ActionOutcome(str, Enum):
    LANDED = "landed"
    MISSED = "missed"
    SUCCESS = "success"
    FAILURE = "failure"
    STUFFED = "stuffed"
    SEPARATED = "separated"
    CONTROLLED = "controlled"
    ESCAPED = "escaped"
    REVERSED = "reversed"
    MAINTAINED = "maintained"
    TACTICAL = "tactical"


class TransitionKind(str, Enum):
    ENTER_CLINCH = "enter_clinch"
    DIRECT_TAKEDOWN = "direct_takedown"
    CLINCH_TAKEDOWN = "clinch_takedown"
    BREAK_CLINCH = "break_clinch"
    ESCAPE_GROUND = "escape_ground"
    REVERSE_GROUND = "reverse_ground"
    DISENGAGE_GROUND = "disengage_ground"


_TRANSITION_TOPOLOGY = {
    TransitionKind.ENTER_CLINCH: (Phase.STANDING, Phase.CLINCH),
    TransitionKind.DIRECT_TAKEDOWN: (Phase.STANDING, Phase.GROUND),
    TransitionKind.CLINCH_TAKEDOWN: (Phase.CLINCH, Phase.GROUND),
    TransitionKind.BREAK_CLINCH: (Phase.CLINCH, Phase.STANDING),
    TransitionKind.ESCAPE_GROUND: (Phase.GROUND, Phase.STANDING),
    TransitionKind.REVERSE_GROUND: (Phase.GROUND, Phase.GROUND),
    TransitionKind.DISENGAGE_GROUND: (Phase.GROUND, Phase.STANDING),
}


@dataclass(frozen=True)
class TransitionRequest:
    kind: TransitionKind
    source_phase: Phase
    target_phase: Phase
    controller: Side | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TransitionKind):
            raise ValueError("kind must be a TransitionKind")
        if not isinstance(self.source_phase, Phase) or not isinstance(
            self.target_phase, Phase
        ):
            raise ValueError("transition phases must be Phase values")
        expected_source, expected_target = _TRANSITION_TOPOLOGY[self.kind]
        if (self.source_phase, self.target_phase) != (expected_source, expected_target):
            raise ValueError(
                f"{self.kind.value} requires "
                f"{expected_source.value} -> {expected_target.value}"
            )
        if self.controller is not None and not isinstance(self.controller, Side):
            raise ValueError("transition controller must be a Side value or None")
        if self.target_phase is Phase.STANDING and self.controller is not None:
            raise ValueError("standing transition cannot request a controller")
        if (
            self.target_phase in {Phase.CLINCH, Phase.GROUND}
            and self.controller is None
        ):
            raise ValueError("clinch and ground transitions require a controller")


@dataclass(frozen=True)
class StrikeConsequence:
    """Complete immutable strike consequence for authoritative engine application."""

    landed: bool
    impact: float = 0.0
    trauma_increment: float = 0.0
    knockdown_probability: float = 0.0
    knockdown: bool = False
    acute_increment: float = 0.0
    termination: FightTerminationRequest | None = None
    ko_probability: float = 0.0
    prior_defender_kds: int = 0
    ko_kd_architecture: str = "legacy_stage10"


class FinishMethod(str, Enum):
    """Finish methods that Stage 3 mechanics can currently request."""

    SUBMISSION = "submission"
    KO_TKO = "ko_tko"


@dataclass(frozen=True)
class FightTerminationRequest:
    """Submission termination intent for the future engine to apply."""

    winner: Side
    finish_method: FinishMethod = FinishMethod.SUBMISSION

    def __post_init__(self) -> None:
        if not isinstance(self.winner, Side):
            raise ValueError("winner must be a Side value")
        if not isinstance(self.finish_method, FinishMethod):
            raise ValueError("finish_method must be a FinishMethod value")


MechanicsConsequence = StrikeConsequence | FightTerminationRequest


@dataclass(frozen=True)
class ActionResolution:
    event: ActionEvent
    outcome: ActionOutcome
    transition: TransitionRequest | None = None
    consequence: MechanicsConsequence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event, ActionEvent):
            raise ValueError("event must be an ActionEvent")
        if not isinstance(self.outcome, ActionOutcome):
            raise ValueError("outcome must be an ActionOutcome")
        if self.transition is not None and not isinstance(
            self.transition, TransitionRequest
        ):
            raise ValueError("transition must be a TransitionRequest or None")
        if (
            self.transition is not None
            and self.transition.source_phase is not self.event.source_phase
        ):
            raise ValueError("transition source_phase must match event source_phase")
        if self.consequence is not None and not isinstance(
            self.consequence, (StrikeConsequence, FightTerminationRequest)
        ):
            raise ValueError(
                "consequence must be a typed mechanics consequence or None"
            )
