"""DISTANCE primary candidates and their minimal typed observations."""

from dataclasses import dataclass

import numpy as np

from ..contracts import FightContext, Resolution
from ..events import ConsequenceEvent
from ..rng import RNGStream
from ..state import FightState, Phase, StateDelta
from ..modifiers import DynamicModifierProvider, DynamicModifiers
from ..stamina import StaminaModel
from ..calibration import DEFAULT_CALIBRATION, EventMCCalibration
from .formulas import phase_strike_landing_probability, strike_landing_probability, td_success_probability
from .profiles import MatchupProfiles, Side


@dataclass(frozen=True)
class ActionAttempt:
    side: Side
    action_family: str
    dynamic_modifiers: DynamicModifiers | None = None
    landed: bool | None = None
    target: str | None = None


@dataclass(frozen=True)
class ActionOutcome:
    side: Side
    action_family: str
    outcome: str


@dataclass(frozen=True)
class DistanceCandidate:
    side: Side
    action_family: str
    profiles: MatchupProfiles
    stamina_model: StaminaModel | None = None
    modifier_provider: DynamicModifierProvider | None = None
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    @property
    def candidate_id(self) -> str:
        return f"{self.side.value}_{self.action_family}"

    @property
    def rng_stream(self) -> RNGStream:
        if self.action_family == "strike":
            return RNGStream.STRIKE_RESOLUTION
        if self.action_family == "takedown":
            return RNGStream.TAKEDOWN
        return RNGStream.SCHEDULER

    def resolve(
        self, state: FightState, context: FightContext, rng: np.random.Generator
    ) -> Resolution:
        attacker = self.profiles.fighter(self.side)
        defender = self.profiles.fighter(self.side.opponent)
        timestamp = state.fight_time_seconds
        delta = StateDelta()
        if self.action_family == "strike":
            landed = rng.random() < strike_landing_probability(attacker, defender, self.calibration)
            outcome = "landed" if landed else "missed"
        elif self.action_family == "takedown":
            landed = rng.random() < td_success_probability(attacker, defender, self.calibration)
            outcome = "landed" if landed else "failed"
            if landed:
                delta = StateDelta(
                    phase=Phase.GROUND,
                    ground_controller=self.side.value,
                    set_ground_controller=True,
                    set_clinch_controller=True,
                )
        elif self.action_family == "clinch_entry":
            outcome = "entered"
            delta = StateDelta(
                phase=Phase.CLINCH,
                clinch_controller=self.side.value,
                set_clinch_controller=True,
                set_ground_controller=True,
            )
        else:
            raise ValueError(f"unsupported action family: {self.action_family}")
        modifiers = self.modifier_provider.modifiers(attacker, state, self.side) if self.modifier_provider else None
        cost_delta = self.stamina_model.action_delta(state, self.side, self.action_family) if self.stamina_model else StateDelta()
        delta = _merge_delta(delta, cost_delta)
        consequence = ConsequenceEvent(
            timestamp, "ActionOutcome", ActionOutcome(self.side, self.action_family, outcome)
        )
        return Resolution(
            delta=delta,
            consequence_events=(consequence,),
            payload=ActionAttempt(self.side, self.action_family, modifiers, landed if self.action_family == "strike" else None),
        )


@dataclass(frozen=True)
class PhaseCandidate:
    """A CLINCH/GROUND action whose availability is supplied by phase providers."""

    side: Side
    action_family: str
    profiles: MatchupProfiles
    stamina_model: StaminaModel | None = None
    modifier_provider: DynamicModifierProvider | None = None
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    @property
    def candidate_id(self) -> str:
        return f"{self.side.value}_{self.action_family}"

    @property
    def rng_stream(self) -> RNGStream:
        if "strike" in self.action_family:
            return RNGStream.STRIKE_RESOLUTION
        if "takedown" in self.action_family:
            return RNGStream.TAKEDOWN
        if "submission" in self.action_family:
            return RNGStream.SUBMISSION
        return RNGStream.SCHEDULER

    def resolve(self, state: FightState, context: FightContext, rng: np.random.Generator) -> Resolution:
        attacker = self.profiles.fighter(self.side)
        defender = self.profiles.fighter(self.side.opponent)
        delta = StateDelta()
        outcome = "occurred"
        family = self.action_family
        if family in {"clinch_strike", "ground_strike"}:
            phase = "clinch" if family == "clinch_strike" else "ground"
            landed = rng.random() < phase_strike_landing_probability(attacker, defender, phase, self.calibration)
            outcome = "landed" if landed else "missed"
        elif family == "clinch_takedown":
            outcome = "landed" if rng.random() < td_success_probability(attacker, defender, self.calibration) else "failed"
            if outcome == "landed":
                delta = StateDelta(phase=Phase.GROUND, ground_controller=self.side.value, set_ground_controller=True, set_clinch_controller=True)
        elif family == "clinch_separation":
            outcome = "separated"
            delta = StateDelta(phase=Phase.DISTANCE, set_ground_controller=True, set_clinch_controller=True)
        elif family == "ground_escape":
            outcome = "escaped"
            delta = StateDelta(phase=Phase.DISTANCE, set_ground_controller=True, set_clinch_controller=True)
        elif family == "ground_reversal":
            outcome = "reversed"
            delta = StateDelta(phase=Phase.GROUND, ground_controller=self.side.value, set_ground_controller=True)
        elif family == "submission_attempt":
            outcome = "attempted"  # Observation only: never terminal in Phase 3.
        else:
            raise ValueError(f"unsupported action family: {family}")
        modifiers = self.modifier_provider.modifiers(attacker, state, self.side) if self.modifier_provider else None
        cost_delta = self.stamina_model.action_delta(state, self.side, family) if self.stamina_model else StateDelta()
        delta = _merge_delta(delta, cost_delta)
        consequence = ConsequenceEvent(state.fight_time_seconds, "ActionOutcome", ActionOutcome(self.side, family, outcome))
        return Resolution(delta=delta, consequence_events=(consequence,), payload=ActionAttempt(self.side, family, modifiers, landed if family in {"clinch_strike", "ground_strike"} else None))


def _merge_delta(primary: StateDelta, physiology: StateDelta) -> StateDelta:
    """Combine transition and stamina requests before engine-owned application."""
    return StateDelta(
        phase=primary.phase,
        ground_controller=primary.ground_controller,
        set_ground_controller=primary.set_ground_controller,
        clinch_controller=primary.clinch_controller,
        set_clinch_controller=primary.set_clinch_controller,
        finished=primary.finished,
        finish_reason=primary.finish_reason,
        action_availability=primary.action_availability,
        red_stamina=physiology.red_stamina,
        blue_stamina=physiology.blue_stamina,
    )
