"""DISTANCE primary candidates and their minimal typed observations."""

from dataclasses import dataclass

import numpy as np

from ..contracts import FightContext, Resolution
from ..events import ConsequenceEvent
from ..rng import RNGStream
from ..state import FightState, Phase, StateDelta
from .formulas import phase_strike_landing_probability, strike_landing_probability, td_success_probability
from .profiles import MatchupProfiles, Side


@dataclass(frozen=True)
class ActionAttempt:
    side: Side
    action_family: str


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
            landed = rng.random() < strike_landing_probability(attacker, defender)
            outcome = "landed" if landed else "missed"
        elif self.action_family == "takedown":
            landed = rng.random() < td_success_probability(attacker, defender)
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
        consequence = ConsequenceEvent(
            timestamp, "ActionOutcome", ActionOutcome(self.side, self.action_family, outcome)
        )
        return Resolution(
            delta=delta,
            consequence_events=(consequence,),
            payload=ActionAttempt(self.side, self.action_family),
        )


@dataclass(frozen=True)
class PhaseCandidate:
    """A CLINCH/GROUND action whose availability is supplied by phase providers."""

    side: Side
    action_family: str
    profiles: MatchupProfiles

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
            outcome = "landed" if rng.random() < phase_strike_landing_probability(attacker, defender, phase) else "missed"
        elif family == "clinch_takedown":
            outcome = "landed" if rng.random() < td_success_probability(attacker, defender) else "failed"
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
        consequence = ConsequenceEvent(state.fight_time_seconds, "ActionOutcome", ActionOutcome(self.side, family, outcome))
        return Resolution(delta=delta, consequence_events=(consequence,), payload=ActionAttempt(self.side, family))
