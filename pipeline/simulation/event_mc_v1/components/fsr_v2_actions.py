"""STANDING/GROUND action resolution for the FSR V2 clock path."""

from dataclasses import dataclass
import numpy as np

from ..contracts import FightContext, Resolution
from ..events import ConsequenceEvent
from ..rng import RNGStream
from ..state import FightState, Phase, StateDelta
from ..stamina import StaminaModel
from ..modifiers import DynamicModifierProvider
from ..calibration import DEFAULT_CALIBRATION, EventMCCalibration
from .actions import ActionAttempt, ActionOutcome, _merge_delta
from .fsr_v2 import FSRV2Matchup
from .fsr_v2_mechanics import (
    TAKEDOWN_ATTACKER_AGE_CENTER_YEARS,
    TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR,
    matchup_probability,
)
from .profiles import MatchupProfiles, Side


@dataclass(frozen=True)
class FSRV2Candidate:
    side: Side
    action_family: str
    matchup: FSRV2Matchup
    profiles: MatchupProfiles
    stamina_model: StaminaModel | None = None
    modifier_provider: DynamicModifierProvider | None = None
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    @property
    def candidate_id(self):
        return f"{self.side.value}_{self.action_family}"

    @property
    def rng_stream(self):
        if "strike" in self.action_family: return RNGStream.STRIKE_RESOLUTION
        if self.action_family == "takedown": return RNGStream.TAKEDOWN
        if self.action_family == "submission_attempt": return RNGStream.SUBMISSION
        return RNGStream.SCHEDULER

    def resolve(self, state: FightState, context: FightContext, rng: np.random.Generator) -> Resolution:
        attacker, defender = self.matchup.fighter(self.side), self.matchup.fighter(self.side.opponent)
        c = self.calibration.section("fsr_v2_calibration")
        family, landed, target = self.action_family, None, None
        delta, outcome = StateDelta(), "occurred"
        if family == "standing_strike":
            landed = bool(rng.random() < matchup_probability(
                attacker.standing_accuracy_baseline,
                attacker.standing_striking_offense,
                defender.standing_striking_defense,
                c["standing_accuracy_logit_offset"],
            ))
            target = rng.choice(("head", "body", "leg"), p=attacker.standing_target_probabilities()).item()
            outcome = "landed" if landed else "missed"
        elif family == "takedown":
            age_logit_offset = (
                TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR
                * (
                    attacker.age_years
                    - TAKEDOWN_ATTACKER_AGE_CENTER_YEARS
                )
            )
            landed = bool(rng.random() < matchup_probability(
                attacker.takedown_completion_baseline,
                attacker.takedown_offense,
                defender.takedown_defense,
                age_logit_offset,
            ))
            outcome = "landed" if landed else "failed"
            if landed:
                delta = StateDelta(phase=Phase.GROUND, ground_controller=self.side.value,
                                   set_ground_controller=True, set_clinch_controller=True)
        elif family == "ground_strike":
            landed = bool(rng.random() < matchup_probability(
                attacker.ground_accuracy_baseline,
                attacker.ground_striking_offense,
                defender.ground_striking_defense,
                c["ground_accuracy_logit_offset"],
            ))
            # Ground damage is intentionally restricted to head/body.
            target = rng.choice(("head", "body")).item()
            outcome = "landed" if landed else "missed"
        elif family == "ground_escape":
            outcome = "escaped"
            delta = StateDelta(phase=Phase.STANDING, set_ground_controller=True, set_clinch_controller=True)
        elif family == "submission_attempt":
            outcome = "attempted"
        else:
            raise ValueError(f"unsupported FSR V2 action family: {family}")
        profile = self.profiles.fighter(self.side)
        modifiers = self.modifier_provider.modifiers(profile, state, self.side) if self.modifier_provider else None
        cost_family = "strike" if family == "standing_strike" else family
        cost = self.stamina_model.action_delta(state, self.side, cost_family) if self.stamina_model else StateDelta()
        delta = _merge_delta(delta, cost)
        attempt = ActionAttempt(self.side, family, modifiers, landed, target)
        event = ConsequenceEvent(state.fight_time_seconds, "ActionOutcome", ActionOutcome(self.side, family, outcome))
        return Resolution(delta=delta, consequence_events=(event,), payload=attempt)
