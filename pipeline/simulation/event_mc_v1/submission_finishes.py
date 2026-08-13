"""Phase 4C probabilistic submission conversion over existing attempts."""

from dataclasses import dataclass
from math import exp

import numpy as np

from .calibration import DEFAULT_CALIBRATION, EventMCCalibration
from .components.actions import ActionAttempt
from .components.profiles import MatchupProfiles, Side
from .events import ConsequenceEvent
from .state import StateDelta


@dataclass(frozen=True)
class SubmissionFinishOutcome:
    attacker: Side
    defender: Side
    threat: float
    resistance: float
    position: str
    stamina_context_term: float
    finish_probability: float
    finished: bool
    method: str = "SUB"


@dataclass(frozen=True)
class SubmissionFinishModel:
    profiles: MatchupProfiles
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    def probability(self, state, attacker_side: Side) -> tuple[float, float, float, str, float]:
        c = self.calibration.section("submission_finish")
        attacker = self.profiles.fighter(attacker_side)
        defender = self.profiles.fighter(attacker_side.opponent)
        threat = c["threat_conversion_weight"] * attacker.submission_conversion + c["threat_pressure_weight"] * attacker.submission_pressure
        resistance = c["resistance_submission_weight"] * defender.submission_resistance + c["resistance_control_weight"] * defender.control_resistance
        position = "top" if state.ground_controller == attacker_side.value else "bottom"
        position_term = c[f"{position}_position_bonus"]
        attacker_stamina = getattr(state, f"{attacker_side.value}_stamina")
        defender_stamina = getattr(state, f"{attacker_side.opponent.value}_stamina")
        stamina_term = c["stamina_edge_weight"] * (attacker_stamina - defender_stamina)
        logit = c["intercept"] + (threat - resistance) / c["rating_scale"] + position_term + stamina_term
        probability = 1.0 / (1.0 + exp(-float(np.clip(logit, -c["logit_clip"], c["logit_clip"]))))
        return probability, threat, resistance, position, stamina_term + position_term

    def resolve(self, state, attempt, timestamp: float, rng, *, pre_action_state=None):
        if not isinstance(attempt, ActionAttempt) or attempt.action_family != "submission_attempt":
            return StateDelta(), None
        evaluation_state = pre_action_state if pre_action_state is not None else state
        probability, threat, resistance, position, context_term = self.probability(evaluation_state, attempt.side)
        finished = bool(rng.random() < probability)
        outcome = SubmissionFinishOutcome(attempt.side, attempt.side.opponent, threat, resistance, position, context_term, probability, finished)
        delta = StateDelta(finished=True, finish_reason="SUB", winner=attempt.side.value, finish_method="SUB") if finished else StateDelta()
        return delta, ConsequenceEvent(timestamp, "SubmissionFinishOutcome", outcome)
