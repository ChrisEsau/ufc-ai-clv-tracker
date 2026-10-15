"""Phase 4B2 probabilistic KO/TKO finish mechanics over existing impact."""

from dataclasses import dataclass
from math import exp, log

import numpy as np

from .calibration import DEFAULT_CALIBRATION, EventMCCalibration
from .components.profiles import MatchupProfiles, Side
from .events import ConsequenceEvent
from .physiology import PhysiologyOutcome
from .state import StateDelta


@dataclass(frozen=True)
class FinishOutcome:
    attacker: Side
    defender: Side
    impact_ratio: float
    current_finish_resistance: float
    finish_probability: float
    knockdown: bool
    finished: bool
    method: str = "KO_TKO"


@dataclass(frozen=True)
class KOTKOFinishModel:
    profiles: MatchupProfiles
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    def probability(self, state, physiology: PhysiologyOutcome) -> tuple[float, float]:
        c = self.calibration.section("finish")
        defender = self.profiles.fighter(physiology.defender)
        baseline_rating = c["baseline_durability_weight"] * defender.damage_durability + c["baseline_knockdown_resistance_weight"] * defender.knockdown_resistance
        trauma = getattr(state, f"{physiology.defender.value}_cumulative_trauma")
        acute = getattr(state, f"{physiology.defender.value}_acute_vulnerability")
        resistance = max(1e-9, exp((baseline_rating - 50) / c["resistance_rating_scale"]) * exp(-trauma / c["trauma_erosion_scale"]) * exp(-acute / c["acute_erosion_scale"]))
        ratio = physiology.impact / resistance
        logit = c["slope"] * (log(max(ratio, 1e-12)) - log(c["midpoint_impact_ratio"]))
        if physiology.knockdown:
            logit += c["knockdown_logit_bonus"]
        probability = 1.0 / (1.0 + exp(-float(np.clip(logit, -20, 20))))
        return probability, resistance

    def resolve(self, state, physiology: PhysiologyOutcome, timestamp: float, rng):
        probability, resistance = self.probability(state, physiology)
        finished = bool(rng.random() < probability)
        ratio = physiology.impact / resistance
        outcome = FinishOutcome(physiology.attacker, physiology.defender, ratio, resistance, probability, physiology.knockdown, finished)
        delta = StateDelta(finished=True, finish_reason="KO_TKO", winner=physiology.attacker.value, finish_method="KO_TKO") if finished else StateDelta()
        return delta, ConsequenceEvent(timestamp, "FinishOutcome", outcome)
