"""V2-native migration of the validated Event2 empirical KO/KD hazards.

KO is sampled first from pre-strike state. KD is sampled only after KO survival,
so one strike cannot be both a knockdown and a KO/TKO. The only dynamic damage
memory used by either hazard is the defender's prior knockdowns suffered.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np

from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Side
from pipeline.simulation.event_clock_mc_v2.mechanics.config import FighterMechanics


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-float(np.clip(value, -30.0, 30.0))))


def _logit(probability: float) -> float:
    p = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    return log(p / (1.0 - p))


@dataclass(frozen=True)
class EmpiricalKOKDCoefficients:
    kd_base_probability: float = 0.00640
    kd_power_beta: float = 0.020741
    kd_attacker_age_beta: float = -0.030660
    kd_kdres_beta: float = -0.014421
    kd_defender_age_beta: float = 0.029369
    kd_prior_kd_beta: float = 0.502627
    kd_elapsed_minute_beta: float = -0.050585
    kd_stamina_beta: float = 0.0

    ko_base_probability: float = 0.00250
    ko_power_beta: float = 0.0200
    ko_attacker_age_beta: float = -0.0300
    ko_defender_age_beta: float = 0.0300
    ko_prior_kd_beta: float = 1.00
    ko_elapsed_minute_beta: float = 0.0
    ko_stamina_beta: float = 0.0


@dataclass(frozen=True)
class EmpiricalKOKDResult:
    ko_probability: float
    ko_tko: bool
    kd_probability: float
    knockdown: bool
    prior_defender_kds: int


def ko_probability(
    attacker: FighterMechanics,
    defender: FighterMechanics,
    *,
    prior_defender_kds: int,
    elapsed_seconds: float,
    attacker_stamina: float,
    coefficients: EmpiricalKOKDCoefficients = EmpiricalKOKDCoefficients(),
) -> float:
    c = coefficients
    z = _logit(c.ko_base_probability)
    z += c.ko_power_beta * (attacker.striking_power - 50.0)
    z += c.ko_attacker_age_beta * (attacker.age_years - 30.0)
    z += c.ko_defender_age_beta * (defender.age_years - 30.0)
    z += c.ko_prior_kd_beta * float(prior_defender_kds)
    z += c.ko_elapsed_minute_beta * (float(elapsed_seconds) / 60.0)
    z += c.ko_stamina_beta * log(float(np.clip(attacker_stamina, 1e-6, 1.0)))
    return _sigmoid(z)


def kd_probability(
    attacker: FighterMechanics,
    defender: FighterMechanics,
    *,
    prior_defender_kds: int,
    elapsed_seconds: float,
    attacker_stamina: float,
    coefficients: EmpiricalKOKDCoefficients = EmpiricalKOKDCoefficients(),
) -> float:
    c = coefficients
    z = _logit(c.kd_base_probability)
    z += c.kd_power_beta * (attacker.striking_power - 50.0)
    z += c.kd_attacker_age_beta * (attacker.age_years - 30.0)
    z += c.kd_kdres_beta * (defender.knockdown_resistance - 50.0)
    z += c.kd_defender_age_beta * (defender.age_years - 30.0)
    z += c.kd_prior_kd_beta * float(prior_defender_kds)
    z += c.kd_elapsed_minute_beta * (float(elapsed_seconds) / 60.0)
    z += c.kd_stamina_beta * log(float(np.clip(attacker_stamina, 1e-6, 1.0)))
    return _sigmoid(z)


def resolve_landed_strike(
    *,
    state: FightState,
    attacker_side: Side,
    attacker: FighterMechanics,
    defender: FighterMechanics,
    rng: np.random.Generator,
) -> EmpiricalKOKDResult:
    target = state.physiology.fighter(attacker_side.opponent)
    prior = target.knockdowns_suffered
    stamina = state.physiology.fighter(attacker_side).stamina
    p_ko = ko_probability(
        attacker,
        defender,
        prior_defender_kds=prior,
        elapsed_seconds=state.fight_time_seconds,
        attacker_stamina=stamina,
    )
    if bool(rng.random() < p_ko):
        return EmpiricalKOKDResult(p_ko, True, 0.0, False, prior)
    p_kd = kd_probability(
        attacker,
        defender,
        prior_defender_kds=prior,
        elapsed_seconds=state.fight_time_seconds,
        attacker_stamina=stamina,
    )
    return EmpiricalKOKDResult(p_ko, False, p_kd, bool(rng.random() < p_kd), prior)
