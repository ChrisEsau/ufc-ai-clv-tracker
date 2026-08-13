"""Exact ports of the legacy V0 DISTANCE consumers.

Source: ``scripts.experimental.fsr_static_mc_v0``. Constants and equations are
copied without tuning; only their time representation changes.
"""

from dataclasses import dataclass
from math import exp, log

import numpy as np

from ..scheduler import probability_to_rate
from .profiles import FighterProfile

LEGACY_INTERVAL_SECONDS = 10.0
CALIBRATION_INTERVAL_SECONDS = 30.0
DISTANCE_CLINCH_BASE_30S = 0.04
DISTANCE_TD_ATTEMPT_BASE_30S = 0.10
DISTANCE_STRIKE_ATTEMPTS_PER_30S_BASE = 5.0
DISTANCE_STRIKE_ACCURACY_BASE = 0.40
RATING_SCALE = 12.0
MODIFIER_SCALE = 6.0
TD_SUCCESS_LOGIT_OFFSET = -0.40


def rescale_interval_probability(
    probability: float, from_seconds: float, to_seconds: float
) -> float:
    return 1.0 - (1.0 - probability) ** (to_seconds / from_seconds)


DISTANCE_CLINCH_BASE_10S = rescale_interval_probability(0.04, 30.0, 10.0)
DISTANCE_TD_ATTEMPT_BASE_10S = rescale_interval_probability(0.10, 30.0, 10.0)


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -12.0, 12.0))
    return 1.0 / (1.0 + exp(-value))


def _logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return log(probability / (1.0 - probability))


def _modifier(delta: float, scale: float = MODIFIER_SCALE) -> float:
    return exp(float(np.clip(delta, -8.0, 8.0)) / scale)


def style_preferences(profile: FighterProfile) -> tuple[float, float, float]:
    distance = profile.distance_striking_pressure
    clinch = profile.clinch_striking_pressure
    wrestling = profile.wrestling_entry
    control = profile.control_imposition
    return (
        distance - 0.5 * clinch - 0.5 * wrestling,
        clinch - 0.5 * distance - 0.5 * wrestling,
        0.75 * wrestling
        + 0.25 * control
        - 0.5 * distance
        - 0.5 * clinch,
    )


def strike_attempt_rate_per_second(profile: FighterProfile) -> float:
    """Poisson intensity preserving V0's expected DISTANCE attempt count."""

    expected_per_30s = DISTANCE_STRIKE_ATTEMPTS_PER_30S_BASE * _modifier(
        profile.distance_striking_pressure - 50.0, scale=12.0
    )
    return expected_per_30s / CALIBRATION_INTERVAL_SECONDS


def strike_landing_probability(
    attacker: FighterProfile, defender: FighterProfile
) -> float:
    return _sigmoid(
        _logit(DISTANCE_STRIKE_ACCURACY_BASE)
        + (
            attacker.distance_striking_precision
            - defender.distance_striking_defense
        )
        / RATING_SCALE
    )


def legacy_td_attempt_interval_probability(profile: FighterProfile) -> float:
    """Phase 2A blended initiation consumer retained only for A/B audits."""

    wrestling_preference = style_preferences(profile)[2]
    raw_probability = DISTANCE_TD_ATTEMPT_BASE_10S * exp(
        wrestling_preference / MODIFIER_SCALE
    )
    return float(np.clip(raw_probability, 0.0, 1.0 - 1e-12))


def td_attempt_interval_probability(profile: FighterProfile) -> float:
    """Phase 2B intrinsic initiation driven only by ``wrestling_entry``."""

    entry_delta = profile.wrestling_entry - 50.0
    raw_probability = DISTANCE_TD_ATTEMPT_BASE_10S * _modifier(entry_delta)
    return float(np.clip(raw_probability, 0.0, 1.0 - 1e-12))


def td_success_probability(
    attacker: FighterProfile, defender: FighterProfile
) -> float:
    edge = (attacker.wrestling_conversion - defender.td_defense) / RATING_SCALE
    return _sigmoid(edge + TD_SUCCESS_LOGIT_OFFSET)


def clinch_entry_interval_probability(profile: FighterProfile) -> float:
    distance_preference, clinch_preference, _ = style_preferences(profile)
    probability = (
        DISTANCE_CLINCH_BASE_10S
        * _modifier(clinch_preference)
        * np.sqrt(_modifier(-distance_preference))
    )
    return float(np.clip(probability, 0.0, 0.60))


def interval_hazard_per_second(probability: float) -> float:
    return probability_to_rate(probability, LEGACY_INTERVAL_SECONDS)


@dataclass(frozen=True)
class ActionRateAudit:
    side: str
    action_family: str
    legacy_interval_probability: float | None
    interval_seconds: float
    rate_per_second: float
    major_inputs: dict[str, float]
