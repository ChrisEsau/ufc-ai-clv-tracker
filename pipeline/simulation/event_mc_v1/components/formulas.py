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
CLINCH_SEPARATE_BASE_30S = 0.25
CLINCH_TD_ATTEMPT_BASE_30S = 0.24
GROUND_EXIT_BASE_30S = 0.20
CLINCH_STRIKE_ATTEMPTS_PER_30S_BASE = 1.2
GROUND_STRIKE_ATTEMPTS_PER_30S_BASE = 1.6
CLINCH_STRIKE_ACCURACY_BASE = 0.68
GROUND_STRIKE_ACCURACY_BASE = 0.70
SUB_ATTEMPT_BASE_30S = 0.045
REVERSAL_SHARE_OF_GROUND_EXITS = 0.18
BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER = 0.20
BOTTOM_SUBMISSION_RATE_MULTIPLIER = 0.55


def rescale_interval_probability(
    probability: float, from_seconds: float, to_seconds: float
) -> float:
    return 1.0 - (1.0 - probability) ** (to_seconds / from_seconds)


DISTANCE_CLINCH_BASE_10S = rescale_interval_probability(0.04, 30.0, 10.0)
DISTANCE_TD_ATTEMPT_BASE_10S = rescale_interval_probability(0.10, 30.0, 10.0)
CLINCH_SEPARATE_BASE_10S = rescale_interval_probability(0.25, 30.0, 10.0)
CLINCH_TD_ATTEMPT_BASE_10S = rescale_interval_probability(0.24, 30.0, 10.0)
GROUND_EXIT_BASE_10S = rescale_interval_probability(0.20, 30.0, 10.0)
SUB_ATTEMPT_BASE_10S = rescale_interval_probability(0.045, 30.0, 10.0)


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


def td_attempt_rate_per_second(
    profile: FighterProfile, *, context_multiplier: float = 1.0
) -> float:
    """Intrinsic Phase 2B rate with a neutral seam for future context effects."""

    if context_multiplier < 0.0:
        raise ValueError("context_multiplier must be non-negative")
    intrinsic_rate = interval_hazard_per_second(
        td_attempt_interval_probability(profile)
    )
    return intrinsic_rate * context_multiplier


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


def phase_strike_rate_per_second(profile: FighterProfile, phase: str, *, bottom=False) -> float:
    """Port V0's CLINCH/GROUND Poisson strike-count intensity."""
    if phase == "clinch":
        base, pressure = CLINCH_STRIKE_ATTEMPTS_PER_30S_BASE, profile.clinch_striking_pressure
    elif phase == "ground":
        base, pressure = GROUND_STRIKE_ATTEMPTS_PER_30S_BASE, profile.ground_striking_pressure
    else:
        raise ValueError(f"unsupported phase: {phase}")
    multiplier = BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER if bottom else 1.0
    return base * _modifier(pressure - 50.0, scale=12.0) * multiplier / 30.0


def phase_strike_landing_probability(attacker: FighterProfile, defender: FighterProfile, phase: str) -> float:
    if phase == "clinch":
        base, precision, defense = CLINCH_STRIKE_ACCURACY_BASE, attacker.clinch_striking_precision, defender.clinch_striking_defense
    elif phase == "ground":
        base, precision, defense = GROUND_STRIKE_ACCURACY_BASE, attacker.ground_striking_precision, defender.ground_striking_defense
    else:
        raise ValueError(f"unsupported phase: {phase}")
    return _sigmoid(_logit(base) + (precision - defense) / RATING_SCALE)


def clinch_td_interval_probability(profile: FighterProfile) -> float:
    """V0 clinch TD consumer; legacy blend is intentionally phase-local here."""
    return float(np.clip(CLINCH_TD_ATTEMPT_BASE_10S * exp(style_preferences(profile)[2] / MODIFIER_SCALE), 0, 1 - 1e-12))


def clinch_separation_interval_probability(controller: FighterProfile, opponent: FighterProfile) -> float:
    _, clinch_preference, _ = style_preferences(controller)
    control_edge = (controller.control_imposition - opponent.control_resistance) / RATING_SCALE
    raw = CLINCH_SEPARATE_BASE_10S * _modifier(-clinch_preference) * exp(float(np.clip(-control_edge, -1, 1)) * 0.15)
    return float(np.clip(raw, 0, 0.90))


def ground_exit_interval_probability(controller: FighterProfile, bottom: FighterProfile) -> float:
    escape_edge = (bottom.control_resistance - controller.control_imposition) / RATING_SCALE
    reversal_edge = (bottom.reversal_ability - controller.control_imposition) / RATING_SCALE
    modifier = exp(float(np.clip(0.60 * escape_edge + 0.40 * reversal_edge, -1.5, 1.5)))
    return float(np.clip(GROUND_EXIT_BASE_10S * modifier, 0, 0.90))


def reversal_probability_given_exit(bottom: FighterProfile, controller: FighterProfile) -> float:
    edge = (bottom.reversal_ability - controller.control_imposition) / RATING_SCALE
    return _sigmoid(_logit(REVERSAL_SHARE_OF_GROUND_EXITS) + 0.75 * edge)


def ground_exit_rates(controller: FighterProfile, bottom: FighterProfile) -> tuple[float, float, float]:
    total = interval_hazard_per_second(ground_exit_interval_probability(controller, bottom))
    reversal = total * reversal_probability_given_exit(bottom, controller)
    return total * (1.0 - reversal_probability_given_exit(bottom, controller)), reversal, total


def submission_attempt_interval_probability(profile: FighterProfile, *, bottom=False) -> float:
    multiplier = BOTTOM_SUBMISSION_RATE_MULTIPLIER if bottom else 1.0
    raw = SUB_ATTEMPT_BASE_10S * multiplier * _modifier(profile.submission_pressure - 50.0, scale=10.0)
    return float(np.clip(raw, 0, 0.35))


@dataclass(frozen=True)
class ActionRateAudit:
    side: str
    action_family: str
    legacy_interval_probability: float | None
    interval_seconds: float
    rate_per_second: float
    major_inputs: dict[str, float]
