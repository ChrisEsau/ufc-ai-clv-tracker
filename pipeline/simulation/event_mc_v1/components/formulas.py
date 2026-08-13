"""Exact ports of the legacy V0 DISTANCE consumers.

Source: ``scripts.experimental.fsr_static_mc_v0``. Constants and equations are
copied without tuning; only their time representation changes.
"""

from dataclasses import dataclass
from math import exp, log

import numpy as np

from ..scheduler import probability_to_rate
from ..calibration import DEFAULT_CALIBRATION, EventMCCalibration
from .profiles import FighterProfile

_S, _D, _C, _G, _SUB = (DEFAULT_CALIBRATION.section(name) for name in ("shared", "distance", "clinch", "ground", "submission_attempts"))
LEGACY_INTERVAL_SECONDS = _S["legacy_interval_seconds"]
CALIBRATION_INTERVAL_SECONDS = _S["calibration_interval_seconds"]
DISTANCE_CLINCH_BASE_30S = _D["clinch_base_30s"]
DISTANCE_TD_ATTEMPT_BASE_30S = _D["td_attempt_base_30s"]
DISTANCE_STRIKE_ATTEMPTS_PER_30S_BASE = _D["strike_attempts_per_30s"]
DISTANCE_STRIKE_ACCURACY_BASE = _D["strike_accuracy"]
RATING_SCALE = _S["rating_scale"]
MODIFIER_SCALE = _S["modifier_scale"]
TD_SUCCESS_LOGIT_OFFSET = _D["td_success_logit_offset"]
CLINCH_SEPARATE_BASE_30S = _C["separation_base_30s"]
CLINCH_TD_ATTEMPT_BASE_30S = _C["td_attempt_base_30s"]
GROUND_EXIT_BASE_30S = _G["exit_base_30s"]
CLINCH_STRIKE_ATTEMPTS_PER_30S_BASE = _C["strike_attempts_per_30s"]
GROUND_STRIKE_ATTEMPTS_PER_30S_BASE = _G["strike_attempts_per_30s"]
CLINCH_STRIKE_ACCURACY_BASE = _C["strike_accuracy"]
GROUND_STRIKE_ACCURACY_BASE = _G["strike_accuracy"]
SUB_ATTEMPT_BASE_30S = _SUB["base_30s"]
REVERSAL_SHARE_OF_GROUND_EXITS = _G["reversal_share"]
BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER = _G["bottom_strike_multiplier"]
BOTTOM_SUBMISSION_RATE_MULTIPLIER = _SUB["bottom_multiplier"]


def rescale_interval_probability(
    probability: float, from_seconds: float, to_seconds: float
) -> float:
    return 1.0 - (1.0 - probability) ** (to_seconds / from_seconds)


DISTANCE_CLINCH_BASE_10S = rescale_interval_probability(DISTANCE_CLINCH_BASE_30S, CALIBRATION_INTERVAL_SECONDS, LEGACY_INTERVAL_SECONDS)
DISTANCE_TD_ATTEMPT_BASE_10S = rescale_interval_probability(DISTANCE_TD_ATTEMPT_BASE_30S, CALIBRATION_INTERVAL_SECONDS, LEGACY_INTERVAL_SECONDS)
CLINCH_SEPARATE_BASE_10S = rescale_interval_probability(CLINCH_SEPARATE_BASE_30S, CALIBRATION_INTERVAL_SECONDS, LEGACY_INTERVAL_SECONDS)
CLINCH_TD_ATTEMPT_BASE_10S = rescale_interval_probability(CLINCH_TD_ATTEMPT_BASE_30S, CALIBRATION_INTERVAL_SECONDS, LEGACY_INTERVAL_SECONDS)
GROUND_EXIT_BASE_10S = rescale_interval_probability(GROUND_EXIT_BASE_30S, CALIBRATION_INTERVAL_SECONDS, LEGACY_INTERVAL_SECONDS)
SUB_ATTEMPT_BASE_10S = rescale_interval_probability(SUB_ATTEMPT_BASE_30S, CALIBRATION_INTERVAL_SECONDS, LEGACY_INTERVAL_SECONDS)


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


def strike_attempt_rate_per_second(profile: FighterProfile, calibration: EventMCCalibration = DEFAULT_CALIBRATION) -> float:
    """Poisson intensity preserving V0's expected DISTANCE attempt count."""

    expected_per_30s = calibration.section("distance")["strike_attempts_per_30s"] * _modifier(
        profile.distance_striking_pressure - 50.0, scale=calibration.section("shared")["rating_scale"]
    )
    return expected_per_30s / CALIBRATION_INTERVAL_SECONDS


def strike_landing_probability(
    attacker: FighterProfile, defender: FighterProfile, calibration: EventMCCalibration = DEFAULT_CALIBRATION
) -> float:
    shared, distance = calibration.section("shared"), calibration.section("distance")
    return _sigmoid(
        _logit(distance["strike_accuracy"])
        + (
            attacker.distance_striking_precision
            - defender.distance_striking_defense
        )
        / shared["rating_scale"]
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
    attacker: FighterProfile, defender: FighterProfile, calibration: EventMCCalibration = DEFAULT_CALIBRATION
) -> float:
    edge = (attacker.wrestling_conversion - defender.td_defense) / calibration.section("shared")["rating_scale"]
    return _sigmoid(edge + calibration.section("distance")["td_success_logit_offset"])


def clinch_entry_interval_probability(profile: FighterProfile) -> float:
    distance_preference, clinch_preference, _ = style_preferences(profile)
    probability = (
        DISTANCE_CLINCH_BASE_10S
        * _modifier(clinch_preference)
        * np.sqrt(_modifier(-distance_preference))
    )
    return float(np.clip(probability, 0.0, _D["clinch_cap"]))


def interval_hazard_per_second(probability: float) -> float:
    return probability_to_rate(probability, LEGACY_INTERVAL_SECONDS)


def phase_strike_rate_per_second(profile: FighterProfile, phase: str, *, bottom=False, calibration: EventMCCalibration = DEFAULT_CALIBRATION) -> float:
    """Port V0's CLINCH/GROUND Poisson strike-count intensity."""
    if phase == "clinch":
        base, pressure = calibration.section("clinch")["strike_attempts_per_30s"], profile.clinch_striking_pressure
    elif phase == "ground":
        base, pressure = calibration.section("ground")["strike_attempts_per_30s"], profile.ground_striking_pressure
    else:
        raise ValueError(f"unsupported phase: {phase}")
    multiplier = calibration.section("ground")["bottom_strike_multiplier"] if bottom else 1.0
    return base * _modifier(pressure - 50.0, scale=calibration.section("shared")["rating_scale"]) * multiplier / calibration.section("shared")["calibration_interval_seconds"]


def phase_strike_landing_probability(attacker: FighterProfile, defender: FighterProfile, phase: str, calibration: EventMCCalibration = DEFAULT_CALIBRATION) -> float:
    if phase == "clinch":
        base, precision, defense = calibration.section("clinch")["strike_accuracy"], attacker.clinch_striking_precision, defender.clinch_striking_defense
    elif phase == "ground":
        base, precision, defense = calibration.section("ground")["strike_accuracy"], attacker.ground_striking_precision, defender.ground_striking_defense
    else:
        raise ValueError(f"unsupported phase: {phase}")
    return _sigmoid(_logit(base) + (precision - defense) / calibration.section("shared")["rating_scale"])


def clinch_td_interval_probability(profile: FighterProfile) -> float:
    """V0 clinch TD consumer; legacy blend is intentionally phase-local here."""
    return float(np.clip(CLINCH_TD_ATTEMPT_BASE_10S * exp(style_preferences(profile)[2] / MODIFIER_SCALE), 0, 1 - 1e-12))


def clinch_separation_interval_probability(controller: FighterProfile, opponent: FighterProfile) -> float:
    _, clinch_preference, _ = style_preferences(controller)
    control_edge = (controller.control_imposition - opponent.control_resistance) / RATING_SCALE
    raw = CLINCH_SEPARATE_BASE_10S * _modifier(-clinch_preference) * exp(float(np.clip(-control_edge, -1, 1)) * _C["control_edge_multiplier"])
    return float(np.clip(raw, 0, _C["separation_cap"]))


def ground_exit_interval_probability(controller: FighterProfile, bottom: FighterProfile) -> float:
    escape_edge = (bottom.control_resistance - controller.control_imposition) / RATING_SCALE
    reversal_edge = (bottom.reversal_ability - controller.control_imposition) / RATING_SCALE
    modifier = exp(float(np.clip(0.60 * escape_edge + 0.40 * reversal_edge, -1.5, 1.5)))
    return float(np.clip(GROUND_EXIT_BASE_10S * modifier, 0, _G["exit_cap"]))


def reversal_probability_given_exit(bottom: FighterProfile, controller: FighterProfile) -> float:
    edge = (bottom.reversal_ability - controller.control_imposition) / RATING_SCALE
    return _sigmoid(_logit(REVERSAL_SHARE_OF_GROUND_EXITS) + 0.75 * edge)


def ground_exit_rates(controller: FighterProfile, bottom: FighterProfile) -> tuple[float, float, float]:
    total = interval_hazard_per_second(ground_exit_interval_probability(controller, bottom))
    reversal = total * reversal_probability_given_exit(bottom, controller)
    return total * (1.0 - reversal_probability_given_exit(bottom, controller)), reversal, total


def submission_attempt_interval_probability(profile: FighterProfile, *, bottom=False) -> float:
    multiplier = BOTTOM_SUBMISSION_RATE_MULTIPLIER if bottom else 1.0
    raw = SUB_ATTEMPT_BASE_10S * multiplier * _modifier(profile.submission_pressure - 50.0, scale=_SUB["modifier_scale"])
    return float(np.clip(raw, 0, _SUB["probability_cap"]))


@dataclass(frozen=True)
class ActionRateAudit:
    side: str
    action_family: str
    legacy_interval_probability: float | None
    interval_seconds: float
    rate_per_second: float
    major_inputs: dict[str, float]
