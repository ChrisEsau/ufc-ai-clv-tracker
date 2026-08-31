"""Phase-specific activity-rate contracts for V2.

Rates represent expected events during one 30-second simulation segment.
These parameters control what happens after the shared phase is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


SEGMENT_SECONDS = 30


def _validate_nonnegative(
    name: str,
    value: float,
) -> None:
    """Require a finite nonnegative rate."""

    if not isfinite(value) or value < 0.0:
        raise ValueError(
            f"{name} must be finite and nonnegative"
        )


def _validate_probability(
    name: str,
    value: float,
) -> None:
    """Require a finite probability between zero and one."""

    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1"
        )


def _validate_control_seconds(
    name: str,
    value: float,
) -> None:
    """Require control time within one simulation segment."""

    if (
        not isfinite(value)
        or not 0.0 <= value <= SEGMENT_SECONDS
    ):
        raise ValueError(
            f"{name} must be between 0 and "
            f"{SEGMENT_SECONDS} seconds"
        )


@dataclass(frozen=True)
class DistanceRateParameters:
    """Activity rates available while both fighters are at distance."""

    sig_strike_attempt_rate: float
    sig_strike_accuracy: float
    knockdown_probability_per_landed: float

    def __post_init__(self) -> None:
        _validate_nonnegative(
            "sig_strike_attempt_rate",
            self.sig_strike_attempt_rate,
        )
        _validate_probability(
            "sig_strike_accuracy",
            self.sig_strike_accuracy,
        )
        _validate_probability(
            "knockdown_probability_per_landed",
            self.knockdown_probability_per_landed,
        )


@dataclass(frozen=True)
class ClinchRateParameters:
    """Activity rates available during an owned clinch phase."""

    clinch_strike_attempt_rate: float
    clinch_strike_accuracy: float
    control_seconds_mean: float
    damaging_clinch_probability: float

    def __post_init__(self) -> None:
        _validate_nonnegative(
            "clinch_strike_attempt_rate",
            self.clinch_strike_attempt_rate,
        )
        _validate_probability(
            "clinch_strike_accuracy",
            self.clinch_strike_accuracy,
        )
        _validate_control_seconds(
            "control_seconds_mean",
            self.control_seconds_mean,
        )
        _validate_probability(
            "damaging_clinch_probability",
            self.damaging_clinch_probability,
        )


@dataclass(frozen=True)
class GroundOwnerRateParameters:
    """Activity rates for the controlling fighter on the ground."""

    ground_strike_attempt_rate: float
    ground_strike_accuracy: float
    control_seconds_mean: float
    submission_attempt_rate: float
    position_advancement_probability: float

    def __post_init__(self) -> None:
        _validate_nonnegative(
            "ground_strike_attempt_rate",
            self.ground_strike_attempt_rate,
        )
        _validate_probability(
            "ground_strike_accuracy",
            self.ground_strike_accuracy,
        )
        _validate_control_seconds(
            "control_seconds_mean",
            self.control_seconds_mean,
        )
        _validate_nonnegative(
            "submission_attempt_rate",
            self.submission_attempt_rate,
        )
        _validate_probability(
            "position_advancement_probability",
            self.position_advancement_probability,
        )


@dataclass(frozen=True)
class GroundDefenderRateParameters:
    """Activity rates for the non-owner while grounded."""

    escape_attempt_rate: float
    reversal_attempt_rate: float
    scramble_attempt_rate: float
    submission_defense: float

    def __post_init__(self) -> None:
        _validate_nonnegative(
            "escape_attempt_rate",
            self.escape_attempt_rate,
        )
        _validate_nonnegative(
            "reversal_attempt_rate",
            self.reversal_attempt_rate,
        )
        _validate_nonnegative(
            "scramble_attempt_rate",
            self.scramble_attempt_rate,
        )
        _validate_probability(
            "submission_defense",
            self.submission_defense,
        )


@dataclass(frozen=True)
class FighterPhaseParameters:
    """Complete static phase-specific activity profile for one fighter."""

    distance: DistanceRateParameters
    clinch: ClinchRateParameters
    ground_owner: GroundOwnerRateParameters
    ground_defender: GroundDefenderRateParameters
