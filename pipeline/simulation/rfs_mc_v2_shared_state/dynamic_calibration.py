"""Dynamic-state calibration contracts for RFS Monte Carlo V2.

Calibration describes the universal modeled cost of activity and adversity.
Fighter-specific response traits remain in ``FighterDynamicParameters``.

No values in this module are production-calibrated defaults. Callers must
provide an explicit calibration bundle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_nonnegative_finite(
    name: str,
    value: float,
) -> None:
    """Validate a nonnegative finite calibration value."""

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be numeric"
        )

    if not math.isfinite(float(value)):
        raise ValueError(
            f"{name} must be finite"
        )

    if float(value) < 0.0:
        raise ValueError(
            f"{name} cannot be negative"
        )


def _validate_unit_interval(
    name: str,
    value: float,
) -> None:
    """Validate a finite value constrained to [0, 1]."""

    _validate_nonnegative_finite(
        name,
        value,
    )

    if float(value) > 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1"
        )


@dataclass(frozen=True)
class PhaseWorkloadCalibration:
    """Base workload incurred by occupying each phase role."""

    distance: float
    clinch_owner: float
    clinch_defender: float
    ground_owner: float
    ground_defender: float

    def __post_init__(self) -> None:
        """Validate phase-role workload costs."""

        for name, value in vars(self).items():
            _validate_nonnegative_finite(
                name,
                value,
            )


@dataclass(frozen=True)
class ActivityWorkloadCalibration:
    """Additional workload created by realized activity."""

    strike_attempt: float
    control_second: float
    submission_attempt: float
    position_advancement: float
    escape_attempt: float
    reversal_attempt: float
    scramble_attempt: float

    def __post_init__(self) -> None:
        """Validate activity workload costs."""

        for name, value in vars(self).items():
            _validate_nonnegative_finite(
                name,
                value,
            )


@dataclass(frozen=True)
class AdversityCalibration:
    """Persistent-damage and acute-stress costs from adversity.

    Clinch damage and stress contain both a standard landed-strike cost and
    an additional bonus for strikes classified as damaging.
    """

    distance_landed_damage: float
    clinch_landed_damage: float
    damaging_clinch_bonus_damage: float
    ground_landed_damage: float
    knockdown_damage: float

    distance_landed_stress: float
    clinch_landed_stress: float
    damaging_clinch_bonus_stress: float
    ground_landed_stress: float
    knockdown_stress: float

    control_second_received_stress: float
    submission_attempt_received_stress: float
    position_advancement_received_stress: float

    def __post_init__(self) -> None:
        """Validate adversity costs."""

        for name, value in vars(self).items():
            _validate_nonnegative_finite(
                name,
                value,
            )


@dataclass(frozen=True)
class ResistanceScalingCalibration:
    """Minimum accumulation multipliers at maximum resistance.

    A floor of 0.25 means a fighter with resistance 1.0 still receives
    25 percent of the raw modeled cost. This prevents a normalized trait of
    one from automatically implying complete immunity.
    """

    minimum_fatigue_accumulation_multiplier: float
    minimum_damage_accumulation_multiplier: float
    minimum_acute_stress_accumulation_multiplier: float

    def __post_init__(self) -> None:
        """Validate resistance multiplier floors."""

        for name, value in vars(self).items():
            _validate_unit_interval(
                name,
                value,
            )


@dataclass(frozen=True)
class RecoveryCalibration:
    """Universal segment and round-break recovery scales."""

    low_workload_threshold: float
    segment_fatigue_recovery: float
    round_break_fatigue_recovery: float
    segment_acute_stress_recovery: float
    round_break_acute_stress_recovery: float

    def __post_init__(self) -> None:
        """Validate normalized recovery calibration values."""

        for name, value in vars(self).items():
            _validate_unit_interval(
                name,
                value,
            )


@dataclass(frozen=True)
class DynamicStateCalibration:
    """Complete universal dynamic-state calibration bundle."""

    phase_workload: PhaseWorkloadCalibration
    activity_workload: ActivityWorkloadCalibration
    adversity: AdversityCalibration
    resistance_scaling: ResistanceScalingCalibration
    recovery: RecoveryCalibration

    def __post_init__(self) -> None:
        """Validate nested calibration contract types."""

        expected_types = {
            "phase_workload": PhaseWorkloadCalibration,
            "activity_workload": ActivityWorkloadCalibration,
            "adversity": AdversityCalibration,
            "resistance_scaling": ResistanceScalingCalibration,
            "recovery": RecoveryCalibration,
        }

        for name, expected_type in expected_types.items():
            if not isinstance(
                getattr(self, name),
                expected_type,
            ):
                raise TypeError(
                    f"{name} must be {expected_type.__name__}"
                )
