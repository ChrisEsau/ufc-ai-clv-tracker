"""Segment workload and adversity calculation for RFS Monte Carlo V2.

This module converts one phase-legal activity record into raw per-fighter
dynamic exposures:

- fatigue workload
- persistent damage exposure
- acute stress exposure

The values are universal calibrated costs before fighter-specific resistance,
recovery, or dynamic-state updates are applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    DynamicStateCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundFighterActivity,
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_activity_dispatch import (
    PhaseSegmentActivity,
)


@dataclass(frozen=True)
class FighterSegmentExposure:
    """Raw dynamic inputs produced for one fighter in one segment."""

    fatigue_workload: float
    persistent_damage_exposure: float
    acute_stress_exposure: float

    def __post_init__(self) -> None:
        """Validate nonnegative finite exposure values."""

        values = {
            "fatigue_workload": self.fatigue_workload,
            "persistent_damage_exposure": (
                self.persistent_damage_exposure
            ),
            "acute_stress_exposure": self.acute_stress_exposure,
        }

        for name, value in values.items():
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


@dataclass(frozen=True)
class SegmentDynamicExposure:
    """Raw dynamic exposure for both fighters in one shared segment."""

    state: SharedFightState
    red: FighterSegmentExposure
    blue: FighterSegmentExposure

    def __post_init__(self) -> None:
        """Validate nested exposure contract types."""

        if not isinstance(
            self.state,
            SharedFightState,
        ):
            raise TypeError(
                "state must be SharedFightState"
            )

        if not isinstance(
            self.red,
            FighterSegmentExposure,
        ):
            raise TypeError(
                "red must be FighterSegmentExposure"
            )

        if not isinstance(
            self.blue,
            FighterSegmentExposure,
        ):
            raise TypeError(
                "blue must be FighterSegmentExposure"
            )


def _distance_exposure(
    activity: DistanceSegmentActivity,
    calibration: DynamicStateCalibration,
) -> SegmentDynamicExposure:
    """Calculate raw exposure from a distance segment."""

    workload = calibration.activity_workload
    adversity = calibration.adversity

    red_fatigue = (
        calibration.phase_workload.distance
        + activity.red.sig_str_attempted
        * workload.strike_attempt
    )
    blue_fatigue = (
        calibration.phase_workload.distance
        + activity.blue.sig_str_attempted
        * workload.strike_attempt
    )

    red_damage = (
        activity.blue.sig_str_landed
        * adversity.distance_landed_damage
        + activity.blue.knockdowns
        * adversity.knockdown_damage
    )
    blue_damage = (
        activity.red.sig_str_landed
        * adversity.distance_landed_damage
        + activity.red.knockdowns
        * adversity.knockdown_damage
    )

    red_stress = (
        activity.blue.sig_str_landed
        * adversity.distance_landed_stress
        + activity.blue.knockdowns
        * adversity.knockdown_stress
    )
    blue_stress = (
        activity.red.sig_str_landed
        * adversity.distance_landed_stress
        + activity.red.knockdowns
        * adversity.knockdown_stress
    )

    return SegmentDynamicExposure(
        state=activity.state,
        red=FighterSegmentExposure(
            fatigue_workload=red_fatigue,
            persistent_damage_exposure=red_damage,
            acute_stress_exposure=red_stress,
        ),
        blue=FighterSegmentExposure(
            fatigue_workload=blue_fatigue,
            persistent_damage_exposure=blue_damage,
            acute_stress_exposure=blue_stress,
        ),
    )


def _clinch_exposure(
    activity: ClinchSegmentActivity,
    calibration: DynamicStateCalibration,
) -> SegmentDynamicExposure:
    """Calculate raw exposure from a clinch segment."""

    owner = activity.state.phase_owner

    if owner is FighterSide.RED:
        red_base = calibration.phase_workload.clinch_owner
        blue_base = calibration.phase_workload.clinch_defender
    elif owner is FighterSide.BLUE:
        red_base = calibration.phase_workload.clinch_defender
        blue_base = calibration.phase_workload.clinch_owner
    else:
        raise ValueError(
            "clinch exposure requires a phase owner"
        )

    workload = calibration.activity_workload
    adversity = calibration.adversity

    red_fatigue = (
        red_base
        + activity.red.clinch_str_attempted
        * workload.strike_attempt
        + activity.red.control_seconds
        * workload.control_second
    )
    blue_fatigue = (
        blue_base
        + activity.blue.clinch_str_attempted
        * workload.strike_attempt
        + activity.blue.control_seconds
        * workload.control_second
    )

    red_damage = (
        activity.blue.clinch_str_landed
        * adversity.clinch_landed_damage
        + activity.blue.damaging_clinch_strikes
        * adversity.damaging_clinch_bonus_damage
    )
    blue_damage = (
        activity.red.clinch_str_landed
        * adversity.clinch_landed_damage
        + activity.red.damaging_clinch_strikes
        * adversity.damaging_clinch_bonus_damage
    )

    red_stress = (
        activity.blue.clinch_str_landed
        * adversity.clinch_landed_stress
        + activity.blue.damaging_clinch_strikes
        * adversity.damaging_clinch_bonus_stress
        + activity.blue.control_seconds
        * adversity.control_second_received_stress
    )
    blue_stress = (
        activity.red.clinch_str_landed
        * adversity.clinch_landed_stress
        + activity.red.damaging_clinch_strikes
        * adversity.damaging_clinch_bonus_stress
        + activity.red.control_seconds
        * adversity.control_second_received_stress
    )

    return SegmentDynamicExposure(
        state=activity.state,
        red=FighterSegmentExposure(
            fatigue_workload=red_fatigue,
            persistent_damage_exposure=red_damage,
            acute_stress_exposure=red_stress,
        ),
        blue=FighterSegmentExposure(
            fatigue_workload=blue_fatigue,
            persistent_damage_exposure=blue_damage,
            acute_stress_exposure=blue_stress,
        ),
    )


def _ground_fatigue_workload(
    activity: GroundFighterActivity,
    *,
    phase_base: float,
    calibration: DynamicStateCalibration,
) -> float:
    """Calculate one fighter's ground-segment workload."""

    workload = calibration.activity_workload

    return (
        phase_base
        + activity.ground_str_attempted
        * workload.strike_attempt
        + activity.control_seconds
        * workload.control_second
        + activity.submission_attempts
        * workload.submission_attempt
        + activity.position_advancements
        * workload.position_advancement
        + activity.escape_attempts
        * workload.escape_attempt
        + activity.reversal_attempts
        * workload.reversal_attempt
        + activity.scramble_attempts
        * workload.scramble_attempt
    )


def _ground_exposure(
    activity: GroundSegmentActivity,
    calibration: DynamicStateCalibration,
) -> SegmentDynamicExposure:
    """Calculate raw exposure from a ground segment."""

    owner = activity.state.phase_owner

    if owner is FighterSide.RED:
        red_base = calibration.phase_workload.ground_owner
        blue_base = calibration.phase_workload.ground_defender
    elif owner is FighterSide.BLUE:
        red_base = calibration.phase_workload.ground_defender
        blue_base = calibration.phase_workload.ground_owner
    else:
        raise ValueError(
            "ground exposure requires a phase owner"
        )

    adversity = calibration.adversity

    red_fatigue = _ground_fatigue_workload(
        activity.red,
        phase_base=red_base,
        calibration=calibration,
    )
    blue_fatigue = _ground_fatigue_workload(
        activity.blue,
        phase_base=blue_base,
        calibration=calibration,
    )

    red_damage = (
        activity.blue.ground_str_landed
        * adversity.ground_landed_damage
    )
    blue_damage = (
        activity.red.ground_str_landed
        * adversity.ground_landed_damage
    )

    red_stress = (
        activity.blue.ground_str_landed
        * adversity.ground_landed_stress
        + activity.blue.control_seconds
        * adversity.control_second_received_stress
        + activity.blue.submission_attempts
        * adversity.submission_attempt_received_stress
        + activity.blue.position_advancements
        * adversity.position_advancement_received_stress
    )
    blue_stress = (
        activity.red.ground_str_landed
        * adversity.ground_landed_stress
        + activity.red.control_seconds
        * adversity.control_second_received_stress
        + activity.red.submission_attempts
        * adversity.submission_attempt_received_stress
        + activity.red.position_advancements
        * adversity.position_advancement_received_stress
    )

    return SegmentDynamicExposure(
        state=activity.state,
        red=FighterSegmentExposure(
            fatigue_workload=red_fatigue,
            persistent_damage_exposure=red_damage,
            acute_stress_exposure=red_stress,
        ),
        blue=FighterSegmentExposure(
            fatigue_workload=blue_fatigue,
            persistent_damage_exposure=blue_damage,
            acute_stress_exposure=blue_stress,
        ),
    )


def calculate_segment_dynamic_exposure(
    activity: PhaseSegmentActivity,
    calibration: DynamicStateCalibration,
) -> SegmentDynamicExposure:
    """Convert one phase-legal activity record into raw dynamic exposure."""

    if not isinstance(
        calibration,
        DynamicStateCalibration,
    ):
        raise TypeError(
            "calibration must be DynamicStateCalibration"
        )

    if isinstance(
        activity,
        DistanceSegmentActivity,
    ):
        if activity.state.phase is not FightPhase.DISTANCE:
            raise ValueError(
                "distance activity must use a distance state"
            )

        return _distance_exposure(
            activity,
            calibration,
        )

    if isinstance(
        activity,
        ClinchSegmentActivity,
    ):
        if activity.state.phase is not FightPhase.CLINCH:
            raise ValueError(
                "clinch activity must use a clinch state"
            )

        return _clinch_exposure(
            activity,
            calibration,
        )

    if isinstance(
        activity,
        GroundSegmentActivity,
    ):
        if activity.state.phase is not FightPhase.GROUND:
            raise ValueError(
                "ground activity must use a ground state"
            )

        return _ground_exposure(
            activity,
            calibration,
        )

    raise TypeError(
        "activity must be a supported phase activity record"
    )
