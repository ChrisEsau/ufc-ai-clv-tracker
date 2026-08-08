"""Deterministic finish-probability calculation for RFS Monte Carlo V2.

This module converts one segment's simulated activity into method-specific
finish probabilities. It does not sample a finish or terminate a fight path.

KO/TKO event probabilities are combined as independent event opportunities:

    1 - product((1 - event_probability) ** event_count)

The resulting event probability is amplified by the defending fighter's
dynamic state and capped by the configured segment maximum.

Submission probability is calculated only for the authoritative ground owner.
The adjusted probability for one attempt incorporates:

- authoritative ground position quality
- the defender's effective submission defense
- the defender's dynamic state

Multiple attempts are then combined as independent opportunities.
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
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
    FighterDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_calibration import (
    FinishProbabilityCalibration,
    KnockoutFinishCalibration,
    SubmissionFinishCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_activity_dispatch import (
    PhaseSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    FighterPhaseParameters,
)


def _validate_probability(
    name: str,
    value: float,
) -> float:
    """Validate and return one finite value constrained to [0, 1]."""

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be numeric"
        )

    selected = float(value)

    if not math.isfinite(selected):
        raise ValueError(
            f"{name} must be finite"
        )

    if not 0.0 <= selected <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1"
        )

    return selected


def _validate_nonnegative_multiplier(
    name: str,
    value: float,
) -> float:
    """Validate a finite nonnegative multiplicative factor.

    Unlike probabilities, power multipliers may legally exceed 1.0.
    """

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be numeric"
        )

    selected = float(value)

    if not math.isfinite(selected):
        raise ValueError(
            f"{name} must be finite"
        )

    if selected < 0.0:
        raise ValueError(
            f"{name} cannot be negative"
        )

    return selected


@dataclass(frozen=True)
class FighterSegmentFinishProbabilities:
    """Method-specific finish probabilities for one fighter."""

    ko_tko_probability: float
    submission_probability: float

    def __post_init__(self) -> None:
        """Validate normalized method probabilities."""

        _validate_probability(
            "ko_tko_probability",
            self.ko_tko_probability,
        )
        _validate_probability(
            "submission_probability",
            self.submission_probability,
        )


@dataclass(frozen=True)
class SegmentFinishProbabilities:
    """Method-specific probabilities for both fighters in one segment."""

    state: SharedFightState
    red: FighterSegmentFinishProbabilities
    blue: FighterSegmentFinishProbabilities

    def __post_init__(self) -> None:
        """Validate nested probability contracts and submission legality."""

        if not isinstance(
            self.state,
            SharedFightState,
        ):
            raise TypeError(
                "state must be SharedFightState"
            )

        if not isinstance(
            self.red,
            FighterSegmentFinishProbabilities,
        ):
            raise TypeError(
                "red must be FighterSegmentFinishProbabilities"
            )

        if not isinstance(
            self.blue,
            FighterSegmentFinishProbabilities,
        ):
            raise TypeError(
                "blue must be FighterSegmentFinishProbabilities"
            )

        if self.state.phase is not FightPhase.GROUND:
            if (
                self.red.submission_probability != 0.0
                or self.blue.submission_probability != 0.0
            ):
                raise ValueError(
                    "submission probability requires a ground state"
                )

        elif self.state.phase_owner is FighterSide.RED:
            if self.blue.submission_probability != 0.0:
                raise ValueError(
                    "ground defender cannot have submission probability"
                )

        elif self.state.phase_owner is FighterSide.BLUE:
            if self.red.submission_probability != 0.0:
                raise ValueError(
                    "ground defender cannot have submission probability"
                )


def combine_independent_event_probabilities(
    event_opportunities: tuple[tuple[int, float], ...],
) -> float:
    """Combine independent event-level probabilities into one probability."""

    probability_no_finish = 1.0

    for event_count, event_probability in event_opportunities:
        if not isinstance(event_count, int):
            raise TypeError(
                "event_count must be an integer"
            )

        if event_count < 0:
            raise ValueError(
                "event_count cannot be negative"
            )

        selected_probability = _validate_probability(
            "event_probability",
            event_probability,
        )

        probability_no_finish *= (
            1.0 - selected_probability
        ) ** event_count

    return 1.0 - probability_no_finish


def calculate_defender_state_amplifier(
    defender_state: FighterDynamicState,
    *,
    fatigue_amplifier: float,
    damage_amplifier: float,
    acute_stress_amplifier: float,
) -> float:
    """Calculate a finish-hazard multiplier from defender state."""

    if not isinstance(
        defender_state,
        FighterDynamicState,
    ):
        raise TypeError(
            "defender_state must be FighterDynamicState"
        )

    selected_fatigue_amplifier = _validate_probability(
        "fatigue_amplifier",
        fatigue_amplifier,
    )
    selected_damage_amplifier = _validate_probability(
        "damage_amplifier",
        damage_amplifier,
    )
    selected_stress_amplifier = _validate_probability(
        "acute_stress_amplifier",
        acute_stress_amplifier,
    )

    return (
        1.0
        + defender_state.fatigue
        * selected_fatigue_amplifier
        + defender_state.damage
        * selected_damage_amplifier
        + defender_state.acute_stress
        * selected_stress_amplifier
    )


def calculate_knockout_finish_probability(
    activity: PhaseSegmentActivity,
    attacker: FighterSide,
    defender_state: FighterDynamicState,
    calibration: KnockoutFinishCalibration,
    *,
    attacker_power_multiplier: float = 1.0,
    defender_ko_vulnerability_multiplier: float = 1.0,
) -> float:
    """Calculate one fighter's KO/TKO probability for the segment."""

    if not isinstance(
        activity,
        (
            DistanceSegmentActivity,
            ClinchSegmentActivity,
            GroundSegmentActivity,
        ),
    ):
        raise TypeError(
            "activity must be a supported phase segment activity"
        )

    if not isinstance(
        attacker,
        FighterSide,
    ):
        raise TypeError(
            "attacker must be FighterSide"
        )

    if not isinstance(
        defender_state,
        FighterDynamicState,
    ):
        raise TypeError(
            "defender_state must be FighterDynamicState"
        )

    if not isinstance(
        calibration,
        KnockoutFinishCalibration,
    ):
        raise TypeError(
            "calibration must be KnockoutFinishCalibration"
        )

    selected_attacker_power_multiplier = (
        _validate_nonnegative_multiplier(
            "attacker_power_multiplier",
            attacker_power_multiplier,
        )
    )


    selected_defender_ko_vulnerability_multiplier = (
        _validate_nonnegative_multiplier(
            "defender_ko_vulnerability_multiplier",
            defender_ko_vulnerability_multiplier,
        )
    )

    fighter_activity = (
        activity.red
        if attacker is FighterSide.RED
        else activity.blue
    )

    if isinstance(
        activity,
        DistanceSegmentActivity,
    ):
        event_probability = (
            combine_independent_event_probabilities(
                (
                    (
                        fighter_activity.sig_str_landed,
                        (
                            min(
                                1.0,
                                calibration.distance_landed_probability
                                * selected_attacker_power_multiplier,
                            )
                        ),
                    ),
                    (
                        fighter_activity.knockdowns,
                        calibration.distance_knockdown_probability,
                    ),
                )
            )
        )

    elif isinstance(
        activity,
        ClinchSegmentActivity,
    ):
        event_probability = (
            combine_independent_event_probabilities(
                (
                    (
                        fighter_activity.clinch_str_landed,
                        (
                            min(
                                1.0,
                                calibration.clinch_landed_probability
                                * selected_attacker_power_multiplier,
                            )
                        ),
                    ),
                    (
                        fighter_activity.damaging_clinch_strikes,
                        calibration.damaging_clinch_probability,
                    ),
                )
            )
        )

    elif isinstance(
        activity,
        GroundSegmentActivity,
    ):
        event_probability = (
            combine_independent_event_probabilities(
                (
                    (
                        fighter_activity.ground_str_landed,
                        (
                            min(
                                1.0,
                                calibration.ground_landed_probability
                                * selected_attacker_power_multiplier,
                            )
                        ),
                    ),
                )
            )
        )

    else:
        raise RuntimeError(
            "supported activity type was not dispatched"
        )

    state_amplifier = calculate_defender_state_amplifier(
        defender_state,
        fatigue_amplifier=(
            calibration.defender_fatigue_amplifier
        ),
        damage_amplifier=(
            calibration.defender_damage_amplifier
        ),
        acute_stress_amplifier=(
            calibration.defender_acute_stress_amplifier
        ),
    )

    return min(
        calibration.maximum_segment_probability,
        (
            event_probability
            * selected_defender_ko_vulnerability_multiplier
            * state_amplifier
        ),
    )


def calculate_submission_finish_probability(
    activity: GroundSegmentActivity,
    attacker: FighterSide,
    defender_state: FighterDynamicState,
    effective_submission_defense: float,
    calibration: SubmissionFinishCalibration,
) -> float:
    """Calculate the ground owner's submission probability."""

    if not isinstance(
        activity,
        GroundSegmentActivity,
    ):
        raise TypeError(
            "activity must be GroundSegmentActivity"
        )

    if not isinstance(
        attacker,
        FighterSide,
    ):
        raise TypeError(
            "attacker must be FighterSide"
        )

    if activity.state.phase_owner is not attacker:
        raise ValueError(
            "submission attacker must own the ground phase"
        )

    if not isinstance(
        defender_state,
        FighterDynamicState,
    ):
        raise TypeError(
            "defender_state must be FighterDynamicState"
        )

    if not isinstance(
        calibration,
        SubmissionFinishCalibration,
    ):
        raise TypeError(
            "calibration must be SubmissionFinishCalibration"
        )

    selected_submission_defense = _validate_probability(
        "effective_submission_defense",
        effective_submission_defense,
    )

    fighter_activity = (
        activity.red
        if attacker is FighterSide.RED
        else activity.blue
    )

    attempt_count = fighter_activity.submission_attempts

    if attempt_count == 0:
        return 0.0

    position_multiplier = (
        1.0
        + activity.state.position_quality
        * calibration.position_quality_amplifier
    )

    submission_defense_effect = max(
        calibration.minimum_submission_defense_effect_multiplier,
        1.0 - selected_submission_defense,
    )

    state_amplifier = calculate_defender_state_amplifier(
        defender_state,
        fatigue_amplifier=(
            calibration.defender_fatigue_amplifier
        ),
        damage_amplifier=(
            calibration.defender_damage_amplifier
        ),
        acute_stress_amplifier=(
            calibration.defender_acute_stress_amplifier
        ),
    )

    probability_per_attempt = min(
        calibration.maximum_probability_per_attempt,
        (
            calibration.base_probability_per_attempt
            * position_multiplier
            * submission_defense_effect
            * state_amplifier
        ),
    )

    segment_probability = (
        combine_independent_event_probabilities(
            (
                (
                    attempt_count,
                    probability_per_attempt,
                ),
            )
        )
    )

    return min(
        calibration.maximum_segment_probability,
        segment_probability,
    )


def calculate_segment_finish_probabilities(
    activity: PhaseSegmentActivity,
    dynamic_state: FightDynamicState,
    red_effective_phase: FighterPhaseParameters,
    blue_effective_phase: FighterPhaseParameters,
    calibration: FinishProbabilityCalibration,
    *,
    red_power_multiplier: float = 1.0,
    blue_power_multiplier: float = 1.0,
    red_ko_vulnerability_multiplier: float = 1.0,
    blue_ko_vulnerability_multiplier: float = 1.0,
) -> SegmentFinishProbabilities:
    """Calculate deterministic finish probabilities for one segment."""

    if not isinstance(
        activity,
        (
            DistanceSegmentActivity,
            ClinchSegmentActivity,
            GroundSegmentActivity,
        ),
    ):
        raise TypeError(
            "activity must be a supported phase segment activity"
        )

    if not isinstance(
        dynamic_state,
        FightDynamicState,
    ):
        raise TypeError(
            "dynamic_state must be FightDynamicState"
        )

    if not isinstance(
        red_effective_phase,
        FighterPhaseParameters,
    ):
        raise TypeError(
            "red_effective_phase must be FighterPhaseParameters"
        )

    if not isinstance(
        blue_effective_phase,
        FighterPhaseParameters,
    ):
        raise TypeError(
            "blue_effective_phase must be FighterPhaseParameters"
        )

    if not isinstance(
        calibration,
        FinishProbabilityCalibration,
    ):
        raise TypeError(
            "calibration must be FinishProbabilityCalibration"
        )

    red_knockout = calculate_knockout_finish_probability(
        activity,
        FighterSide.RED,
        dynamic_state.blue,
        calibration.knockout,
        attacker_power_multiplier=red_power_multiplier,
        defender_ko_vulnerability_multiplier=(
            blue_ko_vulnerability_multiplier
        ),
    )
    blue_knockout = calculate_knockout_finish_probability(
        activity,
        FighterSide.BLUE,
        dynamic_state.red,
        calibration.knockout,
        attacker_power_multiplier=blue_power_multiplier,
        defender_ko_vulnerability_multiplier=(
            red_ko_vulnerability_multiplier
        ),
    )

    red_submission = 0.0
    blue_submission = 0.0

    if isinstance(
        activity,
        GroundSegmentActivity,
    ):
        if activity.state.phase_owner is FighterSide.RED:
            red_submission = (
                calculate_submission_finish_probability(
                    activity,
                    FighterSide.RED,
                    dynamic_state.blue,
                    (
                        blue_effective_phase
                        .ground_defender.submission_defense
                    ),
                    calibration.submission,
                )
            )

        elif activity.state.phase_owner is FighterSide.BLUE:
            blue_submission = (
                calculate_submission_finish_probability(
                    activity,
                    FighterSide.BLUE,
                    dynamic_state.red,
                    (
                        red_effective_phase
                        .ground_defender.submission_defense
                    ),
                    calibration.submission,
                )
            )

    return SegmentFinishProbabilities(
        state=activity.state,
        red=FighterSegmentFinishProbabilities(
            ko_tko_probability=red_knockout,
            submission_probability=red_submission,
        ),
        blue=FighterSegmentFinishProbabilities(
            ko_tko_probability=blue_knockout,
            submission_probability=blue_submission,
        ),
    )
