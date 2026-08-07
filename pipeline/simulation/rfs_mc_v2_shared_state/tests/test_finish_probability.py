"""Tests for deterministic V2 finish-probability calculation."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchFighterActivity,
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceFighterActivity,
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
from pipeline.simulation.rfs_mc_v2_shared_state.finish_probability import (
    FighterSegmentFinishProbabilities,
    SegmentFinishProbabilities,
    calculate_defender_state_amplifier,
    calculate_knockout_finish_probability,
    calculate_segment_finish_probabilities,
    calculate_submission_finish_probability,
    combine_independent_event_probabilities,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundFighterActivity,
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)


PROBABILITY_FIELDS = [
    "ko_tko_probability",
    "submission_probability",
]

AMPLIFIER_FIELDS = [
    "fatigue_amplifier",
    "damage_amplifier",
    "acute_stress_amplifier",
]


def shared_state(
    *,
    phase: FightPhase,
    owner: FighterSide | None = None,
    position_quality: float = 0.0,
) -> SharedFightState:
    """Build one valid authoritative shared state."""

    if phase is FightPhase.DISTANCE:
        selected_owner = None
        phase_age_segments = 0
        selected_quality = 0.0
    else:
        selected_owner = owner or FighterSide.RED
        phase_age_segments = 1
        selected_quality = position_quality

    return SharedFightState(
        phase=phase,
        phase_owner=selected_owner,
        phase_age_segments=phase_age_segments,
        position_quality=selected_quality,
        round_number=1,
        segment_number=1,
    )


def distance_activity(
    *,
    red_landed: int = 0,
    red_knockdowns: int = 0,
    blue_landed: int = 0,
    blue_knockdowns: int = 0,
) -> DistanceSegmentActivity:
    """Build controlled distance activity."""

    return DistanceSegmentActivity(
        state=shared_state(
            phase=FightPhase.DISTANCE,
        ),
        red=DistanceFighterActivity(
            sig_str_attempted=red_landed,
            sig_str_landed=red_landed,
            knockdowns=red_knockdowns,
        ),
        blue=DistanceFighterActivity(
            sig_str_attempted=blue_landed,
            sig_str_landed=blue_landed,
            knockdowns=blue_knockdowns,
        ),
    )


def clinch_activity(
    *,
    owner: FighterSide = FighterSide.RED,
    red_landed: int = 0,
    red_damaging: int = 0,
    blue_landed: int = 0,
    blue_damaging: int = 0,
) -> ClinchSegmentActivity:
    """Build controlled clinch activity."""

    return ClinchSegmentActivity(
        state=shared_state(
            phase=FightPhase.CLINCH,
            owner=owner,
            position_quality=0.50,
        ),
        red=ClinchFighterActivity(
            clinch_str_attempted=red_landed,
            clinch_str_landed=red_landed,
            damaging_clinch_strikes=red_damaging,
            control_seconds=(
                10
                if owner is FighterSide.RED
                else 0
            ),
        ),
        blue=ClinchFighterActivity(
            clinch_str_attempted=blue_landed,
            clinch_str_landed=blue_landed,
            damaging_clinch_strikes=blue_damaging,
            control_seconds=(
                10
                if owner is FighterSide.BLUE
                else 0
            ),
        ),
    )


def ground_fighter_activity(
    *,
    landed: int = 0,
    submission_attempts: int = 0,
    owner: bool,
) -> GroundFighterActivity:
    """Build a phase-legal ground activity payload."""

    return GroundFighterActivity(
        ground_str_attempted=landed,
        ground_str_landed=landed,
        control_seconds=10 if owner else 0,
        submission_attempts=(
            submission_attempts
            if owner
            else 0
        ),
        position_advancements=0,
        escape_attempts=0,
        reversal_attempts=0,
        scramble_attempts=0,
    )


def ground_activity(
    *,
    owner: FighterSide = FighterSide.RED,
    red_landed: int = 0,
    red_submission_attempts: int = 0,
    blue_landed: int = 0,
    blue_submission_attempts: int = 0,
    position_quality: float = 0.50,
) -> GroundSegmentActivity:
    """Build controlled ground activity."""

    red_is_owner = owner is FighterSide.RED
    blue_is_owner = owner is FighterSide.BLUE

    return GroundSegmentActivity(
        state=shared_state(
            phase=FightPhase.GROUND,
            owner=owner,
            position_quality=position_quality,
        ),
        red=ground_fighter_activity(
            landed=red_landed,
            submission_attempts=red_submission_attempts,
            owner=red_is_owner,
        ),
        blue=ground_fighter_activity(
            landed=blue_landed,
            submission_attempts=blue_submission_attempts,
            owner=blue_is_owner,
        ),
    )


def fighter_state(
    *,
    fatigue: float = 0.0,
    damage: float = 0.0,
    acute_stress: float = 0.0,
) -> FighterDynamicState:
    """Build controlled fighter dynamic state."""

    return FighterDynamicState(
        fatigue=fatigue,
        damage=damage,
        acute_stress=acute_stress,
    )


def fight_state(
    *,
    red: FighterDynamicState | None = None,
    blue: FighterDynamicState | None = None,
) -> FightDynamicState:
    """Build controlled two-fighter dynamic state."""

    return FightDynamicState(
        red=red or fighter_state(),
        blue=blue or fighter_state(),
    )


def knockout_calibration(
    **overrides: float,
) -> KnockoutFinishCalibration:
    """Build controlled KO/TKO calibration."""

    values = {
        "distance_landed_probability": 0.10,
        "distance_knockdown_probability": 0.20,
        "clinch_landed_probability": 0.05,
        "damaging_clinch_probability": 0.20,
        "ground_landed_probability": 0.10,
        "defender_fatigue_amplifier": 0.0,
        "defender_damage_amplifier": 0.0,
        "defender_acute_stress_amplifier": 0.0,
        "maximum_segment_probability": 0.90,
    }
    values.update(overrides)

    return KnockoutFinishCalibration(
        **values
    )


def submission_calibration(
    **overrides: float,
) -> SubmissionFinishCalibration:
    """Build controlled submission calibration."""

    values = {
        "base_probability_per_attempt": 0.10,
        "position_quality_amplifier": 0.40,
        "minimum_submission_defense_effect_multiplier": 0.10,
        "defender_fatigue_amplifier": 0.0,
        "defender_damage_amplifier": 0.0,
        "defender_acute_stress_amplifier": 0.0,
        "maximum_probability_per_attempt": 0.90,
        "maximum_segment_probability": 0.95,
    }
    values.update(overrides)

    return SubmissionFinishCalibration(
        **values
    )


def finish_calibration(
    *,
    knockout: KnockoutFinishCalibration | None = None,
    submission: SubmissionFinishCalibration | None = None,
) -> FinishProbabilityCalibration:
    """Build complete controlled finish calibration."""

    return FinishProbabilityCalibration(
        knockout=knockout or knockout_calibration(),
        submission=submission or submission_calibration(),
    )


def phase_parameters(
    *,
    submission_defense: float = 0.25,
) -> FighterPhaseParameters:
    """Build complete effective phase parameters."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=4.0,
            sig_strike_accuracy=0.50,
            knockdown_probability_per_landed=0.02,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=2.0,
            clinch_strike_accuracy=0.50,
            control_seconds_mean=8.0,
            damaging_clinch_probability=0.05,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=2.0,
            ground_strike_accuracy=0.50,
            control_seconds_mean=15.0,
            submission_attempt_rate=0.20,
            position_advancement_probability=0.25,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.20,
            reversal_attempt_rate=0.08,
            scramble_attempt_rate=0.15,
            submission_defense=submission_defense,
        ),
    )


def fighter_finish_probabilities(
    *,
    ko_tko_probability: float = 0.20,
    submission_probability: float = 0.10,
) -> FighterSegmentFinishProbabilities:
    """Build controlled fighter finish probabilities."""

    return FighterSegmentFinishProbabilities(
        ko_tko_probability=ko_tko_probability,
        submission_probability=submission_probability,
    )


def test_valid_fighter_finish_probabilities_retain_values() -> None:
    selected = fighter_finish_probabilities()

    assert selected.ko_tko_probability == 0.20
    assert selected.submission_probability == 0.10


def test_fighter_finish_probability_boundaries_are_allowed() -> None:
    low = FighterSegmentFinishProbabilities(
        ko_tko_probability=0.0,
        submission_probability=0.0,
    )
    high = FighterSegmentFinishProbabilities(
        ko_tko_probability=1.0,
        submission_probability=1.0,
    )

    assert low.ko_tko_probability == 0.0
    assert low.submission_probability == 0.0
    assert high.ko_tko_probability == 1.0
    assert high.submission_probability == 1.0


@pytest.mark.parametrize(
    "field_name",
    PROBABILITY_FIELDS,
)
def test_fighter_finish_probabilities_must_be_numeric(
    field_name: str,
) -> None:
    values = {
        "ko_tko_probability": 0.20,
        "submission_probability": 0.10,
    }
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        FighterSegmentFinishProbabilities(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    PROBABILITY_FIELDS,
)
def test_fighter_finish_probabilities_must_be_finite(
    field_name: str,
) -> None:
    values = {
        "ko_tko_probability": 0.20,
        "submission_probability": 0.10,
    }
    values[field_name] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        FighterSegmentFinishProbabilities(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    PROBABILITY_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
    ],
)
def test_fighter_finish_probabilities_must_be_in_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    values = {
        "ko_tko_probability": 0.20,
        "submission_probability": 0.10,
    }
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        FighterSegmentFinishProbabilities(
            **values
        )


def test_fighter_finish_probabilities_are_immutable() -> None:
    selected = fighter_finish_probabilities()

    with pytest.raises(FrozenInstanceError):
        selected.ko_tko_probability = 0.50


def test_segment_finish_probabilities_retain_values() -> None:
    state = shared_state(
        phase=FightPhase.GROUND,
        owner=FighterSide.RED,
        position_quality=0.50,
    )
    red = fighter_finish_probabilities()
    blue = fighter_finish_probabilities(
        submission_probability=0.0,
    )

    selected = SegmentFinishProbabilities(
        state=state,
        red=red,
        blue=blue,
    )

    assert selected.state is state
    assert selected.red is red
    assert selected.blue is blue


def test_segment_finish_probabilities_require_shared_state() -> None:
    with pytest.raises(
        TypeError,
        match="state must be SharedFightState",
    ):
        SegmentFinishProbabilities(
            state="invalid",
            red=fighter_finish_probabilities(
                submission_probability=0.0,
            ),
            blue=fighter_finish_probabilities(
                submission_probability=0.0,
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "red",
        "blue",
    ],
)
def test_segment_finish_probabilities_require_nested_contracts(
    field_name: str,
) -> None:
    values = {
        "state": shared_state(
            phase=FightPhase.DISTANCE,
        ),
        "red": fighter_finish_probabilities(
            submission_probability=0.0,
        ),
        "blue": fighter_finish_probabilities(
            submission_probability=0.0,
        ),
    }
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name} must be "
            "FighterSegmentFinishProbabilities"
        ),
    ):
        SegmentFinishProbabilities(
            **values
        )


@pytest.mark.parametrize(
    ("phase", "side"),
    [
        (
            FightPhase.DISTANCE,
            FighterSide.RED,
        ),
        (
            FightPhase.DISTANCE,
            FighterSide.BLUE,
        ),
        (
            FightPhase.CLINCH,
            FighterSide.RED,
        ),
        (
            FightPhase.CLINCH,
            FighterSide.BLUE,
        ),
    ],
)
def test_submission_probability_requires_ground_state(
    phase: FightPhase,
    side: FighterSide,
) -> None:
    values = {
        "red": fighter_finish_probabilities(
            submission_probability=0.0,
        ),
        "blue": fighter_finish_probabilities(
            submission_probability=0.0,
        ),
    }
    values[
        "red"
        if side is FighterSide.RED
        else "blue"
    ] = fighter_finish_probabilities(
        submission_probability=0.10,
    )

    with pytest.raises(
        ValueError,
        match="submission probability requires a ground state",
    ):
        SegmentFinishProbabilities(
            state=shared_state(
                phase=phase,
                owner=FighterSide.RED,
                position_quality=0.50,
            ),
            **values,
        )


@pytest.mark.parametrize(
    ("owner", "defender"),
    [
        (
            FighterSide.RED,
            FighterSide.BLUE,
        ),
        (
            FighterSide.BLUE,
            FighterSide.RED,
        ),
    ],
)
def test_ground_defender_cannot_have_submission_probability(
    owner: FighterSide,
    defender: FighterSide,
) -> None:
    red_submission = (
        0.10
        if defender is FighterSide.RED
        else 0.0
    )
    blue_submission = (
        0.10
        if defender is FighterSide.BLUE
        else 0.0
    )

    with pytest.raises(
        ValueError,
        match=(
            "ground defender cannot have submission probability"
        ),
    ):
        SegmentFinishProbabilities(
            state=shared_state(
                phase=FightPhase.GROUND,
                owner=owner,
                position_quality=0.50,
            ),
            red=fighter_finish_probabilities(
                submission_probability=red_submission,
            ),
            blue=fighter_finish_probabilities(
                submission_probability=blue_submission,
            ),
        )


@pytest.mark.parametrize(
    "owner",
    [
        FighterSide.RED,
        FighterSide.BLUE,
    ],
)
def test_ground_owner_can_have_submission_probability(
    owner: FighterSide,
) -> None:
    selected = SegmentFinishProbabilities(
        state=shared_state(
            phase=FightPhase.GROUND,
            owner=owner,
            position_quality=0.50,
        ),
        red=fighter_finish_probabilities(
            submission_probability=(
                0.10
                if owner is FighterSide.RED
                else 0.0
            ),
        ),
        blue=fighter_finish_probabilities(
            submission_probability=(
                0.10
                if owner is FighterSide.BLUE
                else 0.0
            ),
        ),
    )

    owner_probabilities = (
        selected.red
        if owner is FighterSide.RED
        else selected.blue
    )

    assert owner_probabilities.submission_probability == 0.10


def test_segment_finish_probabilities_are_immutable() -> None:
    selected = SegmentFinishProbabilities(
        state=shared_state(
            phase=FightPhase.DISTANCE,
        ),
        red=fighter_finish_probabilities(
            submission_probability=0.0,
        ),
        blue=fighter_finish_probabilities(
            submission_probability=0.0,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        selected.red = fighter_finish_probabilities(
            submission_probability=0.0,
        )


def test_empty_event_opportunities_return_zero() -> None:
    result = combine_independent_event_probabilities(
        ()
    )

    assert result == pytest.approx(0.0)


def test_zero_event_counts_return_zero() -> None:
    result = combine_independent_event_probabilities(
        (
            (
                0,
                0.80,
            ),
            (
                0,
                0.50,
            ),
        )
    )

    assert result == pytest.approx(0.0)


def test_independent_event_probabilities_use_exact_arithmetic() -> None:
    result = combine_independent_event_probabilities(
        (
            (
                2,
                0.10,
            ),
            (
                1,
                0.20,
            ),
        )
    )

    assert result == pytest.approx(
        1.0 - (0.90 ** 2 * 0.80)
    )


def test_certain_event_returns_probability_one() -> None:
    result = combine_independent_event_probabilities(
        (
            (
                1,
                1.0,
            ),
        )
    )

    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    "invalid_count",
    [
        1.0,
        "1",
    ],
)
def test_event_count_must_be_integer(
    invalid_count: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="event_count must be an integer",
    ):
        combine_independent_event_probabilities(
            (
                (
                    invalid_count,
                    0.10,
                ),
            )
        )


def test_event_count_cannot_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="event_count cannot be negative",
    ):
        combine_independent_event_probabilities(
            (
                (
                    -1,
                    0.10,
                ),
            )
        )


def test_event_probability_must_be_numeric() -> None:
    with pytest.raises(
        TypeError,
        match="event_probability must be numeric",
    ):
        combine_independent_event_probabilities(
            (
                (
                    1,
                    "invalid",
                ),
            )
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        float("nan"),
        float("inf"),
    ],
)
def test_event_probability_must_be_finite(
    invalid_probability: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="event_probability must be finite",
    ):
        combine_independent_event_probabilities(
            (
                (
                    1,
                    invalid_probability,
                ),
            )
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.01,
        1.01,
    ],
)
def test_event_probability_must_be_in_unit_interval(
    invalid_probability: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="event_probability must be between 0 and 1",
    ):
        combine_independent_event_probabilities(
            (
                (
                    1,
                    invalid_probability,
                ),
            )
        )


def test_fresh_defender_has_neutral_state_amplifier() -> None:
    result = calculate_defender_state_amplifier(
        fighter_state(),
        fatigue_amplifier=1.0,
        damage_amplifier=1.0,
        acute_stress_amplifier=1.0,
    )

    assert result == pytest.approx(1.0)


def test_defender_state_amplifier_uses_exact_arithmetic() -> None:
    result = calculate_defender_state_amplifier(
        fighter_state(
            fatigue=0.50,
            damage=0.40,
            acute_stress=0.30,
        ),
        fatigue_amplifier=0.20,
        damage_amplifier=0.50,
        acute_stress_amplifier=0.10,
    )

    assert result == pytest.approx(
        1.0
        + 0.50 * 0.20
        + 0.40 * 0.50
        + 0.30 * 0.10
    )


def test_defender_state_amplifier_can_reach_four() -> None:
    result = calculate_defender_state_amplifier(
        fighter_state(
            fatigue=1.0,
            damage=1.0,
            acute_stress=1.0,
        ),
        fatigue_amplifier=1.0,
        damage_amplifier=1.0,
        acute_stress_amplifier=1.0,
    )

    assert result == pytest.approx(4.0)


def test_defender_state_amplifier_requires_dynamic_state() -> None:
    with pytest.raises(
        TypeError,
        match="defender_state must be FighterDynamicState",
    ):
        calculate_defender_state_amplifier(
            "invalid",
            fatigue_amplifier=0.20,
            damage_amplifier=0.50,
            acute_stress_amplifier=0.10,
        )


@pytest.mark.parametrize(
    "field_name",
    AMPLIFIER_FIELDS,
)
def test_defender_state_amplifiers_must_be_numeric(
    field_name: str,
) -> None:
    values = {
        "fatigue_amplifier": 0.20,
        "damage_amplifier": 0.50,
        "acute_stress_amplifier": 0.10,
    }
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        calculate_defender_state_amplifier(
            fighter_state(),
            **values,
        )


@pytest.mark.parametrize(
    "field_name",
    AMPLIFIER_FIELDS,
)
def test_defender_state_amplifiers_must_be_finite(
    field_name: str,
) -> None:
    values = {
        "fatigue_amplifier": 0.20,
        "damage_amplifier": 0.50,
        "acute_stress_amplifier": 0.10,
    }
    values[field_name] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        calculate_defender_state_amplifier(
            fighter_state(),
            **values,
        )


@pytest.mark.parametrize(
    "field_name",
    AMPLIFIER_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
    ],
)
def test_defender_state_amplifiers_must_be_in_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    values = {
        "fatigue_amplifier": 0.20,
        "damage_amplifier": 0.50,
        "acute_stress_amplifier": 0.10,
    }
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        calculate_defender_state_amplifier(
            fighter_state(),
            **values,
        )


def test_distance_knockout_probability_uses_landed_and_knockdowns() -> None:
    result = calculate_knockout_finish_probability(
        distance_activity(
            red_landed=2,
            red_knockdowns=1,
        ),
        FighterSide.RED,
        fighter_state(),
        knockout_calibration(),
    )

    assert result == pytest.approx(
        1.0 - (0.90 ** 2 * 0.80)
    )


def test_zero_offense_returns_zero_knockout_probability() -> None:
    result = calculate_knockout_finish_probability(
        distance_activity(),
        FighterSide.RED,
        fighter_state(
            fatigue=1.0,
            damage=1.0,
            acute_stress=1.0,
        ),
        knockout_calibration(
            defender_fatigue_amplifier=1.0,
            defender_damage_amplifier=1.0,
            defender_acute_stress_amplifier=1.0,
        ),
    )

    assert result == pytest.approx(0.0)


def test_clinch_knockout_probability_uses_phase_events() -> None:
    result = calculate_knockout_finish_probability(
        clinch_activity(
            red_landed=3,
            red_damaging=1,
        ),
        FighterSide.RED,
        fighter_state(),
        knockout_calibration(),
    )

    assert result == pytest.approx(
        1.0 - (0.95 ** 3 * 0.80)
    )


def test_ground_knockout_probability_uses_ground_strikes() -> None:
    result = calculate_knockout_finish_probability(
        ground_activity(
            red_landed=4,
        ),
        FighterSide.RED,
        fighter_state(),
        knockout_calibration(),
    )

    assert result == pytest.approx(
        1.0 - 0.90 ** 4
    )


def test_knockout_probability_uses_defender_state_amplifier() -> None:
    result = calculate_knockout_finish_probability(
        distance_activity(
            red_landed=1,
        ),
        FighterSide.RED,
        fighter_state(
            fatigue=0.50,
            damage=0.40,
            acute_stress=0.30,
        ),
        knockout_calibration(
            defender_fatigue_amplifier=0.20,
            defender_damage_amplifier=0.50,
            defender_acute_stress_amplifier=0.10,
        ),
    )

    assert result == pytest.approx(
        0.10 * 1.33
    )


def test_knockout_probability_respects_segment_cap() -> None:
    result = calculate_knockout_finish_probability(
        distance_activity(
            red_landed=1,
        ),
        FighterSide.RED,
        fighter_state(),
        knockout_calibration(
            distance_landed_probability=0.80,
            maximum_segment_probability=0.50,
        ),
    )

    assert result == pytest.approx(0.50)


def test_knockout_probability_selects_attacker_activity() -> None:
    activity = distance_activity(
        red_landed=1,
        blue_landed=3,
    )
    calibration = knockout_calibration()

    red = calculate_knockout_finish_probability(
        activity,
        FighterSide.RED,
        fighter_state(),
        calibration,
    )
    blue = calculate_knockout_finish_probability(
        activity,
        FighterSide.BLUE,
        fighter_state(),
        calibration,
    )

    assert red == pytest.approx(0.10)
    assert blue == pytest.approx(
        1.0 - 0.90 ** 3
    )


def test_knockout_calculation_requires_supported_activity() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "activity must be a supported phase segment activity"
        ),
    ):
        calculate_knockout_finish_probability(
            "invalid",
            FighterSide.RED,
            fighter_state(),
            knockout_calibration(),
        )


def test_knockout_calculation_requires_fighter_side() -> None:
    with pytest.raises(
        TypeError,
        match="attacker must be FighterSide",
    ):
        calculate_knockout_finish_probability(
            distance_activity(),
            "red",
            fighter_state(),
            knockout_calibration(),
        )


def test_knockout_calculation_requires_defender_state() -> None:
    with pytest.raises(
        TypeError,
        match="defender_state must be FighterDynamicState",
    ):
        calculate_knockout_finish_probability(
            distance_activity(),
            FighterSide.RED,
            "invalid",
            knockout_calibration(),
        )


def test_knockout_calculation_requires_calibration() -> None:
    with pytest.raises(
        TypeError,
        match="calibration must be KnockoutFinishCalibration",
    ):
        calculate_knockout_finish_probability(
            distance_activity(),
            FighterSide.RED,
            fighter_state(),
            "invalid",
        )


def test_zero_submission_attempts_return_zero() -> None:
    result = calculate_submission_finish_probability(
        ground_activity(
            red_submission_attempts=0,
        ),
        FighterSide.RED,
        fighter_state(),
        0.25,
        submission_calibration(),
    )

    assert result == pytest.approx(0.0)


def test_submission_probability_uses_exact_attempt_arithmetic() -> None:
    result = calculate_submission_finish_probability(
        ground_activity(
            red_submission_attempts=2,
            position_quality=0.50,
        ),
        FighterSide.RED,
        fighter_state(),
        0.25,
        submission_calibration(),
    )

    probability_per_attempt = (
        0.10
        * (1.0 + 0.50 * 0.40)
        * (1.0 - 0.25)
    )

    assert result == pytest.approx(
        1.0 - (
            1.0 - probability_per_attempt
        ) ** 2
    )


def test_position_quality_increases_submission_probability() -> None:
    low_quality = calculate_submission_finish_probability(
        ground_activity(
            red_submission_attempts=1,
            position_quality=0.0,
        ),
        FighterSide.RED,
        fighter_state(),
        0.0,
        submission_calibration(),
    )
    high_quality = calculate_submission_finish_probability(
        ground_activity(
            red_submission_attempts=1,
            position_quality=1.0,
        ),
        FighterSide.RED,
        fighter_state(),
        0.0,
        submission_calibration(),
    )

    assert low_quality == pytest.approx(0.10)
    assert high_quality == pytest.approx(0.14)
    assert high_quality > low_quality


def test_submission_defense_effect_respects_floor() -> None:
    result = calculate_submission_finish_probability(
        ground_activity(
            red_submission_attempts=1,
            position_quality=0.0,
        ),
        FighterSide.RED,
        fighter_state(),
        1.0,
        submission_calibration(
            minimum_submission_defense_effect_multiplier=0.20,
        ),
    )

    assert result == pytest.approx(
        0.10 * 0.20
    )


def test_submission_probability_uses_defender_state_amplifier() -> None:
    result = calculate_submission_finish_probability(
        ground_activity(
            red_submission_attempts=1,
            position_quality=0.0,
        ),
        FighterSide.RED,
        fighter_state(
            fatigue=0.50,
            damage=0.40,
            acute_stress=0.30,
        ),
        0.0,
        submission_calibration(
            defender_fatigue_amplifier=0.20,
            defender_damage_amplifier=0.50,
            defender_acute_stress_amplifier=0.10,
        ),
    )

    assert result == pytest.approx(
        0.10 * 1.33
    )


def test_submission_probability_respects_attempt_cap() -> None:
    result = calculate_submission_finish_probability(
        ground_activity(
            red_submission_attempts=1,
            position_quality=1.0,
        ),
        FighterSide.RED,
        fighter_state(
            fatigue=1.0,
            damage=1.0,
            acute_stress=1.0,
        ),
        0.0,
        submission_calibration(
            base_probability_per_attempt=0.80,
            position_quality_amplifier=1.0,
            defender_fatigue_amplifier=1.0,
            defender_damage_amplifier=1.0,
            defender_acute_stress_amplifier=1.0,
            maximum_probability_per_attempt=0.40,
        ),
    )

    assert result == pytest.approx(0.40)


def test_submission_probability_respects_segment_cap() -> None:
    result = calculate_submission_finish_probability(
        ground_activity(
            red_submission_attempts=5,
            position_quality=0.0,
        ),
        FighterSide.RED,
        fighter_state(),
        0.0,
        submission_calibration(
            base_probability_per_attempt=0.40,
            maximum_probability_per_attempt=0.40,
            maximum_segment_probability=0.60,
        ),
    )

    assert result == pytest.approx(0.60)


def test_submission_probability_selects_blue_ground_owner() -> None:
    result = calculate_submission_finish_probability(
        ground_activity(
            owner=FighterSide.BLUE,
            blue_submission_attempts=2,
            position_quality=0.0,
        ),
        FighterSide.BLUE,
        fighter_state(),
        0.0,
        submission_calibration(),
    )

    assert result == pytest.approx(
        1.0 - 0.90 ** 2
    )


def test_submission_calculation_requires_ground_activity() -> None:
    with pytest.raises(
        TypeError,
        match="activity must be GroundSegmentActivity",
    ):
        calculate_submission_finish_probability(
            distance_activity(),
            FighterSide.RED,
            fighter_state(),
            0.25,
            submission_calibration(),
        )


def test_submission_calculation_requires_fighter_side() -> None:
    with pytest.raises(
        TypeError,
        match="attacker must be FighterSide",
    ):
        calculate_submission_finish_probability(
            ground_activity(),
            "red",
            fighter_state(),
            0.25,
            submission_calibration(),
        )


def test_submission_attacker_must_own_ground_phase() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "submission attacker must own the ground phase"
        ),
    ):
        calculate_submission_finish_probability(
            ground_activity(
                owner=FighterSide.RED,
            ),
            FighterSide.BLUE,
            fighter_state(),
            0.25,
            submission_calibration(),
        )


def test_submission_calculation_requires_defender_state() -> None:
    with pytest.raises(
        TypeError,
        match="defender_state must be FighterDynamicState",
    ):
        calculate_submission_finish_probability(
            ground_activity(),
            FighterSide.RED,
            "invalid",
            0.25,
            submission_calibration(),
        )


def test_effective_submission_defense_must_be_numeric() -> None:
    with pytest.raises(
        TypeError,
        match="effective_submission_defense must be numeric",
    ):
        calculate_submission_finish_probability(
            ground_activity(),
            FighterSide.RED,
            fighter_state(),
            "invalid",
            submission_calibration(),
        )


@pytest.mark.parametrize(
    "invalid_defense",
    [
        float("nan"),
        float("inf"),
    ],
)
def test_effective_submission_defense_must_be_finite(
    invalid_defense: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="effective_submission_defense must be finite",
    ):
        calculate_submission_finish_probability(
            ground_activity(),
            FighterSide.RED,
            fighter_state(),
            invalid_defense,
            submission_calibration(),
        )


@pytest.mark.parametrize(
    "invalid_defense",
    [
        -0.01,
        1.01,
    ],
)
def test_effective_submission_defense_must_be_in_unit_interval(
    invalid_defense: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "effective_submission_defense must be between 0 and 1"
        ),
    ):
        calculate_submission_finish_probability(
            ground_activity(),
            FighterSide.RED,
            fighter_state(),
            invalid_defense,
            submission_calibration(),
        )


def test_submission_calculation_requires_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "calibration must be SubmissionFinishCalibration"
        ),
    ):
        calculate_submission_finish_probability(
            ground_activity(),
            FighterSide.RED,
            fighter_state(),
            0.25,
            "invalid",
        )


def test_segment_distance_probabilities_map_both_fighters() -> None:
    result = calculate_segment_finish_probabilities(
        distance_activity(
            red_landed=2,
            red_knockdowns=1,
            blue_landed=1,
        ),
        fight_state(),
        phase_parameters(),
        phase_parameters(),
        finish_calibration(),
    )

    assert result.red.ko_tko_probability == pytest.approx(
        1.0 - (0.90 ** 2 * 0.80)
    )
    assert result.blue.ko_tko_probability == pytest.approx(0.10)

    assert result.red.submission_probability == 0.0
    assert result.blue.submission_probability == 0.0


def test_segment_clinch_probabilities_map_both_fighters() -> None:
    result = calculate_segment_finish_probabilities(
        clinch_activity(
            red_landed=2,
            red_damaging=1,
            blue_landed=1,
        ),
        fight_state(),
        phase_parameters(),
        phase_parameters(),
        finish_calibration(),
    )

    assert result.red.ko_tko_probability == pytest.approx(
        1.0 - (0.95 ** 2 * 0.80)
    )
    assert result.blue.ko_tko_probability == pytest.approx(0.05)

    assert result.red.submission_probability == 0.0
    assert result.blue.submission_probability == 0.0


def test_segment_ground_red_owner_receives_submission_probability() -> None:
    result = calculate_segment_finish_probabilities(
        ground_activity(
            owner=FighterSide.RED,
            red_landed=1,
            red_submission_attempts=2,
            position_quality=0.50,
        ),
        fight_state(),
        phase_parameters(),
        phase_parameters(
            submission_defense=0.25,
        ),
        finish_calibration(),
    )

    probability_per_attempt = (
        0.10
        * (1.0 + 0.50 * 0.40)
        * (1.0 - 0.25)
    )

    assert result.red.ko_tko_probability == pytest.approx(0.10)
    assert result.red.submission_probability == pytest.approx(
        1.0 - (
            1.0 - probability_per_attempt
        ) ** 2
    )

    assert result.blue.ko_tko_probability == 0.0
    assert result.blue.submission_probability == 0.0


def test_segment_ground_blue_owner_receives_submission_probability() -> None:
    result = calculate_segment_finish_probabilities(
        ground_activity(
            owner=FighterSide.BLUE,
            blue_landed=1,
            blue_submission_attempts=2,
            position_quality=0.50,
        ),
        fight_state(),
        phase_parameters(
            submission_defense=0.25,
        ),
        phase_parameters(),
        finish_calibration(),
    )

    probability_per_attempt = (
        0.10
        * (1.0 + 0.50 * 0.40)
        * (1.0 - 0.25)
    )

    assert result.blue.ko_tko_probability == pytest.approx(0.10)
    assert result.blue.submission_probability == pytest.approx(
        1.0 - (
            1.0 - probability_per_attempt
        ) ** 2
    )

    assert result.red.ko_tko_probability == 0.0
    assert result.red.submission_probability == 0.0


def test_segment_knockout_uses_opposing_defender_state() -> None:
    result = calculate_segment_finish_probabilities(
        distance_activity(
            red_landed=1,
            blue_landed=1,
        ),
        fight_state(
            red=fighter_state(
                damage=0.0,
            ),
            blue=fighter_state(
                damage=1.0,
            ),
        ),
        phase_parameters(),
        phase_parameters(),
        finish_calibration(
            knockout=knockout_calibration(
                defender_damage_amplifier=1.0,
            ),
        ),
    )

    assert result.red.ko_tko_probability == pytest.approx(0.20)
    assert result.blue.ko_tko_probability == pytest.approx(0.10)


def test_segment_submission_uses_defender_effective_defense() -> None:
    result = calculate_segment_finish_probabilities(
        ground_activity(
            owner=FighterSide.RED,
            red_submission_attempts=1,
            position_quality=0.0,
        ),
        fight_state(),
        phase_parameters(
            submission_defense=0.0,
        ),
        phase_parameters(
            submission_defense=0.80,
        ),
        finish_calibration(
            submission=submission_calibration(
                position_quality_amplifier=0.0,
                minimum_submission_defense_effect_multiplier=0.10,
            ),
        ),
    )

    assert result.red.submission_probability == pytest.approx(
        0.10 * 0.20
    )


def test_segment_finish_calculation_requires_supported_activity() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "activity must be a supported phase segment activity"
        ),
    ):
        calculate_segment_finish_probabilities(
            "invalid",
            fight_state(),
            phase_parameters(),
            phase_parameters(),
            finish_calibration(),
        )


def test_segment_finish_calculation_requires_dynamic_state() -> None:
    with pytest.raises(
        TypeError,
        match="dynamic_state must be FightDynamicState",
    ):
        calculate_segment_finish_probabilities(
            distance_activity(),
            "invalid",
            phase_parameters(),
            phase_parameters(),
            finish_calibration(),
        )


def test_segment_finish_calculation_requires_red_phase_parameters() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "red_effective_phase must be FighterPhaseParameters"
        ),
    ):
        calculate_segment_finish_probabilities(
            distance_activity(),
            fight_state(),
            "invalid",
            phase_parameters(),
            finish_calibration(),
        )


def test_segment_finish_calculation_requires_blue_phase_parameters() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "blue_effective_phase must be FighterPhaseParameters"
        ),
    ):
        calculate_segment_finish_probabilities(
            distance_activity(),
            fight_state(),
            phase_parameters(),
            "invalid",
            finish_calibration(),
        )


def test_segment_finish_calculation_requires_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "calibration must be FinishProbabilityCalibration"
        ),
    ):
        calculate_segment_finish_probabilities(
            distance_activity(),
            fight_state(),
            phase_parameters(),
            phase_parameters(),
            "invalid",
        )


def test_attacker_power_scales_only_generic_landed_ko_hazard() -> None:
    """Power decay scales landed conversion without double-scaling KD bonus."""

    activity = distance_activity(
        red_landed=1,
        red_knockdowns=1,
    )

    calibration = knockout_calibration(
        distance_landed_probability=0.10,
        distance_knockdown_probability=0.20,
        maximum_segment_probability=1.0,
    )

    full_power = calculate_knockout_finish_probability(
        activity,
        FighterSide.RED,
        fighter_state(),
        calibration,
        attacker_power_multiplier=1.0,
    )

    half_power = calculate_knockout_finish_probability(
        activity,
        FighterSide.RED,
        fighter_state(),
        calibration,
        attacker_power_multiplier=0.5,
    )

    # Full power:
    #   1 - (1 - 0.10) * (1 - 0.20) = 0.28
    #
    # Half power:
    #   landed hazard becomes 0.05
    #   KD bonus remains 0.20
    #   1 - (1 - 0.05) * (1 - 0.20) = 0.24
    assert full_power == pytest.approx(0.28)
    assert half_power == pytest.approx(0.24)
