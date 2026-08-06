"""Tests for V2 transition and phase-specific parameter contracts."""

from dataclasses import replace

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


def make_transition_parameters() -> FighterTransitionParameters:
    """Return a valid neutral transition profile."""

    return FighterTransitionParameters(
        distance_retention=0.50,
        clinch_entry_tendency=0.50,
        clinch_entry_resistance=0.50,
        takedown_entry_tendency=0.50,
        takedown_completion_ability=0.50,
        takedown_resistance=0.50,
        takedown_persistence=0.50,
        failed_takedown_persistence=0.50,
        clinch_retention=0.50,
        clinch_escape_ability=0.50,
        ground_retention=0.50,
        ground_escape_ability=0.50,
        reversal_ability=0.50,
        phase_imposition=0.50,
        phase_resistance=0.50,
    )


def make_phase_parameters() -> FighterPhaseParameters:
    """Return a valid phase-specific activity profile."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=3.0,
            sig_strike_accuracy=0.45,
            knockdown_probability_per_landed=0.02,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=1.5,
            clinch_strike_accuracy=0.50,
            control_seconds_mean=8.0,
            damaging_clinch_probability=0.08,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=2.0,
            ground_strike_accuracy=0.52,
            control_seconds_mean=15.0,
            submission_attempt_rate=0.20,
            position_advancement_probability=0.25,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.20,
            reversal_attempt_rate=0.08,
            scramble_attempt_rate=0.15,
            submission_defense=0.70,
        ),
    )


def test_valid_transition_parameters_are_accepted() -> None:
    parameters = make_transition_parameters()

    assert parameters.phase_imposition == 0.50
    assert parameters.ground_escape_ability == 0.50


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("distance_retention", -0.01),
        ("takedown_persistence", 1.01),
        ("phase_imposition", float("nan")),
        ("phase_resistance", float("inf")),
    ],
)
def test_transition_parameters_are_bounded(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        replace(
            make_transition_parameters(),
            **{field_name: value},
        )


def test_valid_phase_parameter_bundle_is_accepted() -> None:
    parameters = make_phase_parameters()

    assert parameters.distance.sig_strike_attempt_rate == 3.0
    assert parameters.ground_owner.submission_attempt_rate == 0.20


@pytest.mark.parametrize(
    ("component", "field_name"),
    [
        ("distance", "sig_strike_attempt_rate"),
        ("clinch", "clinch_strike_attempt_rate"),
        ("ground_owner", "ground_strike_attempt_rate"),
        ("ground_defender", "escape_attempt_rate"),
    ],
)
def test_phase_rates_cannot_be_negative(
    component: str,
    field_name: str,
) -> None:
    parameters = make_phase_parameters()
    original = getattr(parameters, component)

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        replace(
            original,
            **{field_name: -0.01},
        )


@pytest.mark.parametrize(
    ("component", "field_name"),
    [
        ("distance", "sig_strike_accuracy"),
        ("clinch", "damaging_clinch_probability"),
        ("ground_owner", "position_advancement_probability"),
        ("ground_defender", "submission_defense"),
    ],
)
def test_phase_probabilities_are_bounded(
    component: str,
    field_name: str,
) -> None:
    parameters = make_phase_parameters()
    original = getattr(parameters, component)

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        replace(
            original,
            **{field_name: 1.01},
        )


@pytest.mark.parametrize(
    ("component", "value"),
    [
        ("clinch", 30.01),
        ("ground_owner", -0.01),
    ],
)
def test_control_seconds_stay_within_segment(
    component: str,
    value: float,
) -> None:
    parameters = make_phase_parameters()
    original = getattr(parameters, component)

    with pytest.raises(
        ValueError,
        match="control_seconds_mean",
    ):
        replace(
            original,
            control_seconds_mean=value,
        )
