"""Tests for the simulator-oriented RFS Phase Baseline builder."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pipeline.round_stats.build_round_fighter_phase_baseline import (
    PhaseBaselineBuildResult,
    RoundFighterPhaseBaselineBuildError,
    add_prior_phase_baseline_state,
    build_fight_level_observations,
    build_latest_phase_baseline_state,
    build_round_fighter_phase_baseline,
    cumulative_prior_total,
    evidence_state_name,
    ols_slope,
    prior_total_name,
    safe_scalar_div,
    standardize_round_stats,
)
from pipeline.round_stats.rfs_phase_baseline_feature_contracts import (
    PHASE_BASELINE_EVIDENCE_SPECS,
    PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS,
)


def round_row(
    *,
    event_id="event-1",
    event_name="UFC Test",
    event_date="2025-01-01",
    fight_id="fight-1",
    corner="red",
    fighter_id="fighter-a",
    fighter_name="Fighter A",
    opponent_id="fighter-b",
    opponent_name="Fighter B",
    division="Lightweight",
    total_rounds=3,
    round_number=1,
    sig_str_attempted=20,
    distance_attempted=12,
    clinch_attempted=3,
    ground_attempted=5,
    td_landed=1,
    td_attempted=2,
    ctrl_sec=60,
):
    """Return one authoritative-style round-stat row."""

    return {
        "event_id": event_id,
        "event_name": event_name,
        "event_date": event_date,
        "fight_id": fight_id,
        "corner": corner,
        "fighter_id": fighter_id,
        "fighter_name": fighter_name,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,
        "division": division,
        "total_rounds": total_rounds,
        "round": round_number,
        "sig_str_attempted": sig_str_attempted,
        "distance_attempted": distance_attempted,
        "clinch_attempted": clinch_attempted,
        "ground_attempted": ground_attempted,
        "td_landed": td_landed,
        "td_attempted": td_attempted,
        "ctrl_sec": ctrl_sec,
    }


def two_round_fixture() -> pd.DataFrame:
    """Return two rounds with simple, auditable totals."""

    return pd.DataFrame(
        [
            round_row(
                round_number=1,
                sig_str_attempted=20,
                distance_attempted=12,
                clinch_attempted=3,
                ground_attempted=5,
                td_landed=1,
                td_attempted=2,
                ctrl_sec=60,
            ),
            round_row(
                round_number=2,
                sig_str_attempted=30,
                distance_attempted=18,
                clinch_attempted=6,
                ground_attempted=6,
                td_landed=1,
                td_attempted=4,
                ctrl_sec=120,
            ),
        ]
    )


def three_fight_fixture() -> pd.DataFrame:
    """Return three chronologically ordered fights for one fighter."""

    rows = []

    for index, date in enumerate(
        ["2025-01-01", "2025-02-01", "2025-03-01"],
        start=1,
    ):
        rows.append(
            round_row(
                event_id=f"event-{index}",
                event_name=f"UFC Test {index}",
                event_date=date,
                fight_id=f"fight-{index}",
                opponent_id=f"opponent-{index}",
                opponent_name=f"Opponent {index}",
                round_number=1,
                sig_str_attempted=10 * index,
                distance_attempted=5 * index,
                clinch_attempted=2 * index,
                ground_attempted=3 * index,
                td_landed=index - 1,
                td_attempted=index,
                ctrl_sec=30 * index,
            )
        )

    return pd.DataFrame(rows)


def standardized_two_round_fixture() -> pd.DataFrame:
    return standardize_round_stats(two_round_fixture())


def observations_from_three_fights() -> pd.DataFrame:
    standardized = standardize_round_stats(
        three_fight_fixture()
    )
    return build_fight_level_observations(standardized)


def test_safe_scalar_div_returns_ratio():
    assert safe_scalar_div(6.0, 3.0) == 2.0


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (1.0, 0.0),
        (np.nan, 1.0),
        (1.0, np.nan),
    ],
)
def test_safe_scalar_div_returns_nan_for_invalid_inputs(
    numerator,
    denominator,
):
    assert np.isnan(
        safe_scalar_div(numerator, denominator)
    )


def test_ols_slope_calculates_round_trend():
    slope = ols_slope(
        pd.Series([1, 2, 3]),
        pd.Series([1, 3, 5]),
    )

    assert slope == pytest.approx(2.0)


def test_ols_slope_requires_two_valid_points():
    assert np.isnan(
        ols_slope(
            pd.Series([1]),
            pd.Series([4]),
        )
    )


def test_cumulative_prior_total_excludes_current_value():
    result = cumulative_prior_total(
        pd.Series([2.0, 3.0, 5.0])
    )

    assert result.tolist() == [0.0, 2.0, 5.0]


def test_state_name_helpers_use_locked_convention():
    fight_name = (
        "rfs_phase_base_fight_td_attempts"
    )

    assert prior_total_name(fight_name) == (
        "rfs_phase_base_prior_total_td_attempts"
    )
    assert evidence_state_name(
        fight_name,
        "exp",
    ) == "rfs_phase_base_exp_td_attempts"


def test_evidence_state_name_rejects_unknown_state_kind():
    with pytest.raises(
        ValueError,
        match="state_kind must be",
    ):
        evidence_state_name(
            "rfs_phase_base_fight_td_attempts",
            "unknown",
        )


def test_standardize_uses_event_date_alias():
    standardized = standardize_round_stats(
        two_round_fixture()
    )

    assert "date" in standardized.columns
    assert pd.api.types.is_datetime64_any_dtype(
        standardized["date"]
    )


def test_standardize_normalizes_corner():
    source = two_round_fixture()
    source["corner"] = " RED "

    standardized = standardize_round_stats(source)

    assert set(standardized["corner"]) == {"red"}


def test_standardize_does_not_mutate_input():
    source = two_round_fixture()
    original = source.copy(deep=True)

    standardize_round_stats(source)

    pdt.assert_frame_equal(source, original)


def test_standardize_rejects_missing_date():
    source = two_round_fixture().drop(
        columns=["event_date"]
    )

    with pytest.raises(
        RoundFighterPhaseBaselineBuildError,
        match="must include date or event_date",
    ):
        standardize_round_stats(source)


def test_standardize_rejects_invalid_date():
    source = two_round_fixture()
    source["event_date"] = "not-a-date"

    with pytest.raises(
        RoundFighterPhaseBaselineBuildError,
        match="invalid dates",
    ):
        standardize_round_stats(source)


def test_standardize_rejects_missing_required_column():
    source = two_round_fixture().drop(
        columns=["distance_attempted"]
    )

    with pytest.raises(
        RoundFighterPhaseBaselineBuildError,
        match="missing required columns",
    ):
        standardize_round_stats(source)


def test_standardize_rejects_invalid_corner():
    source = two_round_fixture()
    source["corner"] = "green"

    with pytest.raises(
        RoundFighterPhaseBaselineBuildError,
        match="invalid corners",
    ):
        standardize_round_stats(source)


def test_standardize_rejects_negative_counts():
    source = two_round_fixture()
    source.loc[0, "td_attempted"] = -1

    with pytest.raises(
        RoundFighterPhaseBaselineBuildError,
        match="negative values",
    ):
        standardize_round_stats(source)


def test_standardize_rejects_landed_above_attempted():
    source = two_round_fixture()
    source.loc[0, "td_landed"] = 3
    source.loc[0, "td_attempted"] = 2

    with pytest.raises(
        RoundFighterPhaseBaselineBuildError,
        match="td_landed cannot exceed td_attempted",
    ):
        standardize_round_stats(source)


def test_standardize_rejects_duplicate_fighter_round():
    source = two_round_fixture()
    source = pd.concat(
        [source, source.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        RoundFighterPhaseBaselineBuildError,
        match="duplicate",
    ):
        standardize_round_stats(source)


def test_fight_observation_contains_all_locked_features():
    observations = build_fight_level_observations(
        standardized_two_round_fixture()
    )

    for column in PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS:
        assert column in observations.columns


def test_fight_observation_has_one_fighter_fight_row():
    observations = build_fight_level_observations(
        standardized_two_round_fixture()
    )

    assert len(observations) == 1
    assert not observations.duplicated(
        ["fight_id", "fighter_id"]
    ).any()


def test_fight_aggregate_totals_are_correct():
    row = build_fight_level_observations(
        standardized_two_round_fixture()
    ).iloc[0]

    assert row[
        "rfs_phase_base_fight_rounds_observed"
    ] == 2
    assert row[
        "rfs_phase_base_fight_sig_strike_attempts"
    ] == 50
    assert row[
        "rfs_phase_base_fight_distance_attempts"
    ] == 30
    assert row[
        "rfs_phase_base_fight_clinch_attempts"
    ] == 9
    assert row[
        "rfs_phase_base_fight_ground_attempts"
    ] == 11
    assert row[
        "rfs_phase_base_fight_td_attempts"
    ] == 6
    assert row[
        "rfs_phase_base_fight_td_landed"
    ] == 2
    assert row[
        "rfs_phase_base_fight_failed_td_attempts"
    ] == 4
    assert row[
        "rfs_phase_base_fight_control_seconds"
    ] == 180


def test_per_round_evidence_is_correct():
    row = build_fight_level_observations(
        standardized_two_round_fixture()
    ).iloc[0]

    assert row[
        "rfs_phase_base_fight_distance_attempts_per_round"
    ] == 15
    assert row[
        "rfs_phase_base_fight_clinch_attempts_per_round"
    ] == 4.5
    assert row[
        "rfs_phase_base_fight_ground_attempts_per_round"
    ] == 5.5
    assert row[
        "rfs_phase_base_fight_td_attempts_per_round"
    ] == 3
    assert row[
        "rfs_phase_base_fight_control_seconds_per_round"
    ] == 90


def test_phase_attempt_shares_are_correct():
    row = build_fight_level_observations(
        standardized_two_round_fixture()
    ).iloc[0]

    assert row[
        "rfs_phase_base_fight_distance_attempt_share"
    ] == pytest.approx(30 / 50)
    assert row[
        "rfs_phase_base_fight_clinch_attempt_share"
    ] == pytest.approx(9 / 50)
    assert row[
        "rfs_phase_base_fight_ground_attempt_share"
    ] == pytest.approx(11 / 50)


def test_takedown_evidence_is_correct():
    row = build_fight_level_observations(
        standardized_two_round_fixture()
    ).iloc[0]

    assert row[
        "rfs_phase_base_fight_td_completion_rate"
    ] == pytest.approx(2 / 6)
    assert row[
        "rfs_phase_base_fight_failed_td_attempts_per_round"
    ] == 2
    assert row[
        "rfs_phase_base_fight_control_seconds_per_td_attempt"
    ] == 30
    assert row[
        "rfs_phase_base_fight_control_seconds_per_td_landed"
    ] == 90


def test_non_distance_shares_are_complements():
    row = build_fight_level_observations(
        standardized_two_round_fixture()
    ).iloc[0]

    clinch_share = row[
        "rfs_phase_base_fight_non_distance_clinch_share"
    ]
    ground_share = row[
        "rfs_phase_base_fight_non_distance_ground_share"
    ]

    assert clinch_share == pytest.approx(9 / 20)
    assert ground_share == pytest.approx(11 / 20)
    assert clinch_share + ground_share == pytest.approx(1.0)


def test_control_minute_evidence_is_correct():
    row = build_fight_level_observations(
        standardized_two_round_fixture()
    ).iloc[0]

    assert row[
        "rfs_phase_base_fight_clinch_attempts_per_control_min"
    ] == pytest.approx(3.0)
    assert row[
        "rfs_phase_base_fight_ground_attempts_per_control_min"
    ] == pytest.approx(11 / 3)


def test_round_shape_evidence_is_correct():
    row = build_fight_level_observations(
        standardized_two_round_fixture()
    ).iloc[0]

    assert row[
        "rfs_phase_base_fight_td_attempt_slope"
    ] == pytest.approx(2.0)
    assert row[
        "rfs_phase_base_fight_td_persistence_ratio"
    ] == pytest.approx(2.0)
    assert row[
        "rfs_phase_base_fight_failed_td_attempt_slope"
    ] == pytest.approx(2.0)


def test_zero_opportunities_produce_nan_ratios():
    source = two_round_fixture()

    for column in [
        "sig_str_attempted",
        "distance_attempted",
        "clinch_attempted",
        "ground_attempted",
        "td_landed",
        "td_attempted",
        "ctrl_sec",
    ]:
        source[column] = 0

    row = build_fight_level_observations(
        standardize_round_stats(source)
    ).iloc[0]

    assert np.isnan(
        row[
            "rfs_phase_base_fight_td_completion_rate"
        ]
    )
    assert np.isnan(
        row[
            "rfs_phase_base_fight_distance_attempt_share"
        ]
    )
    assert np.isnan(
        row[
            "rfs_phase_base_fight_non_distance_clinch_share"
        ]
    )
    assert np.isnan(
        row[
            "rfs_phase_base_fight_clinch_attempts_per_control_min"
        ]
    )


def test_unit_interval_evidence_stays_in_range():
    observations = build_fight_level_observations(
        standardized_two_round_fixture()
    )

    for spec in PHASE_BASELINE_EVIDENCE_SPECS:
        if spec.unit_interval:
            values = observations[
                spec.feature_name
            ].dropna()

            assert values.between(0.0, 1.0).all()


def test_first_history_row_has_no_prior_state():
    history = add_prior_phase_baseline_state(
        observations_from_three_fights()
    )

    first = history.iloc[0]

    assert first[
        "rfs_phase_base_prior_fight_count"
    ] == 0
    assert first[
        "rfs_phase_base_prior_total_td_attempts"
    ] == 0
    assert first[
        "rfs_phase_base_has_state"
    ] == 0


def test_second_history_row_uses_only_first_fight():
    observations = observations_from_three_fights()
    history = add_prior_phase_baseline_state(
        observations
    )

    first = observations.iloc[0]
    second = history.iloc[1]

    assert second[
        "rfs_phase_base_prior_fight_count"
    ] == 1
    assert second[
        "rfs_phase_base_prior_total_td_attempts"
    ] == first[
        "rfs_phase_base_fight_td_attempts"
    ]
    assert second[
        "rfs_phase_base_exp_td_attempts_per_round"
    ] == first[
        "rfs_phase_base_fight_td_attempts_per_round"
    ]
    assert second[
        "rfs_phase_base_has_state"
    ] == 1


def test_third_history_row_has_cumulative_prior_totals():
    observations = observations_from_three_fights()
    history = add_prior_phase_baseline_state(
        observations
    )

    expected = observations.iloc[:2][
        "rfs_phase_base_fight_td_attempts"
    ].sum()

    assert history.iloc[2][
        "rfs_phase_base_prior_total_td_attempts"
    ] == expected


def test_current_fight_does_not_leak_into_prior_state():
    observations = observations_from_three_fights()
    history = add_prior_phase_baseline_state(
        observations
    )

    observations_changed = observations.copy()
    observations_changed.loc[
        2,
        "rfs_phase_base_fight_td_attempts",
    ] = 10000

    changed_history = add_prior_phase_baseline_state(
        observations_changed
    )

    state_columns = [
        column
        for column in history.columns
        if (
            column.startswith(
                "rfs_phase_base_prior_"
            )
            or column.startswith(
                "rfs_phase_base_exp_"
            )
            or column.startswith(
                "rfs_phase_base_last3_"
            )
            or column.startswith(
                "rfs_phase_base_ewm_"
            )
        )
    ]

    pdt.assert_series_equal(
        history.loc[2, state_columns],
        changed_history.loc[2, state_columns],
    )


def test_latest_state_uses_complete_fighter_history():
    observations = observations_from_three_fights()
    history = add_prior_phase_baseline_state(
        observations
    )
    latest = build_latest_phase_baseline_state(
        history
    )

    assert len(latest) == 1

    row = latest.iloc[0]

    assert row[
        "rfs_phase_base_prior_fight_count"
    ] == 3
    assert row[
        "rfs_phase_base_prior_total_td_attempts"
    ] == observations[
        "rfs_phase_base_fight_td_attempts"
    ].sum()
    assert row["latest_event_name"] == "UFC Test 3"
    assert row["rfs_phase_base_has_state"] == 1


def test_full_builder_returns_history_and_latest():
    result = build_round_fighter_phase_baseline(
        three_fight_fixture()
    )

    assert isinstance(result, PhaseBaselineBuildResult)
    assert len(result.history) == 3
    assert len(result.latest) == 1


def test_build_result_is_immutable():
    result = build_round_fighter_phase_baseline(
        three_fight_fixture()
    )

    with pytest.raises(FrozenInstanceError):
        result.history = pd.DataFrame()


def test_full_builder_is_deterministic():
    source = three_fight_fixture()

    first = build_round_fighter_phase_baseline(source)
    second = build_round_fighter_phase_baseline(source)

    pdt.assert_frame_equal(
        first.history,
        second.history,
    )
    pdt.assert_frame_equal(
        first.latest,
        second.latest,
    )
