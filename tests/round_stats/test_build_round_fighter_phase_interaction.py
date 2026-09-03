"""Focused tests for the RFS Phase Interaction builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.round_stats.build_round_fighter_phase_interaction import (
    RoundFighterPhaseInteractionBuildError,
    build_fight_level_observations,
    build_round_fighter_phase_interaction,
    evidence_state_name,
    prior_total_name,
    standardize_round_stats,
)
from pipeline.round_stats.rfs_phase_interaction_feature_contracts import (
    PHASE_INTERACTION_AGGREGATE_SPECS,
    PHASE_INTERACTION_EVIDENCE_SPECS,
    PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS,
)


def _round_row(
    *,
    fight_id: str,
    event_id: str,
    event_name: str,
    date: str,
    fighter_id: str,
    fighter_name: str,
    opponent_id: str,
    opponent_name: str,
    corner: str,
    round_number: int,
    sig_str_attempted: int,
    distance_landed: int,
    distance_attempted: int,
    clinch_landed: int,
    clinch_attempted: int,
    ground_landed: int,
    ground_attempted: int,
    td_landed: int,
    td_attempted: int,
    sub_att: int,
    rev: int,
    ctrl_sec: int,
) -> dict[str, object]:
    """Create one authoritative-style fighter-round row."""

    return {
        "event_id": event_id,
        "event_name": event_name,
        "date": date,
        "fight_id": fight_id,
        "corner": corner,
        "fighter_id": fighter_id,
        "fighter_name": fighter_name,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,
        "division": "Lightweight",
        "total_rounds": 3,
        "round": round_number,
        "sig_str_attempted": sig_str_attempted,
        "distance_landed": distance_landed,
        "distance_attempted": distance_attempted,
        "clinch_landed": clinch_landed,
        "clinch_attempted": clinch_attempted,
        "ground_landed": ground_landed,
        "ground_attempted": ground_attempted,
        "td_landed": td_landed,
        "td_attempted": td_attempted,
        "sub_att": sub_att,
        "rev": rev,
        "ctrl_sec": ctrl_sec,
    }


def _two_fight_round_stats() -> pd.DataFrame:
    """Return two reciprocal fights involving fighter A."""

    rows = [
        _round_row(
            fight_id="fight-1",
            event_id="event-1",
            event_name="Event One",
            date="2024-01-01",
            fighter_id="fighter-a",
            fighter_name="Fighter A",
            opponent_id="fighter-b",
            opponent_name="Fighter B",
            corner="red",
            round_number=1,
            sig_str_attempted=20,
            distance_landed=5,
            distance_attempted=10,
            clinch_landed=2,
            clinch_attempted=4,
            ground_landed=3,
            ground_attempted=6,
            td_landed=1,
            td_attempted=2,
            sub_att=1,
            rev=1,
            ctrl_sec=60,
        ),
        _round_row(
            fight_id="fight-1",
            event_id="event-1",
            event_name="Event One",
            date="2024-01-01",
            fighter_id="fighter-b",
            fighter_name="Fighter B",
            opponent_id="fighter-a",
            opponent_name="Fighter A",
            corner="blue",
            round_number=1,
            sig_str_attempted=16,
            distance_landed=3,
            distance_attempted=8,
            clinch_landed=1,
            clinch_attempted=3,
            ground_landed=2,
            ground_attempted=5,
            td_landed=1,
            td_attempted=4,
            sub_att=0,
            rev=0,
            ctrl_sec=30,
        ),
        _round_row(
            fight_id="fight-1",
            event_id="event-1",
            event_name="Event One",
            date="2024-01-01",
            fighter_id="fighter-a",
            fighter_name="Fighter A",
            opponent_id="fighter-b",
            opponent_name="Fighter B",
            corner="red",
            round_number=2,
            sig_str_attempted=30,
            distance_landed=6,
            distance_attempted=15,
            clinch_landed=3,
            clinch_attempted=6,
            ground_landed=4,
            ground_attempted=9,
            td_landed=1,
            td_attempted=3,
            sub_att=0,
            rev=0,
            ctrl_sec=90,
        ),
        _round_row(
            fight_id="fight-1",
            event_id="event-1",
            event_name="Event One",
            date="2024-01-01",
            fighter_id="fighter-b",
            fighter_name="Fighter B",
            opponent_id="fighter-a",
            opponent_name="Fighter A",
            corner="blue",
            round_number=2,
            sig_str_attempted=24,
            distance_landed=4,
            distance_attempted=12,
            clinch_landed=2,
            clinch_attempted=5,
            ground_landed=3,
            ground_attempted=7,
            td_landed=0,
            td_attempted=2,
            sub_att=1,
            rev=1,
            ctrl_sec=45,
        ),
        _round_row(
            fight_id="fight-2",
            event_id="event-2",
            event_name="Event Two",
            date="2024-02-01",
            fighter_id="fighter-a",
            fighter_name="Fighter A",
            opponent_id="fighter-c",
            opponent_name="Fighter C",
            corner="blue",
            round_number=1,
            sig_str_attempted=40,
            distance_landed=10,
            distance_attempted=20,
            clinch_landed=4,
            clinch_attempted=8,
            ground_landed=5,
            ground_attempted=12,
            td_landed=2,
            td_attempted=4,
            sub_att=2,
            rev=1,
            ctrl_sec=120,
        ),
        _round_row(
            fight_id="fight-2",
            event_id="event-2",
            event_name="Event Two",
            date="2024-02-01",
            fighter_id="fighter-c",
            fighter_name="Fighter C",
            opponent_id="fighter-a",
            opponent_name="Fighter A",
            corner="red",
            round_number=1,
            sig_str_attempted=25,
            distance_landed=5,
            distance_attempted=14,
            clinch_landed=2,
            clinch_attempted=4,
            ground_landed=1,
            ground_attempted=7,
            td_landed=1,
            td_attempted=5,
            sub_att=0,
            rev=0,
            ctrl_sec=40,
        ),
    ]

    return pd.DataFrame(rows)


def test_builds_one_row_per_fighter_fight_with_contract_columns() -> None:
    """The observation table must preserve fighter-fight grain."""

    standardized = standardize_round_stats(
        _two_fight_round_stats()
    )

    observations = build_fight_level_observations(
        standardized
    )

    assert len(observations) == 4

    assert not observations.duplicated(
        subset=["fight_id", "fighter_id"]
    ).any()

    assert set(
        PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS
    ).issubset(observations.columns)


def test_reciprocal_aggregate_columns_match_exactly() -> None:
    """Each fighter aggregate must equal the opponent mirror value."""

    standardized = standardize_round_stats(
        _two_fight_round_stats()
    )

    observations = build_fight_level_observations(
        standardized
    )

    fight = (
        observations.loc[
            observations["fight_id"] == "fight-1"
        ]
        .set_index("fighter_id")
    )

    fighter_a = fight.loc["fighter-a"]
    fighter_b = fight.loc["fighter-b"]

    for spec in PHASE_INTERACTION_AGGREGATE_SPECS:
        assert (
            fighter_a[spec.feature_name]
            == fighter_b[spec.opponent_feature_name]
        )

        assert (
            fighter_a[spec.opponent_feature_name]
            == fighter_b[spec.feature_name]
        )


def test_locked_interaction_formulas_are_calculated() -> None:
    """Representative interaction formulas must match hand calculations."""

    standardized = standardize_round_stats(
        _two_fight_round_stats()
    )

    observations = build_fight_level_observations(
        standardized
    )

    row = observations.loc[
        (observations["fight_id"] == "fight-1")
        & (observations["fighter_id"] == "fighter-a")
    ].iloc[0]

    assert row[
        "rfs_phase_interact_fight_distance_accuracy"
    ] == pytest.approx(11 / 25)

    assert row[
        "rfs_phase_interact_fight_td_completion_allowed"
    ] == pytest.approx(1 / 6)

    assert row[
        "rfs_phase_interact_fight_td_defense_rate"
    ] == pytest.approx(5 / 6)

    assert row[
        "rfs_phase_interact_fight_distance_pressure_share"
    ] == pytest.approx(25 / 45)

    assert row[
        "rfs_phase_interact_fight_control_share"
    ] == pytest.approx(150 / 225)

    assert row[
        "rfs_phase_interact_fight_control_exchange_balance"
    ] == pytest.approx(2 / 3)

    assert row[
        "rfs_phase_interact_fight_control_differential_per_round"
    ] == pytest.approx((150 - 75) / 2)

    assert row[
        "rfs_phase_interact_fight_reversal_rate_per_opponent_control_min"
    ] == pytest.approx(1 / (75 / 60))


def test_zero_denominators_produce_nan_not_infinity() -> None:
    """Unavailable opportunities should remain missing, not infinite."""

    frame = _two_fight_round_stats()

    mask = frame["fight_id"] == "fight-2"

    frame.loc[mask, "ctrl_sec"] = 0
    frame.loc[mask, "rev"] = 0
    frame.loc[mask, "ground_landed"] = 0
    frame.loc[mask, "ground_attempted"] = 0
    frame.loc[mask, "sub_att"] = 0

    result = build_round_fighter_phase_interaction(frame)

    fight_two = result.history.loc[
        result.history["fight_id"] == "fight-2"
    ]

    control_rate_columns = [
        spec.feature_name
        for spec in PHASE_INTERACTION_EVIDENCE_SPECS
        if "control_min" in spec.feature_name
    ]

    values = fight_two[control_rate_columns].to_numpy(
        dtype=float
    )

    assert not np.isinf(values).any()


def test_history_prior_state_excludes_current_fight() -> None:
    """Second-fight prior state must contain only the first fight."""

    result = build_round_fighter_phase_interaction(
        _two_fight_round_stats()
    )

    fighter_a_history = (
        result.history.loc[
            result.history["fighter_id"] == "fighter-a"
        ]
        .sort_values(["date", "fight_id"])
        .reset_index(drop=True)
    )

    first = fighter_a_history.iloc[0]
    second = fighter_a_history.iloc[1]

    assert first[
        "rfs_phase_interact_prior_fight_count"
    ] == 0

    assert first[
        "rfs_phase_interact_has_state"
    ] == 0

    assert second[
        "rfs_phase_interact_prior_fight_count"
    ] == 1

    assert second[
        "rfs_phase_interact_has_state"
    ] == 1

    aggregate_column = (
        "rfs_phase_interact_fight_td_attempts"
    )

    assert second[
        prior_total_name(aggregate_column)
    ] == pytest.approx(
        first[aggregate_column]
    )

    evidence_column = (
        "rfs_phase_interact_fight_td_defense_rate"
    )

    for state_kind in (
        "exp",
        "last3",
        "ewm",
    ):
        assert second[
            evidence_state_name(
                evidence_column,
                state_kind,
            )
        ] == pytest.approx(
            first[evidence_column]
        )


def test_latest_state_includes_all_completed_fights() -> None:
    """Latest future-matchup state includes full completed history."""

    result = build_round_fighter_phase_interaction(
        _two_fight_round_stats()
    )

    fighter_a_latest = result.latest.loc[
        result.latest["fighter_id"] == "fighter-a"
    ].iloc[0]

    fighter_a_history = result.history.loc[
        result.history["fighter_id"] == "fighter-a"
    ]

    aggregate_column = (
        "rfs_phase_interact_fight_td_attempts"
    )

    opponent_aggregate_column = (
        "rfs_phase_interact_fight_opp_td_attempts"
    )

    assert fighter_a_latest[
        "rfs_phase_interact_prior_fight_count"
    ] == 2

    assert fighter_a_latest[
        prior_total_name(aggregate_column)
    ] == pytest.approx(
        fighter_a_history[aggregate_column].sum()
    )

    assert fighter_a_latest[
        prior_total_name(opponent_aggregate_column)
    ] == pytest.approx(
        fighter_a_history[
            opponent_aggregate_column
        ].sum()
    )

    assert fighter_a_latest[
        "rfs_phase_interact_has_state"
    ] == 1


def test_missing_reciprocal_fighter_row_fails() -> None:
    """A fight cannot build with only one fighter perspective."""

    frame = _two_fight_round_stats()

    frame = frame.loc[
        ~(
            (frame["fight_id"] == "fight-2")
            & (frame["fighter_id"] == "fighter-c")
        )
    ].copy()

    with pytest.raises(
        RoundFighterPhaseInteractionBuildError,
        match="exactly two fighters",
    ):
        build_round_fighter_phase_interaction(frame)


def test_nonreciprocal_opponent_identity_fails() -> None:
    """Opponent IDs must point back to the reciprocal fighter."""

    frame = _two_fight_round_stats()

    mask = (
        (frame["fight_id"] == "fight-1")
        & (frame["fighter_id"] == "fighter-b")
    )

    frame.loc[mask, "opponent_id"] = "fighter-x"

    with pytest.raises(
        RoundFighterPhaseInteractionBuildError,
        match="not reciprocal",
    ):
        build_round_fighter_phase_interaction(frame)
