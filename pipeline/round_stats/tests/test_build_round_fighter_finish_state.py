"""Focused tests for the Finish State builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.round_stats.build_round_fighter_finish_state import (
    RoundFighterFinishStateBuildError,
    build_round_fighter_finish_state,
)
from pipeline.round_stats.rfs_finish_state_feature_contracts import (
    FINISH_STATE_FIGHT_OBSERVATION_COLUMNS,
)


def _round_row(
    *,
    fight_id: str,
    date: str,
    fighter_id: str,
    fighter_name: str,
    opponent_id: str,
    opponent_name: str,
    corner: str,
    round_number: int,
    sig_landed: int,
    sig_attempted: int,
    td_landed: int,
    sub_att: int,
    ctrl_sec: int,
    kd: int,
    head_landed: int,
    distance_landed: int,
    clinch_landed: int,
    clinch_attempted: int,
    ground_landed: int,
    ground_attempted: int,
) -> dict[str, object]:
    """Create one authoritative fighter-round test row."""

    return {
        "event_id": f"event_{fight_id}",
        "event_name": f"Event {fight_id}",
        "event_date": date,
        "fight_id": fight_id,
        "division": "Lightweight",
        "corner": corner,
        "fighter_id": fighter_id,
        "fighter_name": fighter_name,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,
        "round": round_number,
        "sig_str_landed": sig_landed,
        "sig_str_attempted": sig_attempted,
        "td_landed": td_landed,
        "sub_att": sub_att,
        "ctrl_sec": ctrl_sec,
        "kd": kd,
        "head_landed": head_landed,
        "distance_landed": distance_landed,
        "clinch_landed": clinch_landed,
        "clinch_attempted": clinch_attempted,
        "ground_landed": ground_landed,
        "ground_attempted": ground_attempted,
    }


def _round_fixture() -> pd.DataFrame:
    """Return two reciprocal fights for fighters A and B."""

    rows: list[dict[str, object]] = []

    rows.extend(
        [
            _round_row(
                fight_id="f1",
                date="2024-01-01",
                fighter_id="A",
                fighter_name="Alpha",
                opponent_id="B",
                opponent_name="Beta",
                corner="red",
                round_number=1,
                sig_landed=10,
                sig_attempted=20,
                td_landed=1,
                sub_att=1,
                ctrl_sec=60,
                kd=1,
                head_landed=6,
                distance_landed=7,
                clinch_landed=2,
                clinch_attempted=4,
                ground_landed=1,
                ground_attempted=3,
            ),
            _round_row(
                fight_id="f1",
                date="2024-01-01",
                fighter_id="B",
                fighter_name="Beta",
                opponent_id="A",
                opponent_name="Alpha",
                corner="blue",
                round_number=1,
                sig_landed=8,
                sig_attempted=16,
                td_landed=0,
                sub_att=0,
                ctrl_sec=10,
                kd=0,
                head_landed=4,
                distance_landed=6,
                clinch_landed=1,
                clinch_attempted=2,
                ground_landed=1,
                ground_attempted=2,
            ),
            _round_row(
                fight_id="f1",
                date="2024-01-01",
                fighter_id="A",
                fighter_name="Alpha",
                opponent_id="B",
                opponent_name="Beta",
                corner="red",
                round_number=2,
                sig_landed=15,
                sig_attempted=30,
                td_landed=1,
                sub_att=0,
                ctrl_sec=30,
                kd=0,
                head_landed=10,
                distance_landed=12,
                clinch_landed=2,
                clinch_attempted=5,
                ground_landed=1,
                ground_attempted=3,
            ),
            _round_row(
                fight_id="f1",
                date="2024-01-01",
                fighter_id="B",
                fighter_name="Beta",
                opponent_id="A",
                opponent_name="Alpha",
                corner="blue",
                round_number=2,
                sig_landed=6,
                sig_attempted=12,
                td_landed=0,
                sub_att=0,
                ctrl_sec=5,
                kd=0,
                head_landed=3,
                distance_landed=5,
                clinch_landed=1,
                clinch_attempted=2,
                ground_landed=0,
                ground_attempted=1,
            ),
        ]
    )

    rows.extend(
        [
            _round_row(
                fight_id="f2",
                date="2024-06-01",
                fighter_id="A",
                fighter_name="Alpha",
                opponent_id="B",
                opponent_name="Beta",
                corner="blue",
                round_number=1,
                sig_landed=12,
                sig_attempted=24,
                td_landed=0,
                sub_att=0,
                ctrl_sec=15,
                kd=0,
                head_landed=7,
                distance_landed=10,
                clinch_landed=1,
                clinch_attempted=2,
                ground_landed=1,
                ground_attempted=2,
            ),
            _round_row(
                fight_id="f2",
                date="2024-06-01",
                fighter_id="B",
                fighter_name="Beta",
                opponent_id="A",
                opponent_name="Alpha",
                corner="red",
                round_number=1,
                sig_landed=9,
                sig_attempted=18,
                td_landed=1,
                sub_att=1,
                ctrl_sec=25,
                kd=0,
                head_landed=5,
                distance_landed=6,
                clinch_landed=1,
                clinch_attempted=3,
                ground_landed=2,
                ground_attempted=4,
            ),
        ]
    )

    return pd.DataFrame(rows)


def _outcome_fixture() -> pd.DataFrame:
    """Return one valid finish and one valid decision."""

    return pd.DataFrame(
        [
            {
                "fight_id": "f1",
                "winner": "Alpha",
                "winner_id": "A",
                "method": "KO/TKO",
                "finish_round": 2,
            },
            {
                "fight_id": "f2",
                "winner": "Beta",
                "winner_id": "B",
                "method": "Decision - Unanimous",
                "finish_round": 1,
            },
        ]
    )


def test_builder_returns_one_row_per_fighter_fight() -> None:
    """History must preserve exact fighter-fight grain."""

    result = build_round_fighter_finish_state(
        _round_fixture(),
        _outcome_fixture(),
    )

    assert len(result.history) == 4
    assert not result.history.duplicated(
        subset=["fight_id", "fighter_id"]
    ).any()

    assert set(
        FINISH_STATE_FIGHT_OBSERVATION_COLUMNS
    ).issubset(result.history.columns)


def test_opponent_aggregates_are_reciprocal() -> None:
    """Absorbed statistics must come from the reciprocal fighter."""

    result = build_round_fighter_finish_state(
        _round_fixture(),
        _outcome_fixture(),
    )

    alpha = result.history[
        (result.history["fight_id"] == "f1")
        & (result.history["fighter_id"] == "A")
    ].iloc[0]

    assert (
        alpha[
            "rfs_finish_state_fight_sig_strikes_absorbed"
        ]
        == 14.0
    )
    assert (
        alpha[
            "rfs_finish_state_fight_head_strikes_absorbed"
        ]
        == 7.0
    )
    assert (
        alpha[
            "rfs_finish_state_fight_opponent_submission_attempts"
        ]
        == 0.0
    )
    assert (
        alpha[
            "rfs_finish_state_fight_opponent_control_seconds"
        ]
        == 15.0
    )


def test_outcome_indicators_are_fighter_specific() -> None:
    """KO/TKO loss and survival must reflect fighter perspective."""

    result = build_round_fighter_finish_state(
        _round_fixture(),
        _outcome_fixture(),
    )

    alpha = result.history[
        (result.history["fight_id"] == "f1")
        & (result.history["fighter_id"] == "A")
    ].iloc[0]

    beta = result.history[
        (result.history["fight_id"] == "f1")
        & (result.history["fighter_id"] == "B")
    ].iloc[0]

    assert (
        alpha[
            "rfs_finish_state_fight_ko_tko_loss_indicator"
        ]
        == 0.0
    )
    assert (
        alpha[
            "rfs_finish_state_fight_ko_tko_survival_indicator"
        ]
        == 1.0
    )
    assert (
        beta[
            "rfs_finish_state_fight_ko_tko_loss_indicator"
        ]
        == 1.0
    )
    assert (
        beta[
            "rfs_finish_state_fight_ko_tko_survival_indicator"
        ]
        == 0.0
    )


def test_invalid_outcome_preserves_missing_indicators() -> None:
    """Overturned outcomes must not manufacture wins or losses."""

    outcomes = _outcome_fixture()
    outcomes.loc[
        outcomes["fight_id"] == "f1",
        ["winner", "winner_id", "method"],
    ] = [pd.NA, pd.NA, "Overturned"]

    result = build_round_fighter_finish_state(
        _round_fixture(),
        outcomes,
    )

    fight = result.history[
        result.history["fight_id"] == "f1"
    ]

    assert (
        fight[
            "rfs_finish_state_fight_valid_outcome"
        ]
        == 0.0
    ).all()

    indicator_columns = [
        "rfs_finish_state_fight_ko_tko_loss_indicator",
        "rfs_finish_state_fight_ko_tko_survival_indicator",
        "rfs_finish_state_fight_submission_loss_indicator",
        "rfs_finish_state_fight_submission_survival_indicator",
    ]

    assert fight[indicator_columns].isna().all().all()


def test_prior_state_excludes_current_fight() -> None:
    """Second-fight prior totals must equal first-fight evidence only."""

    result = build_round_fighter_finish_state(
        _round_fixture(),
        _outcome_fixture(),
    )

    alpha_second = result.history[
        (result.history["fight_id"] == "f2")
        & (result.history["fighter_id"] == "A")
    ].iloc[0]

    assert (
        alpha_second[
            "rfs_finish_state_prior_fight_count"
        ]
        == 1
    )

    assert (
        alpha_second[
            "rfs_finish_state_prior_total_knockdowns_scored"
        ]
        == 1.0
    )

    assert (
        alpha_second[
            "rfs_finish_state_exp_ko_tko_loss_indicator"
        ]
        == 0.0
    )


def test_latest_state_includes_all_completed_fights() -> None:
    """Latest state must include the fighter's complete history."""

    result = build_round_fighter_finish_state(
        _round_fixture(),
        _outcome_fixture(),
    )

    alpha = result.latest[
        result.latest["fighter_id"] == "A"
    ].iloc[0]

    assert (
        alpha[
            "rfs_finish_state_prior_fight_count"
        ]
        == 2
    )

    assert (
        alpha[
            "rfs_finish_state_prior_total_knockdowns_scored"
        ]
        == 1.0
    )

    assert (
        alpha[
            "rfs_finish_state_exp_ko_tko_loss_indicator"
        ]
        == 0.0
    )


def test_missing_outcome_fails() -> None:
    """Every round-data fight must have an authoritative outcome."""

    outcomes = _outcome_fixture()
    outcomes = outcomes[
        outcomes["fight_id"] != "f2"
    ]

    with pytest.raises(
        RoundFighterFinishStateBuildError,
        match="missing authoritative outcomes",
    ):
        build_round_fighter_finish_state(
            _round_fixture(),
            outcomes,
        )


def test_duplicate_outcome_fails() -> None:
    """Outcome table must contain one row per fight."""

    outcomes = pd.concat(
        [
            _outcome_fixture(),
            _outcome_fixture().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        RoundFighterFinishStateBuildError,
        match="duplicate fight_id",
    ):
        build_round_fighter_finish_state(
            _round_fixture(),
            outcomes,
        )


def test_nonreciprocal_opponent_identity_fails() -> None:
    """Round rows must preserve reciprocal opponent identities."""

    rounds = _round_fixture()
    rounds.loc[
        (
            (rounds["fight_id"] == "f1")
            & (rounds["fighter_id"] == "A")
        ),
        "opponent_id",
    ] = "C"

    with pytest.raises(
        RoundFighterFinishStateBuildError,
        match="nonreciprocal opponent identity",
    ):
        build_round_fighter_finish_state(
            rounds,
            _outcome_fixture(),
        )


def test_missing_winner_decision_is_invalid_not_fatal() -> None:
    """Draw-like decisions remain unavailable rather than failing."""

    outcomes = _outcome_fixture()
    outcomes.loc[
        outcomes["fight_id"] == "f2",
        ["winner", "winner_id"],
    ] = [pd.NA, pd.NA]

    result = build_round_fighter_finish_state(
        _round_fixture(),
        outcomes,
    )

    fight = result.history[
        result.history["fight_id"] == "f2"
    ]

    assert (
        fight["rfs_finish_state_fight_valid_outcome"]
        == 0.0
    ).all()

    indicator_columns = [
        "rfs_finish_state_fight_ko_tko_loss_indicator",
        "rfs_finish_state_fight_ko_tko_survival_indicator",
        "rfs_finish_state_fight_submission_loss_indicator",
        "rfs_finish_state_fight_submission_survival_indicator",
    ]

    assert fight[indicator_columns].isna().all().all()
