"""Focused tests for the Dynamic Response builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.round_stats.build_round_fighter_dynamic_response import (
    RoundFighterDynamicResponseBuildError,
    build_round_fighter_dynamic_response,
)
from pipeline.round_stats.rfs_dynamic_response_feature_contracts import (
    DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS,
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
    total_landed: int,
    total_attempted: int,
    td_landed: int,
    td_attempted: int,
    ctrl_sec: int,
    kd: int,
    head_landed: int,
    ground_landed: int,
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
        "total_str_landed": total_landed,
        "total_str_attempted": total_attempted,
        "td_landed": td_landed,
        "td_attempted": td_attempted,
        "ctrl_sec": ctrl_sec,
        "kd": kd,
        "head_landed": head_landed,
        "ground_landed": ground_landed,
    }


def _two_fight_fixture() -> pd.DataFrame:
    """Return two reciprocal fights for fighters A and B."""

    rows: list[dict[str, object]] = []

    # Fight 1: two rounds.
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
                total_landed=15,
                total_attempted=30,
                td_landed=1,
                td_attempted=2,
                ctrl_sec=30,
                kd=0,
                head_landed=6,
                ground_landed=2,
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
                total_landed=12,
                total_attempted=24,
                td_landed=0,
                td_attempted=1,
                ctrl_sec=10,
                kd=0,
                head_landed=4,
                ground_landed=1,
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
                total_landed=20,
                total_attempted=40,
                td_landed=1,
                td_attempted=3,
                ctrl_sec=50,
                kd=1,
                head_landed=10,
                ground_landed=4,
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
                total_landed=9,
                total_attempted=18,
                td_landed=0,
                td_attempted=1,
                ctrl_sec=5,
                kd=0,
                head_landed=3,
                ground_landed=0,
            ),
        ]
    )

    # Fight 2: same pair, later date.
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
                total_landed=18,
                total_attempted=36,
                td_landed=0,
                td_attempted=1,
                ctrl_sec=15,
                kd=0,
                head_landed=7,
                ground_landed=1,
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
                total_landed=13,
                total_attempted=26,
                td_landed=1,
                td_attempted=2,
                ctrl_sec=25,
                kd=0,
                head_landed=5,
                ground_landed=2,
            ),
        ]
    )

    return pd.DataFrame(rows)


def test_builder_returns_one_row_per_fighter_fight() -> None:
    """History must preserve exact fighter-fight grain."""

    result = build_round_fighter_dynamic_response(
        _two_fight_fixture()
    )

    assert len(result.history) == 4
    assert not result.history.duplicated(
        subset=["fight_id", "fighter_id"]
    ).any()

    assert set(
        DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS
    ).issubset(result.history.columns)


def test_opponent_aggregates_are_reciprocal() -> None:
    """Absorbed statistics must come from the reciprocal fighter."""

    result = build_round_fighter_dynamic_response(
        _two_fight_fixture()
    )

    alpha = result.history[
        (result.history["fight_id"] == "f1")
        & (result.history["fighter_id"] == "A")
    ].iloc[0]

    assert (
        alpha[
            "rfs_dynamic_response_fight_knockdowns_absorbed"
        ]
        == 0.0
    )
    assert (
        alpha[
            "rfs_dynamic_response_fight_head_strikes_absorbed"
        ]
        == 7.0
    )
    assert (
        alpha[
            "rfs_dynamic_response_fight_ground_strikes_absorbed"
        ]
        == 1.0
    )
    assert (
        alpha[
            "rfs_dynamic_response_fight_opponent_control_seconds"
        ]
        == 15.0
    )


def test_representative_trajectory_formulas() -> None:
    """Hand-check slope, ratio, and accuracy-change evidence."""

    result = build_round_fighter_dynamic_response(
        _two_fight_fixture()
    )

    alpha = result.history[
        (result.history["fight_id"] == "f1")
        & (result.history["fighter_id"] == "A")
    ].iloc[0]

    assert np.isclose(
        alpha[
            "rfs_dynamic_response_fight_"
            "sig_strike_attempt_slope"
        ],
        10.0,
    )

    assert np.isclose(
        alpha[
            "rfs_dynamic_response_fight_"
            "sig_strike_attempt_first_last_ratio"
        ],
        1.5,
    )

    assert np.isclose(
        alpha[
            "rfs_dynamic_response_fight_"
            "sig_strike_accuracy_change"
        ],
        0.0,
    )


def test_single_round_fight_preserves_unavailable_trajectory_nan() -> None:
    """Single-round fights cannot manufacture trajectory evidence."""

    result = build_round_fighter_dynamic_response(
        _two_fight_fixture()
    )

    alpha = result.history[
        (result.history["fight_id"] == "f2")
        & (result.history["fighter_id"] == "A")
    ].iloc[0]

    assert np.isnan(
        alpha[
            "rfs_dynamic_response_fight_"
            "sig_strike_attempt_slope"
        ]
    )
    assert np.isnan(
        alpha[
            "rfs_dynamic_response_fight_"
            "late_early_output_ratio"
        ]
    )


def test_prior_state_excludes_current_fight() -> None:
    """Second-fight prior state must use only the first fight."""

    result = build_round_fighter_dynamic_response(
        _two_fight_fixture()
    )

    alpha_history = (
        result.history[
            result.history["fighter_id"] == "A"
        ]
        .sort_values(["date", "fight_id"])
        .reset_index(drop=True)
    )

    first = alpha_history.iloc[0]
    second = alpha_history.iloc[1]

    assert (
        first["rfs_dynamic_response_prior_fight_count"]
        == 0
    )
    assert (
        first["rfs_dynamic_response_has_state"]
        == 0
    )

    assert (
        second["rfs_dynamic_response_prior_fight_count"]
        == 1
    )

    assert np.isclose(
        second[
            "rfs_dynamic_response_exp_"
            "sig_strike_accuracy"
        ],
        first[
            "rfs_dynamic_response_fight_"
            "sig_strike_accuracy"
        ],
    )


def test_latest_state_includes_all_completed_fights() -> None:
    """Latest state must include the current complete history."""

    result = build_round_fighter_dynamic_response(
        _two_fight_fixture()
    )

    alpha = result.latest[
        result.latest["fighter_id"] == "A"
    ].iloc[0]

    assert (
        alpha["rfs_dynamic_response_prior_fight_count"]
        == 2
    )
    assert (
        alpha["rfs_dynamic_response_has_state"]
        == 1
    )
    assert (
        alpha[
            "rfs_dynamic_response_prior_total_rounds_observed"
        ]
        == 3.0
    )


def test_missing_reciprocal_round_fails() -> None:
    """A fighter-round without its reciprocal opponent row must fail."""

    df = _two_fight_fixture()

    df = df[
        ~(
            (df["fight_id"] == "f1")
            & (df["fighter_id"] == "B")
            & (df["round"] == 2)
        )
    ].copy()

    with pytest.raises(
        RoundFighterDynamicResponseBuildError,
        match="unequal fighter round sets",
    ):
        build_round_fighter_dynamic_response(df)


def test_nonreciprocal_opponent_identity_fails() -> None:
    """Opponent IDs must reference the other fighter in the fight."""

    df = _two_fight_fixture()

    df.loc[
        (
            (df["fight_id"] == "f1")
            & (df["fighter_id"] == "A")
        ),
        "opponent_id",
    ] = "C"

    with pytest.raises(
        RoundFighterDynamicResponseBuildError,
        match="nonreciprocal opponent identity",
    ):
        build_round_fighter_dynamic_response(df)
