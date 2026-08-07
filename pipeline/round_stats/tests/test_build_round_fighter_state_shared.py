"""Tests for shared Round Fighter State family assembly."""

from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline.round_stats import build_round_fighter_state as module


def _base_history() -> pd.DataFrame:
    """Return the legacy history grain used as merge authority."""

    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1"],
            "fighter_id": ["a", "b"],
            "fighter_name": ["Alpha", "Beta"],
            "rfs_traj_prior_fight_count": [0, 0],
        }
    )


def _base_latest() -> pd.DataFrame:
    """Return the legacy latest fighter grain."""

    return pd.DataFrame(
        {
            "fighter_id": ["a", "b"],
            "fighter_name": ["Alpha", "Beta"],
            "rfs_traj_prior_fight_count": [1, 1],
        }
    )


def _family_result(
    history_column: str,
    latest_column: str,
) -> SimpleNamespace:
    """Return one minimal family build result."""

    return SimpleNamespace(
        history=pd.DataFrame(
            {
                "fight_id": ["f1", "f1"],
                "fighter_id": ["a", "b"],
                history_column: [0.1, 0.2],
            }
        ),
        latest=pd.DataFrame(
            {
                "fighter_id": ["a", "b"],
                latest_column: [0.3, 0.4],
            }
        ),
    )


def test_shared_builder_merges_all_four_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared artifacts must contain every completed RFS family."""

    monkeypatch.setattr(
        module,
        "read_round_stats",
        lambda path: pd.DataFrame({"source": [1]}),
    )
    monkeypatch.setattr(
        module,
        "build_round_fighter_state_history",
        lambda df: _base_history(),
    )
    monkeypatch.setattr(
        module,
        "build_latest_round_fighter_state",
        lambda df: _base_latest(),
    )
    monkeypatch.setattr(
        module,
        "_read_finish_state_outcomes",
        lambda path: pd.DataFrame({"fight_id": ["f1"]}),
    )

    monkeypatch.setattr(
        module,
        "build_round_fighter_phase_baseline",
        lambda df: _family_result(
            "rfs_phase_base_fight_distance_attempt_share",
            "rfs_phase_base_ewm_distance_attempt_share",
        ),
    )
    monkeypatch.setattr(
        module,
        "build_round_fighter_phase_interaction",
        lambda df: _family_result(
            "rfs_phase_interact_fight_pressure_proxy",
            "rfs_phase_interact_ewm_pressure_proxy",
        ),
    )
    monkeypatch.setattr(
        module,
        "build_round_fighter_dynamic_response",
        lambda df: _family_result(
            "rfs_dynamic_response_fight_output_change",
            "rfs_dynamic_response_ewm_output_change",
        ),
    )
    monkeypatch.setattr(
        module,
        "build_round_fighter_finish_state",
        lambda rounds, outcomes: _family_result(
            "rfs_finish_state_fight_damage_exposure_composite",
            "rfs_finish_state_ewm_damage_exposure_composite",
        ),
    )

    result = module.build_round_fighter_state(
        round_stats_path="unused-rounds.parquet",
        master_path="unused-master.parquet",
    )

    assert len(result.history_df) == 2
    assert len(result.latest_df) == 2

    expected_history_columns = {
        "rfs_phase_base_fight_distance_attempt_share",
        "rfs_phase_interact_fight_pressure_proxy",
        "rfs_dynamic_response_fight_output_change",
        "rfs_finish_state_fight_damage_exposure_composite",
    }

    expected_latest_columns = {
        "rfs_phase_base_ewm_distance_attempt_share",
        "rfs_phase_interact_ewm_pressure_proxy",
        "rfs_dynamic_response_ewm_output_change",
        "rfs_finish_state_ewm_damage_exposure_composite",
    }

    assert expected_history_columns.issubset(
        result.history_df.columns
    )
    assert expected_latest_columns.issubset(
        result.latest_df.columns
    )

    forbidden_latest_prefixes = (
        "rfs_phase_base_fight_",
        "rfs_phase_interact_fight_",
        "rfs_dynamic_response_fight_",
        "rfs_finish_state_fight_",
    )

    assert not any(
        column.startswith(forbidden_latest_prefixes)
        for column in result.latest_df.columns
    )


def test_family_merge_rejects_unmatched_rows() -> None:
    """A family may not silently omit a base fighter-fight row."""

    base = _base_history()

    incomplete_family = pd.DataFrame(
        {
            "fight_id": ["f1"],
            "fighter_id": ["a"],
            "rfs_phase_base_has_state": [1],
        }
    )

    with pytest.raises(
        module.RoundFighterStateBuildError,
        match="unmatched rows",
    ):
        module._merge_rfs_family(
            base,
            incomplete_family,
            keys=["fight_id", "fighter_id"],
            prefix="rfs_phase_base_",
            label="Phase Baseline history",
        )


def test_family_merge_rejects_feature_collisions() -> None:
    """Existing shared columns must never be overwritten silently."""

    base = _base_history()
    base["rfs_phase_base_has_state"] = [0, 0]

    family = pd.DataFrame(
        {
            "fight_id": ["f1", "f1"],
            "fighter_id": ["a", "b"],
            "rfs_phase_base_has_state": [1, 1],
        }
    )

    with pytest.raises(
        module.RoundFighterStateBuildError,
        match="collisions",
    ):
        module._merge_rfs_family(
            base,
            family,
            keys=["fight_id", "fighter_id"],
            prefix="rfs_phase_base_",
            label="Phase Baseline history",
        )
