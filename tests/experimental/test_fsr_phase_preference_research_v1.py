import pandas as pd

from scripts.experimental import fsr_phase_preference_research_v1 as phase


def _fsr_frame():
    return pd.DataFrame(
        [
            {
                "fight_id": "f1",
                "fighter_id": "a",
                "fighter_name": "A",
                "distance_striking_pressure": 60.0,
                "clinch_striking_pressure": 45.0,
                "wrestling_entry": 44.0,
                "control_imposition": 46.0,
                "td_defense": 52.0,
                "control_resistance": 53.0,
            },
            {
                "fight_id": "f1",
                "fighter_id": "b",
                "fighter_name": "B",
                "distance_striking_pressure": 44.0,
                "clinch_striking_pressure": 55.0,
                "wrestling_entry": 58.0,
                "control_imposition": 57.0,
                "td_defense": 48.0,
                "control_resistance": 47.0,
            },
        ]
    )


def _rfs_frame():
    return pd.DataFrame(
        [
            {
                "fight_id": "f1",
                "fighter_id": "a",
                "rfs_phase_base_fight_distance_attempt_share": 0.80,
                "rfs_phase_base_fight_clinch_attempt_share": 0.15,
                "rfs_phase_base_fight_ground_attempt_share": 0.05,
                "rfs_phase_base_fight_td_attempts_per_round": 0.20,
                "rfs_phase_base_fight_control_seconds_per_round": 5.0,
            },
            {
                "fight_id": "f1",
                "fighter_id": "b",
                "rfs_phase_base_fight_distance_attempt_share": 0.35,
                "rfs_phase_base_fight_clinch_attempt_share": 0.30,
                "rfs_phase_base_fight_ground_attempt_share": 0.35,
                "rfs_phase_base_fight_td_attempts_per_round": 2.20,
                "rfs_phase_base_fight_control_seconds_per_round": 70.0,
            },
        ]
    )


def test_preferences_preserve_fighter_fight_grain():
    out = phase.derive_candidate_preferences(_fsr_frame())
    assert len(out) == 2
    assert not out.duplicated(phase.KEYS).any()


def test_relative_preferences_center_to_zero():
    out = phase.derive_candidate_preferences(_fsr_frame())
    for candidate in phase.CANDIDATES:
        cols = [f"{candidate}_{p}_preference" for p in phase.PHASES]
        assert (out[cols].sum(axis=1).abs() < 1e-9).all()


def test_pressure_only_separates_obvious_styles():
    out = phase.derive_candidate_preferences(_fsr_frame()).set_index("fighter_id")
    assert out.loc["a", "pressure_only_distance_preference"] > 0
    assert out.loc["b", "pressure_only_wrestling_preference"] > 0
    assert out.loc["b", "pressure_only_clinch_preference"] > out.loc["a", "pressure_only_clinch_preference"]


def test_research_frame_joins_outcomes_one_to_one():
    out = phase.build_research_frame(_fsr_frame(), _rfs_frame())
    assert len(out) == 2
    assert out.loc[out["fighter_id"] == "a", phase.OUTCOME_COLUMNS["distance"]].iloc[0] == 0.80
