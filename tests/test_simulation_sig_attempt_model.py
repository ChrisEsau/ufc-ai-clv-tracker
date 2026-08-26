from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.simulation.sig_attempt_model import (
    _fighter_history_prediction,
    prepare_sig_attempt_dataset,
    select_model_columns,
)


def make_training_rows(second_fight_attempts: int = 40) -> pd.DataFrame:
    rows: list[dict] = []
    fight_specs = [
        ("F1", "2023-01-01", 30),
        ("F2", "2024-01-01", second_fight_attempts),
        ("F3", "2025-01-01", 50),
    ]
    for fight_id, date, a_attempts in fight_specs:
        for fighter_id, opponent_id, attempts in (
            ("A", "B", a_attempts),
            ("B", "A", 20),
        ):
            rows.append(
                {
                    "fight_id": fight_id,
                    "event_id": f"E_{fight_id}",
                    "event_name": f"Event {fight_id}",
                    "fighter_id": fighter_id,
                    "fighter_name": fighter_id,
                    "opponent_id": opponent_id,
                    "opponent_name": opponent_id,
                    "date": date,
                    "round": 1,
                    "total_rounds": 3,
                    "title_fight": 0,
                    "division": "Lightweight",
                    "corner": "red" if fighter_id == "A" else "blue",
                    "target_sig_attempted": attempts,
                    "target_finish_time_in_round_seconds": 300.0,
                    "prior_rounds_completed": 0,
                    "rounds_remaining_including_current": 3,
                    "elapsed_seconds_before_round": 0,
                    "scheduled_fight_seconds": 900,
                    "opponent_rounds_remaining_including_current": 3,
                    "opponent_elapsed_seconds_before_round": 0,
                    "prior_sig_str_attempted_cumulative": 0.0,
                    "opponent_prior_sig_str_attempted_cumulative": 0.0,
                    "fighter_rfs_traj_ewm_sig_attempt_slope": 0.1,
                    "opponent_rfs_traj_ewm_sig_attempt_slope": -0.1,
                    "fighter_trajectory_state_available": 1,
                    "opponent_trajectory_state_available": 1,
                }
            )
    return pd.DataFrame(rows)


class SigAttemptModelTests(unittest.TestCase):
    def test_prefight_history_is_shifted_by_complete_fight(self):
        prepared = prepare_sig_attempt_dataset(make_training_rows())
        fighter_a = prepared.loc[prepared["fighter_id"].eq("A")].sort_values("date")

        self.assertTrue(pd.isna(fighter_a.iloc[0]["fighter_prior_sig_attempt_rate_exp"]))
        self.assertAlmostEqual(
            float(fighter_a.iloc[1]["fighter_prior_sig_attempt_rate_exp"]),
            6.0,
        )
        self.assertAlmostEqual(
            float(fighter_a.iloc[2]["fighter_prior_sig_attempt_rate_exp"]),
            7.0,
        )

    def test_current_fight_target_does_not_change_its_own_prior_state(self):
        original = prepare_sig_attempt_dataset(make_training_rows(second_fight_attempts=40))
        mutated = prepare_sig_attempt_dataset(make_training_rows(second_fight_attempts=100))

        original_f2 = original.loc[
            original["fight_id"].eq("F2") & original["fighter_id"].eq("A")
        ].iloc[0]
        mutated_f2 = mutated.loc[
            mutated["fight_id"].eq("F2") & mutated["fighter_id"].eq("A")
        ].iloc[0]
        self.assertAlmostEqual(
            float(original_f2["fighter_prior_sig_attempt_rate_exp"]),
            float(mutated_f2["fighter_prior_sig_attempt_rate_exp"]),
        )

        original_f3 = original.loc[
            original["fight_id"].eq("F3") & original["fighter_id"].eq("A")
        ].iloc[0]
        mutated_f3 = mutated.loc[
            mutated["fight_id"].eq("F3") & mutated["fighter_id"].eq("A")
        ].iloc[0]
        self.assertNotEqual(
            float(original_f3["fighter_prior_sig_attempt_rate_exp"]),
            float(mutated_f3["fighter_prior_sig_attempt_rate_exp"]),
        )

    def test_missing_history_fallback_returns_finite_predictions(self):
        prepared = prepare_sig_attempt_dataset(make_training_rows())
        round_rate = np.full(len(prepared), 5.5, dtype=float)

        prediction = _fighter_history_prediction(prepared, round_rate)

        self.assertTrue(np.isfinite(prediction).all())
        first_fight = prepared["fight_id"].eq("F1").to_numpy()
        np.testing.assert_allclose(prediction[first_fight], 5.5)

    def test_missing_history_fallback_accepts_read_only_float64_view(self):
        prepared = prepare_sig_attempt_dataset(make_training_rows())
        frame = prepared.loc[prepared["fight_id"].isin(["F1", "F2"])].copy()
        frame["fighter_prior_sig_attempt_rate_exp"] = pd.to_numeric(
            frame["fighter_prior_sig_attempt_rate_exp"], errors="coerce"
        ).astype("float64")
        frame["fighter_prior_exposure_minutes"] = pd.to_numeric(
            frame["fighter_prior_exposure_minutes"], errors="coerce"
        ).astype("float64")

        backing = frame["fighter_prior_sig_attempt_rate_exp"].to_numpy(copy=False)
        backing.setflags(write=False)
        round_rate = np.full(len(frame), 5.5, dtype=float)

        prediction = _fighter_history_prediction(frame, round_rate)

        self.assertTrue(np.isfinite(prediction).all())
        cold_start = frame["fight_id"].eq("F1").to_numpy()
        np.testing.assert_allclose(prediction[cold_start], 5.5)

    def test_rfs_ablation_feature_sets_are_separate(self):
        prepared = prepare_sig_attempt_dataset(make_training_rows())
        context_numeric, _ = select_model_columns(prepared, include_rfs=False)
        rfs_numeric, _ = select_model_columns(prepared, include_rfs=True)

        self.assertNotIn("fighter_rfs_traj_ewm_sig_attempt_slope", context_numeric)
        self.assertIn("fighter_rfs_traj_ewm_sig_attempt_slope", rfs_numeric)
        self.assertFalse(any(column.startswith("target_") for column in rfs_numeric))


if __name__ == "__main__":
    unittest.main()
