from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.simulation.finish_hazard_model import (
    FINISH_CLASSES,
    PROBABILITY_COLUMNS,
    _sequential_calibration,
    prepare_finish_hazard_dataset,
    select_finish_rfs_columns,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    fights = [
        ("f2024", "2024-01-01", "decision", 2),
        ("f2025", "2025-01-01", "red_ko", 2),
        ("f2026", "2026-01-01", "blue_sub", 1),
    ]
    for fight_id, date, outcome, finish_round in fights:
        for round_number in (1, 2):
            if round_number > finish_round:
                continue
            for corner, fighter_id, opponent_id in (
                ("red", "fighter-a", "fighter-b"),
                ("blue", "fighter-b", "fighter-a"),
            ):
                fighter_ko = int(
                    outcome == "red_ko"
                    and corner == "red"
                    and round_number == finish_round
                )
                opponent_ko = int(
                    outcome == "red_ko"
                    and corner == "blue"
                    and round_number == finish_round
                )
                fighter_sub = int(
                    outcome == "blue_sub"
                    and corner == "blue"
                    and round_number == finish_round
                )
                opponent_sub = int(
                    outcome == "blue_sub"
                    and corner == "red"
                    and round_number == finish_round
                )
                rows.append(
                    {
                        "fight_id": fight_id,
                        "fighter_id": fighter_id,
                        "opponent_id": opponent_id,
                        "corner": corner,
                        "date": date,
                        "round": round_number,
                        "total_rounds": 3,
                        "division": "Lightweight",
                        "title_fight": 0,
                        "target_finish_time_in_round_seconds": 180.0
                        if round_number == finish_round and outcome != "decision"
                        else 300.0,
                        "target_sig_attempted": 25.0 + round_number,
                        "target_sig_landed": 12.0,
                        "target_td_attempted": 1.0,
                        "target_td_landed": 0.5,
                        "target_control_seconds": 20.0,
                        "target_knockdowns": float(fighter_ko),
                        "target_submission_attempts": float(fighter_sub),
                        "target_fighter_ko_tko_finish": fighter_ko,
                        "target_opponent_ko_tko_finish": opponent_ko,
                        "target_fighter_submission_finish": fighter_sub,
                        "target_opponent_submission_finish": opponent_sub,
                        "fighter_rfs_defense_exp_head_absorbed_slope": 0.2,
                        "opponent_rfs_wrestle_ewm_submission_pressure_score": 0.3,
                        "fighter_rfs_traj_last3_sig_attempt_slope": 0.4,
                        "fighter_defense_state_available": 1.0,
                    }
                )
    return pd.DataFrame(rows)


class FinishHazardModelTests(unittest.TestCase):
    def test_prepare_pairs_fight_rounds_and_maps_competing_events(self):
        prepared = prepare_finish_hazard_dataset(_training_frame())
        self.assertEqual(len(prepared), 5)
        self.assertFalse(prepared.duplicated(["fight_id", "round"]).any())

        red_ko = prepared.loc[
            prepared["fight_id"].eq("f2025") & prepared["round"].eq(2),
            "finish_class",
        ].iloc[0]
        blue_sub = prepared.loc[
            prepared["fight_id"].eq("f2026") & prepared["round"].eq(1),
            "finish_class",
        ].iloc[0]
        self.assertEqual(red_ko, "red_ko_tko")
        self.assertEqual(blue_sub, "blue_submission")
        self.assertEqual(
            prepared.loc[
                prepared["fight_id"].eq("f2025") & prepared["round"].eq(1),
                "finish_class",
            ].iloc[0],
            "no_finish",
        )

    def test_compact_finish_rfs_selection_excludes_last3_noise(self):
        frame = _training_frame()
        selected = select_finish_rfs_columns(frame)
        self.assertIn("fighter_rfs_defense_exp_head_absorbed_slope", selected)
        self.assertIn(
            "opponent_rfs_wrestle_ewm_submission_pressure_score", selected
        )
        self.assertIn("fighter_defense_state_available", selected)
        self.assertNotIn("fighter_rfs_traj_last3_sig_attempt_slope", selected)

    def test_current_year_labels_do_not_change_its_calibration_factors(self):
        rows = []
        for year, actual in ((2022, "no_finish"), (2023, "red_ko_tko")):
            for index in range(4):
                row = {
                    "fight_id": f"{year}-{index}",
                    "round": 1,
                    "finish_class": actual,
                    "finish_class_index": FINISH_CLASSES.index(actual),
                    "model_name": "candidate",
                    "test_year": year,
                }
                for column, value in zip(
                    PROBABILITY_COLUMNS,
                    (0.60, 0.15, 0.10, 0.10, 0.05),
                ):
                    row[column] = value
                rows.append(row)
        predictions = pd.DataFrame(rows)
        first, schedule_first = _sequential_calibration(
            predictions, minimum_prior_rows=1
        )

        mutated = predictions.copy()
        mask = mutated["test_year"].eq(2023)
        mutated.loc[mask, "finish_class"] = "blue_submission"
        mutated.loc[mask, "finish_class_index"] = FINISH_CLASSES.index(
            "blue_submission"
        )
        second, schedule_second = _sequential_calibration(
            mutated, minimum_prior_rows=1
        )

        first_2023 = first.loc[first["test_year"].eq(2023)].sort_values("fight_id")
        second_2023 = second.loc[second["test_year"].eq(2023)].sort_values("fight_id")
        calibrated_columns = [f"calibrated_{column}" for column in PROBABILITY_COLUMNS]
        np.testing.assert_allclose(
            first_2023[calibrated_columns].to_numpy(),
            second_2023[calibrated_columns].to_numpy(),
        )
        first_schedule = schedule_first.loc[
            schedule_first["test_year"].eq(2023)
        ].reset_index(drop=True)
        second_schedule = schedule_second.loc[
            schedule_second["test_year"].eq(2023)
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(first_schedule, second_schedule)
        np.testing.assert_allclose(
            first_2023[calibrated_columns].sum(axis=1).to_numpy(),
            np.ones(len(first_2023)),
        )


if __name__ == "__main__":
    unittest.main()
