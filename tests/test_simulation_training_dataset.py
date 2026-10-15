from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.parameter_models import (
    DEFAULT_PARAMETER_MODEL_SPECS,
    ParameterContractError,
    SimulationParameter,
    pivot_parameter_predictions,
    validate_parameter_prediction_frame,
    validate_training_targets,
)
from pipeline.simulation.training_dataset import (
    SimulationTrainingDataError,
    build_simulation_training_dataset,
)


def make_round_stats() -> pd.DataFrame:
    rows = []
    f1 = {
        1: {"A": (20, 40, 0, 1, 20, 0), "B": (15, 35, 1, 2, 60, 0)},
        2: {"A": (22, 44, 1, 2, 55, 0), "B": (17, 38, 0, 1, 30, 0)},
        3: {"A": (18, 39, 0, 0, 10, 0), "B": (19, 41, 0, 0, 5, 0)},
    }
    for round_number, fighters in f1.items():
        for corner, fighter_id, opponent_id in (("red", "A", "B"), ("blue", "B", "A")):
            landed, attempted, td_landed, td_attempted, control, kd = fighters[fighter_id]
            rows.append(
                {
                    "event_id": "E1",
                    "event_name": "Event 1",
                    "date": "2025-01-01",
                    "fight_id": "F1",
                    "corner": corner,
                    "fighter_id": fighter_id,
                    "fighter_name": f"Fighter {fighter_id}",
                    "opponent_id": opponent_id,
                    "opponent_name": f"Fighter {opponent_id}",
                    "round": round_number,
                    "sig_str_landed": landed,
                    "sig_str_attempted": attempted,
                    "total_str_landed": landed + 5,
                    "total_str_attempted": attempted + 7,
                    "td_landed": td_landed,
                    "td_attempted": td_attempted,
                    "control_seconds": control,
                    "kd": kd,
                    "sub_att": 0,
                    "rev": 0,
                    "ground_landed": 2,
                    "ground_attempted": 4,
                }
            )

    f2 = {
        1: {"C": (14, 31, 0, 0, 0, 0), "D": (12, 29, 0, 1, 15, 0)},
        2: {"C": (11, 20, 0, 0, 0, 1), "D": (5, 15, 0, 0, 0, 0)},
    }
    for round_number, fighters in f2.items():
        for corner, fighter_id, opponent_id in (("red", "C", "D"), ("blue", "D", "C")):
            landed, attempted, td_landed, td_attempted, control, kd = fighters[fighter_id]
            rows.append(
                {
                    "event_id": "E2",
                    "event_name": "Event 2",
                    "date": "2025-02-01",
                    "fight_id": "F2",
                    "corner": corner,
                    "fighter_id": fighter_id,
                    "fighter_name": f"Fighter {fighter_id}",
                    "opponent_id": opponent_id,
                    "opponent_name": f"Fighter {opponent_id}",
                    "round": round_number,
                    "sig_str_landed": landed,
                    "sig_str_attempted": attempted,
                    "total_str_landed": landed + 3,
                    "total_str_attempted": attempted + 4,
                    "td_landed": td_landed,
                    "td_attempted": td_attempted,
                    "control_seconds": control,
                    "kd": kd,
                    "sub_att": 0,
                    "rev": 0,
                    "ground_landed": 0,
                    "ground_attempted": 0,
                }
            )
    return pd.DataFrame(rows)


def make_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "E1",
                "event_name": "Event 1",
                "date": "2025-01-01",
                "fight_id": "F1",
                "division": "Lightweight",
                "title_fight": 0,
                "method": "U-DEC",
                "finish_round": 3,
                "match_time_sec": 900,
                "total_rounds": 3,
                "winner_id": "A",
            },
            {
                "event_id": "E2",
                "event_name": "Event 2",
                "date": "2025-02-01",
                "fight_id": "F2",
                "division": "Welterweight",
                "title_fight": 0,
                "method": "KO/TKO",
                "finish_round": 2,
                "match_time_sec": 150,
                "total_rounds": 3,
                "winner_id": "C",
            },
        ]
    )


def make_state_source() -> pd.DataFrame:
    rows = []
    for fight_id, date, fighters in (
        ("F1", "2025-01-01", ("A", "B")),
        ("F2", "2025-02-01", ("C", "D")),
    ):
        for index, fighter_id in enumerate(fighters):
            rows.append(
                {
                    "fight_id": fight_id,
                    "fighter_id": fighter_id,
                    "date": date,
                    "rfs_traj_ewm_sig_attempt_slope": 0.1 + index,
                    "rfs_traj_prior_fight_count": 4 + index,
                    "rfs_traj_fight_sig_attempt_slope": 99.0,
                }
            )
    return pd.DataFrame(rows)


class SimulationTrainingDatasetTests(unittest.TestCase):
    def build(self):
        return build_simulation_training_dataset(
            make_round_stats(),
            make_master(),
            state_sources={"trajectory": make_state_source()},
        )

    def test_builds_one_row_per_fighter_round(self):
        result = self.build()
        dataset = result.dataset
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset[["fight_id", "fighter_id", "round"]].drop_duplicates().shape[0], 10)
        self.assertTrue(result.audit["passed"].all())
        validate_training_targets(dataset)

    def test_finish_targets_and_elapsed_time_are_correct(self):
        dataset = self.build().dataset
        c_round_two = dataset[(dataset["fight_id"] == "F2") & (dataset["fighter_id"] == "C") & (dataset["round"] == 2)].iloc[0]
        d_round_two = dataset[(dataset["fight_id"] == "F2") & (dataset["fighter_id"] == "D") & (dataset["round"] == 2)].iloc[0]
        c_round_one = dataset[(dataset["fight_id"] == "F2") & (dataset["fighter_id"] == "C") & (dataset["round"] == 1)].iloc[0]

        self.assertEqual(c_round_two["target_fighter_ko_tko_finish"], 1)
        self.assertEqual(d_round_two["target_opponent_ko_tko_finish"], 1)
        self.assertEqual(c_round_two["target_any_stoppage"], 1)
        self.assertEqual(c_round_two["target_finish_time_in_round_seconds"], 150)
        self.assertEqual(c_round_two["target_elapsed_fight_seconds"], 450)
        self.assertEqual(c_round_one["target_fight_reaches_next_round"], 1)
        self.assertEqual(c_round_two["target_fight_reaches_next_round"], 0)

    def test_prior_round_context_excludes_current_round(self):
        dataset = self.build().dataset
        row = dataset[(dataset["fight_id"] == "F2") & (dataset["fighter_id"] == "C") & (dataset["round"] == 2)].iloc[0]
        self.assertEqual(row["prior_sig_str_attempted_cumulative"], 31)
        self.assertEqual(row["prior_sig_str_attempted_last_round"], 31)
        self.assertEqual(row["opponent_prior_sig_str_attempted_cumulative"], 29)
        self.assertNotIn("sig_str_attempted", dataset.columns)

    def test_prefight_state_is_side_aware_and_excludes_realized_fight_columns(self):
        dataset = self.build().dataset
        row = dataset[(dataset["fight_id"] == "F1") & (dataset["fighter_id"] == "A") & (dataset["round"] == 1)].iloc[0]
        self.assertEqual(row["fighter_rfs_traj_ewm_sig_attempt_slope"], 0.1)
        self.assertEqual(row["opponent_rfs_traj_ewm_sig_attempt_slope"], 1.1)
        self.assertEqual(row["fighter_trajectory_state_available"], 1)
        self.assertFalse(any(
            "_fight_" in column
            for column in dataset.columns
            if column.startswith(("fighter_rfs_", "opponent_rfs_"))
        ))

    def test_future_dated_state_is_rejected(self):
        state = make_state_source()
        state.loc[(state["fight_id"] == "F1") & (state["fighter_id"] == "A"), "date"] = "2025-01-02"
        with self.assertRaises(SimulationTrainingDataError):
            build_simulation_training_dataset(
                make_round_stats(),
                make_master(),
                state_sources={"trajectory": state},
            )

    def test_duplicate_round_key_is_rejected(self):
        rounds = make_round_stats()
        rounds = pd.concat([rounds, rounds.iloc[[0]]], ignore_index=True)
        with self.assertRaises(SimulationTrainingDataError):
            build_simulation_training_dataset(rounds, make_master())


class ParameterContractTests(unittest.TestCase):
    def make_prediction_frame(self) -> pd.DataFrame:
        rows = []
        for parameter in DEFAULT_PARAMETER_MODEL_SPECS:
            row = {
                "fight_id": "LIVE1",
                "fighter_id": "A",
                "opponent_id": "B",
                "round": 1,
                "parameter": parameter.value,
                "prediction_mean": pd.NA,
                "prediction_probability": pd.NA,
                "prediction_dispersion": pd.NA,
                "prediction_zero_probability": pd.NA,
                "model_name": f"{parameter.value}_model",
                "model_version": "v1",
            }
            if parameter in {
                SimulationParameter.SIG_ATTEMPTS,
                SimulationParameter.TD_ATTEMPTS,
                SimulationParameter.KNOCKDOWNS,
            }:
                row["prediction_mean"] = 10.0 if parameter == SimulationParameter.SIG_ATTEMPTS else 1.0
                row["prediction_dispersion"] = 0.3
            elif parameter in {SimulationParameter.SIG_ACCURACY, SimulationParameter.TD_ACCURACY}:
                row["prediction_probability"] = 0.45
            elif parameter == SimulationParameter.CONTROL_SECONDS:
                row["prediction_mean"] = 42.0
                row["prediction_zero_probability"] = 0.35
            elif parameter == SimulationParameter.KO_TKO_FINISH:
                row["prediction_probability"] = 0.08
            elif parameter == SimulationParameter.SUBMISSION_FINISH:
                row["prediction_probability"] = 0.03
            rows.append(row)
        return pd.DataFrame(rows)

    def test_long_predictions_pivot_to_complete_estimate(self):
        predictions = self.make_prediction_frame()
        validate_parameter_prediction_frame(predictions)
        estimates = pivot_parameter_predictions(predictions)
        self.assertEqual(len(estimates), 1)
        estimate = estimates[0]
        self.assertEqual(estimate.sig_attempts_mean, 10.0)
        self.assertEqual(estimate.control_seconds_mean, 42.0)
        self.assertAlmostEqual(estimate.ko_tko_finish_probability, 0.08)

    def test_prediction_probability_bounds_are_enforced(self):
        predictions = self.make_prediction_frame()
        predictions.loc[
            predictions["parameter"] == SimulationParameter.KO_TKO_FINISH.value,
            "prediction_probability",
        ] = 1.2
        with self.assertRaises(ParameterContractError):
            validate_parameter_prediction_frame(predictions)


if __name__ == "__main__":
    unittest.main()
