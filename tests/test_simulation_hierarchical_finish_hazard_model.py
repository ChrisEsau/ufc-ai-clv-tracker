from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.simulation.finish_hazard_model import FINISH_CLASSES, PROBABILITY_COLUMNS
from pipeline.simulation.hierarchical_finish_hazard_model import (
    CALIBRATED_STAGE_COLUMNS,
    STAGE_PROBABILITY_COLUMNS,
    _fit_binary_stage,
    combine_hierarchical_probabilities,
    sequential_hierarchical_calibration,
)


def _prediction_frame() -> pd.DataFrame:
    rows = []
    classes = (
        "no_finish",
        "red_ko_tko",
        "red_submission",
        "blue_ko_tko",
        "blue_submission",
    )
    for year in (2022, 2023):
        for index, finish_class in enumerate(classes * 2):
            finish = 0.30 + 0.02 * index
            submission = 0.25 + 0.01 * index
            red_ko = 0.55
            red_sub = 0.62
            combined = combine_hierarchical_probabilities(
                np.array([finish]),
                np.array([submission]),
                np.array([red_ko]),
                np.array([red_sub]),
            )[0]
            row = {
                "fight_id": f"{year}-{index}",
                "round": 1,
                "total_rounds": 3,
                "finish_class": finish_class,
                "finish_class_index": FINISH_CLASSES.index(finish_class),
                "model_name": "hierarchical-test",
                "test_year": year,
                "prob_finish": finish,
                "prob_submission_given_finish": submission,
                "prob_red_given_ko_tko": red_ko,
                "prob_red_given_submission": red_sub,
            }
            for column, value in zip(PROBABILITY_COLUMNS, combined):
                row[column] = value
            rows.append(row)
    return pd.DataFrame(rows)


class HierarchicalFinishHazardModelTests(unittest.TestCase):
    def test_hierarchical_probabilities_preserve_conditional_structure(self):
        combined = combine_hierarchical_probabilities(
            np.array([0.40]),
            np.array([0.25]),
            np.array([0.60]),
            np.array([0.70]),
        )[0]
        self.assertAlmostEqual(float(combined.sum()), 1.0)
        self.assertAlmostEqual(combined[0], 0.60)
        self.assertAlmostEqual(combined[1] + combined[3], 0.30)
        self.assertAlmostEqual(combined[2] + combined[4], 0.10)
        self.assertAlmostEqual(combined[1] / (combined[1] + combined[3]), 0.60)
        self.assertAlmostEqual(combined[2] / (combined[2] + combined[4]), 0.70)

    def test_current_year_labels_do_not_change_current_year_calibration(self):
        predictions = _prediction_frame()
        first, first_schedule = sequential_hierarchical_calibration(
            predictions,
            minimum_prior_rows=1,
        )

        mutated = predictions.copy()
        current = mutated["test_year"].eq(2023)
        mutated.loc[current, "finish_class"] = "blue_submission"
        mutated.loc[current, "finish_class_index"] = FINISH_CLASSES.index(
            "blue_submission"
        )
        second, second_schedule = sequential_hierarchical_calibration(
            mutated,
            minimum_prior_rows=1,
        )

        stage_columns = list(CALIBRATED_STAGE_COLUMNS.values())
        class_columns = [f"calibrated_{column}" for column in PROBABILITY_COLUMNS]
        first_current = first.loc[first["test_year"].eq(2023)].sort_values("fight_id")
        second_current = second.loc[second["test_year"].eq(2023)].sort_values("fight_id")
        np.testing.assert_allclose(
            first_current[[*stage_columns, *class_columns]].to_numpy(dtype=float),
            second_current[[*stage_columns, *class_columns]].to_numpy(dtype=float),
        )
        pd.testing.assert_frame_equal(
            first_schedule.loc[first_schedule["test_year"].eq(2023)].reset_index(
                drop=True
            ),
            second_schedule.loc[second_schedule["test_year"].eq(2023)].reset_index(
                drop=True
            ),
        )

    def test_calibrated_probabilities_remain_on_simplex(self):
        calibrated, schedule = sequential_hierarchical_calibration(
            _prediction_frame(),
            minimum_prior_rows=1,
        )
        class_columns = [f"calibrated_{column}" for column in PROBABILITY_COLUMNS]
        np.testing.assert_allclose(
            calibrated[class_columns].sum(axis=1).to_numpy(dtype=float),
            np.ones(len(calibrated)),
        )
        self.assertTrue(
            np.isfinite(
                schedule[
                    [
                        "factor_finish",
                        "factor_submission_given_finish",
                        "factor_red_given_ko_tko",
                        "factor_red_given_submission",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        )

    def test_one_class_stage_uses_smoothed_constant_fallback(self):
        x_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
        x_test = pd.DataFrame({"feature": [3.0, 4.0]})
        predicted, importance, estimator = _fit_binary_stage(
            x_train,
            np.ones(3, dtype=int),
            x_test,
            seed=7,
        )
        np.testing.assert_allclose(predicted, np.full(2, 0.8))
        np.testing.assert_allclose(importance, np.zeros(1))
        self.assertEqual(estimator, "smoothed_constant_fallback")

    def test_raw_stage_column_contract_is_complete(self):
        self.assertEqual(
            set(STAGE_PROBABILITY_COLUMNS),
            {
                "finish",
                "submission_given_finish",
                "red_given_ko_tko",
                "red_given_submission",
            },
        )


if __name__ == "__main__":
    unittest.main()
