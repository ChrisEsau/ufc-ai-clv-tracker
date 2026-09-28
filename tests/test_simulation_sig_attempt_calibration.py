from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.simulation.sig_attempt_calibration import (
    calibrate_walk_forward_predictions,
    gamma_poisson_overdispersion,
    multiplicative_mean_factor,
)


def make_predictions(year_2023_actual: float = 20.0) -> pd.DataFrame:
    rows: list[dict] = []
    for model_name in ("xgb_context", "xgb_context_rfs"):
        for year, actual in ((2022, 20.0), (2023, year_2023_actual), (2024, 20.0)):
            for index in range(1_200):
                predicted_count = 10.0
                rows.append(
                    {
                        "model_name": model_name,
                        "test_year": year,
                        "fight_id": f"{model_name}_{year}_{index // 2}",
                        "fighter_id": f"fighter_{index % 100}",
                        "round": 1,
                        "target_sig_attempted": actual,
                        "round_exposure_seconds": 300.0,
                        "predicted_rate_per_min": predicted_count / 5.0,
                        "predicted_count_at_actual_exposure": predicted_count,
                    }
                )
    return pd.DataFrame(rows)


class SigAttemptCalibrationTests(unittest.TestCase):
    def test_mean_factor_aligns_totals(self):
        factor = multiplicative_mean_factor(
            np.array([20.0, 20.0]),
            np.array([10.0, 10.0]),
        )
        self.assertEqual(factor, 2.0)

    def test_dispersion_is_positive_and_bounded(self):
        alpha = gamma_poisson_overdispersion(
            np.array([0.0, 10.0, 50.0, 100.0]),
            np.array([5.0, 10.0, 40.0, 80.0]),
        )
        self.assertGreater(alpha, 0.0)
        self.assertLessEqual(alpha, 5.0)

    def test_calibration_uses_only_prior_walk_forward_years(self):
        result = calibrate_walk_forward_predictions(
            make_predictions(),
            minimum_prior_rows=1_000,
        )
        schedule = result.schedule.loc[
            result.schedule["model_name"].eq("xgb_context")
        ].set_index("test_year")

        self.assertEqual(schedule.loc[2022, "calibration_source"], "cold_start_default")
        self.assertEqual(float(schedule.loc[2022, "calibration_factor"]), 1.0)
        self.assertAlmostEqual(float(schedule.loc[2023, "calibration_factor"]), 2.0)
        self.assertAlmostEqual(float(schedule.loc[2024, "calibration_factor"]), 2.0)

    def test_current_year_outcomes_do_not_change_current_year_factor(self):
        original = calibrate_walk_forward_predictions(
            make_predictions(year_2023_actual=20.0),
            minimum_prior_rows=1_000,
        )
        mutated = calibrate_walk_forward_predictions(
            make_predictions(year_2023_actual=10.0),
            minimum_prior_rows=1_000,
        )

        def factor(result, year):
            row = result.schedule.loc[
                result.schedule["model_name"].eq("xgb_context")
                & result.schedule["test_year"].eq(year)
            ].iloc[0]
            return float(row["calibration_factor"])

        self.assertAlmostEqual(factor(original, 2023), factor(mutated, 2023))
        self.assertNotEqual(factor(original, 2024), factor(mutated, 2024))

    def test_calibrated_predictions_include_distribution_contract(self):
        result = calibrate_walk_forward_predictions(
            make_predictions(),
            minimum_prior_rows=1_000,
        )
        self.assertIn("calibrated_rate_per_min", result.predictions.columns)
        self.assertIn("gamma_poisson_overdispersion", result.predictions.columns)
        self.assertTrue(
            result.predictions["gamma_poisson_overdispersion"].gt(0).all()
        )
        self.assertEqual(set(result.final_parameters["model_name"]), {
            "xgb_context",
            "xgb_context_rfs",
        })


if __name__ == "__main__":
    unittest.main()
