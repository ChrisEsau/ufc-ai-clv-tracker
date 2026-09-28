from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.simulation.finish_survival_calibration import (
    apply_finish_survival_schedule,
    fit_finish_survival_schedule,
)


def _oof_predictions() -> pd.DataFrame:
    rows = []
    for year in (2022, 2023, 2024, 2025, 2026):
        for index in range(20):
            actual_finish = index < (8 if year < 2026 else 2)
            rows.append(
                {
                    "fight_id": f"{year}-{index}",
                    "round": 1,
                    "total_rounds": 3,
                    "model_name": "candidate",
                    "test_year": year,
                    "finish_class": "red_ko_tko" if actual_finish else "no_finish",
                    "calibrated_prob_no_finish": 0.80,
                    "calibrated_prob_red_ko_tko": 0.10,
                    "calibrated_prob_red_submission": 0.04,
                    "calibrated_prob_blue_ko_tko": 0.04,
                    "calibrated_prob_blue_submission": 0.02,
                }
            )
        for index in range(12):
            actual_finish = index < 3
            rows.append(
                {
                    "fight_id": f"{year}-r2-{index}",
                    "round": 2,
                    "total_rounds": 3,
                    "model_name": "candidate",
                    "test_year": year,
                    "finish_class": "blue_submission" if actual_finish else "no_finish",
                    "calibrated_prob_no_finish": 0.70,
                    "calibrated_prob_red_ko_tko": 0.10,
                    "calibrated_prob_red_submission": 0.05,
                    "calibrated_prob_blue_ko_tko": 0.08,
                    "calibrated_prob_blue_submission": 0.07,
                }
            )
    return pd.DataFrame(rows)


def _counterfactual() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fight_id": "holdout",
                "round": 1,
                "total_rounds": 3,
                "model_name": "candidate",
                "calibration_source": "prior_class_calibration",
                "calibrated_prob_no_finish": 0.80,
                "calibrated_prob_red_ko_tko": 0.10,
                "calibrated_prob_red_submission": 0.04,
                "calibrated_prob_blue_ko_tko": 0.04,
                "calibrated_prob_blue_submission": 0.02,
            },
            {
                "fight_id": "holdout",
                "round": 2,
                "total_rounds": 3,
                "model_name": "candidate",
                "calibration_source": "prior_class_calibration",
                "calibrated_prob_no_finish": 0.70,
                "calibrated_prob_red_ko_tko": 0.10,
                "calibrated_prob_red_submission": 0.05,
                "calibrated_prob_blue_ko_tko": 0.08,
                "calibrated_prob_blue_submission": 0.07,
            },
        ]
    )


class FinishSurvivalCalibrationTests(unittest.TestCase):
    def test_schedule_uses_only_years_before_target(self):
        original = _oof_predictions()
        first = fit_finish_survival_schedule(
            original,
            model_name="candidate",
            target_year=2026,
            group_prior_rows=0.0,
        )

        mutated = original.copy()
        mask = mutated["test_year"].eq(2026)
        mutated.loc[mask, "finish_class"] = "red_ko_tko"
        second = fit_finish_survival_schedule(
            mutated,
            model_name="candidate",
            target_year=2026,
            group_prior_rows=0.0,
        )
        pd.testing.assert_frame_equal(first, second)

    def test_round_specific_terminal_factor_adjusts_only_terminal_mass(self):
        schedule = fit_finish_survival_schedule(
            _oof_predictions(),
            model_name="candidate",
            target_year=2026,
            group_prior_rows=0.0,
        )
        result = apply_finish_survival_schedule(_counterfactual(), schedule)
        adjusted = result.predictions.sort_values("round").reset_index(drop=True)

        # Prior round-one actual finish rate is 40% while predicted is 20%,
        # so calibrated terminal probability must increase.
        self.assertGreater(
            1.0 - adjusted.loc[0, "calibrated_prob_no_finish"],
            0.20,
        )
        # Conditional terminal-class mix is preserved.
        before_mix = np.array([0.10, 0.04, 0.04, 0.02]) / 0.20
        after_terminal = 1.0 - adjusted.loc[0, "calibrated_prob_no_finish"]
        after_mix = adjusted.loc[
            0,
            [
                "calibrated_prob_red_ko_tko",
                "calibrated_prob_red_submission",
                "calibrated_prob_blue_ko_tko",
                "calibrated_prob_blue_submission",
            ],
        ].to_numpy(dtype=float) / after_terminal
        np.testing.assert_allclose(before_mix, after_mix)
        np.testing.assert_allclose(
            adjusted[
                [
                    "calibrated_prob_no_finish",
                    "calibrated_prob_red_ko_tko",
                    "calibrated_prob_red_submission",
                    "calibrated_prob_blue_ko_tko",
                    "calibrated_prob_blue_submission",
                ]
            ].sum(axis=1),
            np.ones(len(adjusted)),
        )

    def test_missing_round_factor_is_rejected(self):
        schedule = fit_finish_survival_schedule(
            _oof_predictions(),
            model_name="candidate",
            target_year=2026,
        )
        incomplete = schedule.loc[schedule["round"].eq(1)].copy()
        with self.assertRaises(Exception):
            apply_finish_survival_schedule(_counterfactual(), incomplete)


if __name__ == "__main__":
    unittest.main()
