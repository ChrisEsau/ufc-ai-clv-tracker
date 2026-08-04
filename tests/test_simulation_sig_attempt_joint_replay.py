from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.simulation.sig_attempt_joint_replay import (
    _pair_rows,
    estimate_gaussian_copula_rho,
    sample_gaussian_copula_pairs,
    sequential_joint_strike_replay,
)


def make_predictions(year_2023_shift: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows: list[dict] = []
    for year in (2022, 2023, 2024):
        for pair_number in range(600):
            shared = int(rng.poisson(12.0))
            first_actual = int(rng.poisson(18.0) + shared)
            second_actual = int(rng.poisson(16.0) + shared)
            if year == 2023:
                second_actual += year_2023_shift
            for fighter_id, opponent_id, actual, mean in (
                (f"A_{year}_{pair_number}", f"B_{year}_{pair_number}", first_actual, 30.0),
                (f"B_{year}_{pair_number}", f"A_{year}_{pair_number}", second_actual, 28.0),
            ):
                rows.append(
                    {
                        "fight_id": f"F_{year}_{pair_number}",
                        "fighter_id": fighter_id,
                        "opponent_id": opponent_id,
                        "round": 1,
                        "test_year": year,
                        "model_name": "xgb_context_rfs",
                        "target_sig_attempted": actual,
                        "calibrated_count_at_actual_exposure": mean,
                        "gamma_poisson_overdispersion": 0.20,
                    }
                )
    return pd.DataFrame(rows)


class JointStrikeReplayTests(unittest.TestCase):
    def test_estimated_rho_is_positive_for_shared_counts(self):
        pairs = _pair_rows(make_predictions())
        rho = estimate_gaussian_copula_rho(pairs)
        self.assertGreater(rho, 0.10)
        self.assertLess(rho, 0.95)

    def test_gaussian_copula_preserves_marginal_means_and_adds_dependence(self):
        pairs = _pair_rows(make_predictions()).head(100)
        rng = np.random.default_rng(9)
        independent_1, independent_2 = sample_gaussian_copula_pairs(
            rng,
            pairs,
            rho=0.0,
            simulations=2_000,
        )
        rng = np.random.default_rng(9)
        shared_1, shared_2 = sample_gaussian_copula_pairs(
            rng,
            pairs,
            rho=0.55,
            simulations=2_000,
        )

        self.assertAlmostEqual(shared_1.mean(), pairs["mean_1"].mean(), delta=0.8)
        self.assertAlmostEqual(shared_2.mean(), pairs["mean_2"].mean(), delta=0.8)
        independent_corr = float(
            np.corrcoef(independent_1.ravel(), independent_2.ravel())[0, 1]
        )
        shared_corr = float(np.corrcoef(shared_1.ravel(), shared_2.ravel())[0, 1])
        self.assertLess(abs(independent_corr), 0.05)
        self.assertGreater(shared_corr, 0.25)

    def test_current_year_outcomes_do_not_change_current_year_rho(self):
        original = sequential_joint_strike_replay(
            make_predictions(year_2023_shift=0),
            minimum_prior_pairs=500,
            simulations=25,
        )
        mutated = sequential_joint_strike_replay(
            make_predictions(year_2023_shift=20),
            minimum_prior_pairs=500,
            simulations=25,
        )

        def rho(result, year):
            return float(
                result.dependence_schedule.loc[
                    result.dependence_schedule["test_year"].eq(year),
                    "gaussian_copula_rho",
                ].iloc[0]
            )

        self.assertEqual(rho(original, 2022), 0.0)
        self.assertAlmostEqual(rho(original, 2023), rho(mutated, 2023))
        self.assertNotEqual(rho(original, 2024), rho(mutated, 2024))

    def test_joint_replay_compares_independent_and_copula_models(self):
        result = sequential_joint_strike_replay(
            make_predictions(),
            minimum_prior_pairs=500,
            simulations=50,
        )
        self.assertEqual(
            set(result.correlation_metrics["joint_model"]),
            {"independent", "gaussian_copula"},
        )
        self.assertEqual(
            set(result.total_interval_coverage["nominal_coverage"]),
            {0.5, 0.8, 0.9},
        )
        self.assertEqual(len(result.final_dependence), 1)
        self.assertGreater(
            float(result.final_dependence.iloc[0]["gaussian_copula_rho"]),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
