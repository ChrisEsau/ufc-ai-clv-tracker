from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from pipeline.simulation.round_parameter_provider import (
    HistoricalSignificantStrikeProvider,
    RoundParameterKey,
    RoundParameterProviderError,
    SignificantStrikeAttemptParameters,
)
from pipeline.simulation.sig_attempt_replay import (
    gamma_poisson_logpmf,
    replay_calibrated_strike_distribution,
)


def make_predictions() -> pd.DataFrame:
    rows: list[dict] = []
    actual_pairs = [
        (20, 18),
        (40, 35),
        (60, 55),
        (25, 30),
        (75, 65),
        (15, 12),
    ]
    for fight_number, (red_actual, blue_actual) in enumerate(actual_pairs, start=1):
        for fighter_id, opponent_id, actual, calibrated_mean in (
            (f"R{fight_number}", f"B{fight_number}", red_actual, red_actual * 0.95 + 1.0),
            (f"B{fight_number}", f"R{fight_number}", blue_actual, blue_actual * 0.95 + 1.0),
        ):
            rows.append(
                {
                    "fight_id": f"F{fight_number}",
                    "fighter_id": fighter_id,
                    "opponent_id": opponent_id,
                    "round": 1,
                    "test_year": 2024 + (fight_number % 2),
                    "model_name": "xgb_context_rfs",
                    "target_sig_attempted": actual,
                    "round_exposure_seconds": 300.0,
                    "predicted_rate_per_min": calibrated_mean / 1.1 / 5.0,
                    "predicted_count_at_actual_exposure": calibrated_mean / 1.1,
                    "calibration_factor": 1.1,
                    "gamma_poisson_overdispersion": 0.20,
                    "calibrated_rate_per_min": calibrated_mean / 5.0,
                    "calibrated_count_at_actual_exposure": calibrated_mean,
                }
            )
    return pd.DataFrame(rows)


class StrikeProviderTests(unittest.TestCase):
    def test_provider_expected_count_matches_calibrated_mean(self):
        predictions = make_predictions()
        provider = HistoricalSignificantStrikeProvider(
            predictions,
            model_name="xgb_context_rfs",
        )
        row = predictions.iloc[0]
        parameters = provider.significant_strike_attempts(
            RoundParameterKey(
                fight_id=row["fight_id"],
                fighter_id=row["fighter_id"],
                round=int(row["round"]),
            )
        )
        self.assertAlmostEqual(
            parameters.expected_count(row["round_exposure_seconds"]),
            row["calibrated_count_at_actual_exposure"],
        )

    def test_provider_sampler_matches_mean_and_variance_contract(self):
        parameters = SignificantStrikeAttemptParameters(
            key=RoundParameterKey("F", "A", 1),
            mean_rate_per_minute=8.0,
            gamma_poisson_overdispersion=0.20,
            model_name="test",
            model_version="v0",
            calibration_factor=1.0,
        )
        rng = np.random.default_rng(9)
        samples = np.asarray(
            [parameters.sample_count(rng, 300.0) for _ in range(50_000)],
            dtype=float,
        )
        expected_mean = 40.0
        expected_variance = expected_mean + 0.20 * expected_mean**2
        self.assertAlmostEqual(samples.mean(), expected_mean, delta=0.40)
        # Sampling variance itself is noisy. This bound remains tight relative to
        # the 360-count theoretical variance while avoiding seed-specific flakes.
        self.assertAlmostEqual(samples.var(), expected_variance, delta=12.0)

    def test_provider_rejects_duplicate_keys(self):
        predictions = make_predictions()
        duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
        with self.assertRaises(RoundParameterProviderError):
            HistoricalSignificantStrikeProvider(
                duplicate,
                model_name="xgb_context_rfs",
            )


class StrikeReplayTests(unittest.TestCase):
    def test_gamma_poisson_logpmf_matches_scipy(self):
        observed = np.array([0.0, 5.0, 20.0])
        mean = np.array([2.0, 8.0, 18.0])
        alpha = np.array([0.2, 0.2, 0.2])
        size = 1.0 / alpha
        probability = size / (size + mean)
        expected = nbinom.logpmf(observed, size, probability)
        actual = gamma_poisson_logpmf(observed, mean, alpha)
        np.testing.assert_allclose(actual, expected)

    def test_replay_produces_distribution_and_pair_diagnostics(self):
        result = replay_calibrated_strike_distribution(make_predictions())
        self.assertEqual(
            set(result.aggregate_metrics["distribution"]),
            {
                "raw_poisson",
                "calibrated_poisson",
                "calibrated_gamma_poisson",
            },
        )
        self.assertEqual(len(result.interval_coverage), 9)
        self.assertEqual(
            set(result.pair_diagnostics["diagnostic"]),
            {
                "paired_actual_count_correlation",
                "paired_predicted_mean_correlation",
                "paired_residual_correlation",
            },
        )
        self.assertTrue(
            np.isfinite(result.aggregate_metrics["mean_negative_log_likelihood"]).all()
        )
        self.assertEqual(len(result.calibration_deciles), 10)


if __name__ == "__main__":
    unittest.main()
