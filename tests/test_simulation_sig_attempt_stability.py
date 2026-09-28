from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.sig_attempt_stability import (
    build_recency_weights,
    evaluate_stability_gates,
    select_compact_rfs_columns,
)


class SigAttemptStabilityTests(unittest.TestCase):
    def test_compact_selector_is_deterministic_and_excludes_last3_noise(self):
        frame = pd.DataFrame(
            {
                "fighter_rfs_suppress_exp_opp_sig_attempt_delta": [0.1],
                "opponent_rfs_traj_ewm_sig_attempt_slope": [0.2],
                "fighter_rfs_traj_last3_sig_attempt_slope": [0.3],
                "fighter_rfs_wrestle_exp_submission_pressure_score": [0.4],
                "fighter_rfs_traj_prior_valid_trajectory_count": [3.0],
                "fighter_trajectory_state_available": [1.0],
                "target_sig_attempted": [20.0],
            }
        )
        selected = select_compact_rfs_columns(frame)
        self.assertIn(
            "fighter_rfs_suppress_exp_opp_sig_attempt_delta", selected
        )
        self.assertIn("opponent_rfs_traj_ewm_sig_attempt_slope", selected)
        self.assertIn(
            "fighter_rfs_traj_prior_valid_trajectory_count", selected
        )
        self.assertIn("fighter_trajectory_state_available", selected)
        self.assertNotIn(
            "fighter_rfs_traj_last3_sig_attempt_slope", selected
        )
        self.assertNotIn(
            "fighter_rfs_wrestle_exp_submission_pressure_score", selected
        )

    def test_recency_weights_favor_newer_rows_and_preserve_mean(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2016-01-01", "2020-01-01", "2025-01-01"]
                ),
                "exposure_weight": [1.0, 1.0, 1.0],
            }
        )
        weights = build_recency_weights(frame, test_year=2026)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_gate_blocks_latest_year_regression(self):
        aggregate = pd.DataFrame(
            [
                {
                    "model_name": "xgb_context",
                    "calibration": "sequential_mean_calibrated",
                    "count_poisson_deviance": 10.0,
                    "mean_count_bias": -0.5,
                },
                {
                    "model_name": "xgb_context_rfs_compact",
                    "calibration": "sequential_mean_calibrated",
                    "count_poisson_deviance": 9.8,
                    "mean_count_bias": -0.4,
                },
                {
                    "model_name": "xgb_context_rfs_compact_recent",
                    "calibration": "sequential_mean_calibrated",
                    "count_poisson_deviance": 9.7,
                    "mean_count_bias": -0.3,
                },
            ]
        )
        fold_rows = []
        for year in (2022, 2023, 2024, 2025, 2026):
            fold_rows.append(
                {
                    "model_name": "xgb_context",
                    "calibration": "sequential_mean_calibrated",
                    "test_year": year,
                    "count_poisson_deviance": 10.0,
                }
            )
            fold_rows.append(
                {
                    "model_name": "xgb_context_rfs_compact",
                    "calibration": "sequential_mean_calibrated",
                    "test_year": year,
                    "count_poisson_deviance": 9.8 if year < 2026 else 10.2,
                }
            )
            fold_rows.append(
                {
                    "model_name": "xgb_context_rfs_compact_recent",
                    "calibration": "sequential_mean_calibrated",
                    "test_year": year,
                    "count_poisson_deviance": 9.7,
                }
            )
        subgroup = pd.DataFrame(
            [
                {
                    "model_name": model,
                    "calibration": "sequential_mean_calibrated",
                    "group_name": "round",
                    "group_value": "1",
                    "count_poisson_deviance": value,
                }
                for model, value in (
                    ("xgb_context", 10.0),
                    ("xgb_context_rfs_compact", 9.9),
                    ("xgb_context_rfs_compact_recent", 9.8),
                )
            ]
        )
        gates = evaluate_stability_gates(
            aggregate,
            pd.DataFrame(fold_rows),
            subgroup,
        ).set_index("candidate_model")
        self.assertEqual(
            gates.loc["xgb_context_rfs_compact", "gate_status"], "blocked"
        )
        self.assertFalse(
            bool(gates.loc["xgb_context_rfs_compact", "latest_year_pass"])
        )
        self.assertEqual(
            gates.loc["xgb_context_rfs_compact_recent", "gate_status"], "pass"
        )

    def test_recency_weight_validation(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01"]),
                "exposure_weight": [1.0],
            }
        )
        with self.assertRaises(Exception):
            build_recency_weights(frame, test_year=2026, half_life_years=0.0)
        with self.assertRaises(Exception):
            build_recency_weights(frame, test_year=2026, floor=1.1)


if __name__ == "__main__":
    unittest.main()
