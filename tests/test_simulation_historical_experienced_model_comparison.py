from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.historical_experienced_model_comparison import (
    REFERENCE_VARIANT,
    compare_experienced_model_candidates,
)
from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
)


def _frame(probabilities: list[float], variant: str) -> pd.DataFrame:
    rows = []
    for index, probability in enumerate(probabilities):
        actual_red = index % 2 == 0
        rows.append(
            {
                "fight_id": f"fight-{index}",
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index),
                "scheduled_rounds": 3,
                "red_prior_fights": [0, 3, 4, 6][index],
                "blue_prior_fights": [5, 3, 7, 8][index],
                "actual_winner_corner": "red" if actual_red else "blue",
                "actual_method": "decision" if index % 2 == 0 else "ko_tko",
                "actual_fight_time_seconds": 700.0,
                "actual_red_sig_attempted": 90.0,
                "actual_blue_sig_attempted": 80.0,
                "sim_red_win_probability": probability,
                "sim_decision_probability": 0.65 if index % 2 == 0 else 0.20,
                "sim_ko_tko_probability": 0.20 if index % 2 == 0 else 0.65,
                "sim_submission_probability": 0.15,
                "sim_fight_time_seconds": 690.0,
                "sim_red_sig_attempted": 88.0,
                "sim_blue_sig_attempted": 82.0,
                "baseline_red_win_probability": 0.50,
                "baseline_decision_probability": 0.45,
                "baseline_ko_tko_probability": 0.40,
                "baseline_submission_probability": 0.15,
                "baseline_fight_time_seconds": 650.0,
                "baseline_red_sig_attempted": 85.0,
                "baseline_blue_sig_attempted": 85.0,
                "simulator_version": variant,
            }
        )
    return pd.DataFrame(rows)


class ExperiencedModelComparisonTests(unittest.TestCase):
    def test_only_both_fighters_three_plus_determine_metrics(self):
        reference = _frame([0.10, 0.40, 0.70, 0.30], REFERENCE_VARIANT)
        candidate = _frame([0.99, 0.20, 0.80, 0.20], "candidate")
        result = compare_experienced_model_candidates(
            {
                REFERENCE_VARIANT: reference,
                "candidate": candidate,
            },
            minimum_prior_fights=3,
            bootstrap_samples=20,
            seed=7,
        )
        self.assertEqual(result.summary["eligible_fights"], 3)
        self.assertNotIn("fight-0", set(result.eligible_fights["fight_id"]))
        candidate_row = result.metrics.loc[
            result.metrics["variant"].eq("candidate")
        ].iloc[0]
        self.assertEqual(int(candidate_row["fights"]), 3)
        self.assertGreater(candidate_row["winner_accuracy"], 0.9)

    def test_mismatched_experienced_fight_sets_are_rejected(self):
        reference = _frame([0.60, 0.40, 0.70, 0.30], REFERENCE_VARIANT)
        candidate = _frame([0.65, 0.35, 0.75, 0.25], "candidate").iloc[:-1]
        with self.assertRaises(HistoricalSimulatorReplayError):
            compare_experienced_model_candidates(
                {
                    REFERENCE_VARIANT: reference,
                    "candidate": candidate,
                },
                bootstrap_samples=10,
            )


if __name__ == "__main__":
    unittest.main()
