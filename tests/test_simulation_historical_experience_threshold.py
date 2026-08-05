from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.historical_experience_threshold_diagnostics import (
    audit_experience_threshold,
)
from tests.test_simulation_historical_replay_evaluation import _prediction_frame


class HistoricalExperienceThresholdTests(unittest.TestCase):
    def test_flags_either_fighter_below_three_prior_fights(self):
        predictions = _prediction_frame(
            [0.80, 0.20, 0.75, 0.25, 0.70, 0.30, 0.65, 0.35]
        )
        result = audit_experience_threshold(
            predictions,
            minimum_prior_fights=3,
        )

        flagged = result.flagged_predictions.loc[
            result.flagged_predictions["low_experience_flag"]
        ]
        experienced = result.flagged_predictions.loc[
            ~result.flagged_predictions["low_experience_flag"]
        ]
        self.assertEqual(set(flagged["fight_id"]), {"fight-0", "fight-1", "fight-4", "fight-5"})
        self.assertEqual(set(experienced["fight_id"]), {"fight-2", "fight-3", "fight-6", "fight-7"})
        self.assertTrue((experienced["minimum_prior_fights"] >= 3).all())
        self.assertEqual(result.summary["flagged_fights"], 4)
        self.assertEqual(result.summary["experienced_only_fights"], 4)
        self.assertFalse(result.summary["probabilities_changed"])

    def test_metrics_include_all_flagged_and_excluded_cohorts(self):
        predictions = _prediction_frame(
            [0.80, 0.20, 0.75, 0.25, 0.70, 0.30, 0.65, 0.35]
        )
        result = audit_experience_threshold(predictions, minimum_prior_fights=3)
        metrics = result.metrics.set_index("cohort")

        self.assertEqual(
            set(metrics.index),
            {
                "all_fights_included",
                "flagged_either_under_3",
                "excluding_flagged_both_3_plus",
            },
        )
        self.assertEqual(int(metrics.loc["all_fights_included", "fights"]), 8)
        self.assertEqual(int(metrics.loc["flagged_either_under_3", "fights"]), 4)
        self.assertEqual(
            int(metrics.loc["excluding_flagged_both_3_plus", "fights"]), 4
        )
        self.assertAlmostEqual(
            float(metrics.loc["all_fights_included", "winner_accuracy"]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
