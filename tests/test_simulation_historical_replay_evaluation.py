from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.historical_replay_evaluation import (
    HISTORICAL_BASELINE,
    RECOMMENDED_VARIANT,
    evaluate_historical_replay_cohort,
)
from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
)


def _prediction_frame(red_probabilities: list[float]) -> pd.DataFrame:
    rows = []
    actual_red = [True, False, True, False, True, False, True, False]
    methods = [
        "decision",
        "ko_tko",
        "submission",
        "decision",
        "ko_tko",
        "submission",
        "decision",
        "ko_tko",
    ]
    prior_pairs = [(0, 4), (1, 2), (3, 5), (6, 8), (0, 0), (2, 7), (4, 4), (9, 11)]
    for index, probability in enumerate(red_probabilities):
        method = methods[index]
        method_probabilities = {
            "decision": 0.70 if method == "decision" else 0.15,
            "ko_tko": 0.70 if method == "ko_tko" else 0.15,
            "submission": 0.70 if method == "submission" else 0.15,
        }
        red_prior, blue_prior = prior_pairs[index]
        rows.append(
            {
                "fight_id": f"fight-{index}",
                "event_id": f"event-{index // 2}",
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index * 14),
                "scheduled_rounds": 5 if index == 7 else 3,
                "red_fighter_id": f"red-{index}",
                "red_fighter_name": f"Red {index}",
                "blue_fighter_id": f"blue-{index}",
                "blue_fighter_name": f"Blue {index}",
                "red_prior_fights": red_prior,
                "blue_prior_fights": blue_prior,
                "actual_winner_corner": "red" if actual_red[index] else "blue",
                "actual_method": method,
                "actual_fight_time_seconds": 700.0 + index,
                "actual_red_sig_attempted": 100.0 + index,
                "actual_blue_sig_attempted": 90.0 + index,
                "sim_red_win_probability": probability,
                "sim_decision_probability": method_probabilities["decision"],
                "sim_ko_tko_probability": method_probabilities["ko_tko"],
                "sim_submission_probability": method_probabilities["submission"],
                "sim_fight_time_seconds": 705.0 + index,
                "sim_red_sig_attempted": 102.0 + index,
                "sim_blue_sig_attempted": 88.0 + index,
                "baseline_red_win_probability": 0.50,
                "baseline_decision_probability": 0.45,
                "baseline_ko_tko_probability": 0.40,
                "baseline_submission_probability": 0.15,
                "baseline_fight_time_seconds": 650.0,
                "baseline_red_sig_attempted": 95.0,
                "baseline_blue_sig_attempted": 95.0,
                "simulations": 500,
                "simulator_version": "test",
            }
        )
    return pd.DataFrame(rows)


def _training_metadata() -> pd.DataFrame:
    rows = []
    for index in range(8):
        for corner in ("red", "blue"):
            rows.append(
                {
                    "fight_id": f"fight-{index}",
                    "fighter_id": f"{corner}-{index}",
                    "round": 1,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index * 14),
                    "event_id": f"event-{index // 2}",
                    "event_name": f"Event {index // 2}",
                    "division": "Women's Strawweight" if index < 2 else "Lightweight",
                    "title_fight": index == 7,
                }
            )
    return pd.DataFrame(rows)


class HistoricalReplayEvaluationTests(unittest.TestCase):
    def test_evaluation_adds_baseline_and_subgroup_diagnostics(self):
        recommended = _prediction_frame([0.80, 0.20, 0.75, 0.25, 0.70, 0.30, 0.65, 0.35])
        heuristic = _prediction_frame([0.55, 0.55, 0.52, 0.52, 0.51, 0.51, 0.50, 0.50])

        result = evaluate_historical_replay_cohort(
            {
                RECOMMENDED_VARIANT: recommended,
                "heuristic_simulator": heuristic,
            },
            _training_metadata(),
            minimum_group_size=2,
            bootstrap_samples=40,
            seed=17,
        )

        self.assertIn(HISTORICAL_BASELINE, result.enriched_predictions)
        self.assertEqual(result.summary["fights"], 8)
        self.assertEqual(result.summary["single_card_role"], "smoke_test_only")
        experience = result.subgroup_metrics.loc[
            (result.subgroup_metrics["variant"] == RECOMMENDED_VARIANT)
            & (result.subgroup_metrics["segment"] == "experience")
        ]
        self.assertEqual(
            set(experience["subgroup"]),
            {"0_cold_start", "1_2_prior", "3_5_prior", "6_plus_prior"},
        )
        self.assertIn("women", set(result.enriched_predictions[RECOMMENDED_VARIANT]["sex_segment_inferred"]))
        self.assertFalse(result.calibration.empty)
        self.assertFalse(result.stability_metrics.empty)

    def test_recommended_variant_has_lower_paired_winner_brier(self):
        recommended = _prediction_frame([0.80, 0.20, 0.75, 0.25, 0.70, 0.30, 0.65, 0.35])
        heuristic = _prediction_frame([0.60, 0.60, 0.55, 0.55, 0.52, 0.52, 0.51, 0.51])
        result = evaluate_historical_replay_cohort(
            {
                RECOMMENDED_VARIANT: recommended,
                "heuristic_simulator": heuristic,
            },
            _training_metadata(),
            minimum_group_size=1,
            bootstrap_samples=40,
            seed=23,
        )
        row = result.paired_variant_deltas.loc[
            (result.paired_variant_deltas["comparator_variant"] == "heuristic_simulator")
            & (result.paired_variant_deltas["metric"] == "winner_brier")
        ].iloc[0]
        self.assertLess(row["reference_minus_comparator"], 0.0)
        self.assertTrue(row["favors_reference"])

    def test_mismatched_variant_fight_sets_are_rejected(self):
        recommended = _prediction_frame([0.80, 0.20, 0.75, 0.25, 0.70, 0.30, 0.65, 0.35])
        incomplete = recommended.iloc[:-1].copy()
        with self.assertRaises(HistoricalSimulatorReplayError):
            evaluate_historical_replay_cohort(
                {
                    RECOMMENDED_VARIANT: recommended,
                    "heuristic_simulator": incomplete,
                },
                _training_metadata(),
                bootstrap_samples=10,
            )


if __name__ == "__main__":
    unittest.main()
