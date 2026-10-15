from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.historical_submission_diagnostics import (
    HistoricalSubmissionDiagnosticError,
    _recommendation,
    aggregate_finish_hazards,
    audit_submission_failures,
)


def _fighter_round(
    *,
    fight_id: str,
    date: str,
    fighter_id: str,
    opponent_id: str,
    corner: str,
    winner_id: str,
    method: str,
    round_number: int,
    attempts: float,
    td_attempts: float,
    td_landed: float,
    control: float,
    submission_attempts: float,
) -> dict[str, object]:
    return {
        "fight_id": fight_id,
        "fighter_id": fighter_id,
        "opponent_id": opponent_id,
        "corner": corner,
        "date": date,
        "round": round_number,
        "total_rounds": 3,
        "winner_id": winner_id,
        "method_family": method,
        "match_time_sec": float((round_number - 1) * 300 + 180),
        "target_finish_time_in_round_seconds": 180.0,
        "target_sig_attempted": attempts,
        "target_sig_landed": attempts * 0.45,
        "target_td_attempted": td_attempts,
        "target_td_landed": td_landed,
        "target_control_seconds": control,
        "target_knockdowns": 0.0,
        "target_submission_attempts": submission_attempts,
        "division": "lightweight",
        "title_fight": False,
        "fighter_name": fighter_id,
        "opponent_name": opponent_id,
    }


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fight_id, date, red, blue, winner, method in (
        ("pre-ab", "2025-02-01", "a", "b", "a", "decision"),
        ("pre-cd", "2025-03-01", "c", "d", "d", "decision"),
        ("holdout-1", "2026-02-01", "a", "b", "a", "submission"),
        ("holdout-2", "2026-03-01", "c", "d", "d", "submission"),
    ):
        observed_rounds = 3 if method == "decision" else 1
        for round_number in range(1, observed_rounds + 1):
            rows.append(
                _fighter_round(
                    fight_id=fight_id,
                    date=date,
                    fighter_id=red,
                    opponent_id=blue,
                    corner="red",
                    winner_id=winner,
                    method=method,
                    round_number=round_number,
                    attempts=30.0,
                    td_attempts=2.0,
                    td_landed=1.0,
                    control=45.0,
                    submission_attempts=1.0,
                )
            )
            rows.append(
                _fighter_round(
                    fight_id=fight_id,
                    date=date,
                    fighter_id=blue,
                    opponent_id=red,
                    corner="blue",
                    winner_id=winner,
                    method=method,
                    round_number=round_number,
                    attempts=26.0,
                    td_attempts=1.0,
                    td_landed=0.0,
                    control=15.0,
                    submission_attempts=0.0,
                )
            )
    return pd.DataFrame(rows)


def _finish_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fight_id in ("holdout-1", "holdout-2"):
        for round_number in (1, 2, 3):
            rows.append(
                {
                    "fight_id": fight_id,
                    "round": round_number,
                    "total_rounds": 3,
                    "calibrated_prob_no_finish": 0.70,
                    "calibrated_prob_red_ko_tko": 0.05,
                    "calibrated_prob_red_submission": 0.15,
                    "calibrated_prob_blue_ko_tko": 0.05,
                    "calibrated_prob_blue_submission": 0.05,
                    "model_name": "test",
                    "model_version": "test-v0",
                }
            )
    return pd.DataFrame(rows)


def _simulator_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fight_id": "holdout-1",
                "date": "2026-02-01",
                "scheduled_rounds": 3,
                "red_fighter_id": "a",
                "blue_fighter_id": "b",
                "red_prior_fights": 1,
                "blue_prior_fights": 1,
                "actual_winner_corner": "red",
                "actual_method": "submission",
                "sim_red_win_probability": 0.70,
                "sim_decision_probability": 0.20,
                "sim_ko_tko_probability": 0.20,
                "sim_submission_probability": 0.60,
            },
            {
                "fight_id": "holdout-2",
                "date": "2026-03-01",
                "scheduled_rounds": 3,
                "red_fighter_id": "c",
                "blue_fighter_id": "d",
                "red_prior_fights": 1,
                "blue_prior_fights": 1,
                "actual_winner_corner": "blue",
                "actual_method": "submission",
                "sim_red_win_probability": 0.65,
                "sim_decision_probability": 0.20,
                "sim_ko_tko_probability": 0.60,
                "sim_submission_probability": 0.20,
            },
        ]
    )


class HistoricalSubmissionDiagnosticTests(unittest.TestCase):
    def test_round_hazards_convert_to_unconditional_fight_mass(self):
        frame = pd.DataFrame(
            [
                {
                    "fight_id": "f",
                    "round": 1,
                    "total_rounds": 2,
                    "calibrated_prob_no_finish": 0.50,
                    "calibrated_prob_red_ko_tko": 0.10,
                    "calibrated_prob_red_submission": 0.20,
                    "calibrated_prob_blue_ko_tko": 0.10,
                    "calibrated_prob_blue_submission": 0.10,
                },
                {
                    "fight_id": "f",
                    "round": 2,
                    "total_rounds": 2,
                    "calibrated_prob_no_finish": 0.60,
                    "calibrated_prob_red_ko_tko": 0.10,
                    "calibrated_prob_red_submission": 0.10,
                    "calibrated_prob_blue_ko_tko": 0.10,
                    "calibrated_prob_blue_submission": 0.10,
                },
            ]
        )
        result = aggregate_finish_hazards(frame).iloc[0]
        self.assertAlmostEqual(result["provider_decision_probability"], 0.30)
        self.assertAlmostEqual(
            result["provider_red_submission_probability"],
            0.25,
        )
        self.assertAlmostEqual(
            result["provider_blue_submission_probability"],
            0.15,
        )
        self.assertAlmostEqual(
            result["provider_total_submission_probability"],
            0.40,
        )
        self.assertAlmostEqual(
            result["provider_conditional_red_submission_share"],
            0.625,
        )
        self.assertAlmostEqual(
            result["provider_expected_submission_round"],
            1.25,
        )

    def test_audit_assigns_method_and_side_error_classes(self):
        result = audit_submission_failures(
            _simulator_predictions(),
            _finish_predictions(),
            _training_frame(),
            test_year=2026,
            minimum_group_size=1,
        )
        diagnostics = result.fight_diagnostics.set_index("fight_id")
        self.assertEqual(
            diagnostics.loc["holdout-1", "error_class"],
            "correct_method_correct_side",
        )
        self.assertEqual(
            diagnostics.loc["holdout-2", "error_class"],
            "wrong_method_wrong_side",
        )
        self.assertEqual(result.summary["status"], "evaluation_only")
        self.assertEqual(result.summary["metrics"]["actual_submission_fights"], 2)

    def test_holdout_target_mutation_does_not_change_prefight_state(self):
        original = _training_frame()
        first = audit_submission_failures(
            _simulator_predictions(),
            _finish_predictions(),
            original,
            test_year=2026,
            minimum_group_size=1,
        ).fight_diagnostics.sort_values("fight_id")

        mutated = original.copy()
        holdout = pd.to_datetime(mutated["date"]).dt.year.eq(2026)
        mutated.loc[holdout, "target_td_attempted"] = 5000.0
        mutated.loc[holdout, "target_td_landed"] = 4000.0
        mutated.loc[holdout, "target_control_seconds"] = 20000.0
        mutated.loc[holdout, "target_submission_attempts"] = 3000.0
        second = audit_submission_failures(
            _simulator_predictions(),
            _finish_predictions(),
            mutated,
            test_year=2026,
            minimum_group_size=1,
        ).fight_diagnostics.sort_values("fight_id")

        state_columns = [
            column for column in first.columns if column.startswith("state_")
        ]
        pd.testing.assert_frame_equal(
            first[state_columns].reset_index(drop=True),
            second[state_columns].reset_index(drop=True),
        )

    def test_mismatched_fight_sets_are_rejected(self):
        finish = _finish_predictions().loc[
            lambda frame: frame["fight_id"].eq("holdout-1")
        ]
        with self.assertRaises(HistoricalSubmissionDiagnosticError):
            audit_submission_failures(
                _simulator_predictions(),
                finish,
                _training_frame(),
                test_year=2026,
            )

    def test_recommendation_routes_side_winner_disagreement_to_grappling(self):
        action, _ = _recommendation(
            {
                "actual_submission_fights": 60,
                "submission_method_detection_rate": 0.60,
                "submission_side_accuracy": 0.70,
                "simulator_winner_accuracy_on_submissions": 0.45,
                "submission_side_vs_simulator_winner_disagreement_rate": 0.35,
            }
        )
        self.assertEqual(action, "stateful_grappling_and_scoring_provider")


if __name__ == "__main__":
    unittest.main()
