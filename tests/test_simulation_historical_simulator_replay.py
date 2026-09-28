from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.historical_simulator_replay import (
    build_fighter_fight_history,
    population_priors,
    run_historical_simulator_replay,
)


def make_training_rows() -> pd.DataFrame:
    rows: list[dict] = []
    fights = [
        ("F1", "2024-01-01", "A", "B", "A", "decision", 900, 3),
        ("F2", "2024-06-01", "C", "D", "C", "ko_tko", 220, 3),
        ("F3", "2025-01-01", "A", "C", "C", "submission", 410, 3),
        ("F4", "2025-08-01", "B", "D", "B", "decision", 900, 3),
        ("F5", "2026-02-01", "A", "D", "A", "ko_tko", 430, 3),
        ("F6", "2026-05-01", "B", "C", "C", "decision", 900, 3),
    ]
    for fight_id, date, red, blue, winner, method, fight_time, scheduled in fights:
        finish_round = min(scheduled, max(1, (fight_time - 1) // 300 + 1))
        for corner, fighter, opponent in (("red", red, blue), ("blue", blue, red)):
            for round_number in range(1, finish_round + 1):
                completed_before = (round_number - 1) * 300
                exposure = min(300, max(1, fight_time - completed_before))
                base = 28 + (ord(fighter[0]) - ord("A")) * 2 + round_number
                rows.append(
                    {
                        "fight_id": fight_id,
                        "fighter_id": fighter,
                        "opponent_id": opponent,
                        "fighter_name": f"Fighter {fighter}",
                        "opponent_name": f"Fighter {opponent}",
                        "corner": corner,
                        "date": date,
                        "round": round_number,
                        "total_rounds": scheduled,
                        "winner_id": winner,
                        "method_family": method,
                        "match_time_sec": fight_time,
                        "target_finish_time_in_round_seconds": exposure,
                        "target_sig_attempted": base,
                        "target_sig_landed": int(base * 0.45),
                        "target_td_attempted": 2 + (round_number % 2),
                        "target_td_landed": 1,
                        "target_control_seconds": 35.0,
                        "target_knockdowns": int(method == "ko_tko" and fighter == winner and round_number == finish_round),
                        "target_submission_attempts": int(method == "submission" and fighter == winner),
                        "event_id": f"E_{date}",
                    }
                )
    return pd.DataFrame(rows)


class HistoricalSimulatorReplayTests(unittest.TestCase):
    def test_current_fight_targets_do_not_change_prefight_history(self):
        original = make_training_rows()
        mutated = original.copy()
        mask = mutated["fight_id"].eq("F5")
        mutated.loc[mask, "target_sig_attempted"] = 999
        mutated.loc[mask, "target_sig_landed"] = 900

        original_history = build_fighter_fight_history(original)
        mutated_history = build_fighter_fight_history(mutated)
        columns = [column for column in original_history.columns if column.startswith("prior_")]
        original_f5 = original_history.loc[
            original_history["fight_id"].eq("F5"),
            ["fighter_id", *columns],
        ].sort_values("fighter_id").reset_index(drop=True)
        mutated_f5 = mutated_history.loc[
            mutated_history["fight_id"].eq("F5"),
            ["fighter_id", *columns],
        ].sort_values("fighter_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(original_f5, mutated_f5)

    def test_population_priors_use_only_pre_holdout_rows(self):
        history = build_fighter_fight_history(make_training_rows())
        original = population_priors(history, test_year=2026)
        mutated = history.copy()
        mask = mutated["date"].dt.year.eq(2026)
        mutated.loc[mask, "fight_sig_attempted"] = 100000
        updated = population_priors(mutated, test_year=2026)
        self.assertEqual(original, updated)

    def test_small_end_to_end_replay(self):
        result = run_historical_simulator_replay(
            make_training_rows(),
            test_year=2026,
            simulations_per_fight=20,
            seed=3,
        )
        self.assertEqual(len(result.fight_predictions), 2)
        self.assertIn("winner", set(result.metrics["task"]))
        self.assertIn("method", set(result.metrics["task"]))
        self.assertIn("fight_time_seconds", set(result.metrics["task"]))
        self.assertIn("fighter_sig_attempted", set(result.metrics["task"]))
        self.assertEqual(
            set(result.fight_predictions["simulator_version"]),
            {"round_simulator_v0_1"},
        )
        self.assertTrue(
            result.fight_predictions["sim_red_win_probability"].between(0, 1).all()
        )


if __name__ == "__main__":
    unittest.main()
