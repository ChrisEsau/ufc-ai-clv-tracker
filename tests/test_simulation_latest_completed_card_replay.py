from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from pipeline.simulation.run_latest_completed_card_replay import (
    BLOCKED_STRIKE_VARIANT,
    RECOMMENDED_VARIANT,
    _add_experience_diagnostics,
    select_latest_card_records,
)


class LatestCompletedCardReplayTests(unittest.TestCase):
    def test_selects_every_fight_from_latest_event(self):
        records = [
            {
                "date": pd.Timestamp("2026-07-18"),
                "matchup": SimpleNamespace(event_id="event_old", fight_id="old_1"),
            },
            {
                "date": pd.Timestamp("2026-08-01"),
                "matchup": SimpleNamespace(event_id="event_latest", fight_id="new_2"),
            },
            {
                "date": pd.Timestamp("2026-08-01"),
                "matchup": SimpleNamespace(event_id="event_latest", fight_id="new_1"),
            },
        ]
        selected = select_latest_card_records(records)
        self.assertEqual(
            [row["matchup"].fight_id for row in selected],
            ["new_1", "new_2"],
        )
        self.assertEqual(
            {row["matchup"].event_id for row in selected},
            {"event_latest"},
        )

    def test_prefers_largest_event_when_same_date_has_multiple_ids(self):
        records = [
            {
                "date": pd.Timestamp("2026-08-01"),
                "matchup": SimpleNamespace(event_id="small", fight_id="small_1"),
            },
            {
                "date": pd.Timestamp("2026-08-01"),
                "matchup": SimpleNamespace(event_id="main", fight_id="main_1"),
            },
            {
                "date": pd.Timestamp("2026-08-01"),
                "matchup": SimpleNamespace(event_id="main", fight_id="main_2"),
            },
        ]
        selected = select_latest_card_records(records)
        self.assertEqual({row["matchup"].event_id for row in selected}, {"main"})
        self.assertEqual(len(selected), 2)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            select_latest_card_records([])

    def test_latest_card_uses_recommended_survival_variant(self):
        self.assertEqual(RECOMMENDED_VARIANT, "survival_finish_hazard_provider")
        self.assertEqual(
            BLOCKED_STRIKE_VARIANT,
            "strike_and_survival_finish_providers",
        )
        self.assertNotEqual(RECOMMENDED_VARIANT, BLOCKED_STRIKE_VARIANT)

    def test_experience_diagnostics_identify_cold_and_limited_history(self):
        predictions = pd.DataFrame(
            {
                "red_prior_fights": [0, 1, 4],
                "blue_prior_fights": [3, 2, 8],
                "predicted_winner_probability": [0.61, 0.58, 0.72],
            }
        )
        diagnosed = _add_experience_diagnostics(predictions)

        self.assertEqual(
            diagnosed["experience_band"].tolist(),
            ["cold_start", "limited_history", "established_history"],
        )
        self.assertEqual(diagnosed["cold_start_fighters"].tolist(), [1, 0, 0])
        self.assertEqual(diagnosed["minimum_prior_fights"].tolist(), [0, 1, 4])
        self.assertAlmostEqual(
            float(diagnosed.loc[0, "winner_edge_from_coin_flip"]),
            0.11,
        )


if __name__ == "__main__":
    unittest.main()
