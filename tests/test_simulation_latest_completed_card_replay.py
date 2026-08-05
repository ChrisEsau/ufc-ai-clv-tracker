from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from pipeline.simulation.run_latest_completed_card_replay import (
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
        self.assertEqual([row["matchup"].fight_id for row in selected], ["new_1", "new_2"])
        self.assertEqual({row["matchup"].event_id for row in selected}, {"event_latest"})

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


if __name__ == "__main__":
    unittest.main()
