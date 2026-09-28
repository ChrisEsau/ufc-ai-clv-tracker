from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.historical_training import (
    SOURCE_ONLY_ROUND_COLUMNS,
    UNREGISTERED_TARGET_ROUND_OBSERVATION_COLUMNS,
    build_historical_simulation_training_dataset,
    select_standard_round_history,
)


def make_round_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for fight_id, event_id, date, fighters, rounds in (
        ("F1", "E1", "2025-01-01", (("red", "A", "B"), ("blue", "B", "A")), (1, 2, 3)),
        ("F2", "E2", "2025-02-01", (("red", "C", "D"), ("blue", "D", "C")), (1,)),
    ):
        for round_number in rounds:
            for corner, fighter_id, opponent_id in fighters:
                rows.append(
                    {
                        "event_id": event_id,
                        "event_name": f"Event {event_id}",
                        "event_date": date,
                        "date": date,
                        "fight_id": fight_id,
                        "corner": corner,
                        "fighter_id": fighter_id,
                        "fighter_name": f"Fighter {fighter_id}",
                        "opponent_id": opponent_id,
                        "opponent_name": f"Fighter {opponent_id}",
                        "round": round_number,
                        "sig_str_landed": 10 + round_number,
                        "sig_str_attempted": 20 + round_number,
                        "total_str_landed": 12 + round_number,
                        "total_str_attempted": 23 + round_number,
                        "td_landed": 0,
                        "td_attempted": 1,
                        # The real historical artifact uses this alias rather than
                        # the canonical control_seconds column.
                        "ctrl_sec": 15,
                        "kd": 0,
                        "sub_att": 0,
                        "rev": 0,
                        "ground_landed": 0,
                        "ground_attempted": 0,
                        "head_landed": 8,
                        "head_attempted": 15,
                        "body_landed": 2,
                        "body_attempted": 4,
                        "leg_landed": 1,
                        "leg_attempted": 2,
                        "distance_landed": 9,
                        "distance_attempted": 18,
                        "clinch_landed": 2,
                        "clinch_attempted": 3,
                        "event_url": "https://example.test/event",
                        "fight_url": "https://example.test/fight",
                        "fighter_url": "https://example.test/fighter",
                        "opponent_url": "https://example.test/opponent",
                        "fight_order": None,
                        "division": "Lightweight",
                        "title_fight": 0,
                        "total_rounds": 3,
                    }
                )
    return pd.DataFrame(rows)


def make_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "E1",
                "event_name": "Event E1",
                "date": "2025-01-01",
                "fight_id": "F1",
                "division": "Lightweight",
                "title_fight": 0,
                "method": "U-DEC",
                "finish_round": 3,
                "match_time_sec": 900,
                "total_rounds": 3,
                "winner_id": "A",
            },
            {
                "event_id": "E2",
                "event_name": "Event E2",
                "date": "2025-02-01",
                "fight_id": "F2",
                "division": "Lightweight",
                "title_fight": 0,
                "method": "U-DEC",
                "finish_round": 1,
                "match_time_sec": 300,
                "total_rounds": None,
                "winner_id": "C",
            },
        ]
    )


class HistoricalTrainingBoundaryTests(unittest.TestCase):
    def test_filters_unsupported_fights_and_source_only_columns(self):
        rounds, master, _, summary = select_standard_round_history(
            make_round_rows(),
            make_master(),
        )
        self.assertEqual(set(rounds["fight_id"]), {"F1"})
        self.assertEqual(set(master["fight_id"]), {"F1"})
        self.assertEqual(summary.excluded_fights, 1)
        self.assertEqual(summary.excluded_round_rows, 2)
        self.assertEqual(summary.control_seconds_source_column, "ctrl_sec")
        self.assertIn("control_seconds", rounds.columns)

        forbidden = set(UNREGISTERED_TARGET_ROUND_OBSERVATION_COLUMNS)
        forbidden.update(SOURCE_ONLY_ROUND_COLUMNS)
        forbidden.update({"division", "title_fight", "total_rounds"})
        self.assertFalse(forbidden.intersection(rounds.columns))

    def test_final_dataset_contains_targets_but_no_raw_phase_or_url_columns(self):
        result = build_historical_simulation_training_dataset(
            make_round_rows(),
            make_master(),
        )
        dataset = result.dataset

        self.assertEqual(len(dataset), 6)
        self.assertIn("target_sig_attempted", dataset.columns)
        self.assertIn("target_control_seconds", dataset.columns)
        self.assertIn("prior_sig_str_attempted_cumulative", dataset.columns)
        self.assertFalse(
            set(UNREGISTERED_TARGET_ROUND_OBSERVATION_COLUMNS).intersection(dataset.columns)
        )
        self.assertFalse(any(column.endswith("_url") for column in dataset.columns))
        self.assertNotIn("ctrl_sec", dataset.columns)
        self.assertTrue(result.audit["passed"].all())


if __name__ == "__main__":
    unittest.main()
