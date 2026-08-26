from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.run_historical_simulator_replay import (
    _attach_scoring_labels,
)


class ReplayElapsedLabelTests(unittest.TestCase):
    def test_final_round_clock_is_repaired_and_matches_training_target(self):
        training = pd.DataFrame(
            {
                "fight_id": ["fight-1", "fight-1"],
                "fighter_id": ["red", "blue"],
                "target_elapsed_fight_seconds": [710.0, 710.0],
            }
        )
        master = pd.DataFrame(
            {
                "fight_id": ["fight-1"],
                "winner_id": ["red"],
                "method": ["KO/TKO"],
                "finish_round": [3],
                "match_time_sec": [110.0],
            }
        )
        labeled = _attach_scoring_labels(training, master)
        self.assertEqual(set(labeled["match_time_sec"]), {710.0})
        self.assertEqual(set(labeled["method_family"]), {"ko_tko"})

    def test_repaired_master_time_must_agree_with_training_target(self):
        training = pd.DataFrame(
            {
                "fight_id": ["fight-1", "fight-1"],
                "fighter_id": ["red", "blue"],
                "target_elapsed_fight_seconds": [700.0, 700.0],
            }
        )
        master = pd.DataFrame(
            {
                "fight_id": ["fight-1"],
                "winner_id": ["red"],
                "method": ["KO/TKO"],
                "finish_round": [3],
                "match_time_sec": [110.0],
            }
        )
        with self.assertRaises(ValueError):
            _attach_scoring_labels(training, master)

    def test_elapsed_target_must_be_constant_within_fight(self):
        training = pd.DataFrame(
            {
                "fight_id": ["fight-1", "fight-1"],
                "fighter_id": ["red", "blue"],
                "target_elapsed_fight_seconds": [710.0, 711.0],
            }
        )
        master = pd.DataFrame(
            {
                "fight_id": ["fight-1"],
                "winner_id": ["red"],
                "method": ["KO/TKO"],
                "finish_round": [3],
                "match_time_sec": [110.0],
            }
        )
        with self.assertRaises(ValueError):
            _attach_scoring_labels(training, master)


if __name__ == "__main__":
    unittest.main()
