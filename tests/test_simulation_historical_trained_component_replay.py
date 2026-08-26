from __future__ import annotations

import unittest

import numpy as np

from pipeline.simulation.contracts import FightSimulationOutcome, FighterFightTotals
from pipeline.simulation.finish_hazard_provider import (
    FinishHazardProbabilities,
)
from pipeline.simulation.historical_trained_component_replay import (
    apply_finish_provider_to_decision_path,
)


class _ScheduledFinishProvider:
    def __init__(self, by_round: dict[int, tuple[float, float, float, float, float]]):
        self.by_round = by_round

    def finish_hazards(self, key):
        values = self.by_round[int(key.round)]
        return FinishHazardProbabilities(
            key=key,
            no_finish=values[0],
            red_ko_tko=values[1],
            red_submission=values[2],
            blue_ko_tko=values[3],
            blue_submission=values[4],
            model_name="test_finish_provider",
            model_version="test_v0",
        )


def _decision_outcome() -> FightSimulationOutcome:
    return FightSimulationOutcome(
        winner_corner="red",
        method="decision",
        finish_round=3,
        finish_time_seconds=300.0,
        total_fight_seconds=900.0,
        red_rounds_won=2,
        blue_rounds_won=1,
        red_totals=FighterFightTotals(
            sig_attempted=120,
            sig_landed=60,
            takedowns_attempted=6,
            takedowns_landed=3,
            control_seconds=180.0,
            knockdowns=1,
        ),
        blue_totals=FighterFightTotals(
            sig_attempted=100,
            sig_landed=40,
            takedowns_attempted=4,
            takedowns_landed=1,
            control_seconds=60.0,
            knockdowns=0,
        ),
        regime="mixed",
    )


class HistoricalTrainedComponentReplayTests(unittest.TestCase):
    def test_no_finish_preserves_complete_decision_path(self):
        provider = _ScheduledFinishProvider(
            {
                1: (1.0, 0.0, 0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0, 0.0, 0.0),
                3: (1.0, 0.0, 0.0, 0.0, 0.0),
            }
        )
        original = _decision_outcome()
        result = apply_finish_provider_to_decision_path(
            original,
            fight_id="fight-1",
            scheduled_rounds=3,
            round_seconds=300,
            finish_provider=provider,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(result.method, "decision")
        self.assertEqual(result.winner_corner, "red")
        self.assertEqual(result.total_fight_seconds, 900.0)
        self.assertEqual(result.red_totals, original.red_totals)
        self.assertEqual(result.blue_totals, original.blue_totals)

    def test_round_two_submission_uses_conditional_reach_path(self):
        provider = _ScheduledFinishProvider(
            {
                1: (1.0, 0.0, 0.0, 0.0, 0.0),
                2: (0.0, 0.0, 0.0, 0.0, 1.0),
                3: (1.0, 0.0, 0.0, 0.0, 0.0),
            }
        )
        original = _decision_outcome()
        result = apply_finish_provider_to_decision_path(
            original,
            fight_id="fight-2",
            scheduled_rounds=3,
            round_seconds=300,
            finish_provider=provider,
            rng=np.random.default_rng(11),
        )
        self.assertEqual(result.method, "submission")
        self.assertEqual(result.winner_corner, "blue")
        self.assertEqual(result.finish_round, 2)
        self.assertGreater(result.total_fight_seconds, 300.0)
        self.assertLessEqual(result.total_fight_seconds, 600.0)
        self.assertLessEqual(
            result.red_totals.sig_attempted,
            original.red_totals.sig_attempted,
        )
        self.assertLessEqual(
            result.blue_totals.sig_attempted,
            original.blue_totals.sig_attempted,
        )
        self.assertLessEqual(
            result.red_totals.sig_landed,
            result.red_totals.sig_attempted,
        )
        self.assertLessEqual(
            result.blue_totals.takedowns_landed,
            result.blue_totals.takedowns_attempted,
        )

    def test_overlay_rejects_nondecision_mechanics_path(self):
        original = _decision_outcome()
        terminal = FightSimulationOutcome(
            **{
                **original.to_dict(),
                "method": "ko_tko",
                "finish_round": 1,
                "finish_time_seconds": 90.0,
                "total_fight_seconds": 90.0,
            }
        )
        provider = _ScheduledFinishProvider(
            {1: (1.0, 0.0, 0.0, 0.0, 0.0)}
        )
        with self.assertRaises(Exception):
            apply_finish_provider_to_decision_path(
                terminal,
                fight_id="fight-3",
                scheduled_rounds=3,
                round_seconds=300,
                finish_provider=provider,
                rng=np.random.default_rng(13),
            )


if __name__ == "__main__":
    unittest.main()
