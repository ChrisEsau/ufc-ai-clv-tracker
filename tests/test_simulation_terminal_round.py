from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from pipeline.simulation.contracts import (
    FighterSimulationState,
    MatchupSimulationInput,
    SimulatorConfig,
)
from pipeline.simulation.engine import simulate_fight
from pipeline.simulation.terminal_round import (
    RoundPerformance,
    TerminalRoundError,
    thin_round_performance,
)


class _DeterministicRng:
    """Small deterministic RNG surface used by the terminal-round engine test."""

    def choice(self, values, p=None):
        return values[0]

    def normal(self, loc=0.0, scale=1.0):
        return 0.0

    def beta(self, a, b):
        return 0.20

    def binomial(self, n, p):
        return int(np.floor(int(n) * float(p)))


def _fighter(fighter_id: str) -> FighterSimulationState:
    return FighterSimulationState(
        fighter_id=fighter_id,
        fighter_name=fighter_id,
        sig_attempts_per_minute=6.0,
        sig_accuracy=0.45,
        sig_defense=0.55,
        power=0.60,
        durability=0.65,
        td_attempts_per_15=3.0,
        td_accuracy=0.40,
        td_defense=0.70,
        control_seconds_per_takedown=60.0,
        submission_threat=0.40,
        submission_defense=0.65,
        cardio=0.70,
        recovery=0.65,
        pace_sustainability=0.70,
        adaptability=0.60,
    )


class TerminalRoundTests(unittest.TestCase):
    def test_thinning_preserves_count_constraints(self):
        full = RoundPerformance(
            sig_attempted=100,
            sig_landed=45,
            takedowns_attempted=12,
            takedowns_landed=5,
            control_seconds=150.0,
            knockdowns=3,
        )
        for seed in range(100):
            rng = np.random.default_rng(seed)
            partial = thin_round_performance(rng, full, 0.37)
            self.assertGreaterEqual(partial.sig_attempted, 0)
            self.assertLessEqual(partial.sig_attempted, full.sig_attempted)
            self.assertLessEqual(partial.sig_landed, partial.sig_attempted)
            self.assertLessEqual(
                partial.takedowns_landed,
                partial.takedowns_attempted,
            )
            self.assertLessEqual(partial.control_seconds, full.control_seconds)
            self.assertLessEqual(partial.knockdowns, full.knockdowns)

    def test_zero_and_full_exposure_are_exact(self):
        full = RoundPerformance(20, 8, 4, 2, 90.0, 1)
        rng = np.random.default_rng(4)
        self.assertEqual(
            thin_round_performance(rng, full, 0.0),
            RoundPerformance(0, 0, 0, 0, 0.0, 0),
        )
        self.assertEqual(thin_round_performance(rng, full, 1.0), full)

    def test_invalid_exposure_is_rejected(self):
        full = RoundPerformance(20, 8, 4, 2, 90.0, 1)
        with self.assertRaises(TerminalRoundError):
            thin_round_performance(np.random.default_rng(1), full, 1.01)

    def test_engine_credits_only_partial_terminal_round_and_no_judge_round(self):
        matchup = MatchupSimulationInput(
            fight_id="terminal_test",
            event_id="event",
            red=_fighter("red"),
            blue=_fighter("blue"),
            scheduled_rounds=3,
        )
        rng = _DeterministicRng()

        with (
            patch(
                "pipeline.simulation.engine._sample_phase_shares",
                return_value=(1.0, 0.0, 0.0),
            ),
            patch(
                "pipeline.simulation.engine._strike_round",
                side_effect=[(100, 50), (80, 40)],
            ),
            patch(
                "pipeline.simulation.engine._takedown_round",
                side_effect=[(10, 5, 100.0), (8, 4, 80.0)],
            ),
            patch(
                "pipeline.simulation.engine._knockdowns",
                side_effect=[2, 1],
            ),
            patch(
                "pipeline.simulation.engine._finish_hazards",
                side_effect=[(1.0, 0.0), (0.0, 0.0)],
            ),
            patch(
                "pipeline.simulation.engine._sample_finish",
                return_value=("red", "ko_tko"),
            ),
        ):
            outcome = simulate_fight(matchup, rng, SimulatorConfig(simulations=1))

        self.assertEqual(outcome.method, "ko_tko")
        self.assertEqual(outcome.finish_round, 1)
        self.assertEqual(outcome.finish_time_seconds, 60.0)
        self.assertEqual(outcome.total_fight_seconds, 60.0)
        self.assertEqual(outcome.red_rounds_won, 0)
        self.assertEqual(outcome.blue_rounds_won, 0)

        self.assertEqual(outcome.red_totals.sig_attempted, 20)
        self.assertEqual(outcome.red_totals.sig_landed, 10)
        self.assertEqual(outcome.red_totals.takedowns_attempted, 2)
        self.assertEqual(outcome.red_totals.takedowns_landed, 1)
        self.assertEqual(outcome.red_totals.control_seconds, 20.0)
        self.assertEqual(outcome.red_totals.knockdowns, 0)

        self.assertEqual(outcome.blue_totals.sig_attempted, 16)
        self.assertEqual(outcome.blue_totals.sig_landed, 8)
        self.assertEqual(outcome.blue_totals.takedowns_attempted, 0)
        self.assertEqual(outcome.blue_totals.takedowns_landed, 0)
        self.assertEqual(outcome.blue_totals.control_seconds, 16.0)
        self.assertEqual(outcome.blue_totals.knockdowns, 0)


if __name__ == "__main__":
    unittest.main()
