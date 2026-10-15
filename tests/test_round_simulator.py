"""Unit tests for the standalone round-level simulation kernel."""

from __future__ import annotations

import unittest

from pipeline.simulation.contracts import (
    FighterSimulationState,
    MatchupSimulationInput,
    SimulationContractError,
    SimulatorConfig,
)
from pipeline.simulation.engine import run_simulation


def _fighter(fighter_id: str, name: str, strength: float = 0.5) -> FighterSimulationState:
    return FighterSimulationState(
        fighter_id=fighter_id,
        fighter_name=name,
        sig_attempts_per_minute=4.5 + 3.0 * strength,
        sig_accuracy=0.34 + 0.24 * strength,
        sig_defense=0.42 + 0.35 * strength,
        power=0.30 + 0.62 * strength,
        durability=0.38 + 0.55 * strength,
        td_attempts_per_15=1.5 + 5.0 * strength,
        td_accuracy=0.25 + 0.42 * strength,
        td_defense=0.38 + 0.48 * strength,
        control_seconds_per_takedown=35.0 + 85.0 * strength,
        submission_threat=0.22 + 0.62 * strength,
        submission_defense=0.35 + 0.55 * strength,
        cardio=0.40 + 0.52 * strength,
        recovery=0.38 + 0.52 * strength,
        pace_sustainability=0.38 + 0.54 * strength,
        adaptability=0.35 + 0.55 * strength,
        initiative=0.40 + 0.45 * strength,
        phase_imposition=0.38 + 0.48 * strength,
    )


def _matchup(red_strength: float = 0.6, blue_strength: float = 0.5) -> MatchupSimulationInput:
    return MatchupSimulationInput(
        fight_id="test_fight",
        event_id="test_event",
        red=_fighter("red", "Red", red_strength),
        blue=_fighter("blue", "Blue", blue_strength),
        scheduled_rounds=3,
    )


class RoundSimulatorTests(unittest.TestCase):
    def test_input_contract_rejects_invalid_probability(self) -> None:
        with self.assertRaises(SimulationContractError):
            FighterSimulationState(
                fighter_id="bad",
                fighter_name="Bad",
                sig_attempts_per_minute=5.0,
                sig_accuracy=1.2,
                sig_defense=0.5,
                power=0.5,
                durability=0.5,
                td_attempts_per_15=2.0,
                td_accuracy=0.4,
                td_defense=0.5,
                control_seconds_per_takedown=45.0,
                submission_threat=0.4,
                submission_defense=0.5,
                cardio=0.5,
                recovery=0.5,
                pace_sustainability=0.5,
                adaptability=0.5,
            )

    def test_seed_is_deterministic(self) -> None:
        config = SimulatorConfig(simulations=600, seed=123)
        first, _ = run_simulation(_matchup(), config)
        second, _ = run_simulation(_matchup(), config)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_market_probabilities_are_coherent(self) -> None:
        summary, _ = run_simulation(
            _matchup(), SimulatorConfig(simulations=1_200, seed=17)
        )
        p = summary.probabilities
        self.assertAlmostEqual(p["red_win"] + p["blue_win"], 1.0, places=12)
        self.assertAlmostEqual(p["goes_distance"] + p["inside_distance"], 1.0, places=12)
        self.assertAlmostEqual(
            p["red_by_decision"] + p["red_by_ko_tko"] + p["red_by_submission"],
            p["red_win"],
            places=12,
        )
        self.assertAlmostEqual(
            p["blue_by_decision"] + p["blue_by_ko_tko"] + p["blue_by_submission"],
            p["blue_win"],
            places=12,
        )
        for value in p.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_stronger_fighter_has_positive_directional_signal(self) -> None:
        summary, _ = run_simulation(
            _matchup(red_strength=0.92, blue_strength=0.22),
            SimulatorConfig(simulations=2_500, seed=99),
        )
        self.assertGreater(summary.probabilities["red_win"], 0.65)
        self.assertGreater(
            summary.expectations["red_sig_landed"],
            summary.expectations["blue_sig_landed"],
        )

    def test_retained_outcomes_obey_time_and_count_bounds(self) -> None:
        matchup = _matchup()
        summary, outcomes = run_simulation(
            matchup,
            SimulatorConfig(simulations=300, seed=5, retain_outcomes=True),
        )
        self.assertIsNotNone(outcomes)
        assert outcomes is not None
        self.assertEqual(len(outcomes), summary.simulations)
        for outcome in outcomes:
            self.assertIn(outcome.winner_corner, {"red", "blue"})
            self.assertIn(outcome.method, {"decision", "ko_tko", "submission"})
            self.assertGreater(outcome.total_fight_seconds, 0.0)
            self.assertLessEqual(
                outcome.total_fight_seconds,
                matchup.scheduled_rounds * matchup.round_seconds,
            )
            self.assertGreaterEqual(outcome.red_totals.sig_attempted, 0)
            self.assertGreaterEqual(outcome.blue_totals.sig_attempted, 0)
            self.assertLessEqual(
                outcome.red_totals.control_seconds + outcome.blue_totals.control_seconds,
                matchup.scheduled_rounds * matchup.round_seconds + 1e-6,
            )


if __name__ == "__main__":
    unittest.main()
