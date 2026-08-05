from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from pipeline.simulation.contracts import (
    FighterSimulationState,
    MatchupSimulationInput,
    SimulatorConfig,
)
from pipeline.simulation.historical_simulator_replay import (
    build_fighter_fight_history,
    population_priors,
)
from pipeline.simulation.historical_strike_provider_replay import (
    StaticPrefightStrikeProvider,
    estimate_prefight_strike_calibration,
)
from pipeline.simulation.provider_engine import run_simulation_with_strike_provider
from pipeline.simulation.round_parameter_provider import (
    RoundParameterKey,
    RoundParameterProviderError,
)


def _fighter(fighter_id: str, name: str) -> FighterSimulationState:
    return FighterSimulationState(
        fighter_id=fighter_id,
        fighter_name=name,
        sig_attempts_per_minute=1.0,
        sig_accuracy=0.45,
        sig_defense=0.55,
        power=0.05,
        durability=0.95,
        td_attempts_per_15=0.0,
        td_accuracy=0.30,
        td_defense=0.80,
        control_seconds_per_takedown=30.0,
        submission_threat=0.02,
        submission_defense=0.95,
        cardio=0.75,
        recovery=0.75,
        pace_sustainability=0.75,
        adaptability=0.50,
        initiative=0.50,
        phase_imposition=0.95,
    )


def _matchup() -> MatchupSimulationInput:
    return MatchupSimulationInput(
        fight_id="fight-provider-test",
        event_id="event-test",
        red=_fighter("red-id", "Red"),
        blue=_fighter("blue-id", "Blue"),
        scheduled_rounds=3,
    )


def _training_frame() -> pd.DataFrame:
    rows = []
    fight_specs = [
        ("f2024", "2024-02-01", 28.0, 24.0),
        ("f2025", "2025-02-01", 34.0, 30.0),
        ("f2026", "2026-02-01", 40.0, 36.0),
    ]
    for fight_id, date, red_attempts, blue_attempts in fight_specs:
        for corner, fighter_id, opponent_id, attempts in (
            ("red", "fighter-a", "fighter-b", red_attempts),
            ("blue", "fighter-b", "fighter-a", blue_attempts),
        ):
            rows.append(
                {
                    "fight_id": fight_id,
                    "fighter_id": fighter_id,
                    "opponent_id": opponent_id,
                    "corner": corner,
                    "date": date,
                    "round": 1,
                    "total_rounds": 3,
                    "winner_id": "fighter-a",
                    "method_family": "decision",
                    "match_time_sec": 900.0,
                    "target_finish_time_in_round_seconds": 300.0,
                    "target_sig_attempted": attempts,
                    "target_sig_landed": attempts * 0.45,
                    "target_td_attempted": 0.0,
                    "target_td_landed": 0.0,
                    "target_control_seconds": 0.0,
                    "target_knockdowns": 0.0,
                    "target_submission_attempts": 0.0,
                }
            )
    return pd.DataFrame(rows)


class ProviderEngineTests(unittest.TestCase):
    def test_absolute_provider_rate_bypasses_heuristic_pace_discounts(self):
        matchup = _matchup()
        provider = StaticPrefightStrikeProvider(
            matchup,
            mean_calibration_factor=10.0,
            gamma_poisson_alpha=0.001,
        )
        with patch(
            "pipeline.simulation.provider_engine._sample_finish",
            return_value=None,
        ):
            summary, _ = run_simulation_with_strike_provider(
                matchup,
                provider,
                SimulatorConfig(simulations=800, seed=17),
            )

        # Base fighter pace is 1/minute, while the provider exposes 10/minute.
        # Three complete rounds therefore imply roughly 150 attempts per fighter.
        self.assertGreater(summary.expectations["red_sig_attempted"], 145.0)
        self.assertLess(summary.expectations["red_sig_attempted"], 155.0)
        self.assertGreater(summary.expectations["blue_sig_attempted"], 145.0)
        self.assertLess(summary.expectations["blue_sig_attempted"], 155.0)

    def test_static_provider_rejects_unknown_fighter(self):
        matchup = _matchup()
        provider = StaticPrefightStrikeProvider(matchup, 1.0, 0.2)
        with self.assertRaises(RoundParameterProviderError):
            provider.significant_strike_attempts(
                RoundParameterKey(matchup.fight_id, "unknown", 1)
            )

    def test_holdout_target_mutation_does_not_change_pretest_calibration(self):
        original = _training_frame()
        history = build_fighter_fight_history(original)
        priors = population_priors(history, test_year=2026)
        first = estimate_prefight_strike_calibration(
            original,
            history,
            priors,
            test_year=2026,
        )

        mutated = original.copy()
        mutated.loc[
            pd.to_datetime(mutated["date"]).dt.year.eq(2026),
            "target_sig_attempted",
        ] = 5000.0
        mutated_history = build_fighter_fight_history(mutated)
        mutated_priors = population_priors(mutated_history, test_year=2026)
        second = estimate_prefight_strike_calibration(
            mutated,
            mutated_history,
            mutated_priors,
            test_year=2026,
        )

        self.assertAlmostEqual(
            first.mean_calibration_factor,
            second.mean_calibration_factor,
        )
        self.assertAlmostEqual(
            first.gamma_poisson_overdispersion,
            second.gamma_poisson_overdispersion,
        )
        self.assertEqual(first.rows, second.rows)


if __name__ == "__main__":
    unittest.main()
