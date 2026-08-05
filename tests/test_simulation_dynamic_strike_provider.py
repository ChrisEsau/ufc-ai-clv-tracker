from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline.simulation.component_provider_engine import (
    run_simulation_with_component_providers,
)
from pipeline.simulation.contracts import (
    FighterSimulationState,
    MatchupSimulationInput,
    SimulatorConfig,
)
from pipeline.simulation.dynamic_strike_provider import (
    DynamicPrefightStrikeProvider,
    DynamicStrikeRoundContext,
)
from pipeline.simulation.round_parameter_provider import (
    RoundParameterKey,
    SignificantStrikeAttemptParameters,
)


def _fighter(fighter_id: str, name: str) -> FighterSimulationState:
    return FighterSimulationState(
        fighter_id=fighter_id,
        fighter_name=name,
        sig_attempts_per_minute=8.0,
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
        cardio=0.70,
        recovery=0.70,
        pace_sustainability=0.70,
        adaptability=0.50,
        initiative=0.50,
        phase_imposition=0.50,
    )


def _matchup(rounds: int = 3) -> MatchupSimulationInput:
    return MatchupSimulationInput(
        fight_id="dynamic-strike-test",
        event_id="event-test",
        red=_fighter("red-id", "Red"),
        blue=_fighter("blue-id", "Blue"),
        scheduled_rounds=rounds,
    )


def _context(
    round_number: int,
    *,
    fatigue: float = 0.0,
    damage: float = 0.0,
    confidence: float = 0.0,
    attempts: int = 0,
    opponent_control: float = 0.0,
    rounds_won: int = 0,
    opponent_rounds_won: int = 0,
    scheduled_rounds: int = 3,
) -> DynamicStrikeRoundContext:
    return DynamicStrikeRoundContext(
        key=RoundParameterKey("dynamic-strike-test", "red-id", round_number),
        opponent_id="blue-id",
        scheduled_rounds=scheduled_rounds,
        round_seconds=300,
        fighter_fatigue=fatigue,
        fighter_damage=damage,
        fighter_confidence=confidence,
        opponent_fatigue=0.0,
        opponent_damage=0.0,
        opponent_confidence=0.0,
        fighter_sig_attempted=attempts,
        fighter_sig_landed=attempts // 2,
        opponent_sig_attempted=0,
        opponent_sig_landed=0,
        fighter_control_seconds=0.0,
        opponent_control_seconds=opponent_control,
        fighter_knockdowns=0,
        opponent_knockdowns=0,
        fighter_rounds_won=rounds_won,
        opponent_rounds_won=opponent_rounds_won,
    )


class _ContextOnlyProvider:
    simulator_suffix = "test_context"

    def __init__(self) -> None:
        self.contexts: list[DynamicStrikeRoundContext] = []

    def significant_strike_attempts(self, key):
        raise AssertionError("Static provider method should not be called")

    def significant_strike_attempts_with_context(
        self,
        context: DynamicStrikeRoundContext,
    ) -> SignificantStrikeAttemptParameters:
        self.contexts.append(context)
        return SignificantStrikeAttemptParameters(
            key=context.key,
            mean_rate_per_minute=1.0,
            gamma_poisson_overdispersion=0.001,
            model_name="context-test",
            model_version="context-test-v0",
            calibration_factor=1.0,
            source="unit_test",
        )


class DynamicStrikeProviderTests(unittest.TestCase):
    def test_round_one_equals_calibrated_prefight_rate(self):
        provider = DynamicPrefightStrikeProvider(
            _matchup(),
            mean_calibration_factor=1.25,
            gamma_poisson_alpha=0.20,
        )
        static = provider.significant_strike_attempts(
            RoundParameterKey("dynamic-strike-test", "red-id", 1)
        )
        dynamic = provider.significant_strike_attempts_with_context(_context(1))
        self.assertAlmostEqual(static.mean_rate_per_minute, 10.0)
        self.assertAlmostEqual(
            dynamic.mean_rate_per_minute,
            static.mean_rate_per_minute,
        )

    def test_fatigue_damage_and_control_reduce_later_round_rate(self):
        provider = DynamicPrefightStrikeProvider(_matchup(), 1.0, 0.20)
        neutral = provider.significant_strike_attempts_with_context(
            _context(3, attempts=80)
        )
        suppressed = provider.significant_strike_attempts_with_context(
            _context(
                3,
                fatigue=0.75,
                damage=0.60,
                confidence=-0.50,
                attempts=80,
                opponent_control=240.0,
                rounds_won=0,
                opponent_rounds_won=1,
            )
        )
        self.assertLess(
            suppressed.mean_rate_per_minute,
            neutral.mean_rate_per_minute,
        )

    def test_late_score_deficit_adds_urgency(self):
        provider = DynamicPrefightStrikeProvider(_matchup(rounds=5), 1.0, 0.20)
        leading = provider.significant_strike_attempts_with_context(
            _context(
                5,
                attempts=160,
                rounds_won=3,
                opponent_rounds_won=1,
                scheduled_rounds=5,
            )
        )
        trailing = provider.significant_strike_attempts_with_context(
            _context(
                5,
                attempts=160,
                rounds_won=1,
                opponent_rounds_won=3,
                scheduled_rounds=5,
            )
        )
        self.assertGreater(
            trailing.mean_rate_per_minute,
            leading.mean_rate_per_minute,
        )

    def test_component_engine_delivers_path_context_each_round(self):
        provider = _ContextOnlyProvider()
        with patch(
            "pipeline.simulation.component_provider_engine._sample_finish",
            return_value=None,
        ):
            summary, _ = run_simulation_with_component_providers(
                _matchup(),
                SimulatorConfig(simulations=1, seed=19),
                strike_provider=provider,
            )

        self.assertEqual(len(provider.contexts), 6)
        red_contexts = [
            context
            for context in provider.contexts
            if context.key.fighter_id == "red-id"
        ]
        self.assertEqual([context.key.round for context in red_contexts], [1, 2, 3])
        self.assertEqual(red_contexts[0].fighter_sig_attempted, 0)
        self.assertGreaterEqual(red_contexts[1].fighter_sig_attempted, 0)
        self.assertIn("test_context", summary.simulator_version)


if __name__ == "__main__":
    unittest.main()
