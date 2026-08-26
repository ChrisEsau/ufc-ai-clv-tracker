from __future__ import annotations

import unittest

import pandas as pd

from pipeline.simulation.component_provider_engine import (
    run_simulation_with_component_providers,
)
from pipeline.simulation.contracts import (
    FighterSimulationState,
    MatchupSimulationInput,
    SimulatorConfig,
)
from pipeline.simulation.finish_hazard_holdout import (
    _counterfactual_holdout_rows,
)
from pipeline.simulation.finish_hazard_provider import (
    FinishHazardKey,
    FinishHazardProviderError,
    HistoricalFinishHazardProvider,
)


def _fighter(fighter_id: str) -> FighterSimulationState:
    return FighterSimulationState(
        fighter_id=fighter_id,
        fighter_name=fighter_id,
        sig_attempts_per_minute=6.0,
        sig_accuracy=0.45,
        sig_defense=0.55,
        power=0.30,
        durability=0.70,
        td_attempts_per_15=1.0,
        td_accuracy=0.30,
        td_defense=0.70,
        control_seconds_per_takedown=30.0,
        submission_threat=0.20,
        submission_defense=0.75,
        cardio=0.70,
        recovery=0.70,
        pace_sustainability=0.70,
        adaptability=0.50,
    )


def _matchup() -> MatchupSimulationInput:
    return MatchupSimulationInput(
        fight_id="provider-fight",
        event_id="event",
        red=_fighter("red"),
        blue=_fighter("blue"),
        scheduled_rounds=3,
    )


def _provider_frame(round_one_red_ko: bool) -> pd.DataFrame:
    rows = []
    for round_number in (1, 2, 3):
        red_ko = 1.0 if round_one_red_ko and round_number == 1 else 0.0
        rows.append(
            {
                "fight_id": "provider-fight",
                "round": round_number,
                "model_name": "candidate",
                "calibrated_prob_no_finish": 1.0 - red_ko,
                "calibrated_prob_red_ko_tko": red_ko,
                "calibrated_prob_red_submission": 0.0,
                "calibrated_prob_blue_ko_tko": 0.0,
                "calibrated_prob_blue_submission": 0.0,
            }
        )
    return pd.DataFrame(rows)


class FinishHazardProviderTests(unittest.TestCase):
    def test_counterfactual_rows_cover_all_scheduled_rounds(self):
        prepared = pd.DataFrame(
            [
                {
                    "fight_id": "early-finish",
                    "date": pd.Timestamp("2026-01-01"),
                    "round": 1,
                    "total_rounds": 3,
                    "red_fighter_id": "r1",
                    "blue_fighter_id": "b1",
                    "feature": 2.0,
                },
                {
                    "fight_id": "five-round",
                    "date": pd.Timestamp("2026-02-01"),
                    "round": 1,
                    "total_rounds": 5,
                    "red_fighter_id": "r2",
                    "blue_fighter_id": "b2",
                    "feature": 3.0,
                },
                {
                    "fight_id": "five-round",
                    "date": pd.Timestamp("2026-02-01"),
                    "round": 2,
                    "total_rounds": 5,
                    "red_fighter_id": "r2",
                    "blue_fighter_id": "b2",
                    "feature": 3.0,
                },
            ]
        )
        counterfactual = _counterfactual_holdout_rows(prepared, test_year=2026)
        self.assertEqual(len(counterfactual), 8)
        self.assertEqual(
            set(
                counterfactual.loc[
                    counterfactual["fight_id"].eq("early-finish"), "round"
                ]
            ),
            {1, 2, 3},
        )
        self.assertEqual(
            set(
                counterfactual.loc[
                    counterfactual["fight_id"].eq("five-round"), "round"
                ]
            ),
            {1, 2, 3, 4, 5},
        )

    def test_finish_provider_can_force_round_one_red_ko(self):
        provider = HistoricalFinishHazardProvider(
            _provider_frame(round_one_red_ko=True),
            model_name="candidate",
        )
        summary, _ = run_simulation_with_component_providers(
            _matchup(),
            SimulatorConfig(simulations=250, seed=3),
            finish_provider=provider,
        )
        self.assertAlmostEqual(summary.probabilities["red_by_ko_tko"], 1.0)
        self.assertAlmostEqual(summary.probabilities["goes_distance"], 0.0)
        self.assertAlmostEqual(summary.probabilities["reaches_round_2"], 0.0)

    def test_finish_provider_can_force_decision(self):
        provider = HistoricalFinishHazardProvider(
            _provider_frame(round_one_red_ko=False),
            model_name="candidate",
        )
        summary, _ = run_simulation_with_component_providers(
            _matchup(),
            SimulatorConfig(simulations=250, seed=4),
            finish_provider=provider,
        )
        self.assertAlmostEqual(summary.probabilities["goes_distance"], 1.0)
        self.assertAlmostEqual(summary.probabilities["inside_distance"], 0.0)

    def test_finish_provider_rejects_missing_round(self):
        provider = HistoricalFinishHazardProvider(
            _provider_frame(round_one_red_ko=False).loc[lambda frame: frame["round"].lt(3)],
            model_name="candidate",
        )
        with self.assertRaises(FinishHazardProviderError):
            provider.finish_hazards(FinishHazardKey("provider-fight", 3))


if __name__ == "__main__":
    unittest.main()
