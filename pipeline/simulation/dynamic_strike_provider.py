"""Stateful significant-strike provider for simulated round-to-round paths.

The earlier static provider exposed one pre-fight pace for every scheduled round.
That was leakage-safe, but it could not react to the simulated fight path. This
shadow-only provider keeps the same pre-fight calibrated base pace and derives a
conservative round-two-plus multiplier from state already created by prior
simulated rounds.

No realized target-fight round is read. Every dynamic input is generated inside
the Monte Carlo path: fatigue, damage, confidence, cumulative activity, control,
knockdowns, and score position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.simulation.contracts import MatchupSimulationInput
from pipeline.simulation.round_parameter_provider import (
    RoundParameterKey,
    RoundParameterProviderError,
    SignificantStrikeAttemptParameters,
)


@dataclass(frozen=True)
class DynamicStrikeRoundContext:
    """Prior simulated state available before one fighter-round is sampled."""

    key: RoundParameterKey
    opponent_id: str
    scheduled_rounds: int
    round_seconds: int
    fighter_fatigue: float
    fighter_damage: float
    fighter_confidence: float
    opponent_fatigue: float
    opponent_damage: float
    opponent_confidence: float
    fighter_sig_attempted: int
    fighter_sig_landed: int
    opponent_sig_attempted: int
    opponent_sig_landed: int
    fighter_control_seconds: float
    opponent_control_seconds: float
    fighter_knockdowns: int
    opponent_knockdowns: int
    fighter_rounds_won: int
    opponent_rounds_won: int

    def __post_init__(self) -> None:
        if not str(self.opponent_id).strip():
            raise RoundParameterProviderError("opponent_id must be non-empty")
        if int(self.scheduled_rounds) not in (3, 5):
            raise RoundParameterProviderError(
                "scheduled_rounds must be 3 or 5 for the dynamic strike provider"
            )
        if int(self.key.round) > int(self.scheduled_rounds):
            raise RoundParameterProviderError(
                f"Round {self.key.round} exceeds scheduled rounds"
            )
        if int(self.round_seconds) <= 0:
            raise RoundParameterProviderError("round_seconds must be positive")

        bounded = {
            "fighter_fatigue": self.fighter_fatigue,
            "fighter_damage": self.fighter_damage,
            "opponent_fatigue": self.opponent_fatigue,
            "opponent_damage": self.opponent_damage,
        }
        for name, raw in bounded.items():
            value = float(raw)
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise RoundParameterProviderError(
                    f"{name} must be finite and within [0, 1]; received {raw!r}"
                )

        confidence = {
            "fighter_confidence": self.fighter_confidence,
            "opponent_confidence": self.opponent_confidence,
        }
        for name, raw in confidence.items():
            value = float(raw)
            if not np.isfinite(value) or value < -1.0 or value > 1.0:
                raise RoundParameterProviderError(
                    f"{name} must be finite and within [-1, 1]; received {raw!r}"
                )

        nonnegative = {
            "fighter_sig_attempted": self.fighter_sig_attempted,
            "fighter_sig_landed": self.fighter_sig_landed,
            "opponent_sig_attempted": self.opponent_sig_attempted,
            "opponent_sig_landed": self.opponent_sig_landed,
            "fighter_control_seconds": self.fighter_control_seconds,
            "opponent_control_seconds": self.opponent_control_seconds,
            "fighter_knockdowns": self.fighter_knockdowns,
            "opponent_knockdowns": self.opponent_knockdowns,
            "fighter_rounds_won": self.fighter_rounds_won,
            "opponent_rounds_won": self.opponent_rounds_won,
        }
        for name, raw in nonnegative.items():
            value = float(raw)
            if not np.isfinite(value) or value < 0:
                raise RoundParameterProviderError(
                    f"{name} must be finite and nonnegative; received {raw!r}"
                )

    @property
    def completed_rounds(self) -> int:
        return int(self.key.round) - 1


class DynamicPrefightStrikeProvider:
    """Apply conservative path-state adjustments to calibrated pre-fight pace."""

    simulator_suffix = "dynamic_strike"

    def __init__(
        self,
        matchup: MatchupSimulationInput,
        mean_calibration_factor: float,
        gamma_poisson_alpha: float,
    ) -> None:
        if mean_calibration_factor <= 0 or not np.isfinite(mean_calibration_factor):
            raise RoundParameterProviderError(
                "mean_calibration_factor must be finite and positive"
            )
        if gamma_poisson_alpha <= 0 or not np.isfinite(gamma_poisson_alpha):
            raise RoundParameterProviderError(
                "gamma_poisson_alpha must be finite and positive"
            )
        self.matchup = matchup
        self.factor = float(mean_calibration_factor)
        self.alpha = float(gamma_poisson_alpha)
        self._rates = {
            str(matchup.red.fighter_id): float(matchup.red.sig_attempts_per_minute),
            str(matchup.blue.fighter_id): float(matchup.blue.sig_attempts_per_minute),
        }
        self._opponents = {
            str(matchup.red.fighter_id): str(matchup.blue.fighter_id),
            str(matchup.blue.fighter_id): str(matchup.red.fighter_id),
        }

    def _base_rate(self, key: RoundParameterKey) -> float:
        if str(key.fight_id) != str(self.matchup.fight_id):
            raise RoundParameterProviderError(
                f"Provider fight mismatch: {key.fight_id!r}"
            )
        if int(key.round) > int(self.matchup.scheduled_rounds):
            raise RoundParameterProviderError(
                f"Round {key.round} exceeds scheduled rounds"
            )
        try:
            return float(self._rates[str(key.fighter_id)] * self.factor)
        except KeyError as exc:
            raise RoundParameterProviderError(
                f"Unknown fighter for provider: {key.fighter_id!r}"
            ) from exc

    def significant_strike_attempts(
        self,
        key: RoundParameterKey,
    ) -> SignificantStrikeAttemptParameters:
        """Return the calibrated pre-fight rate for compatibility and round one."""
        return SignificantStrikeAttemptParameters(
            key=key,
            mean_rate_per_minute=self._base_rate(key),
            gamma_poisson_overdispersion=self.alpha,
            model_name="dynamic_prefight_career_pace",
            model_version="dynamic_prefight_career_pace_v0",
            calibration_factor=self.factor,
            source="pre_holdout_career_pace_dynamic_path_state",
        )

    def significant_strike_attempts_with_context(
        self,
        context: DynamicStrikeRoundContext,
    ) -> SignificantStrikeAttemptParameters:
        """Return an absolute rate adjusted only by prior simulated-round state."""
        key = context.key
        base_rate = self._base_rate(key)
        expected_opponent = self._opponents[str(key.fighter_id)]
        if str(context.opponent_id) != expected_opponent:
            raise RoundParameterProviderError(
                "Dynamic strike context opponent does not match the matchup"
            )
        if int(context.scheduled_rounds) != int(self.matchup.scheduled_rounds):
            raise RoundParameterProviderError(
                "Dynamic strike context scheduled rounds do not match the matchup"
            )

        completed = context.completed_rounds
        if completed == 0:
            multiplier = 1.0
        else:
            expected_previous_attempts = (
                base_rate * float(context.round_seconds) / 60.0 * completed
            )
            observed_ratio = float(context.fighter_sig_attempted) / max(
                expected_previous_attempts,
                1e-6,
            )
            pace_memory = 1.0 + 0.22 * (
                float(np.clip(observed_ratio, 0.55, 1.45)) - 1.0
            )

            fatigue_drag = 1.0 - 0.28 * float(context.fighter_fatigue)
            damage_drag = 1.0 - 0.18 * float(context.fighter_damage)
            confidence_effect = 1.0 + 0.06 * float(context.fighter_confidence)
            opponent_opening = 1.0 + 0.03 * float(context.opponent_damage)

            elapsed_seconds = float(context.round_seconds * completed)
            opponent_control_share = float(context.opponent_control_seconds) / max(
                elapsed_seconds,
                1.0,
            )
            suppression = 1.0 - 0.18 * float(
                np.clip(opponent_control_share, 0.0, 0.75)
            )

            score_deficit = int(context.opponent_rounds_won) - int(
                context.fighter_rounds_won
            )
            late_weight = completed / max(1, int(context.scheduled_rounds) - 1)
            urgency = 1.0 + 0.06 * float(score_deficit) * late_weight
            round_decay = 1.0 - 0.025 * completed

            multiplier = float(
                np.clip(
                    pace_memory
                    * fatigue_drag
                    * damage_drag
                    * confidence_effect
                    * opponent_opening
                    * suppression
                    * urgency
                    * round_decay,
                    0.50,
                    1.30,
                )
            )

        return SignificantStrikeAttemptParameters(
            key=key,
            mean_rate_per_minute=max(0.25, float(base_rate * multiplier)),
            gamma_poisson_overdispersion=self.alpha,
            model_name="dynamic_prefight_career_pace",
            model_version="dynamic_prefight_career_pace_v0",
            calibration_factor=self.factor,
            source="pre_holdout_career_pace_dynamic_path_state",
        )
