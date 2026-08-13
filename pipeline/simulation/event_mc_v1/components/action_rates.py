"""Composable provider for the six Phase 2A DISTANCE candidates."""

from dataclasses import dataclass

from ..contracts import FightContext
from ..scheduler import EventRate
from ..state import FightState, Phase
from .actions import DistanceCandidate, PhaseCandidate
from .formulas import (
    ActionRateAudit,
    clinch_entry_interval_probability,
    interval_hazard_per_second,
    legacy_td_attempt_interval_probability,
    strike_attempt_rate_per_second,
    style_preferences,
    td_attempt_interval_probability,
    td_attempt_rate_per_second,
    clinch_separation_interval_probability,
    clinch_td_interval_probability,
    ground_exit_rates,
    phase_strike_rate_per_second,
    submission_attempt_interval_probability,
)
from .profiles import MatchupProfiles, Side
from ..modifiers import DynamicModifierProvider
from ..stamina import StaminaModel
from ..calibration import DEFAULT_CALIBRATION, EventMCCalibration


@dataclass(frozen=True)
class DistanceActionRateProvider:
    profiles: MatchupProfiles
    stamina_model: StaminaModel | None = None
    modifier_provider: DynamicModifierProvider | None = None
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    def candidates(self, state: FightState, context: FightContext):
        if state.phase is not Phase.DISTANCE:
            return ()
        output = []
        for side in Side:
            profile = self.profiles.fighter(side)
            rates = {
                "strike": strike_attempt_rate_per_second(profile, self.calibration),
                "takedown": td_attempt_rate_per_second(profile, calibration=self.calibration),
                "clinch_entry": interval_hazard_per_second(
                    clinch_entry_interval_probability(profile, self.calibration)
                ),
            }
            output.extend(
                EventRate(DistanceCandidate(side, family, self.profiles, self.stamina_model, self.modifier_provider, self.calibration), rate * self._output_multiplier(state, side))
                for family, rate in rates.items()
            )
        return tuple(output)

    def _output_multiplier(self, state: FightState, side: Side) -> float:
        if self.modifier_provider is None:
            return 1.0
        return self.modifier_provider.modifiers(self.profiles.fighter(side), state, side).output_multiplier

    def audit_rows(self) -> tuple[ActionRateAudit, ...]:
        rows = []
        for side in Side:
            profile = self.profiles.fighter(side)
            _, _, wrestling_preference = style_preferences(profile)
            strike_rate = strike_attempt_rate_per_second(profile)
            rows.append(
                ActionRateAudit(
                    side.value,
                    "strike",
                    None,
                    30.0,
                    strike_rate,
                    {"distance_striking_pressure": profile.distance_striking_pressure},
                )
            )
            for family, probability, inputs in (
                (
                    "takedown",
                    td_attempt_interval_probability(profile),
                    {
                        "wrestling_entry": profile.wrestling_entry,
                        "context_multiplier": 1.0,
                        "phase_2a_probability": legacy_td_attempt_interval_probability(profile),
                        "phase_2a_legacy_wrestling_preference": wrestling_preference,
                        "legacy_wrestling_preference": wrestling_preference,
                    },
                ),
                (
                    "clinch_entry",
                    clinch_entry_interval_probability(profile),
                    {
                        "distance_striking_pressure": profile.distance_striking_pressure,
                        "clinch_striking_pressure": profile.clinch_striking_pressure,
                        "wrestling_entry": profile.wrestling_entry,
                    },
                ),
            ):
                rows.append(
                    ActionRateAudit(
                        side.value,
                        family,
                        probability,
                        10.0,
                        interval_hazard_per_second(probability),
                        inputs,
                    )
                )
        return tuple(rows)


@dataclass(frozen=True)
class FightFlowRateProvider:
    """Composed phase provider; keeps the generic scheduler UFC-agnostic."""

    profiles: MatchupProfiles
    stamina_model: StaminaModel | None = None
    modifier_provider: DynamicModifierProvider | None = None
    calibration: EventMCCalibration = DEFAULT_CALIBRATION

    def candidates(self, state: FightState, context: FightContext):
        if state.phase is Phase.DISTANCE:
            return DistanceActionRateProvider(self.profiles, self.stamina_model, self.modifier_provider, self.calibration).candidates(state, context)
        if state.phase is Phase.CLINCH:
            if state.clinch_controller is None:
                return ()
            controller = Side(state.clinch_controller)
            controller_profile = self.profiles.fighter(controller)
            opponent = self.profiles.fighter(controller.opponent)
            output = []
            for side in Side:
                fighter = self.profiles.fighter(side)
                output.extend((
                    EventRate(self._candidate(side, "clinch_strike"), phase_strike_rate_per_second(fighter, "clinch", calibration=self.calibration) * self._output(state, side)),
                    EventRate(self._candidate(side, "clinch_takedown"), interval_hazard_per_second(clinch_td_interval_probability(fighter, self.calibration)) * self._output(state, side)),
                ))
            output.append(EventRate(self._candidate(controller.opponent, "clinch_separation"), interval_hazard_per_second(clinch_separation_interval_probability(controller_profile, opponent, self.calibration))))
            return tuple(output)
        if state.ground_controller is None:
            return ()
        top = Side(state.ground_controller)
        bottom = top.opponent
        top_profile, bottom_profile = self.profiles.fighter(top), self.profiles.fighter(bottom)
        escape_rate, reversal_rate, _ = ground_exit_rates(top_profile, bottom_profile, self.calibration)
        return (
            EventRate(self._candidate(top, "ground_strike"), phase_strike_rate_per_second(top_profile, "ground", calibration=self.calibration) * self._output(state, top)),
            EventRate(self._candidate(bottom, "ground_strike"), phase_strike_rate_per_second(bottom_profile, "ground", bottom=True, calibration=self.calibration) * self._output(state, bottom)),
            EventRate(self._candidate(top, "submission_attempt"), interval_hazard_per_second(submission_attempt_interval_probability(top_profile, calibration=self.calibration)) * self._output(state, top)),
            EventRate(self._candidate(bottom, "submission_attempt"), interval_hazard_per_second(submission_attempt_interval_probability(bottom_profile, bottom=True, calibration=self.calibration)) * self._output(state, bottom)),
            EventRate(self._candidate(bottom, "ground_escape"), escape_rate),
            EventRate(self._candidate(bottom, "ground_reversal"), reversal_rate),
        )

    def _candidate(self, side: Side, family: str) -> PhaseCandidate:
        return PhaseCandidate(side, family, self.profiles, self.stamina_model, self.modifier_provider, self.calibration)

    def _output(self, state: FightState, side: Side) -> float:
        return 1.0 if self.modifier_provider is None else self.modifier_provider.modifiers(self.profiles.fighter(side), state, side).output_multiplier
