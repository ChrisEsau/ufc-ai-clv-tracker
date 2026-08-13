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


@dataclass(frozen=True)
class DistanceActionRateProvider:
    profiles: MatchupProfiles

    def candidates(self, state: FightState, context: FightContext):
        if state.phase is not Phase.DISTANCE:
            return ()
        output = []
        for side in Side:
            profile = self.profiles.fighter(side)
            rates = {
                "strike": strike_attempt_rate_per_second(profile),
                "takedown": td_attempt_rate_per_second(profile),
                "clinch_entry": interval_hazard_per_second(
                    clinch_entry_interval_probability(profile)
                ),
            }
            output.extend(
                EventRate(DistanceCandidate(side, family, self.profiles), rate)
                for family, rate in rates.items()
            )
        return tuple(output)

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

    def candidates(self, state: FightState, context: FightContext):
        if state.phase is Phase.DISTANCE:
            return DistanceActionRateProvider(self.profiles).candidates(state, context)
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
                    EventRate(PhaseCandidate(side, "clinch_strike", self.profiles), phase_strike_rate_per_second(fighter, "clinch")),
                    EventRate(PhaseCandidate(side, "clinch_takedown", self.profiles), interval_hazard_per_second(clinch_td_interval_probability(fighter))),
                ))
            output.append(EventRate(PhaseCandidate(controller.opponent, "clinch_separation", self.profiles), interval_hazard_per_second(clinch_separation_interval_probability(controller_profile, opponent))))
            return tuple(output)
        if state.ground_controller is None:
            return ()
        top = Side(state.ground_controller)
        bottom = top.opponent
        top_profile, bottom_profile = self.profiles.fighter(top), self.profiles.fighter(bottom)
        escape_rate, reversal_rate, _ = ground_exit_rates(top_profile, bottom_profile)
        return (
            EventRate(PhaseCandidate(top, "ground_strike", self.profiles), phase_strike_rate_per_second(top_profile, "ground")),
            EventRate(PhaseCandidate(bottom, "ground_strike", self.profiles), phase_strike_rate_per_second(bottom_profile, "ground", bottom=True)),
            EventRate(PhaseCandidate(top, "submission_attempt", self.profiles), interval_hazard_per_second(submission_attempt_interval_probability(top_profile))),
            EventRate(PhaseCandidate(bottom, "submission_attempt", self.profiles), interval_hazard_per_second(submission_attempt_interval_probability(bottom_profile, bottom=True))),
            EventRate(PhaseCandidate(bottom, "ground_escape", self.profiles), escape_rate),
            EventRate(PhaseCandidate(bottom, "ground_reversal", self.profiles), reversal_rate),
        )
