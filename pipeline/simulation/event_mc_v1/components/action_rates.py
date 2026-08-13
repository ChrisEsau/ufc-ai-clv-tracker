"""Composable provider for the six Phase 2A DISTANCE candidates."""

from dataclasses import dataclass

from ..contracts import FightContext
from ..scheduler import EventRate
from ..state import FightState, Phase
from .actions import DistanceCandidate
from .formulas import (
    ActionRateAudit,
    clinch_entry_interval_probability,
    interval_hazard_per_second,
    strike_attempt_rate_per_second,
    style_preferences,
    td_attempt_interval_probability,
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
                "takedown": interval_hazard_per_second(
                    td_attempt_interval_probability(profile)
                ),
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
                        "control_imposition": profile.control_imposition,
                        "distance_striking_pressure": profile.distance_striking_pressure,
                        "clinch_striking_pressure": profile.clinch_striking_pressure,
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
