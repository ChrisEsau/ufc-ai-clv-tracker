"""Phase 2A-versus-2B takedown-initiation A/B diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..components.formulas import (
    interval_hazard_per_second,
    legacy_td_attempt_interval_probability,
    style_preferences,
    strike_attempt_rate_per_second,
    strike_landing_probability,
    td_attempt_interval_probability,
    td_success_probability,
)
from ..components.profiles import FighterProfile, MatchupProfiles, Side
from .distance_parity import FROZEN_FIXTURES, FSR_32_PATH, load_fixture_matchup


@dataclass(frozen=True)
class OntologyAuditRow:
    side: str
    fighter_name: str
    wrestling_entry: float
    control_imposition: float
    distance_striking_pressure: float
    clinch_striking_pressure: float
    legacy_wrestling_preference: float
    phase_2a_probability_10s: float
    phase_2a_hazard_per_second: float
    phase_2b_probability_10s: float
    phase_2b_hazard_per_second: float
    hazard_ratio_2b_to_2a: float
    hazard_percent_change: float
    context_multiplier: float = 1.0


def ontology_audit_rows(profiles: MatchupProfiles) -> tuple[OntologyAuditRow, ...]:
    rows = []
    for side in Side:
        fighter = profiles.fighter(side)
        legacy_probability = legacy_td_attempt_interval_probability(fighter)
        intrinsic_probability = td_attempt_interval_probability(fighter)
        legacy_hazard = interval_hazard_per_second(legacy_probability)
        intrinsic_hazard = interval_hazard_per_second(intrinsic_probability)
        ratio = intrinsic_hazard / legacy_hazard
        rows.append(
            OntologyAuditRow(
                side.value,
                fighter.fighter_name,
                fighter.wrestling_entry,
                fighter.control_imposition,
                fighter.distance_striking_pressure,
                fighter.clinch_striking_pressure,
                style_preferences(fighter)[2],
                legacy_probability,
                legacy_hazard,
                intrinsic_probability,
                intrinsic_hazard,
                ratio,
                (ratio - 1.0) * 100.0,
            )
        )
    return tuple(rows)


def matched_ontology_summary(
    profiles: MatchupProfiles,
    *,
    paths: int = 10_000,
    exposure_seconds: float = 900.0,
    seed: int = 20260811,
) -> dict[str, object]:
    """Compare independent Phase 2A and 2B clocks over equal DISTANCE time."""

    result = {}
    for index, side in enumerate(Side):
        fighter = profiles.fighter(side)
        opponent = profiles.fighter(side.opponent)
        success = td_success_probability(fighter, opponent)
        unchanged_striking = {
            "strike_attempts_per_minute": strike_attempt_rate_per_second(fighter)
            * 60.0,
            "strike_landing_probability": strike_landing_probability(
                fighter, opponent
            ),
        }
        arms = {}
        for arm_index, (name, probability) in enumerate(
            (
                ("phase_2a", legacy_td_attempt_interval_probability(fighter)),
                ("phase_2b", td_attempt_interval_probability(fighter)),
            )
        ):
            rng = np.random.default_rng(seed + index * 10 + arm_index)
            hazard = interval_hazard_per_second(probability)
            attempts = rng.poisson(hazard * exposure_seconds, size=paths)
            landed = rng.binomial(attempts, success)
            arms[name] = {
                "td_attempts_per_15_minutes": float(
                    attempts.mean() * 900.0 / exposure_seconds
                ),
                "td_success_percentage": float(landed.sum() / max(attempts.sum(), 1)),
            }
        arms["td_attempt_change_per_15_minutes"] = (
            arms["phase_2b"]["td_attempts_per_15_minutes"]
            - arms["phase_2a"]["td_attempts_per_15_minutes"]
        )
        result[side.value] = arms
        arms["unchanged_phase_2a_2b_striking"] = unchanged_striking
    return result


def fixture_report(path: Path = FSR_32_PATH, *, paths: int = 10_000):
    frame = pd.read_parquet(path)
    report = {}
    for red, blue, date in FROZEN_FIXTURES:
        profiles = load_fixture_matchup(frame, red, blue, date)
        report[f"{red} vs {blue}"] = {
            "ontology_audit": [asdict(row) for row in ontology_audit_rows(profiles)],
            "matched_distance_exposure": matched_ontology_summary(
                profiles, paths=paths
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--fsr-path", type=Path, default=FSR_32_PATH)
    args = parser.parse_args()
    print(
        json.dumps(
            fixture_report(args.fsr_path, paths=args.paths), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
