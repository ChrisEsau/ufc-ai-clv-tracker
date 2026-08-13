"""Matched-exposure legacy-vs-continuous Phase 2A diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..components.action_rates import DistanceActionRateProvider
from ..components.formulas import (
    LEGACY_INTERVAL_SECONDS,
    clinch_entry_interval_probability,
    interval_hazard_per_second,
    strike_attempt_rate_per_second,
    strike_landing_probability,
    td_attempt_interval_probability,
    td_success_probability,
)
from ..components.profiles import FighterProfile, MatchupProfiles, Side

FSR_32_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)

FROZEN_FIXTURES = (
    ("Rob Font", "Raul Rosas Jr.", "2026-03-07"),
    ("Derrick Lewis", "Chris Daukaus", "2021-12-18"),
    ("Max Holloway", "Calvin Kattar", "2021-01-16"),
    ("Merab Dvalishvili", "Petr Yan", "2023-03-11"),
)


def matched_exposure_summary(
    profiles: MatchupProfiles,
    *,
    paths: int = 10_000,
    exposure_seconds: float = 900.0,
    seed: int = 20260811,
) -> dict[str, object]:
    """Compare V0 segment sampling with equivalent continuous Poisson clocks.

    Both arms hold the phase at DISTANCE so downstream missing mechanics do not
    contaminate the comparison. The legacy transition arm preserves V0's at-most
    one competing transition per 10-second segment.
    """

    rng_legacy = np.random.default_rng(seed)
    rng_event = np.random.default_rng(seed + 1)
    segments = int(exposure_seconds / LEGACY_INTERVAL_SECONDS)
    result: dict[str, object] = {}
    for side in Side:
        fighter = profiles.fighter(side)
        opponent = profiles.fighter(side.opponent)
        strike_rate = strike_attempt_rate_per_second(fighter)
        accuracy = strike_landing_probability(fighter, opponent)
        td_probability = td_attempt_interval_probability(fighter)
        td_rate = interval_hazard_per_second(td_probability)
        td_success = td_success_probability(fighter, opponent)
        clinch_probability = clinch_entry_interval_probability(fighter)
        clinch_rate = interval_hazard_per_second(clinch_probability)

        legacy_strikes = rng_legacy.poisson(
            strike_rate * LEGACY_INTERVAL_SECONDS, size=(paths, segments)
        ).sum(axis=1)
        legacy_landed = rng_legacy.binomial(legacy_strikes, accuracy)
        event_strikes = rng_event.poisson(strike_rate * exposure_seconds, size=paths)
        event_landed = rng_event.binomial(event_strikes, accuracy)

        # V0's competing sampler converts each probability to integrated hazard,
        # then permits at most one transition event per segment across both sides.
        all_transition_rates = []
        labels = []
        for candidate_side in Side:
            candidate = profiles.fighter(candidate_side)
            all_transition_rates.extend(
                [
                    interval_hazard_per_second(td_attempt_interval_probability(candidate)),
                    interval_hazard_per_second(clinch_entry_interval_probability(candidate)),
                ]
            )
            labels.extend([(candidate_side, "td"), (candidate_side, "clinch")])
        total_integrated = sum(all_transition_rates) * LEGACY_INTERVAL_SECONDS
        event_probability = 1.0 - np.exp(-total_integrated)
        event_count = rng_legacy.binomial(segments, event_probability, size=paths)
        chosen_share = np.array(all_transition_rates) / sum(all_transition_rates)
        chosen = np.array(
            [rng_legacy.multinomial(count, chosen_share) for count in event_count]
        )
        td_index = labels.index((side, "td"))
        clinch_index = labels.index((side, "clinch"))
        legacy_td = chosen[:, td_index]
        legacy_clinch = chosen[:, clinch_index]
        legacy_td_landed = rng_legacy.binomial(legacy_td, td_success)

        event_td = rng_event.poisson(td_rate * exposure_seconds, size=paths)
        event_clinch = rng_event.poisson(clinch_rate * exposure_seconds, size=paths)
        event_td_landed = rng_event.binomial(event_td, td_success)
        result[side.value] = {
            "legacy": _metrics(
                legacy_strikes,
                legacy_landed,
                legacy_td,
                legacy_td_landed,
                legacy_clinch,
                exposure_seconds,
            ),
            "event_mc": _metrics(
                event_strikes,
                event_landed,
                event_td,
                event_td_landed,
                event_clinch,
                exposure_seconds,
            ),
        }
    return result


def _metrics(strikes, landed, td, td_landed, clinch, exposure_seconds):
    minutes = exposure_seconds / 60.0
    return {
        "strike_attempts_per_minute": float(np.mean(strikes) / minutes),
        "strike_landing_percentage": float(np.sum(landed) / np.sum(strikes)),
        "td_attempts_per_15_minutes": float(np.mean(td) * 900.0 / exposure_seconds),
        "td_success_percentage": float(np.sum(td_landed) / max(np.sum(td), 1)),
        "clinch_entries_per_minute": float(np.mean(clinch) / minutes),
    }


def load_fixture_matchup(
    frame: pd.DataFrame, red_name: str, blue_name: str, event_date: str
) -> MatchupProfiles:
    date = pd.Timestamp(event_date)
    matches = frame[
        (pd.to_datetime(frame["date"]) == date)
        & frame["fighter_name"].isin([red_name, blue_name])
    ]
    common = set(matches.loc[matches["fighter_name"] == red_name, "fight_id"]) & set(
        matches.loc[matches["fighter_name"] == blue_name, "fight_id"]
    )
    if len(common) != 1:
        raise ValueError(f"fixture did not resolve uniquely: {red_name} vs {blue_name}")
    rows = matches[matches["fight_id"] == common.pop()].set_index("fighter_name")
    red_row = rows.loc[red_name].to_dict()
    blue_row = rows.loc[blue_name].to_dict()
    red_row["fighter_name"] = red_name
    blue_row["fighter_name"] = blue_name
    return MatchupProfiles(
        FighterProfile.from_mapping(red_row),
        FighterProfile.from_mapping(blue_row),
    )


def fixture_report(path: Path = FSR_32_PATH, *, paths: int = 5_000):
    frame = pd.read_parquet(path)
    report = {}
    for red, blue, date in FROZEN_FIXTURES:
        profiles = load_fixture_matchup(frame, red, blue, date)
        provider = DistanceActionRateProvider(profiles)
        report[f"{red} vs {blue}"] = {
            "rate_audit": [asdict(row) for row in provider.audit_rows()],
            "matched_distance_exposure": matched_exposure_summary(
                profiles, paths=paths
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=5_000)
    args = parser.parse_args()
    print(json.dumps(fixture_report(paths=args.paths), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
