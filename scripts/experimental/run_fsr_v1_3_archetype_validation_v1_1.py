"""Robust fight-resolution wrapper for FSR/MC V1.3 archetype validation.

Shadow/research only.

The original archetype validator required an exact event_date match before
checking fighter names. Some local UFCStats datasets can store an event date one
day differently from the curated reference date. This wrapper changes only the
fight-ID resolver:

1. find every historical fight containing the exact unordered normalized
   fighter pair;
2. compute the absolute date distance from the curated reference date;
3. select the unique nearest fight;
4. reject ambiguous ties.

All FSR equations, PRE-fight leakage rules, V1.2 activity conversion, V1.3
finish hazards, cardio, judging, simulator mechanics, seeds, and output metrics
remain unchanged.
"""

from __future__ import annotations

import pandas as pd

from scripts.experimental import run_fsr_v1_3_archetype_validation as validation


def resolve_fight_id(
    rounds: pd.DataFrame,
    spec: validation.FightSpec,
) -> str:
    """Resolve the exact fighter pair, using date only to disambiguate fights."""

    wanted = {
        validation.normalize_name(spec.fighter_a),
        validation.normalize_name(spec.fighter_b),
    }
    target_date = pd.Timestamp(spec.event_date).normalize()

    matches: list[tuple[pd.Timedelta, pd.Timestamp, str]] = []

    for fight_id, fight_rows in rounds.groupby("fight_id", sort=False):
        names = {
            validation.normalize_name(name)
            for name in fight_rows["fighter_name"].dropna().unique()
        }
        if names != wanted:
            continue

        fight_dates = (
            pd.to_datetime(fight_rows["event_date"])
            .dt.normalize()
            .dropna()
            .unique()
        )
        if len(fight_dates) != 1:
            raise RuntimeError(
                "Expected one event date for curated fight candidate "
                f"{fight_id}; found {len(fight_dates)}"
            )

        fight_date = pd.Timestamp(fight_dates[0]).normalize()
        date_distance = abs(fight_date - target_date)
        matches.append((date_distance, fight_date, str(fight_id)))

    if not matches:
        raise RuntimeError(
            "No UFCStats fight found for exact normalized fighter pair: "
            f"{spec.fighter_a} vs {spec.fighter_b}"
        )

    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    best_distance = matches[0][0]
    nearest = [item for item in matches if item[0] == best_distance]

    if len(nearest) != 1:
        details = [
            f"{fight_date.date()}:{fight_id}"
            for _, fight_date, fight_id in nearest
        ]
        raise RuntimeError(
            "Ambiguous nearest curated fight for "
            f"{spec.fighter_a} vs {spec.fighter_b} around {spec.event_date}: "
            f"{details}"
        )

    _, resolved_date, resolved_fight_id = nearest[0]

    if resolved_date != target_date:
        print(
            "  NOTE: curated date "
            f"{target_date.date()} resolved to dataset date "
            f"{resolved_date.date()} for fight {resolved_fight_id}"
        )

    return resolved_fight_id


def main() -> None:
    validation.resolve_fight_id = resolve_fight_id
    validation.main()


if __name__ == "__main__":
    main()
