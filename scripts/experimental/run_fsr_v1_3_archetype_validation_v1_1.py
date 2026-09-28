"""Robust fight-resolution wrapper for FSR/MC V1.3 archetype validation.

Shadow/research only.

This wrapper changes only curated-fight resolution behavior. It does not change
FSR equations, PRE-fight leakage rules, V1.2 activity conversion, V1.3 finish
hazards, cardio, judging, simulator mechanics, seeds, or output metrics.

Behavior:
1. find every historical fight containing the exact unordered normalized
   fighter pair;
2. use the curated date only to disambiguate repeat matchups;
3. before launching the validator, audit every curated fight against the local
   UFCStats parquet;
4. report and skip unavailable curated fights instead of aborting the whole
   cross-archetype batch.
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


def available_curated_fights() -> tuple[validation.FightSpec, ...]:
    """Return only curated fights that exist unambiguously in local round data."""

    rounds = pd.read_parquet(validation.base.ROUND_PATH)
    rounds["event_date"] = pd.to_datetime(rounds["event_date"])
    rounds["fight_id"] = rounds["fight_id"].astype(str)

    available: list[validation.FightSpec] = []
    skipped: list[tuple[validation.FightSpec, str]] = []

    print()
    print("CURATED FIGHT AVAILABILITY AUDIT")
    print("-" * 90)

    for spec in validation.CURATED_FIGHTS:
        try:
            fight_id = resolve_fight_id(rounds, spec)
        except RuntimeError as exc:
            skipped.append((spec, str(exc)))
            print(
                f"SKIP  {spec.archetype}: {spec.fighter_a} vs {spec.fighter_b}"
            )
            print(f"      {exc}")
            continue

        available.append(spec)
        print(
            f"PASS  {spec.archetype}: {spec.fighter_a} vs {spec.fighter_b} "
            f"-> {fight_id}"
        )

    print()
    print(
        f"Available curated fights: {len(available)} / "
        f"{len(validation.CURATED_FIGHTS)}"
    )

    if skipped:
        print(
            "Unavailable curated examples are excluded from this diagnostic "
            "batch only; no simulator/model behavior is changed."
        )

    if len(available) < 4:
        raise RuntimeError(
            "Fewer than four curated fights are available; insufficient "
            "cross-archetype coverage for this checkpoint."
        )

    return tuple(available)


def main() -> None:
    # Install the robust resolver used by the underlying validator.
    validation.resolve_fight_id = resolve_fight_id

    # Filter only the diagnostic cohort. The model/simulator configuration is
    # untouched and every surviving fight still uses leakage-safe PRE-fight
    # cards and the frozen V1.3 calibration.
    validation.CURATED_FIGHTS = available_curated_fights()

    validation.main()


if __name__ == "__main__":
    main()
