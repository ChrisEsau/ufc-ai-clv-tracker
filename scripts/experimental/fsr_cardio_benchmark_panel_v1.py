"""Small benchmark panel for FSR Cardio V1.

Purpose
-------
Run the existing leakage-safe Cardio V1 target-card builder for a user-supplied
set of historical fight IDs and print one compact comparison table.

This script does not change simulator behavior or choose benchmark fights.
It is only a convenient inspection panel for deciding which fights/fighters
should be used to validate the current cardio bridge.

Guardrails
----------
- Evaluation fights before 2018-01-01 are rejected.
- Cardio values come directly from ``fsr_cardio_v1.build_target_card``.
- No production RFS artifacts are modified.
- Shadow/research only.

Usage
-----
PYTHONPATH=. python \
    scripts/experimental/fsr_cardio_benchmark_panel_v1.py \
    <fight_id> [<fight_id> ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.experimental.fsr_cardio_v1 import (
    build_target_card,
)


MIN_EVALUATION_DATE = pd.Timestamp("2018-01-01")
OUTPUT_DIR = Path(
    "data/simulation/rfs_mc_v2_shared_state"
)
OUTPUT_PATH = OUTPUT_DIR / "fsr_cardio_v1_benchmark_panel.csv"


PANEL_COLUMNS = [
    "fight_id",
    "date",
    "fighter_name",
    "prior_ufc_fights",
    "fatigue_accumulation_resistance_rating",
    "fatigue_performance_resilience_rating",
    "recovery_ability_rating",
    "fatigue_accumulation_resistance_reliability",
    "fatigue_performance_resilience_reliability",
    "recovery_ability_reliability",
]


def build_panel(
    fight_ids: list[str],
) -> pd.DataFrame:
    """Build one compact Cardio V1 comparison panel."""

    if not fight_ids:
        raise ValueError("At least one fight ID is required.")

    cards: list[pd.DataFrame] = []

    for fight_id in fight_ids:
        card = build_target_card(
            str(fight_id)
        ).copy()

        card["date"] = pd.to_datetime(
            card["date"],
            errors="coerce",
        )

        if card["date"].isna().any():
            raise RuntimeError(
                f"{fight_id}: Cardio V1 returned an invalid target date"
            )

        target_dates = card["date"].drop_duplicates()

        if len(target_dates) != 1:
            raise RuntimeError(
                f"{fight_id}: expected exactly one target date; "
                f"found {len(target_dates)}"
            )

        target_date = pd.Timestamp(
            target_dates.iloc[0]
        )

        if target_date < MIN_EVALUATION_DATE:
            raise RuntimeError(
                f"{fight_id}: evaluation date {target_date.date()} "
                "is before the locked 2018-01-01 benchmark cutoff"
            )

        cards.append(card)

    panel = pd.concat(
        cards,
        ignore_index=True,
    )

    missing = [
        column
        for column in PANEL_COLUMNS
        if column not in panel.columns
    ]

    if missing:
        raise RuntimeError(
            "Cardio V1 panel is missing expected columns: "
            f"{missing}"
        )

    return (
        panel[PANEL_COLUMNS]
        .sort_values(
            [
                "date",
                "fight_id",
                "fighter_name",
            ]
        )
        .reset_index(drop=True)
    )


def print_panel(
    panel: pd.DataFrame,
) -> None:
    """Print ratings and reliability in a compact side-by-side table."""

    display = panel.copy()

    display = display.rename(
        columns={
            "fighter_name": "Fighter",
            "prior_ufc_fights": "Prior",
            "fatigue_accumulation_resistance_rating": "FatigueRes",
            "fatigue_performance_resilience_rating": "PerfRes",
            "recovery_ability_rating": "Recovery",
            "fatigue_accumulation_resistance_reliability": "FatRel",
            "fatigue_performance_resilience_reliability": "PerfRel",
            "recovery_ability_reliability": "RecRel",
        }
    )

    display["date"] = (
        pd.to_datetime(display["date"])
        .dt.strftime("%Y-%m-%d")
    )

    rating_columns = [
        "FatigueRes",
        "PerfRes",
        "Recovery",
    ]
    reliability_columns = [
        "FatRel",
        "PerfRel",
        "RecRel",
    ]

    for column in rating_columns:
        display[column] = display[column].map(
            lambda value: f"{float(value):.2f}"
        )

    for column in reliability_columns:
        display[column] = display[column].map(
            lambda value: f"{float(value):.3f}"
        )

    print()
    print("=" * 120)
    print("FSR CARDIO V1 — SMALL BENCHMARK PANEL")
    print("=" * 120)
    print(
        display.to_string(
            index=False,
        )
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fight_ids",
        nargs="+",
        help="One or more historical UFC fight IDs (evaluation date >= 2018-01-01)",
    )
    args = parser.parse_args()

    panel = build_panel(
        [
            str(fight_id)
            for fight_id in args.fight_ids
        ]
    )

    print_panel(panel)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved benchmark panel: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
