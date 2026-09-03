"""Compatibility-fixed launcher for the V1.5 2026 winner-signal audit.

Shadow/research only.

The replay predictions already contain a small subset of diagnostic FSR columns
(e.g. control_imposition and striking_power).  The full winner-signal audit then
joins the authoritative cached PRE-fight FSR card, which uses the same column
names. Pandas consequently suffixes the duplicate columns to ``_x`` / ``_y``
and the audit's exact-name validation fails.

This launcher removes only those pre-existing partial ``red_fsr_*`` /
``blue_fsr_*`` columns before the authoritative full cached FSR card is joined.
It changes no ratings, labels, simulator output, or scoring logic.
"""

from __future__ import annotations

import pandas as pd

from scripts.experimental import diagnose_fsr_v1_5_winner_signal_2026 as audit


_ORIGINAL_ATTACH_FSR_CARDS = audit._attach_fsr_cards


def attach_fsr_cards_without_column_collisions(
    predictions: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Drop partial replay FSR diagnostics before joining the full cached card."""

    collision_columns = [
        column
        for column in predictions.columns
        if column.startswith(("red_fsr_", "blue_fsr_"))
    ]

    cleaned = predictions.drop(
        columns=collision_columns,
        errors="ignore",
    )

    return _ORIGINAL_ATTACH_FSR_CARDS(
        cleaned,
        snapshots,
    )


def main() -> None:
    audit._attach_fsr_cards = (
        attach_fsr_cards_without_column_collisions
    )
    audit.main()


if __name__ == "__main__":
    main()
