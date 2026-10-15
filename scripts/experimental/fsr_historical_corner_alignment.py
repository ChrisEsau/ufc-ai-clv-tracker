"""Strictly align historical two-fighter FSR pairs to master red/blue corners.

The legacy historical pair builder preserves parquet row order, which is not a
corner contract. Any caller that combines a returned pair with master ``r_*`` /
``b_*`` metadata must align by fighter_id first.
"""
from __future__ import annotations

import pandas as pd


def align_pair_to_master_corners(
    bout: pd.Series,
    pair: tuple[pd.Series, pd.Series],
) -> tuple[pd.Series, pd.Series]:
    """Return ``(red, blue)`` by matching FSR fighter_id to master r_id/b_id.

    Raises on any mismatch rather than silently trusting parquet row order.
    """
    first, second = pair
    r_id = str(bout["r_id"])
    b_id = str(bout["b_id"])
    by_id = {
        str(first["fighter_id"]): first,
        str(second["fighter_id"]): second,
    }

    if len(by_id) != 2:
        raise ValueError(
            f"Bout {bout['bout_id']}: expected two distinct FSR fighter_ids; "
            f"got {list(by_id)}"
        )
    missing = [fighter_id for fighter_id in (r_id, b_id) if fighter_id not in by_id]
    if missing:
        raise ValueError(
            f"Bout {bout['bout_id']}: FSR pair ids {list(by_id)} do not match "
            f"master corners r_id={r_id}, b_id={b_id}; missing={missing}"
        )
    return by_id[r_id], by_id[b_id]


def align_pair_dict_to_master_corners(
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Align every historical pair dictionary entry to cohort master corners."""
    aligned: dict[str, tuple[pd.Series, pd.Series]] = {}
    for _, bout in cohort.iterrows():
        bout_id = str(bout["bout_id"])
        if bout_id not in pairs:
            raise ValueError(f"Bout {bout_id}: missing FSR pair")
        aligned[bout_id] = align_pair_to_master_corners(bout, pairs[bout_id])
    return aligned
