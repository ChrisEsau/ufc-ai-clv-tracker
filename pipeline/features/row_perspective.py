from __future__ import annotations

from typing import Any

import pandas as pd


def apply_row_perspective_to_prepared_fights(prepared_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    mode = str((config.get("row_perspective", {}) or {}).get("mode", "fight_level")).strip().lower()
    if mode == "fight_level":
        out = prepared_df.copy()
        out["base_fight_id"] = out["fight_id"].astype(str)
        out["state_fight_id"] = out["fight_id"].astype(str)
        out["row_perspective"] = "original"
        return out
    if mode != "both_fighter_perspectives":
        raise ValueError(f"Unsupported row_perspective.mode: {mode}")

    original = prepared_df.copy()
    original["base_fight_id"] = original["fight_id"].astype(str)
    original["state_fight_id"] = original["fight_id"].astype(str)
    original["row_perspective"] = "original"

    flipped = original.copy()
    swap_column_pairs(flipped, "r_", "b_")
    swap_column_pairs(flipped, "red_", "blue_")
    flipped["fight_id"] = flipped["base_fight_id"].astype(str) + "__flip"
    flipped["row_perspective"] = "flipped"
    if "target" in flipped.columns:
        flipped["target"] = 1 - pd.to_numeric(flipped["target"], errors="coerce")

    return pd.concat([original, flipped], ignore_index=True)


def swap_column_pairs(df: pd.DataFrame, left_prefix: str, right_prefix: str) -> None:
    for left_column in [column for column in df.columns if column.startswith(left_prefix)]:
        right_column = f"{right_prefix}{left_column[len(left_prefix):]}"
        if right_column in df.columns:
            left_values = df[left_column].copy()
            df[left_column] = df[right_column]
            df[right_column] = left_values
