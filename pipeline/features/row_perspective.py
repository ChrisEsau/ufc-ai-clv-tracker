from __future__ import annotations

import pandas as pd


def original_rows(prepared_df: pd.DataFrame) -> pd.DataFrame:
    out = prepared_df.copy()
    out["base_fight_id"] = out["fight_id"].astype(str)
    out["state_fight_id"] = out["fight_id"].astype(str)
    out["row_perspective"] = "original"
    return out


def both_perspectives(prepared_df: pd.DataFrame) -> pd.DataFrame:
    original = original_rows(prepared_df)
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
