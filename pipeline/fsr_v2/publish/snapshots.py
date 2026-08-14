"""Assemble leakage-safe prefight snapshots from independent histories."""

from pathlib import Path
import pandas as pd
from pipeline.common.paths import FSR_V2_HISTORY_DIR

KEYS = ["event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "opponent_name"]


def load_histories(history_dir: Path = FSR_V2_HISTORY_DIR) -> pd.DataFrame:
    paths = sorted(history_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No FSR V2 histories found in {history_dir}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def assemble_prefight(histories: pd.DataFrame) -> pd.DataFrame:
    duplicate = histories.duplicated(KEYS + ["trait"])
    if duplicate.any():
        sample = histories.loc[duplicate, KEYS + ["trait"]].head().to_dict("records")
        raise ValueError(f"Duplicate trait snapshots detected: {sample}")
    wide = histories.pivot(index=KEYS, columns="trait", values="pre_rating").reset_index()
    wide.columns.name = None
    return wide.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def assemble_latest(histories: pd.DataFrame) -> pd.DataFrame:
    latest = histories.sort_values(["event_date", "fight_id"]).groupby(
        ["fighter_id", "fighter_name", "trait"], as_index=False
    ).tail(1)
    wide = latest.pivot(index=["fighter_id", "fighter_name"], columns="trait", values="post_rating").reset_index()
    wide.columns.name = None
    return wide.sort_values("fighter_id").reset_index(drop=True)
