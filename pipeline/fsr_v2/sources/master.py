"""Minimal UFC master fields used only where round stats are insufficient."""

from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH

MASTER_COLUMNS = ["fight_id", "finish_round", "match_time_sec", "method", "winner_id"]


def load_master(path: Path = MASTER_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"FSR V2 master source is missing: {path}")
    frame = pd.read_parquet(path, columns=MASTER_COLUMNS)
    if frame["fight_id"].duplicated().any():
        raise ValueError("UFC master contains duplicate fight_id values")
    return frame
