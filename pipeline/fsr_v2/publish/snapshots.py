"""Assemble leakage-safe prefight snapshots from independent histories."""

from pathlib import Path
import pandas as pd
from pipeline.common.paths import FSR_V2_HISTORY_DIR
from pipeline.fsr_v2.traits.registry import resolve_groups

KEYS = ["event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "opponent_name"]
POPULATION_METADATA = {
    "standing_striking_offense": ("population_baseline", "standing_accuracy_baseline"),
    "takedown_offense": ("population_baseline", "takedown_completion_baseline"),
    "ground_striking_offense": ("population_baseline", "ground_accuracy_baseline"),
    "submission_offense": ("population_baseline", "submission_conversion_baseline"),
    "escape_offense": ("population_duration_baseline_seconds", "escape_population_mean_seconds"),
}


def load_histories(history_dir: Path = FSR_V2_HISTORY_DIR) -> pd.DataFrame:
    paths = [history_dir / f"{group.name}.parquet" for group in resolve_groups(None)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required core FSR V2 histories: " + ", ".join(missing))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def assemble_prefight(histories: pd.DataFrame) -> pd.DataFrame:
    duplicate = histories.duplicated(KEYS + ["trait"])
    if duplicate.any():
        sample = histories.loc[duplicate, KEYS + ["trait"]].head().to_dict("records")
        raise ValueError(f"Duplicate trait snapshots detected: {sample}")
    wide = histories.pivot(index=KEYS, columns="trait", values="pre_rating").reset_index()
    wide.columns.name = None
    for trait, (source, target) in POPULATION_METADATA.items():
        if source not in histories or not histories["trait"].eq(trait).any():
            continue
        selected = histories.loc[histories["trait"].eq(trait), KEYS + [source]].rename(
            columns={source: target}
        )
        wide = wide.merge(selected, on=KEYS, how="left", validate="one_to_one")
    return wide.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def assemble_latest(histories: pd.DataFrame) -> pd.DataFrame:
    # Fighter names can change across UFC history. Stable fighter_id is the
    # canonical profile key; grouping by the display name produced duplicate
    # latest rows for renamed fighters.
    latest = histories.sort_values(["event_date", "fight_id"]).groupby(
        ["fighter_id", "trait"], as_index=False
    ).tail(1)
    latest = latest.copy()
    if "latest_rating" in latest:
        latest["published_rating"] = latest["latest_rating"].fillna(latest["post_rating"])
    else:
        latest["published_rating"] = latest["post_rating"]
    wide = latest.pivot(index=["fighter_id", "fighter_name"], columns="trait", values="published_rating").reset_index()
    wide.columns.name = None
    for trait, (_, target) in POPULATION_METADATA.items():
        source = ("latest_population_duration_baseline_seconds" if trait == "escape_offense"
                  else "latest_population_baseline")
        if source not in latest or not latest["trait"].eq(trait).any():
            continue
        selected = latest.loc[latest["trait"].eq(trait),
                              ["fighter_id", source]].rename(columns={source: target})
        wide = wide.merge(selected, on=["fighter_id"], how="left", validate="one_to_one")
    return wide.sort_values("fighter_id").reset_index(drop=True)
