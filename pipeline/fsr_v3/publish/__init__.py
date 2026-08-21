"""Publish FSR V3 as a safe overlay on the frozen FSR V2 snapshot.

At this stage only the fully validated ground-striking family is replaced.
Every other field is copied verbatim from FSR V2.  The rejected V2
``ground_striking_defense`` field is explicitly removed so downstream V3 code
cannot accidentally continue using it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.common.paths import FSR_V2_LATEST_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.paths import (
    FSR_V3_LATEST_PATH,
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
    FSR_V3_PREFIGHT_UNCERTAINTY_PATH,
    GROUND_EFFECTIVENESS_HISTORY_PATH,
    GROUND_SUPPRESSION_HISTORY_PATH,
    GROUND_TENDENCY_HISTORY_PATH,
)

KEYS = ["event_date", "fight_id", "fighter_id"]


def _read_history(path):
    if not path.is_file():
        raise FileNotFoundError(f"missing FSR V3 history: {path}")
    frame = pd.read_parquet(path).copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    return frame


def _replacement(history, rating_name, extra_columns=None):
    columns = KEYS + ["pre_rating"] + list((extra_columns or {}).keys())
    selected = history[columns].copy().rename(columns={"pre_rating": rating_name, **(extra_columns or {})})
    duplicate = selected.duplicated(KEYS)
    if duplicate.any():
        raise ValueError(f"duplicate V3 replacement rows for {rating_name}")
    return selected


def assemble_prefight() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FSR_V2_PREFIGHT_SNAPSHOTS_PATH.is_file():
        raise FileNotFoundError(
            "FSR V3 publication requires the frozen FSR V2 prefight snapshot: "
            f"{FSR_V2_PREFIGHT_SNAPSHOTS_PATH}"
        )

    base = pd.read_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH).copy()
    base["event_date"] = pd.to_datetime(base["event_date"], errors="raise").dt.normalize()
    base["fight_id"] = base["fight_id"].astype(str)
    base["fighter_id"] = base["fighter_id"].astype(str)

    tendency = _read_history(GROUND_TENDENCY_HISTORY_PATH)
    suppression = _read_history(GROUND_SUPPRESSION_HISTORY_PATH)
    effectiveness = _read_history(GROUND_EFFECTIVENESS_HISTORY_PATH)

    # Remove every old V2 ground field that V3 redefines or rejects.
    drop = [
        "ground_striking_tendency",
        "ground_striking_suppression",
        "ground_striking_offense",
        "ground_striking_defense",
        "ground_accuracy_baseline",
        "ground_striking_burst_baseline",
        "ground_striking_population_slope_15m",
    ]
    base = base.drop(columns=[c for c in drop if c in base.columns])

    replacements = [
        _replacement(
            tendency,
            "ground_striking_tendency",
            {
                "population_burst": "ground_striking_burst_baseline",
                "population_rate_15m": "ground_striking_population_slope_15m",
            },
        ),
        _replacement(suppression, "ground_striking_suppression"),
        _replacement(
            effectiveness,
            "ground_striking_offense",
            {"population_baseline": "ground_accuracy_baseline"},
        ),
    ]

    out = base
    for replacement in replacements:
        out = out.merge(replacement, on=KEYS, how="left", validate="one_to_one")

    required = [
        "ground_striking_tendency",
        "ground_striking_suppression",
        "ground_striking_offense",
        "ground_accuracy_baseline",
        "ground_striking_burst_baseline",
    ]
    missing = [name for name in required if name not in out or out[name].isna().any()]
    if missing:
        raise ValueError(f"FSR V3 ground overlay has missing values: {missing}")
    if "ground_striking_defense" in out.columns:
        raise AssertionError("rejected ground_striking_defense leaked into FSR V3")

    uncertainty_frames = []
    for history in (tendency, suppression, effectiveness):
        u = history[
            KEYS
            + [
                "trait",
                "pre_rating",
                "pre_posterior_sd",
                "variance_multiplier",
                "sampling_enabled",
            ]
        ].copy()
        u = u.rename(
            columns={
                "pre_rating": "posterior_mean",
                "pre_posterior_sd": "posterior_sd",
            }
        )
        uncertainty_frames.append(u)

    uncertainty = pd.concat(uncertainty_frames, ignore_index=True)
    uncertainty = uncertainty.sort_values(KEYS + ["trait"]).reset_index(drop=True)
    if uncertainty[KEYS + ["trait"]].duplicated().any():
        raise ValueError("duplicate FSR V3 uncertainty rows")

    return (
        out.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True),
        uncertainty,
    )


def assemble_latest(prefight: pd.DataFrame) -> pd.DataFrame:
    """Build a conservative latest table from the newest published prefight row.

    This intentionally preserves all non-ground V2 latest fields and replaces
    the validated ground family with the newest leakage-safe V3 prefight state.
    A later V3 phase can add a dedicated post-final-event publisher if needed.
    """
    if not FSR_V2_LATEST_PATH.is_file():
        raise FileNotFoundError(f"missing frozen FSR V2 latest profiles: {FSR_V2_LATEST_PATH}")
    base = pd.read_parquet(FSR_V2_LATEST_PATH).copy()
    base["fighter_id"] = base["fighter_id"].astype(str)

    latest_ground = (
        prefight.sort_values(["event_date", "fight_id"])
        .groupby("fighter_id", as_index=False)
        .tail(1)
    )
    ground_columns = [
        "fighter_id",
        "ground_striking_tendency",
        "ground_striking_suppression",
        "ground_striking_offense",
        "ground_accuracy_baseline",
        "ground_striking_burst_baseline",
        "ground_striking_population_slope_15m",
    ]
    latest_ground = latest_ground[ground_columns]

    drop = [c for c in ground_columns[1:] + ["ground_striking_defense"] if c in base.columns]
    base = base.drop(columns=drop)
    return base.merge(latest_ground, on="fighter_id", how="left", validate="one_to_one")


def publish() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefight, uncertainty = assemble_prefight()
    latest = assemble_latest(prefight)
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH.parent.joinpath("history").mkdir(parents=True, exist_ok=True)
    prefight.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
    uncertainty.to_parquet(FSR_V3_PREFIGHT_UNCERTAINTY_PATH, index=False)
    latest.to_parquet(FSR_V3_LATEST_PATH, index=False)
    return prefight, latest, uncertainty


def main() -> None:
    prefight, latest, uncertainty = publish()
    print(
        f"published FSR V3 ground overlay: prefight={len(prefight):,}, "
        f"latest={len(latest):,}, uncertainty={len(uncertainty):,}"
    )


if __name__ == "__main__":
    main()
