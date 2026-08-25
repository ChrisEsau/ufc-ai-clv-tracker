"""Frozen chronological mature-fighter cohort construction and validation."""

from __future__ import annotations
from collections import Counter
from pathlib import Path
import pandas as pd
from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

SPLIT_COUNTS = {
    "development": 400,
    "calibration": 200,
    "validation": 200,
    "final_holdout": 200,
}
MATURITY_THRESHOLD = 2


def build_manifest(
    master_path: Path = MASTER_PATH,
    round_stats_path: Path = ROUND_STATS_PATH,
    prefight_path: Path = FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
) -> tuple[pd.DataFrame, dict]:
    master = pd.read_parquet(master_path).drop_duplicates("fight_id").copy()
    master["date"] = pd.to_datetime(master["date"], errors="coerce").dt.normalize()
    master = master.dropna(subset=["date", "fight_id", "r_id", "b_id"]).sort_values(
        ["date", "fight_id"]
    )
    rounds = pd.read_parquet(round_stats_path)
    available = set(rounds["fight_id"].astype(str))
    snapshots = _canonical_prefight_keys(pd.read_parquet(prefight_path))
    prior: Counter[str] = Counter()
    candidates = []
    excluded = 0
    missing_round_stats = 0
    threshold_counts = {1: 0, 2: 0, 3: 0, 5: 0}
    missing_prefight = 0
    # Process a complete event date before updating history. This makes every
    # count strictly prefight even if a fighter has multiple rows on one date.
    for _, same_date in master.groupby("date", sort=True):
        for _, row in same_date.iterrows():
            fid, red, blue = str(row["fight_id"]), str(row["r_id"]), str(row["b_id"])
            rp, bp = prior[red], prior[blue]
            exact_prefight = _has_exact_corners(snapshots, row["date"], fid, red, blue)
            if fid not in available:
                missing_round_stats += 1
                continue
            if not exact_prefight:
                missing_prefight += 1
                continue
            for threshold in threshold_counts:
                threshold_counts[threshold] += int(rp >= threshold and bp >= threshold)
            if rp >= MATURITY_THRESHOLD and bp >= MATURITY_THRESHOLD:
                candidates.append(
                    {
                        "bout_id": fid,
                        "date": row["date"].date().isoformat(),
                        "red_fighter_id": red,
                        "red_fighter": str(row["r_name"]),
                        "blue_fighter_id": blue,
                        "blue_fighter": str(row["b_name"]),
                        "red_prior_ufc_fights": rp,
                        "blue_prior_ufc_fights": bp,
                        "cohort_split": "",
                        "included": True,
                        "exclusion_reason": "",
                    }
                )
            else:
                excluded += 1
        for fighter_id in pd.concat(
            [same_date["r_id"], same_date["b_id"]], ignore_index=True
        ).astype(str):
            prior[fighter_id] += 1
    need = sum(SPLIT_COUNTS.values())
    if len(candidates) < need:
        raise RuntimeError(f"only {len(candidates)} mature fights; need {need}")
    selected = candidates[-need:]
    offset = 0
    for split, count in SPLIT_COUNTS.items():
        for row in selected[offset : offset + count]:
            row["cohort_split"] = split
        offset += count
    manifest = pd.DataFrame(selected)
    validate_manifest(manifest)
    mins = manifest[["red_prior_ufc_fights", "blue_prior_ufc_fights"]].min(axis=1)
    audit = {
        "master_fights_with_valid_identity_and_date": len(master),
        "fights_with_round_stats": len(master) - missing_round_stats,
        "excluded_missing_round_stats": missing_round_stats,
        "excluded_missing_or_nonunique_exact_prefight": missing_prefight,
        "exclusion_counts": {
            "missing_round_stats": missing_round_stats,
            "missing_or_nonunique_exact_prefight": missing_prefight,
            "below_maturity_threshold_2": excluded,
        },
        "eligible_fights_threshold_2": len(candidates),
        "excluded_fights_threshold_2": excluded,
        "eligible_counts_by_threshold": {
            str(k): v for k, v in threshold_counts.items()
        },
        "additional_vs_threshold_2": {
            str(k): v - threshold_counts[2] for k, v in threshold_counts.items()
        },
        "selected_prior_fight_min_distribution": {
            k: float(v)
            for k, v in mins.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .to_dict()
            .items()
        },
    }
    return manifest, audit


def _canonical_prefight_keys(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"event_date", "fight_id", "fighter_id"}
    if missing := required - set(frame):
        raise ValueError(f"prefight snapshots missing {sorted(missing)}")
    keys = frame.loc[:, sorted(required)].copy()
    keys["event_date"] = pd.to_datetime(
        keys["event_date"], errors="coerce"
    ).dt.normalize()
    keys["fight_id"] = keys["fight_id"].astype(str)
    keys["fighter_id"] = keys["fighter_id"].astype(str)
    return keys


def _has_exact_corners(
    keys: pd.DataFrame, event_date, fight_id: str, red_id: str, blue_id: str
) -> bool:
    matched = keys.loc[
        keys["event_date"].eq(pd.Timestamp(event_date).normalize())
        & keys["fight_id"].eq(str(fight_id))
        & keys["fighter_id"].isin((str(red_id), str(blue_id)))
    ]
    counts = matched["fighter_id"].value_counts()
    return counts.to_dict() == {str(red_id): 1, str(blue_id): 1}


def validate_manifest_prefight_contract(
    manifest: pd.DataFrame, snapshots: pd.DataFrame
) -> None:
    """Require one exact canonical prefight row for both manifest corners."""
    keys = _canonical_prefight_keys(snapshots)
    failures = [
        str(row.bout_id)
        for row in manifest.itertuples(index=False)
        if not _has_exact_corners(
            keys,
            row.date,
            str(row.bout_id),
            str(row.red_fighter_id),
            str(row.blue_fighter_id),
        )
    ]
    if failures:
        raise ValueError(
            "manifest violates exact historical prefight contract for bouts: "
            + ", ".join(failures[:10])
        )


def validate_manifest(frame: pd.DataFrame) -> None:
    required = {"bout_id", "date", "cohort_split", "red_fighter_id", "blue_fighter_id"}
    if missing := required - set(frame):
        raise ValueError(f"manifest missing {sorted(missing)}")
    if frame["bout_id"].astype(str).duplicated().any():
        raise ValueError("cohort splits overlap")
    if set(frame["cohort_split"]) != set(SPLIT_COUNTS):
        raise ValueError("all four frozen splits are required")
    ranges = []
    for split in SPLIT_COUNTS:
        dates = pd.to_datetime(frame.loc[frame.cohort_split.eq(split), "date"])
        if not dates.is_monotonic_increasing:
            raise ValueError(f"{split} is not chronological")
        ranges.append((dates.min(), dates.max()))
    if any(ranges[i][1] > ranges[i + 1][0] for i in range(3)):
        raise ValueError("split chronology overlaps")


def select_split(
    frame: pd.DataFrame, split: str, *, allow_final_holdout: bool = False
) -> pd.DataFrame:
    if split == "final_holdout" and not allow_final_holdout:
        raise PermissionError(
            "final holdout is dark; explicit future authorization is required"
        )
    if split not in SPLIT_COUNTS:
        raise ValueError(f"unknown cohort split: {split}")
    return frame.loc[frame.cohort_split.eq(split)].copy()
