"""Preserve the validated FSR-32 physical/stamina contract inside FSR V2.

FSR V2 redesigned the fight-mechanics traits, but EVENT MC still requires the
fighter-specific physical traits that were already researched and frozen in
FSR-32.  This module deliberately reuses those exact leakage-safe builders so
we do not silently change their equations while migrating the data contract.

Simulator-facing physical fields:

- stamina_capacity (fixed 100.0 profile parameter)
- stamina_depletion_resistance (alias of fatigue_accumulation_resistance)
- stamina_performance_resilience (alias of fatigue_performance_resilience)
- striking_power (fresh FSR-32 hierarchical power model)
- damage_durability (FSR reservoir model)
- knockdown_resistance (FSR reservoir model)

The historical output is a true pre-fight snapshot.  The current/latest output
uses synthetic post-history sentinel bouts so every fighter is snapshotted
after all of their real evidence, including their most recent fight.  Sentinel
observations are never allowed to affect a returned rating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    ROUND_FIGHTER_STATE_HISTORY_PATH,
    ROUND_STATS_PATH,
)
from scripts.experimental import build_fsr_32_database as power32
from scripts.experimental import fsr_dynamic_families_v1 as dynamic
from scripts.experimental import fsr_finish_reservoir_traits_v1 as reservoir


STAMINA_CAPACITY = 100.0

PHYSICAL_COLUMNS = (
    "stamina_capacity",
    "stamina_depletion_resistance",
    "stamina_performance_resilience",
    "striking_power",
    "damage_durability",
    "knockdown_resistance",
)

LEARNED_PHYSICAL_COLUMNS = (
    "stamina_depletion_resistance",
    "stamina_performance_resilience",
    "striking_power",
    "damage_durability",
    "knockdown_resistance",
)

_DYNAMIC_RENAMES = {
    "fatigue_accumulation_resistance": "stamina_depletion_resistance",
    "fatigue_performance_resilience": "stamina_performance_resilience",
}

_SENTINEL_PREFIX = "__fsr_v2_latest_physical__"
_DUMMY_FIGHTER_ID = "__fsr_v2_latest_physical_dummy__"


@dataclass(frozen=True)
class PhysicalSnapshots:
    prefight: pd.DataFrame
    latest: pd.DataFrame


def _date_column(frame: pd.DataFrame) -> str:
    if "date" in frame.columns:
        return "date"
    if "event_date" in frame.columns:
        return "event_date"
    raise ValueError("frame has neither date nor event_date")


def _normalize_inputs(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rfs = rfs.copy()
    rounds = rounds.copy()
    master = master.copy()

    for frame in (rfs, rounds, master):
        frame["fight_id"] = frame["fight_id"].astype(str)

    rfs["fighter_id"] = rfs["fighter_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)

    rfs_date = _date_column(rfs)
    rounds_date = _date_column(rounds)
    master_date = _date_column(master)

    rfs[rfs_date] = pd.to_datetime(rfs[rfs_date], errors="raise")
    rounds[rounds_date] = pd.to_datetime(rounds[rounds_date], errors="raise")
    master[master_date] = pd.to_datetime(master[master_date], errors="raise")

    # The frozen physical builders use ``date`` for the RFS/master chronology.
    if "date" not in rfs.columns:
        rfs["date"] = rfs[rfs_date]
    if "date" not in master.columns:
        master["date"] = master[master_date]

    if rfs.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("RFS history violates fighter-fight grain")

    return rfs, rounds, master


def _physical_from_prefight_builders(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Replay the exact FSR-32 physical families at historical prefight grain."""
    dynamic_snapshots = dynamic.build_prefight_snapshots(rfs, rounds)
    reservoir_snapshots = reservoir.build_prefight_snapshots(rfs)

    base_columns = ["fight_id", "fighter_id"]
    if "fighter_name" in rfs.columns:
        base_columns.append("fighter_name")
    base = rfs[base_columns].copy()

    dynamic_keep = [
        "fight_id",
        "fighter_id",
        "fatigue_accumulation_resistance",
        "fatigue_performance_resilience",
    ]
    dynamic_part = dynamic_snapshots[dynamic_keep].rename(columns=_DYNAMIC_RENAMES)

    reservoir_part = reservoir_snapshots[
        ["fight_id", "fighter_id", "damage_durability", "knockdown_resistance"]
    ].copy()

    fight_order = power32._fight_order_table(master)
    evidence = power32._power_evidence_by_fighter_fight(master, rounds, fight_order)
    power_part = power32.build_prefight_striking_power(
        base[["fight_id", "fighter_id"]],
        evidence,
        fight_order,
    )

    out = base.merge(
        dynamic_part,
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    out = out.merge(
        reservoir_part,
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    out = out.merge(
        power_part,
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    out["stamina_capacity"] = STAMINA_CAPACITY

    if len(out) != len(base):
        raise RuntimeError(
            "physical prefight replay lost fighter-fight rows: "
            f"base={len(base):,}, physical={len(out):,}"
        )

    return _validate_physical(out)


def _latest_sentinel_inputs(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append one no-leakage sentinel snapshot row for every real fighter.

    The frozen dynamic and reservoir builders snapshot a date before applying
    observations from that date.  By placing all sentinel bouts after the full
    real history, their prefight values are exactly current post-history states.
    Power uses only real evidence with the augmented fight-order table, so no
    synthetic event can contribute evidence.
    """
    ordered = rfs.sort_values(["date", "fight_id", "fighter_id"])
    latest_real = ordered.groupby("fighter_id", as_index=False, sort=False).tail(1).copy()
    latest_real = latest_real.sort_values("fighter_id").reset_index(drop=True)

    if latest_real.empty:
        raise RuntimeError("cannot build latest physical state from empty RFS history")

    rows_for_pairing = latest_real.copy()
    rows_for_pairing["_source_fighter_id"] = rows_for_pairing["fighter_id"].astype(str)
    rows_for_pairing["_is_dummy"] = False

    if len(rows_for_pairing) % 2:
        dummy = rows_for_pairing.iloc[[0]].copy()
        dummy["fighter_id"] = _DUMMY_FIGHTER_ID
        if "fighter_name" in dummy.columns:
            dummy["fighter_name"] = "FSR V2 latest physical dummy"
        dummy["_is_dummy"] = True
        rows_for_pairing = pd.concat([rows_for_pairing, dummy], ignore_index=True)

    dates = [
        pd.Timestamp(rfs["date"].max()),
        pd.Timestamp(rounds[_date_column(rounds)].max()),
        pd.Timestamp(master["date"].max()),
    ]
    sentinel_date = max(dates) + pd.Timedelta(days=1)

    sentinel_rfs_rows: list[pd.DataFrame] = []
    sentinel_round_rows: list[pd.DataFrame] = []
    sentinel_master_rows: list[dict[str, object]] = []
    sentinel_map_rows: list[dict[str, object]] = []

    round_date = _date_column(rounds)

    for pair_index in range(0, len(rows_for_pairing), 2):
        pair = rows_for_pairing.iloc[pair_index : pair_index + 2].copy()
        sentinel_id = f"{_SENTINEL_PREFIX}{pair_index // 2:05d}"
        pair_ids = pair["fighter_id"].astype(str).tolist()

        for local_index, (_, source_row) in enumerate(pair.iterrows()):
            target_fighter_id = str(source_row["fighter_id"])
            source_fighter_id = str(source_row["_source_fighter_id"])
            source_fight_id = str(source_row["fight_id"])
            opponent_target_id = pair_ids[1 - local_index]

            rfs_row = source_row.drop(labels=["_source_fighter_id", "_is_dummy"]).to_frame().T
            rfs_row["fight_id"] = sentinel_id
            rfs_row["fighter_id"] = target_fighter_id
            rfs_row["date"] = sentinel_date
            if "event_date" in rfs_row.columns:
                rfs_row["event_date"] = sentinel_date
            if "opponent_id" in rfs_row.columns:
                rfs_row["opponent_id"] = opponent_target_id
            sentinel_rfs_rows.append(rfs_row)

            fighter_rounds = rounds.loc[
                rounds["fight_id"].eq(source_fight_id)
                & rounds["fighter_id"].eq(source_fighter_id)
            ].copy()
            if fighter_rounds.empty:
                raise RuntimeError(
                    "latest physical sentinel missing source round rows for "
                    f"fighter={source_fighter_id} fight={source_fight_id}"
                )
            fighter_rounds["fight_id"] = sentinel_id
            fighter_rounds["fighter_id"] = target_fighter_id
            fighter_rounds[round_date] = sentinel_date
            if "date" in fighter_rounds.columns:
                fighter_rounds["date"] = sentinel_date
            if "event_date" in fighter_rounds.columns:
                fighter_rounds["event_date"] = sentinel_date
            if "opponent_id" in fighter_rounds.columns:
                fighter_rounds["opponent_id"] = opponent_target_id
            sentinel_round_rows.append(fighter_rounds)

            if not bool(source_row["_is_dummy"]):
                sentinel_map_rows.append(
                    {
                        "fight_id": sentinel_id,
                        "fighter_id": target_fighter_id,
                        "fighter_name": str(source_row.get("fighter_name", target_fighter_id)),
                    }
                )

        sentinel_master_rows.append({"fight_id": sentinel_id, "date": sentinel_date})

    rfs_augmented = pd.concat(
        [rfs, *sentinel_rfs_rows],
        ignore_index=True,
        sort=False,
    )
    rounds_augmented = pd.concat(
        [rounds, *sentinel_round_rows],
        ignore_index=True,
        sort=False,
    )
    master_order = pd.concat(
        [master[["fight_id", "date"]], pd.DataFrame(sentinel_master_rows)],
        ignore_index=True,
        sort=False,
    )
    sentinel_map = pd.DataFrame(sentinel_map_rows)

    if sentinel_map.duplicated(["fighter_id"]).any():
        raise RuntimeError("latest physical sentinel map contains duplicate fighters")

    return rfs_augmented, rounds_augmented, master_order, sentinel_map


def _physical_latest_from_sentinels(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    rfs_aug, rounds_aug, master_order_frame, sentinel_map = _latest_sentinel_inputs(
        rfs,
        rounds,
        master,
    )

    sentinel_ids = set(sentinel_map["fight_id"].astype(str))

    dynamic_aug = dynamic.build_prefight_snapshots(rfs_aug, rounds_aug)
    dynamic_part = dynamic_aug.loc[
        dynamic_aug["fight_id"].astype(str).isin(sentinel_ids),
        [
            "fight_id",
            "fighter_id",
            "fatigue_accumulation_resistance",
            "fatigue_performance_resilience",
        ],
    ].rename(columns=_DYNAMIC_RENAMES)

    reservoir_aug = reservoir.build_prefight_snapshots(rfs_aug)
    reservoir_part = reservoir_aug.loc[
        reservoir_aug["fight_id"].astype(str).isin(sentinel_ids),
        ["fight_id", "fighter_id", "damage_durability", "knockdown_resistance"],
    ].copy()

    # Build the chronology with sentinel fights, but power evidence from REAL
    # master/round rows only.  Therefore all returned sentinel snapshots see all
    # prior real power evidence and zero synthetic evidence.
    fight_order = power32._fight_order_table(master_order_frame)
    evidence = power32._power_evidence_by_fighter_fight(master, rounds, fight_order)
    power_part = power32.build_prefight_striking_power(
        sentinel_map[["fight_id", "fighter_id"]],
        evidence,
        fight_order,
    )

    latest = sentinel_map.merge(
        dynamic_part,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )
    latest = latest.merge(
        reservoir_part,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )
    latest = latest.merge(
        power_part,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )
    latest["stamina_capacity"] = STAMINA_CAPACITY
    latest = latest.drop(columns=["fight_id"])

    return _validate_physical(latest).sort_values("fighter_id").reset_index(drop=True)


def _validate_physical(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PHYSICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"physical FSR frame missing columns: {missing}")

    numeric = frame[list(PHYSICAL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise RuntimeError(f"physical FSR contains missing/non-finite values: {bad}")

    if not np.allclose(numeric["stamina_capacity"].to_numpy(float), STAMINA_CAPACITY):
        raise RuntimeError("stamina_capacity must remain fixed at 100")

    ten_to_ninety = [
        "stamina_depletion_resistance",
        "stamina_performance_resilience",
        "damage_durability",
        "knockdown_resistance",
    ]
    if ((numeric[ten_to_ninety] < 10.0) | (numeric[ten_to_ninety] > 90.0)).any().any():
        raise RuntimeError("physical FSR contains out-of-range 10-90 ratings")
    if ((numeric["striking_power"] < 35.0) | (numeric["striking_power"] > 90.0)).any():
        raise RuntimeError("physical FSR striking_power is outside 35-90")

    return frame


def build_physical_snapshots(
    *,
    rfs: pd.DataFrame | None = None,
    rounds: pd.DataFrame | None = None,
    master: pd.DataFrame | None = None,
    rfs_path: Path = ROUND_FIGHTER_STATE_HISTORY_PATH,
    round_path: Path = ROUND_STATS_PATH,
    master_path: Path = MASTER_PATH,
) -> PhysicalSnapshots:
    """Build historical prefight and true current physical FSR snapshots."""
    if rfs is None:
        rfs = pd.read_parquet(rfs_path)
    if rounds is None:
        rounds = pd.read_parquet(round_path)
    if master is None:
        master = pd.read_parquet(master_path)

    rfs, rounds, master = _normalize_inputs(rfs, rounds, master)

    required = {"fight_id", "fighter_id", "fighter_name", *dynamic.C.values()}
    required |= set(reservoir.REQUIRED_COLUMNS)
    missing = sorted(required.difference(rfs.columns))
    if missing:
        raise FileNotFoundError(
            "exact FSR-32 physical replay prerequisites are unavailable: "
            f"{rfs_path} is missing {len(missing)} frozen dynamic/finish columns. "
            "Supply the original FSR-32 enriched RFS history; physical values "
            "must not be approximated. Missing columns: " + ", ".join(missing)
        )

    prefight = _physical_from_prefight_builders(rfs, rounds, master)
    latest = _physical_latest_from_sentinels(rfs, rounds, master)

    return PhysicalSnapshots(prefight=prefight, latest=latest)


def attach_physical_prefight(
    core_prefight: pd.DataFrame,
    physical_prefight: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]
    if core_prefight.duplicated(keys).any():
        raise ValueError("core prefight snapshot violates fighter-fight grain")
    if physical_prefight.duplicated(keys).any():
        raise ValueError("physical prefight snapshot violates fighter-fight grain")

    core_keys = set(map(tuple, core_prefight[keys].astype(str).to_numpy()))
    physical_keys = set(map(tuple, physical_prefight[keys].astype(str).to_numpy()))
    if core_keys != physical_keys:
        raise RuntimeError(
            "core/physical prefight key mismatch: "
            f"core={len(core_keys):,}, physical={len(physical_keys):,}, "
            f"missing_physical={len(core_keys - physical_keys):,}, "
            f"extra_physical={len(physical_keys - core_keys):,}"
        )

    return core_prefight.merge(
        physical_prefight[[*keys, *PHYSICAL_COLUMNS]],
        on=keys,
        how="left",
        validate="one_to_one",
    )


def attach_physical_latest(
    core_latest: pd.DataFrame,
    physical_latest: pd.DataFrame,
) -> pd.DataFrame:
    key = ["fighter_id"]
    if core_latest.duplicated(key).any():
        raise ValueError("core latest snapshot contains duplicate fighter_id")
    if physical_latest.duplicated(key).any():
        raise ValueError("physical latest snapshot contains duplicate fighter_id")

    core_ids = set(core_latest["fighter_id"].astype(str))
    physical_ids = set(physical_latest["fighter_id"].astype(str))
    if core_ids != physical_ids:
        raise RuntimeError(
            "core/physical latest fighter mismatch: "
            f"core={len(core_ids):,}, physical={len(physical_ids):,}, "
            f"missing_physical={len(core_ids - physical_ids):,}, "
            f"extra_physical={len(physical_ids - core_ids):,}"
        )

    return core_latest.merge(
        physical_latest[["fighter_id", *PHYSICAL_COLUMNS]],
        on="fighter_id",
        how="left",
        validate="one_to_one",
    )
