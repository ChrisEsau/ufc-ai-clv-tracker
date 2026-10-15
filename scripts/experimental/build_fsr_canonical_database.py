"""Build one canonical FSR database directly from source history.

This replaces the numbered wrapper chain (FSR-18 -> 21 -> 22 -> 25 -> 26 ->
28 -> 32) with one research-only orchestrator over the underlying trait-family
replay engines.

Canonical ontology
------------------
The canonical learned FSR contains 25 unique ratings. Compatibility aliases and
superseded traits are intentionally excluded:

* ``distance_precision`` / ``distance_defense`` are emitted only under the
  canonical names ``distance_striking_precision`` /
  ``distance_striking_defense``.
* legacy locked ``striking_power`` is replaced by the fresh hierarchical power
  model used by FSR-32.
* legacy ``chin_resistance`` / ``damage_resistance`` are replaced by
  ``knockdown_resistance`` / ``damage_durability``.
* ``recovery_ability`` is excluded (the later FSR-32 contract dropped the
  fighter-specific recovery proxy).
* ``stamina_depletion_resistance`` and ``stamina_performance_resilience`` are
  not separate learned traits because they are aliases of the two fatigue
  ratings.
* ``stamina_capacity`` is retained as a simulator parameter, not counted as a
  learned FSR rating.

The script can also recover an ACTUAL post-fight FSR state for a historical
fight, including a fighter's latest fight.  It does this without changing any
rating equations: the selected fight is replayed normally, then a synthetic
sentinel fight is inserted immediately after that event date.  Every family
snapshots its state before applying the sentinel, so the sentinel pre-fight
state is exactly the state after the selected real fight under the current
chronological replay contract.

Examples
--------
Target-only pre/post extraction (fastest useful mode for Ricci/Kline):

    PYTHONPATH=. python scripts/experimental/build_fsr_canonical_database.py \
      --target-postfight-id 52ddf20a10890b41 --target-only

Full canonical pre-fight database:

    PYTHONPATH=. python scripts/experimental/build_fsr_canonical_database.py

Shadow/research only.  No production artifacts are modified.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as power32
from scripts.experimental import fsr_clinch_striking_v1 as clinch
from scripts.experimental import fsr_distance_striking_pressure_v1 as distance
from scripts.experimental import fsr_dynamic_families_v1 as dynamic
from scripts.experimental import fsr_finish_reservoir_traits_v1 as reservoir
from scripts.experimental import fsr_ground_striking_v1 as ground
from scripts.experimental import fsr_reversal_v1 as reversal
from scripts.experimental.run_fsr_v1_5_2026_replay import build_locked_prefight_snapshots


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
ROUND_PATH = Path("data/fight_details/ufc_round_stats.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_canonical_shadow")
PREFIGHT_OUTPUT_PATH = OUTPUT_DIR / "fsr_canonical_prefight_snapshots.parquet"

STAMINA_CAPACITY = 100.0

LOCKED_CANONICAL = (
    "distance_striking_precision",
    "distance_striking_defense",
    "wrestling_entry",
    "wrestling_conversion",
    "td_defense",
    "control_imposition",
    "control_resistance",
    "submission_pressure",
    "submission_conversion",
    "submission_resistance",
)
DYNAMIC_CANONICAL = (
    "fatigue_accumulation_resistance",
    "fatigue_performance_resilience",
    "adversity_resistance",
    "adversity_recovery",
)
GROUND_CANONICAL = (
    "ground_striking_pressure",
    "ground_striking_precision",
    "ground_striking_defense",
)
REVERSAL_CANONICAL = ("reversal_ability",)
CLINCH_CANONICAL = (
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
)
DISTANCE_CANONICAL = ("distance_striking_pressure",)
RESERVOIR_CANONICAL = (
    "knockdown_resistance",
    "damage_durability",
)
POWER_CANONICAL = ("striking_power",)

CANONICAL_RATINGS = (
    *LOCKED_CANONICAL,
    *DYNAMIC_CANONICAL,
    *GROUND_CANONICAL,
    *REVERSAL_CANONICAL,
    *CLINCH_CANONICAL,
    *DISTANCE_CANONICAL,
    *RESERVOIR_CANONICAL,
    *POWER_CANONICAL,
)

if len(CANONICAL_RATINGS) != 25 or len(set(CANONICAL_RATINGS)) != 25:
    raise RuntimeError("Canonical FSR ontology must contain exactly 25 unique ratings")


def _elapsed(start: float) -> str:
    seconds = max(0.0, time.perf_counter() - start)
    return f"{seconds:.1f}s"


def _date_column(df: pd.DataFrame) -> str:
    if "date" in df.columns:
        return "date"
    if "event_date" in df.columns:
        return "event_date"
    raise RuntimeError("frame has neither date nor event_date")


def _normalize_inputs(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rfs = rfs.copy()
    rounds = rounds.copy()
    master = master.copy()

    rfs["fight_id"] = rfs["fight_id"].astype(str)
    rfs["fighter_id"] = rfs["fighter_id"].astype(str)
    rfs_date = _date_column(rfs)
    rfs[rfs_date] = pd.to_datetime(rfs[rfs_date], errors="raise")
    if "date" not in rfs.columns:
        rfs["date"] = rfs[rfs_date]

    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)
    rounds_date = _date_column(rounds)
    rounds[rounds_date] = pd.to_datetime(rounds[rounds_date], errors="raise")

    master["fight_id"] = master["fight_id"].astype(str)
    master_date = _date_column(master)
    master[master_date] = pd.to_datetime(master[master_date], errors="raise")
    if "date" not in master.columns:
        master["date"] = master[master_date]

    return rfs, rounds, master


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, frame[["fight_id", "fighter_id"]].astype(str).to_numpy()))


def _merge_family(
    base: pd.DataFrame,
    family: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]
    if family.duplicated(keys).any():
        raise RuntimeError(f"{label} snapshots violate fighter-fight grain")
    if _key_set(base) != _key_set(family):
        raise RuntimeError(
            f"canonical/{label} key mismatch: base={len(_key_set(base)):,}, "
            f"{label}={len(_key_set(family)):,}"
        )
    missing = [column for column in columns if column not in family.columns]
    if missing:
        raise RuntimeError(f"{label} missing canonical columns: {missing}")
    return base.merge(
        family[[*keys, *columns]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )


def build_canonical_prefight(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
    *,
    progress: bool = True,
) -> pd.DataFrame:
    """Replay all unique canonical families directly from source inputs."""
    start = time.perf_counter()
    if progress:
        print(
            f"[canonical FSR] start | rfs={len(rfs):,} round_rows={len(rounds):,} "
            f"master_fights={len(master):,}",
            flush=True,
        )

    if progress:
        print("[canonical FSR] 1/8 locked core (10 retained ratings)...", flush=True)
    locked = build_locked_prefight_snapshots(rfs)
    locked = locked.rename(
        columns={
            "distance_precision": "distance_striking_precision",
            "distance_defense": "distance_striking_defense",
        }
    )
    metadata = ["fight_id", "date", "fighter_id", "fighter_name", "prior_ufc_fights"]
    missing_meta = [c for c in metadata if c not in locked.columns]
    if missing_meta:
        raise RuntimeError(f"locked replay missing metadata: {missing_meta}")
    base = locked[[*metadata, *LOCKED_CANONICAL]].copy()

    if progress:
        print(f"[canonical FSR] 2/8 dynamic response | elapsed={_elapsed(start)}", flush=True)
    dynamic_snapshots = dynamic.build_prefight_snapshots(rfs, rounds)
    base = _merge_family(base, dynamic_snapshots, DYNAMIC_CANONICAL, "dynamic")

    if progress:
        print(f"[canonical FSR] 3/8 ground striking | elapsed={_elapsed(start)}", flush=True)
    ground_snapshots = ground.build_prefight_snapshots(rfs)
    base = _merge_family(base, ground_snapshots, GROUND_CANONICAL, "ground")

    if progress:
        print(f"[canonical FSR] 4/8 reversal | elapsed={_elapsed(start)}", flush=True)
    reversal_dependency = base[["fight_id", "fighter_id", "control_imposition"]].copy()
    reversal_snapshots = reversal.build_prefight_snapshots(rfs, reversal_dependency)
    base = _merge_family(base, reversal_snapshots, REVERSAL_CANONICAL, "reversal")

    if progress:
        print(f"[canonical FSR] 5/8 clinch striking | elapsed={_elapsed(start)}", flush=True)
    clinch_snapshots = clinch.build_prefight_snapshots(
        rfs,
        progress_every_dates=100 if progress else None,
    )
    base = _merge_family(base, clinch_snapshots, CLINCH_CANONICAL, "clinch")

    if progress:
        print(f"[canonical FSR] 6/8 distance pressure | elapsed={_elapsed(start)}", flush=True)
    distance_snapshots = distance.build_prefight_snapshots(rfs, progress=progress)
    base = _merge_family(base, distance_snapshots, DISTANCE_CANONICAL, "distance")

    if progress:
        print(f"[canonical FSR] 7/8 finish reservoir | elapsed={_elapsed(start)}", flush=True)
    reservoir_snapshots = reservoir.build_prefight_snapshots(rfs)
    base = _merge_family(base, reservoir_snapshots, RESERVOIR_CANONICAL, "reservoir")

    if progress:
        print(f"[canonical FSR] 8/8 fresh striking power | elapsed={_elapsed(start)}", flush=True)
    fight_order = power32._fight_order_table(master)
    evidence = power32._power_evidence_by_fighter_fight(master, rounds, fight_order)
    fresh_power = power32.build_prefight_striking_power(
        base[["fight_id", "fighter_id"]],
        evidence,
        fight_order,
    )
    base = _merge_family(base, fresh_power, POWER_CANONICAL, "fresh_power")

    base["stamina_capacity"] = float(STAMINA_CAPACITY)

    numeric = base[list(CANONICAL_RATINGS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise RuntimeError(f"canonical FSR contains missing/non-finite ratings: {bad}")

    # Fresh power is intentionally 35-90; all other learned ratings are 10-90.
    non_power = [c for c in CANONICAL_RATINGS if c != "striking_power"]
    if ((numeric[non_power] < 10.0) | (numeric[non_power] > 90.0)).any().any():
        raise RuntimeError("canonical FSR contains out-of-range 10-90 ratings")
    if ((numeric["striking_power"] < 35.0) | (numeric["striking_power"] > 90.0)).any():
        raise RuntimeError("canonical fresh striking_power is outside 35-90")

    out = base.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)
    if progress:
        print(
            f"[canonical FSR] complete | rows={len(out):,} | learned_ratings=25 | "
            f"elapsed={_elapsed(start)}",
            flush=True,
        )
    return out


def _truncate_and_add_target_sentinel(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
    target_fight_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, pd.Timestamp]:
    """Keep history through target date and append one post-target sentinel bout."""
    target_fight_id = str(target_fight_id)
    target_rows = rfs.loc[rfs["fight_id"].eq(target_fight_id)].copy()
    if len(target_rows) != 2:
        raise RuntimeError(
            f"target {target_fight_id} must have exactly two RFS fighter rows; "
            f"found {len(target_rows)}"
        )
    target_date = pd.Timestamp(target_rows["date"].iloc[0])
    if not (pd.to_datetime(target_rows["date"]) == target_date).all():
        raise RuntimeError("target RFS fighter rows disagree on date")

    sentinel_id = f"__postfight__{target_fight_id}"
    sentinel_date = target_date + pd.Timedelta(nanoseconds=1)

    rfs_work = rfs.loc[pd.to_datetime(rfs["date"]) <= target_date].copy()
    rfs_sentinel = target_rows.copy()
    rfs_sentinel["fight_id"] = sentinel_id
    rfs_sentinel["date"] = sentinel_date
    if "event_date" in rfs_sentinel.columns:
        rfs_sentinel["event_date"] = sentinel_date
    rfs_work = pd.concat([rfs_work, rfs_sentinel], ignore_index=True)

    round_date_col = _date_column(rounds)
    rounds_work = rounds.loc[pd.to_datetime(rounds[round_date_col]) <= target_date].copy()
    target_rounds = rounds.loc[rounds["fight_id"].eq(target_fight_id)].copy()
    if target_rounds.empty:
        raise RuntimeError(f"target {target_fight_id} has no authoritative round rows")
    rounds_sentinel = target_rounds.copy()
    rounds_sentinel["fight_id"] = sentinel_id
    rounds_sentinel[round_date_col] = sentinel_date
    if "date" in rounds_sentinel.columns:
        rounds_sentinel["date"] = sentinel_date
    if "event_date" in rounds_sentinel.columns:
        rounds_sentinel["event_date"] = sentinel_date
    rounds_work = pd.concat([rounds_work, rounds_sentinel], ignore_index=True)

    master_date_col = _date_column(master)
    master_work = master.loc[pd.to_datetime(master[master_date_col]) <= target_date].copy()
    target_master = master.loc[master["fight_id"].eq(target_fight_id)].copy()
    if len(target_master) != 1:
        raise RuntimeError(
            f"target {target_fight_id} must have exactly one master row; found {len(target_master)}"
        )
    master_sentinel = target_master.copy()
    master_sentinel["fight_id"] = sentinel_id
    master_sentinel[master_date_col] = sentinel_date
    if "date" in master_sentinel.columns:
        master_sentinel["date"] = sentinel_date
    master_work = pd.concat([master_work, master_sentinel], ignore_index=True)

    return rfs_work, rounds_work, master_work, sentinel_id, target_date


def extract_target_pre_post(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
    target_fight_id: str,
    *,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide target states and a trait-by-trait comparison table."""
    work_rfs, work_rounds, work_master, sentinel_id, target_date = (
        _truncate_and_add_target_sentinel(rfs, rounds, master, target_fight_id)
    )
    if progress:
        print(
            f"[canonical FSR] target postfight extraction | fight={target_fight_id} | "
            f"date={target_date.date()} | sentinel={sentinel_id}",
            flush=True,
        )

    replay = build_canonical_prefight(
        work_rfs,
        work_rounds,
        work_master,
        progress=progress,
    )

    pre = replay.loc[replay["fight_id"].eq(str(target_fight_id))].copy()
    post = replay.loc[replay["fight_id"].eq(sentinel_id)].copy()
    if len(pre) != 2 or len(post) != 2:
        raise RuntimeError(
            f"expected two pre and two post rows; got pre={len(pre)}, post={len(post)}"
        )

    pre = pre.set_index("fighter_id", drop=False)
    post = post.set_index("fighter_id", drop=False)
    if set(pre.index) != set(post.index):
        raise RuntimeError("target pre/post fighter IDs do not align")

    wide_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for fighter_id in pre.index:
        pre_row = pre.loc[fighter_id]
        post_row = post.loc[fighter_id]
        fighter_name = str(pre_row["fighter_name"])

        wide: dict[str, object] = {
            "fight_id": str(target_fight_id),
            "date": target_date,
            "fighter_id": str(fighter_id),
            "fighter_name": fighter_name,
            "prior_ufc_fights_pre": int(pre_row["prior_ufc_fights"]),
            "prior_ufc_fights_post": int(post_row["prior_ufc_fights"]),
            "stamina_capacity": float(post_row["stamina_capacity"]),
        }
        for trait in CANONICAL_RATINGS:
            pre_value = float(pre_row[trait])
            post_value = float(post_row[trait])
            wide[f"{trait}_pre"] = pre_value
            wide[f"{trait}_post"] = post_value
            wide[f"{trait}_delta"] = post_value - pre_value
            comparison_rows.append(
                {
                    "fight_id": str(target_fight_id),
                    "date": target_date,
                    "fighter_id": str(fighter_id),
                    "fighter_name": fighter_name,
                    "trait": trait,
                    "pre_fsr": pre_value,
                    "post_fsr": post_value,
                    "delta": post_value - pre_value,
                }
            )
        wide_rows.append(wide)

    return pd.DataFrame(wide_rows), pd.DataFrame(comparison_rows)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (RFS_PATH, ROUND_PATH, MASTER_PATH):
        if not path.exists():
            raise RuntimeError(f"required canonical FSR input not found: {path}")

    print(f"[canonical FSR] loading RFS: {RFS_PATH}", flush=True)
    rfs = pd.read_parquet(RFS_PATH)
    print(f"[canonical FSR] loading rounds: {ROUND_PATH}", flush=True)
    rounds = pd.read_parquet(ROUND_PATH)
    print(f"[canonical FSR] loading master: {MASTER_PATH}", flush=True)
    master = pd.read_parquet(MASTER_PATH)
    return _normalize_inputs(rfs, rounds, master)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-postfight-id",
        help="Historical fight ID whose actual post-fight canonical FSR state should be extracted.",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Skip writing the full canonical pre-fight parquet; useful for targeted diagnostics.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress family progress messages where supported.",
    )
    args = parser.parse_args()

    if args.target_only and not args.target_postfight_id:
        raise RuntimeError("--target-only requires --target-postfight-id")

    rfs, rounds, master = _load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = not args.quiet

    print("\nCANONICAL FSR ONTOLOGY (25 UNIQUE LEARNED RATINGS)", flush=True)
    for index, trait in enumerate(CANONICAL_RATINGS, start=1):
        print(f"  {index:02d}. {trait}", flush=True)
    print("  simulator parameter: stamina_capacity = 100.0 (not a learned rating)\n", flush=True)

    if not args.target_only:
        database = build_canonical_prefight(rfs, rounds, master, progress=progress)
        database.to_parquet(PREFIGHT_OUTPUT_PATH, index=False)
        print(
            f"[canonical FSR] wrote {len(database):,} pre-fight rows -> {PREFIGHT_OUTPUT_PATH}",
            flush=True,
        )

    if args.target_postfight_id:
        wide, comparison = extract_target_pre_post(
            rfs,
            rounds,
            master,
            str(args.target_postfight_id),
            progress=progress,
        )
        stem = f"fsr_canonical_{args.target_postfight_id}_pre_post"
        wide_path = OUTPUT_DIR / f"{stem}_wide.csv"
        comparison_path = OUTPUT_DIR / f"{stem}_traits.csv"
        wide.to_csv(wide_path, index=False)
        comparison.to_csv(comparison_path, index=False)

        print("\nACTUAL TARGET PRE -> POST FSR", flush=True)
        display = comparison.pivot(
            index="trait",
            columns="fighter_name",
            values=["pre_fsr", "post_fsr", "delta"],
        )
        print(display.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)
        print(f"\n[canonical FSR] wrote: {wide_path}", flush=True)
        print(f"[canonical FSR] wrote: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
