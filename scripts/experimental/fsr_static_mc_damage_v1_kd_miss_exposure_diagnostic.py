"""Diagnose actual-KD misses using historical vs simulated striking exposure.

This research-only audit focuses on the actual-KD bouts that the current MC rated
below the configured KD-probability threshold (30% by default).

For each missed bout it:
- reloads the leakage-safe pre-fight FSR pair;
- reruns the current locked Damage Reservoir V1 / KD=80 engine;
- captures simulated significant strikes landed/attempted;
- computes simulated significant-strike rates per minute;
- joins actual UFCStats significant strikes for the same bout;
- normalizes actual totals by actual elapsed fight time from the master fight file;
- keeps the existing power-vs-KD-resistance matchup diagnostics beside exposure.

Why rates rather than raw totals?
Historical KD fights may end early while the no-KO shadow simulator always runs
its scheduled duration. Comparing raw totals would therefore confound exposure
with fight length. Per-minute rates make the comparison materially cleaner.

No simulator constants or architecture are changed by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.fight_time import repair_elapsed_match_time
from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from scripts.experimental import fsr_static_mc_damage_v1 as damage


VALIDATION_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_kd_actual_vs_mc.parquet"
)
MISS_DIAGNOSTIC_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_kd_miss_diagnostic.parquet"
)
FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_kd_miss_exposure_diagnostic.parquet"
)
DEFAULT_THRESHOLD = 0.30
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810


def _load_missed_bouts(validation_path: Path, threshold: float) -> pd.DataFrame:
    frame = pd.read_parquet(validation_path)
    required = {
        "bout_id",
        "actual_any_kd",
        "actual_total_kd",
        "mc_p_any_kd",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Validation artifact missing columns: {missing}")

    frame = frame.copy()
    frame["bout_id"] = frame["bout_id"].astype(str)
    missed = frame[
        (frame["actual_any_kd"] == 1)
        & (frame["mc_p_any_kd"] < threshold)
    ].copy()
    if missed.empty:
        raise ValueError("No actual-KD missed bouts found at the requested threshold.")
    return missed


def _load_fsr_pairs(path: Path, bout_ids: set[str]) -> dict[str, tuple[pd.Series, pd.Series]]:
    frame = pd.read_parquet(path)
    bout_key = "fight_id" if "fight_id" in frame.columns else "bout_id"
    required = {bout_key, "fighter_id"} | set(damage.base.REQUIRED_COLUMNS) | damage.REQUIRED_DAMAGE_COLUMNS
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FSR artifact missing required columns: {missing}")

    work = frame.copy()
    work[bout_key] = work[bout_key].astype(str)
    work["fighter_id"] = work["fighter_id"].astype(str)
    work = work[work[bout_key].isin(bout_ids)].copy()

    pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    for key, group in work.groupby(bout_key, sort=False):
        group = group.reset_index(drop=True)
        if len(group) == 2 and group["fighter_id"].nunique() == 2:
            pairs[str(key)] = (group.iloc[0], group.iloc[1])
    return pairs


def _actual_exposure(round_stats_path: Path, master_path: Path, bout_ids: set[str]) -> pd.DataFrame:
    rounds = pd.read_parquet(round_stats_path)
    required_round = {"fight_id", "sig_str_landed", "sig_str_attempted"}
    missing = sorted(required_round - set(rounds.columns))
    if missing:
        raise ValueError(f"Round stats missing exposure columns: {missing}")

    rounds = rounds.copy()
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds = rounds[rounds["fight_id"].isin(bout_ids)].copy()
    rounds["sig_str_landed"] = pd.to_numeric(rounds["sig_str_landed"], errors="coerce").fillna(0.0)
    rounds["sig_str_attempted"] = pd.to_numeric(rounds["sig_str_attempted"], errors="coerce").fillna(0.0)

    actual = (
        rounds.groupby("fight_id", as_index=False)
        .agg(
            actual_sig_landed=("sig_str_landed", "sum"),
            actual_sig_attempted=("sig_str_attempted", "sum"),
        )
        .rename(columns={"fight_id": "bout_id"})
    )

    master = pd.read_parquet(master_path, columns=["fight_id", "finish_round", "match_time_sec"])
    master = master.copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master = master[master["fight_id"].isin(bout_ids)].copy()
    master["finish_round"] = pd.to_numeric(master["finish_round"], errors="coerce")
    master["match_time_sec"] = pd.to_numeric(master["match_time_sec"], errors="coerce")
    master = repair_elapsed_match_time(master)
    master = master.drop_duplicates("fight_id")
    master = master.rename(columns={"fight_id": "bout_id", "match_time_sec": "actual_elapsed_sec"})

    actual = actual.merge(
        master[["bout_id", "actual_elapsed_sec"]],
        on="bout_id",
        how="left",
        validate="one_to_one",
    )
    if actual["actual_elapsed_sec"].isna().any():
        raise ValueError("Missing elapsed fight time for one or more missed KD bouts.")

    minutes = actual["actual_elapsed_sec"].clip(lower=1.0) / 60.0
    actual["actual_sig_landed_per_min"] = actual["actual_sig_landed"] / minutes
    actual["actual_sig_attempted_per_min"] = actual["actual_sig_attempted"] / minutes
    return actual


def _rounds_for_bout(row: pd.Series) -> int:
    value = row.get("rounds")
    try:
        rounds = int(round(float(value)))
    except (TypeError, ValueError):
        rounds = 3
    return rounds if rounds in (3, 5) else 3


def _simulate_exposure(
    missed: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(missed) * paths_per_bout
    counter = 0

    for _, bout in missed.iterrows():
        bout_id = str(bout["bout_id"])
        if bout_id not in pairs:
            raise ValueError(f"Missing leakage-safe FSR pair for bout {bout_id}")
        red, blue = pairs[bout_id]
        rounds = _rounds_for_bout(bout)
        scheduled_minutes = rounds * 5.0

        path_rows: list[dict[str, float]] = []
        for _path in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = damage.StaticFSRMCDamageV1(red, blue, rounds=rounds, seed=path_seed)
            sim.run()

            sig_landed = float(sim.stats[0].sig_landed + sim.stats[1].sig_landed)
            sig_attempted = float(sim.stats[0].sig_attempted + sim.stats[1].sig_attempted)
            total_kd = float(sim.stats[0].knockdowns_scored + sim.stats[1].knockdowns_scored)
            path_rows.append(
                {
                    "sig_landed": sig_landed,
                    "sig_attempted": sig_attempted,
                    "total_kd": total_kd,
                    "mean_res_depletion": 1.0
                    - np.mean([
                        sim.damage_state[0].reservoir_fraction,
                        sim.damage_state[1].reservoir_fraction,
                    ]),
                }
            )

            counter += 1
            if counter % 500 == 0 or counter == total_paths:
                print(
                    f"[KD miss exposure] paths {counter:,}/{total_paths:,}; "
                    f"bouts_started={len(rows)+1:,}/{len(missed):,}",
                    flush=True,
                )

        pf = pd.DataFrame(path_rows)
        rows.append(
            {
                "bout_id": bout_id,
                "sim_paths": paths_per_bout,
                "sim_sig_landed": pf["sig_landed"].mean(),
                "sim_sig_attempted": pf["sig_attempted"].mean(),
                "sim_sig_landed_per_min": pf["sig_landed"].mean() / scheduled_minutes,
                "sim_sig_attempted_per_min": pf["sig_attempted"].mean() / scheduled_minutes,
                "sim_p_any_kd_rerun": float((pf["total_kd"] > 0).mean()),
                "sim_expected_total_kd_rerun": pf["total_kd"].mean(),
                "sim_mean_reservoir_depletion_rerun": pf["mean_res_depletion"].mean(),
            }
        )

    return pd.DataFrame(rows)


def _attach_existing_danger_features(frame: pd.DataFrame, diagnostic_path: Path) -> pd.DataFrame:
    if not diagnostic_path.exists():
        return frame
    diagnostic = pd.read_parquet(diagnostic_path).copy()
    diagnostic["bout_id"] = diagnostic["bout_id"].astype(str)
    keep = [
        c
        for c in [
            "bout_id",
            "max_power_minus_opp_kd_res",
            "mean_power_minus_opp_kd_res",
            "max_striking_power",
            "min_kd_resistance",
            "mean_damage_durability",
            "max_distance_pressure",
            "max_distance_precision",
            "min_distance_defense",
        ]
        if c in diagnostic.columns
    ]
    return frame.merge(diagnostic[keep], on="bout_id", how="left", validate="one_to_one")


def _print_summary(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["landed_rate_ratio_actual_to_mc"] = (
        frame["actual_sig_landed_per_min"] / frame["sim_sig_landed_per_min"].replace(0.0, np.nan)
    )
    frame["attempt_rate_ratio_actual_to_mc"] = (
        frame["actual_sig_attempted_per_min"] / frame["sim_sig_attempted_per_min"].replace(0.0, np.nan)
    )
    frame["landed_rate_gap_actual_minus_mc"] = (
        frame["actual_sig_landed_per_min"] - frame["sim_sig_landed_per_min"]
    )

    print("\n" + "=" * 120)
    print("ACTUAL-KD MISSES — HISTORICAL VS MC STRIKING EXPOSURE")
    print("=" * 120)
    print(f"missed actual-KD bouts: {len(frame):,}")
    print(f"rerun MC paths: {int(frame['sim_paths'].sum()):,}")
    print(f"KD shock coefficient: {damage.KD_SHOCK_COEFFICIENT:g}")

    print("\nAGGREGATE EXPOSURE")
    for actual_col, sim_col, label in [
        ("actual_sig_landed_per_min", "sim_sig_landed_per_min", "sig landed/min"),
        ("actual_sig_attempted_per_min", "sim_sig_attempted_per_min", "sig attempted/min"),
    ]:
        actual_mean = frame[actual_col].mean()
        sim_mean = frame[sim_col].mean()
        ratio = actual_mean / sim_mean if sim_mean > 0 else np.nan
        print(f"{label}: actual={actual_mean:.3f}; MC={sim_mean:.3f}; actual/MC={ratio:.3f}x")

    print("\nEXPOSURE-GAP DISTRIBUTION")
    print(
        "actual landed rate >= 1.25x MC: "
        f"{int((frame['landed_rate_ratio_actual_to_mc'] >= 1.25).sum())} / {len(frame)}"
    )
    print(
        "actual landed rate within 0.80-1.25x MC: "
        f"{int(frame['landed_rate_ratio_actual_to_mc'].between(0.80, 1.25, inclusive='both').sum())} / {len(frame)}"
    )
    print(
        "actual landed rate <= 0.80x MC: "
        f"{int((frame['landed_rate_ratio_actual_to_mc'] <= 0.80).sum())} / {len(frame)}"
    )

    if "max_power_minus_opp_kd_res" in frame.columns:
        corr = frame[["landed_rate_gap_actual_minus_mc", "max_power_minus_opp_kd_res"]].corr().iloc[0, 1]
        print(f"\ncorrelation(exposure gap, max power-resistance edge): {corr:.4f}")

    display = [
        c
        for c in [
            "bout_id", "event_date", "red_name", "blue_name", "mc_p_any_kd", "actual_total_kd",
            "actual_sig_landed_per_min", "sim_sig_landed_per_min", "landed_rate_ratio_actual_to_mc",
            "actual_sig_attempted_per_min", "sim_sig_attempted_per_min",
            "max_power_minus_opp_kd_res", "max_striking_power", "min_kd_resistance",
            "mean_damage_durability", "sim_mean_reservoir_depletion_rerun",
        ]
        if c in frame.columns
    ]

    print("\nMISSES WITH LARGEST ACTUAL > MC LANDED-STRIKE RATE GAP")
    print(
        frame.sort_values("landed_rate_ratio_actual_to_mc", ascending=False)
        .head(20)[display]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nMISSES WHERE MC EXPOSURE WAS ALREADY AS HIGH OR HIGHER THAN ACTUAL")
    print(
        frame.sort_values("landed_rate_ratio_actual_to_mc", ascending=True)
        .head(20)[display]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nINTERPRETATION BOUNDARY")
    print("- High actual/MC exposure ratios support an opportunity/exposure miss.")
    print("- Ratios near 1.0 with weak danger traits support a trait/severity miss.")
    print("- Ratios near 1.0 with strong danger traits point back toward stochastic KD conversion/calibration.")
    print("- These are diagnostics, not automatic causal labels; no constants are changed here.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare historical vs MC significant-strike exposure for actual-KD misses"
    )
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--miss-diagnostic", type=Path, default=MISS_DIAGNOSTIC_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--round-stats", type=Path, default=Path(ROUND_STATS_PATH))
    parser.add_argument("--master", type=Path, default=Path(MASTER_PATH))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    missed = _load_missed_bouts(args.validation, args.threshold)
    bout_ids = set(missed["bout_id"])
    print(
        f"[KD miss exposure] actual-KD misses={len(missed):,}; "
        f"paths_per_bout={args.paths_per_bout}; total_paths={len(missed)*args.paths_per_bout:,}",
        flush=True,
    )

    pairs = _load_fsr_pairs(args.fsr_path, bout_ids)
    if len(pairs) != len(missed):
        missing = sorted(bout_ids - set(pairs))
        raise ValueError(f"Missing valid FSR pairs for {len(missing)} missed bouts: {missing[:10]}")

    actual = _actual_exposure(args.round_stats, args.master, bout_ids)
    simulated = _simulate_exposure(
        missed,
        pairs,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )

    merged = missed.merge(actual, on="bout_id", how="left", validate="one_to_one")
    merged = merged.merge(simulated, on="bout_id", how="left", validate="one_to_one")
    merged = _attach_existing_danger_features(merged, args.miss_diagnostic)
    merged["landed_rate_ratio_actual_to_mc"] = (
        merged["actual_sig_landed_per_min"]
        / merged["sim_sig_landed_per_min"].replace(0.0, np.nan)
    )
    merged["attempt_rate_ratio_actual_to_mc"] = (
        merged["actual_sig_attempted_per_min"]
        / merged["sim_sig_attempted_per_min"].replace(0.0, np.nan)
    )
    merged["landed_rate_gap_actual_minus_mc"] = (
        merged["actual_sig_landed_per_min"] - merged["sim_sig_landed_per_min"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    _print_summary(merged)
    print(f"\n[KD miss exposure] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
