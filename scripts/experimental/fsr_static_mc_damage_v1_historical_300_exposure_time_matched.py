"""Time-match MC striking exposure to each historical bout's actual elapsed time.

This research-only audit corrects a confound in the prior 300-bout exposure
study. Historical striking rates were normalized by actual elapsed fight time,
while the shadow MC always ran the full scheduled duration. Early finishes could
therefore appear artificially high-pace relative to the MC.

For the exact same 300 historical matchups, this script:
- reloads actual UFCStats exposure and elapsed fight time;
- reruns Damage Reservoir V1 / KD=80 for 100 paths per bout by default;
- stops each simulated path at the same elapsed fight time as history;
- handles the final partial 10-second segment by scaling only strike opportunity
  within that segment to the remaining fraction of 10 seconds;
- compares actual vs MC significant strikes landed/attempted per minute;
- reports matchup correlations, MAE, aggregate bias, exposure quartiles, and
  high-exposure ROC-AUC.

No simulator constants or architecture are changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_predictive_value as prior


VALIDATION_PATH = prior.VALIDATION_PATH
FSR_PATH = prior.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_300_exposure_time_matched.parquet"
)
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810


def _reset_round(sim: damage.StaticFSRMCDamageV1) -> None:
    sim.phase = "DISTANCE"
    sim.ground_controller = None
    sim.clinch_controller = None
    sim.clinch_initiator = None


def _run_full_segment(sim: damage.StaticFSRMCDamageV1) -> None:
    """Advance one unchanged 10-second simulator segment."""
    phase_start = sim.phase
    for stats in sim.stats:
        stats.phase_segments[phase_start] += 1

    sim._generate_striking(phase_start)
    if phase_start == "DISTANCE":
        sim._distance_transition()
    elif phase_start == "CLINCH":
        sim._clinch_transition()
    else:
        sim._ground_transition()


def _run_partial_striking_segment(
    sim: damage.StaticFSRMCDamageV1,
    *,
    fraction: float,
) -> None:
    """Generate only the final partial segment's strikes.

    No phase transition is needed because the historical comparison window ends
    immediately after this partial segment. Strike Poisson intensity is scaled
    by the remaining fraction of a 10-second segment.
    """
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return

    sim._advance_damage_timers()
    phase = sim.phase

    if phase == "GROUND" and sim.ground_controller is not None:
        controller = sim.ground_controller
        bottom = sim._other(controller)
        sim._generate_strikes_for_fighter(
            controller,
            "GROUND",
            rate_multiplier=fraction,
        )
        sim._generate_strikes_for_fighter(
            bottom,
            "GROUND",
            rate_multiplier=damage.base.BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER * fraction,
        )
        return

    for fighter in (0, 1):
        sim._generate_strikes_for_fighter(
            fighter,
            phase,
            rate_multiplier=fraction,
        )


def _simulate_to_elapsed(
    red: pd.Series,
    blue: pd.Series,
    *,
    elapsed_sec: float,
    rounds: int,
    seed: int,
) -> damage.StaticFSRMCDamageV1:
    """Run one path only through the historical elapsed-time window."""
    max_seconds = int(rounds) * 5 * 60
    elapsed_sec = float(np.clip(elapsed_sec, 0.0, max_seconds))

    sim = damage.StaticFSRMCDamageV1(red, blue, rounds=rounds, seed=seed)
    if elapsed_sec <= 0.0:
        return sim

    full_segments = int(elapsed_sec // damage.base.SEGMENT_SECONDS)
    remainder = elapsed_sec - full_segments * damage.base.SEGMENT_SECONDS

    current_round = 0
    for segment_index in range(full_segments):
        round_no = segment_index // damage.base.SEGMENTS_PER_ROUND + 1
        if round_no != current_round:
            _reset_round(sim)
            current_round = round_no
        _run_full_segment(sim)

    if remainder > 1e-9:
        round_no = full_segments // damage.base.SEGMENTS_PER_ROUND + 1
        if round_no != current_round:
            _reset_round(sim)
        _run_partial_striking_segment(
            sim,
            fraction=remainder / damage.base.SEGMENT_SECONDS,
        )

    return sim


def _simulate_exposure_time_matched(
    validation_with_actual: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(validation_with_actual) * paths_per_bout
    counter = 0

    for bout_number, (_, bout) in enumerate(validation_with_actual.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        if bout_id not in pairs:
            raise ValueError(f"Missing leakage-safe FSR pair for bout {bout_id}")

        red, blue = pairs[bout_id]
        rounds = prior._rounds_for_bout(bout)
        elapsed_sec = float(bout["actual_elapsed_sec"])
        elapsed_minutes = max(elapsed_sec, 1.0) / 60.0

        landed: list[float] = []
        attempted: list[float] = []
        kds: list[float] = []

        for _ in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = _simulate_to_elapsed(
                red,
                blue,
                elapsed_sec=elapsed_sec,
                rounds=rounds,
                seed=path_seed,
            )

            landed.append(float(sim.stats[0].sig_landed + sim.stats[1].sig_landed))
            attempted.append(float(sim.stats[0].sig_att + sim.stats[1].sig_att))
            kds.append(float(sim.stats[0].knockdowns_scored + sim.stats[1].knockdowns_scored))

            counter += 1
            if counter % 1000 == 0 or counter == total_paths:
                print(
                    f"[time-matched exposure] paths {counter:,}/{total_paths:,}; "
                    f"bouts_started={bout_number:,}/{len(validation_with_actual):,}",
                    flush=True,
                )

        rows.append(
            {
                "bout_id": bout_id,
                "sim_paths_time_matched": paths_per_bout,
                "sim_elapsed_sec": elapsed_sec,
                "sim_sig_landed_time_matched": float(np.mean(landed)),
                "sim_sig_attempted_time_matched": float(np.mean(attempted)),
                "sim_sig_landed_per_min_time_matched": float(np.mean(landed) / elapsed_minutes),
                "sim_sig_attempted_per_min_time_matched": float(np.mean(attempted) / elapsed_minutes),
                "sim_p_any_kd_time_matched": float(np.mean(np.asarray(kds) > 0)),
                "sim_expected_total_kd_time_matched": float(np.mean(kds)),
            }
        )

    return pd.DataFrame(rows)


def _print_summary(frame: pd.DataFrame) -> None:
    a_land = frame["actual_sig_landed_per_min"].astype(float)
    s_land = frame["sim_sig_landed_per_min_time_matched"].astype(float)
    a_att = frame["actual_sig_attempted_per_min"].astype(float)
    s_att = frame["sim_sig_attempted_per_min_time_matched"].astype(float)

    pearson_land = float(a_land.corr(s_land, method="pearson"))
    spearman_land = float(a_land.corr(s_land, method="spearman"))
    pearson_att = float(a_att.corr(s_att, method="pearson"))
    spearman_att = float(a_att.corr(s_att, method="spearman"))

    actual_high_cut = float(a_land.quantile(0.75))
    high_actual = (a_land >= actual_high_cut).astype(int)
    high_auc = float(roc_auc_score(high_actual, s_land))

    print("\n" + "=" * 120)
    print("HISTORICAL 300-BOUT TIME-MATCHED STRIKING EXPOSURE")
    print("=" * 120)
    print(f"historical bouts: {len(frame):,}")
    print(f"rerun MC paths: {int(frame['sim_paths_time_matched'].sum()):,}")
    print(f"KD shock coefficient unchanged: {damage.KD_SHOCK_COEFFICIENT:g}")

    print("\nAGGREGATE EXPOSURE — SAME ELAPSED-TIME WINDOW")
    print(
        f"sig landed/min: actual={a_land.mean():.4f}; MC={s_land.mean():.4f}; "
        f"MC-actual={s_land.mean()-a_land.mean():+.4f}; "
        f"actual/MC={a_land.mean()/s_land.mean():.4f}x"
    )
    print(
        f"sig attempted/min: actual={a_att.mean():.4f}; MC={s_att.mean():.4f}; "
        f"MC-actual={s_att.mean()-a_att.mean():+.4f}; "
        f"actual/MC={a_att.mean()/s_att.mean():.4f}x"
    )

    print("\nMATCHUP-LEVEL EXPOSURE PREDICTIVE VALUE")
    print(f"landed/min Pearson r:   {pearson_land:.6f}")
    print(f"landed/min Spearman r:  {spearman_land:.6f}")
    print(f"attempt/min Pearson r:  {pearson_att:.6f}")
    print(f"attempt/min Spearman r: {spearman_att:.6f}")
    print(f"landed/min MAE:         {mean_absolute_error(a_land, s_land):.6f}")
    print(f"attempt/min MAE:        {mean_absolute_error(a_att, s_att):.6f}")
    print(
        f"ROC-AUC for historical top-quartile landed exposure "
        f"(cut >= {actual_high_cut:.4f}/min): {high_auc:.6f}"
    )

    work = frame.copy()
    work["mc_exposure_quartile"] = pd.qcut(
        work["sim_sig_landed_per_min_time_matched"],
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    quart = (
        work.groupby("mc_exposure_quartile", observed=True, as_index=False)
        .agg(
            bouts=("bout_id", "size"),
            mc_landed_per_min=("sim_sig_landed_per_min_time_matched", "mean"),
            actual_landed_per_min=("actual_sig_landed_per_min", "mean"),
            actual_attempted_per_min=("actual_sig_attempted_per_min", "mean"),
            actual_kd_rate=("actual_any_kd", "mean"),
        )
    )
    print("\nACTUAL EXPOSURE BY MC-PREDICTED EXPOSURE QUARTILE")
    print(quart.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    frame = frame.copy()
    frame["landed_gap_actual_minus_mc"] = a_land - s_land
    frame["landed_ratio_actual_to_mc"] = a_land / s_land.replace(0.0, np.nan)

    display = [
        c
        for c in [
            "bout_id", "event_date", "red_name", "blue_name", "actual_elapsed_sec",
            "actual_any_kd", "actual_total_kd", "actual_sig_landed_per_min",
            "sim_sig_landed_per_min_time_matched", "landed_ratio_actual_to_mc",
            "actual_sig_attempted_per_min", "sim_sig_attempted_per_min_time_matched",
            "mean_distance_pressure", "mean_clinch_pressure", "wrestling_tendency", "weight_class",
        ]
        if c in frame.columns
    ]

    print("\n20 LARGEST UNDERPREDICTED EXPOSURE MATCHUPS")
    print(
        frame.sort_values("landed_gap_actual_minus_mc", ascending=False)
        .head(20)[display]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n20 LARGEST OVERPREDICTED EXPOSURE MATCHUPS")
    print(
        frame.sort_values("landed_gap_actual_minus_mc", ascending=True)
        .head(20)[display]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nINTERPRETATION")
    print("- Both actual and MC exposure now use the same historical elapsed-time window.")
    print("- Early finishes no longer get compared against a full 15/25-minute MC path.")
    print("- Correlation/ranking now tests matchup pace generation rather than finish-time mismatch.")
    print("- No simulator constants were changed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run time-matched 300-bout historical striking-exposure audit"
    )
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--round-stats", type=Path, default=Path(prior.ROUND_STATS_PATH))
    parser.add_argument("--master", type=Path, default=Path(prior.MASTER_PATH))
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    validation = prior._load_validation(args.validation)
    bout_ids = set(validation["bout_id"].astype(str))

    print(f"[time-matched exposure] loading {len(validation):,} historical bouts", flush=True)
    pairs, style = prior._load_fsr_pairs(args.fsr_path, bout_ids)
    actual = prior._actual_exposure(args.round_stats, args.master, bout_ids)

    base = validation.merge(actual, on="bout_id", how="left", validate="one_to_one")
    base = base.merge(style, on="bout_id", how="left", validate="one_to_one")

    if base["actual_elapsed_sec"].isna().any():
        raise ValueError("Missing actual elapsed time after historical exposure join.")
    if len(pairs) != len(validation):
        missing = sorted(bout_ids - set(pairs))
        raise ValueError(
            f"Only {len(pairs)} / {len(validation)} FSR pairs resolved. "
            f"First missing IDs: {missing[:10]}"
        )

    print(
        f"[time-matched exposure] paths_per_bout={args.paths_per_bout}; "
        f"total_paths={len(base) * args.paths_per_bout:,}",
        flush=True,
    )
    simulated = _simulate_exposure_time_matched(
        base,
        pairs,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )

    merged = base.merge(simulated, on="bout_id", how="left", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    _print_summary(merged)
    print(f"\n[time-matched exposure] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
