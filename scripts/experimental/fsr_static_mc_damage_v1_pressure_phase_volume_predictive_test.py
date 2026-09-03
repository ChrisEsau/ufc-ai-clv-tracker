"""Validate FSR phase-pressure traits against subsequent phase-specific strike volume.

Research-only diagnostic. No simulator constants or architecture are changed.

Goal
----
Before changing strike-attempt generation, test whether the leakage-safe FSR
pressure traits have the right predictive relationship with subsequent UFCStats
phase-specific significant-strike attempts, and whether the current shadow MC
reproduces that relationship.

The audit is fighter-level across the exact same 300 historical bouts used by
the current KD/exposure validation cohort (normally 600 fighter-bout rows).
For each fighter it compares:

- distance_striking_pressure -> distance significant-strike attempts/min;
- clinch_striking_pressure   -> clinch significant-strike attempts/min;
- ground_striking_pressure   -> ground significant-strike attempts/min.

Historical phase-specific attempt rates use total actual elapsed fight time as
the denominator because UFCStats does not provide exact phase residence time.
The MC is time-matched to the same historical elapsed window and is instrumented
inside this script to record phase-specific attempts. The simulator module itself
is not modified.

Interpretation
--------------
If historical Q1->Q4 separation is materially stronger than MC separation, the
pressure-to-attempt mapping may be compressed. If the relationships are similar,
pressure mapping should be left alone and other mechanics investigated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_predictive_value as prior
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_time_matched as tm


VALIDATION_PATH = prior.VALIDATION_PATH
FSR_PATH = prior.FSR_PATH
ROUND_STATS_PATH = Path(prior.ROUND_STATS_PATH)
MASTER_PATH = Path(prior.MASTER_PATH)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_pressure_phase_volume_predictive_test.parquet"
)
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810

PHASES = ("DISTANCE", "CLINCH", "GROUND")
PRESSURE_COLUMNS = {
    "DISTANCE": "distance_striking_pressure",
    "CLINCH": "clinch_striking_pressure",
    "GROUND": "ground_striking_pressure",
}
ACTUAL_ATTEMPT_COLUMNS = {
    "DISTANCE": "distance_attempted",
    "CLINCH": "clinch_attempted",
    "GROUND": "ground_attempted",
}


class PhaseAttemptTrackingSim(damage.StaticFSRMCDamageV1):
    """Damage V1 simulator with research-only phase-attempt counters."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.phase_attempts = [
            {phase: 0 for phase in PHASES},
            {phase: 0 for phase in PHASES},
        ]

    def _generate_strikes_for_fighter(
        self,
        fighter: int,
        phase: str,
        *,
        rate_multiplier: float = 1.0,
    ) -> str | None:
        before = self.stats[fighter].sig_att
        note = super()._generate_strikes_for_fighter(
            fighter,
            phase,
            rate_multiplier=rate_multiplier,
        )
        generated = int(self.stats[fighter].sig_att - before)
        phase_key = str(phase).upper()
        if phase_key in self.phase_attempts[fighter]:
            self.phase_attempts[fighter][phase_key] += generated
        return note


def _simulate_to_elapsed(
    red: pd.Series,
    blue: pd.Series,
    *,
    elapsed_sec: float,
    rounds: int,
    seed: int,
) -> PhaseAttemptTrackingSim:
    """Run the unchanged segment mechanics through the historical time window."""
    max_seconds = int(rounds) * 5 * 60
    elapsed_sec = float(np.clip(elapsed_sec, 0.0, max_seconds))

    sim = PhaseAttemptTrackingSim(red, blue, rounds=rounds, seed=seed)
    if elapsed_sec <= 0.0:
        return sim

    full_segments = int(elapsed_sec // damage.base.SEGMENT_SECONDS)
    remainder = elapsed_sec - full_segments * damage.base.SEGMENT_SECONDS

    current_round = 0
    for segment_index in range(full_segments):
        round_no = segment_index // damage.base.SEGMENTS_PER_ROUND + 1
        if round_no != current_round:
            tm._reset_round(sim)
            current_round = round_no
        tm._run_full_segment(sim)

    if remainder > 1e-9:
        round_no = full_segments // damage.base.SEGMENTS_PER_ROUND + 1
        if round_no != current_round:
            tm._reset_round(sim)
        tm._run_partial_striking_segment(
            sim,
            fraction=remainder / damage.base.SEGMENT_SECONDS,
        )

    return sim


def _load_fsr_rows(path: Path, bout_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    bout_key = "fight_id" if "fight_id" in frame.columns else "bout_id"
    required = {bout_key, "fighter_id"} | set(PRESSURE_COLUMNS.values())
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FSR artifact missing pressure-test columns: {missing}")

    frame[bout_key] = frame[bout_key].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame = frame[frame[bout_key].isin(bout_ids)].copy()
    frame = frame.rename(columns={bout_key: "bout_id"})

    if frame.duplicated(["bout_id", "fighter_id"]).any():
        raise ValueError("FSR cohort has duplicate bout/fighter rows.")

    counts = frame.groupby("bout_id")["fighter_id"].nunique()
    bad = counts[counts != 2]
    if not bad.empty:
        raise ValueError(f"Expected two leakage-safe FSR fighters per bout; bad bouts={len(bad)}")

    keep = ["bout_id", "fighter_id", *PRESSURE_COLUMNS.values()]
    for optional in ("fighter_name", "name"):
        if optional in frame.columns:
            keep.append(optional)
    return frame[keep].copy()


def _load_actual_fighter_phase_volume(
    round_stats_path: Path,
    master_path: Path,
    bout_ids: set[str],
) -> pd.DataFrame:
    rounds = pd.read_parquet(round_stats_path).copy()
    required = {
        "fight_id",
        "fighter_id",
        "fighter_name",
        "distance_attempted",
        "clinch_attempted",
        "ground_attempted",
    }
    missing = sorted(required - set(rounds.columns))
    if missing:
        raise ValueError(f"Round stats missing phase-attempt columns: {missing}")

    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)
    rounds = rounds[rounds["fight_id"].isin(bout_ids)].copy()

    for col in ACTUAL_ATTEMPT_COLUMNS.values():
        rounds[col] = pd.to_numeric(rounds[col], errors="coerce").fillna(0.0)

    actual = (
        rounds.groupby(["fight_id", "fighter_id"], as_index=False)
        .agg(
            fighter_name=("fighter_name", "first"),
            distance_attempted=("distance_attempted", "sum"),
            clinch_attempted=("clinch_attempted", "sum"),
            ground_attempted=("ground_attempted", "sum"),
        )
        .rename(columns={"fight_id": "bout_id"})
    )

    # Reuse the already-audited elapsed-time loader so historical and simulated
    # denominators remain identical to the corrected exposure test.
    elapsed = prior._actual_exposure(round_stats_path, master_path, bout_ids)[
        ["bout_id", "actual_elapsed_sec"]
    ].drop_duplicates("bout_id")
    actual = actual.merge(elapsed, on="bout_id", how="left", validate="many_to_one")
    if actual["actual_elapsed_sec"].isna().any():
        raise ValueError("Missing historical elapsed time in fighter phase-volume rows.")

    minutes = actual["actual_elapsed_sec"].clip(lower=1.0) / 60.0
    for phase, col in ACTUAL_ATTEMPT_COLUMNS.items():
        actual[f"actual_{phase.lower()}_attempts_per_min"] = actual[col] / minutes

    return actual


def _simulate_fighter_phase_volume(
    validation: pd.DataFrame,
    fsr_rows: pd.DataFrame,
    elapsed_by_bout: pd.DataFrame,
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grouped = {
        str(bout_id): group.reset_index(drop=True)
        for bout_id, group in fsr_rows.groupby("bout_id", sort=False)
    }
    elapsed_map = elapsed_by_bout.set_index("bout_id")["actual_elapsed_sec"].to_dict()

    accum: dict[tuple[str, str], dict[str, float]] = {}
    total_paths = len(validation) * paths_per_bout
    counter = 0

    for bout_number, (_, bout) in enumerate(validation.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        group = grouped.get(bout_id)
        if group is None or len(group) != 2:
            raise ValueError(f"Missing two FSR rows for bout {bout_id}")

        red = group.iloc[0]
        blue = group.iloc[1]
        fighter_ids = [str(red["fighter_id"]), str(blue["fighter_id"])]
        elapsed_sec = float(elapsed_map[bout_id])
        elapsed_min = max(elapsed_sec, 1.0) / 60.0
        rounds = prior._rounds_for_bout(bout)

        for fighter_id in fighter_ids:
            accum[(bout_id, fighter_id)] = {
                "paths": 0.0,
                "elapsed_min": elapsed_min,
                **{f"{phase.lower()}_attempts": 0.0 for phase in PHASES},
            }

        for _ in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = _simulate_to_elapsed(
                red,
                blue,
                elapsed_sec=elapsed_sec,
                rounds=rounds,
                seed=path_seed,
            )

            for i, fighter_id in enumerate(fighter_ids):
                row = accum[(bout_id, fighter_id)]
                row["paths"] += 1.0
                for phase in PHASES:
                    row[f"{phase.lower()}_attempts"] += float(sim.phase_attempts[i][phase])

            counter += 1
            if counter % 1000 == 0 or counter == total_paths:
                print(
                    f"[pressure phase-volume] paths {counter:,}/{total_paths:,}; "
                    f"bouts_started={bout_number:,}/{len(validation):,}",
                    flush=True,
                )

    rows: list[dict[str, object]] = []
    for (bout_id, fighter_id), values in accum.items():
        paths = max(values["paths"], 1.0)
        elapsed_min = float(values["elapsed_min"])
        out: dict[str, object] = {
            "bout_id": bout_id,
            "fighter_id": fighter_id,
            "sim_paths": int(paths),
        }
        for phase in PHASES:
            mean_attempts = values[f"{phase.lower()}_attempts"] / paths
            out[f"sim_{phase.lower()}_attempts"] = mean_attempts
            out[f"sim_{phase.lower()}_attempts_per_min"] = mean_attempts / elapsed_min
        rows.append(out)

    return pd.DataFrame(rows)


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else float("nan")


def _linear_slope_per_10_pressure(x: pd.Series, y: pd.Series) -> float:
    work = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(work) < 3 or work["x"].nunique() < 2:
        return float("nan")
    slope = np.polyfit(work["x"].to_numpy(), work["y"].to_numpy(), 1)[0]
    return float(slope * 10.0)


def _phase_summary(frame: pd.DataFrame, phase: str) -> tuple[dict[str, float], pd.DataFrame]:
    key = phase.lower()
    pressure_col = PRESSURE_COLUMNS[phase]
    actual_col = f"actual_{key}_attempts_per_min"
    sim_col = f"sim_{key}_attempts_per_min"

    work = frame[[pressure_col, actual_col, sim_col]].dropna().copy()
    actual_spear = float(work[pressure_col].corr(work[actual_col], method="spearman"))
    sim_spear = float(work[pressure_col].corr(work[sim_col], method="spearman"))
    actual_pear = float(work[pressure_col].corr(work[actual_col], method="pearson"))
    sim_pear = float(work[pressure_col].corr(work[sim_col], method="pearson"))
    matchup_spear = float(work[actual_col].corr(work[sim_col], method="spearman"))

    work["pressure_quartile"] = pd.qcut(
        work[pressure_col],
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    quart = (
        work.groupby("pressure_quartile", observed=True, as_index=False)
        .agg(
            fighters=(pressure_col, "size"),
            pressure_mean=(pressure_col, "mean"),
            actual_attempts_per_min=(actual_col, "mean"),
            sim_attempts_per_min=(sim_col, "mean"),
        )
    )
    quart.insert(0, "phase", phase)

    q1 = quart.iloc[0] if len(quart) else None
    q4 = quart.iloc[-1] if len(quart) else None
    actual_q4_q1 = _safe_ratio(float(q4["actual_attempts_per_min"]), float(q1["actual_attempts_per_min"])) if q1 is not None else float("nan")
    sim_q4_q1 = _safe_ratio(float(q4["sim_attempts_per_min"]), float(q1["sim_attempts_per_min"])) if q1 is not None else float("nan")

    metrics = {
        "actual_mean": float(work[actual_col].mean()),
        "sim_mean": float(work[sim_col].mean()),
        "actual_pressure_spearman": actual_spear,
        "sim_pressure_spearman": sim_spear,
        "actual_pressure_pearson": actual_pear,
        "sim_pressure_pearson": sim_pear,
        "actual_slope_per_10_pressure": _linear_slope_per_10_pressure(work[pressure_col], work[actual_col]),
        "sim_slope_per_10_pressure": _linear_slope_per_10_pressure(work[pressure_col], work[sim_col]),
        "actual_q4_q1": actual_q4_q1,
        "sim_q4_q1": sim_q4_q1,
        "actual_vs_sim_matchup_spearman": matchup_spear,
    }
    return metrics, quart


def _print_summary(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 118)
    print("FSR PRESSURE -> SUBSEQUENT PHASE-SPECIFIC STRIKE VOLUME")
    print("=" * 118)
    print(f"fighter-bout rows: {len(frame):,}; bouts: {frame['bout_id'].nunique():,}")
    print(f"time-matched MC paths represented: {int(frame['sim_paths'].sum() / 2):,}")
    print("NOTE: historical phase attempts/min use total fight time; exact historical phase residence time is unavailable.")

    all_quarts: list[pd.DataFrame] = []
    for phase in PHASES:
        metrics, quart = _phase_summary(frame, phase)
        all_quarts.append(quart)
        print(f"\n{phase}")
        print("-" * 118)
        print(
            f"attempts/min actual={metrics['actual_mean']:.4f}; MC={metrics['sim_mean']:.4f}; "
            f"MC-actual={metrics['sim_mean']-metrics['actual_mean']:+.4f}"
        )
        print(
            f"pressure -> actual: Spearman={metrics['actual_pressure_spearman']:.4f}; "
            f"Pearson={metrics['actual_pressure_pearson']:.4f}; "
            f"slope/10 rating={metrics['actual_slope_per_10_pressure']:+.4f} att/min"
        )
        print(
            f"pressure -> MC:     Spearman={metrics['sim_pressure_spearman']:.4f}; "
            f"Pearson={metrics['sim_pressure_pearson']:.4f}; "
            f"slope/10 rating={metrics['sim_slope_per_10_pressure']:+.4f} att/min"
        )
        print(
            f"Q4/Q1 volume ratio: actual={metrics['actual_q4_q1']:.4f}x; "
            f"MC={metrics['sim_q4_q1']:.4f}x"
        )
        print(
            f"actual-vs-MC fighter matchup volume Spearman={metrics['actual_vs_sim_matchup_spearman']:.4f}"
        )
        print("\nPRESSURE QUARTILES")
        print(quart.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDECISION RULE")
    print("- Historical Q1->Q4 separation materially stronger than MC => pressure mapping may be compressed.")
    print("- Similar historical and MC separation => leave pressure mapping alone.")
    print("- Weak historical pressure relationship => do not strengthen the mapping merely to fix aggregate pace.")
    print("- This audit changes no simulator constants; any candidate change must be shadow-tested against moneyline and props regression gates.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test FSR pressure traits against historical and simulated phase strike volume"
    )
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--round-stats", type=Path, default=ROUND_STATS_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    validation = prior._load_validation(args.validation)
    bout_ids = set(validation["bout_id"].astype(str))
    print(f"[pressure phase-volume] loading {len(validation):,} historical bouts", flush=True)

    fsr_rows = _load_fsr_rows(args.fsr_path, bout_ids)
    actual = _load_actual_fighter_phase_volume(args.round_stats, args.master, bout_ids)
    elapsed = actual[["bout_id", "actual_elapsed_sec"]].drop_duplicates("bout_id")

    base = fsr_rows.merge(
        actual,
        on=["bout_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_fsr", "_actual"),
    )
    expected_rows = len(validation) * 2
    if len(base) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} fighter-bout rows after FSR/UFCStats join; got {len(base)}"
        )

    sim = _simulate_fighter_phase_volume(
        validation,
        fsr_rows,
        elapsed,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )
    out = base.merge(sim, on=["bout_id", "fighter_id"], how="left", validate="one_to_one")
    if out.filter(regex=r"^sim_.*attempts_per_min$").isna().any().any():
        raise ValueError("Missing simulated phase-volume values after join.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    _print_summary(out)
    print(f"\n[pressure phase-volume] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
