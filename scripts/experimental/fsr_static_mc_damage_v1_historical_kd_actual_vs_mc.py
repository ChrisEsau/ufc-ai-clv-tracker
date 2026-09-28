"""Compare historical UFCStats knockdowns against the locked KD=80 MC audit.

This script consumes the exact 300-bout sample produced by
``fsr_static_mc_damage_v1_historical_300_kd_audit.py`` and joins those bout IDs
back to the raw UFCStats round-level data. No matchups are resampled here.

For each historical bout it compares:
- actual UFCStats total knockdowns;
- actual indicator for at least one knockdown;
- MC probability of at least one knockdown;
- MC expected total knockdowns.

It then reports aggregate calibration, KD-count distribution, Brier score, MAE,
and calibration buckets. KO/TKO logic is not used or modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH


MC_AUDIT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_300_kd_audit.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_kd_actual_vs_mc.parquet"
)


def _load_mc_bouts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"MC historical KD audit not found: {path}. Run the 300-bout audit first."
        )

    paths = pd.read_parquet(path)
    required = {"bout_id", "any_knockdown", "total_knockdowns", "path_index"}
    missing = sorted(required - set(paths.columns))
    if missing:
        raise ValueError(f"MC audit missing required columns: {missing}")

    paths = paths.copy()
    paths["bout_id"] = paths["bout_id"].astype(str)

    agg_spec: dict[str, tuple[str, str]] = {
        "mc_paths": ("path_index", "size"),
        "mc_p_any_kd": ("any_knockdown", "mean"),
        "mc_expected_total_kd": ("total_knockdowns", "mean"),
    }
    for col in ("event_date", "rounds", "red_name", "blue_name"):
        if col in paths.columns:
            agg_spec[col] = (col, "first")

    return paths.groupby("bout_id", as_index=False).agg(**agg_spec)


def _load_actual_bouts(path: Path, selected_ids: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Round stats input not found: {path}. Historical actual KD validation "
            "requires UFCStats round-level data."
        )

    round_stats = pd.read_parquet(path)
    required = {"fight_id", "kd"}
    missing = sorted(required - set(round_stats.columns))
    if missing:
        raise ValueError(f"Round stats missing required columns: {missing}")

    work = round_stats.copy()
    work["fight_id"] = work["fight_id"].astype(str)
    work = work[work["fight_id"].isin(selected_ids)].copy()
    work["kd"] = pd.to_numeric(work["kd"], errors="coerce").fillna(0.0)

    if work.empty:
        raise ValueError(
            "None of the 300 sampled MC bout IDs matched raw round-stats fight_id. "
            "Inspect identifier lineage before proceeding; do not substitute fuzzy matches."
        )

    actual = (
        work.groupby("fight_id", as_index=False)
        .agg(actual_total_kd=("kd", "sum"))
        .rename(columns={"fight_id": "bout_id"})
    )
    actual["actual_total_kd"] = actual["actual_total_kd"].round().astype(int)
    actual["actual_any_kd"] = (actual["actual_total_kd"] > 0).astype(int)
    return actual


def _calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    # Fixed-width probability bins preserve interpretability across reruns.
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{int(100*bins[i])}-{int(100*bins[i+1])}%" for i in range(10)]
    work = frame.copy()
    work["mc_probability_bucket"] = pd.cut(
        work["mc_p_any_kd"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    table = (
        work.groupby("mc_probability_bucket", observed=True, as_index=False)
        .agg(
            bouts=("bout_id", "size"),
            mean_mc_probability=("mc_p_any_kd", "mean"),
            actual_kd_rate=("actual_any_kd", "mean"),
            expected_total_kd=("mc_expected_total_kd", "mean"),
            actual_total_kd=("actual_total_kd", "mean"),
        )
    )
    table["calibration_error"] = (
        table["mean_mc_probability"] - table["actual_kd_rate"]
    )
    return table


def _print_summary(frame: pd.DataFrame) -> None:
    n = len(frame)
    actual_any = int(frame["actual_any_kd"].sum())
    expected_any = float(frame["mc_p_any_kd"].sum())
    actual_total_kd = int(frame["actual_total_kd"].sum())
    expected_total_kd = float(frame["mc_expected_total_kd"].sum())

    y = frame["actual_any_kd"].astype(float)
    p = frame["mc_p_any_kd"].astype(float)
    brier = float(np.mean((p - y) ** 2))
    abs_error = float(abs(expected_any - actual_any))
    aggregate_rate_error = float(p.mean() - y.mean())
    total_kd_mae = float(
        np.mean(np.abs(frame["mc_expected_total_kd"] - frame["actual_total_kd"]))
    )

    print("\n" + "=" * 120)
    print("HISTORICAL ACTUAL VS MC — DAMAGE RESERVOIR / KD=80")
    print("=" * 120)
    print(f"matched historical bouts: {n:,}")

    print("\nANY-KD AGGREGATE")
    print(f"actual bouts with >=1 KD:   {actual_any:,} / {n} ({y.mean():.2%})")
    print(f"MC expected bouts >=1 KD:   {expected_any:.1f} / {n} ({p.mean():.2%})")
    print(f"aggregate rate error MC-actual: {aggregate_rate_error:+.2%}")
    print(f"absolute expected-count error: {abs_error:.1f} bouts")
    print(f"Brier score: {brier:.6f}")

    print("\nTOTAL KD AGGREGATE")
    print(f"actual total knockdowns:    {actual_total_kd:,}")
    print(f"MC expected total KD:        {expected_total_kd:.1f}")
    print(f"actual KD per bout:          {frame['actual_total_kd'].mean():.4f}")
    print(f"MC expected KD per bout:     {frame['mc_expected_total_kd'].mean():.4f}")
    print(f"bout-level expected-KD MAE:  {total_kd_mae:.4f}")

    print("\nACTUAL KD COUNT DISTRIBUTION")
    actual_dist = (
        frame["actual_total_kd"]
        .clip(upper=3)
        .replace({3: "3+"})
        .value_counts()
        .rename_axis("actual_KD")
        .reset_index(name="bouts")
    )
    actual_dist["share"] = actual_dist["bouts"] / n
    print(actual_dist.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nMC PROBABILITY CALIBRATION")
    calibration = _calibration_table(frame)
    print(calibration.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDISCRIMINATION CHECK")
    # Compare the MC probabilities assigned to actual KD and no-KD bouts.
    for actual_value, label in ((0, "actual no-KD"), (1, "actual KD")):
        g = frame[frame["actual_any_kd"] == actual_value]
        print(
            f"{label}: n={len(g):,}; mean MC p(KD)={g['mc_p_any_kd'].mean():.4f}; "
            f"median={g['mc_p_any_kd'].median():.4f}"
        )

    print("\nRESEARCH BOUNDARY")
    print("- Same 300 historical matchups as the prior MC audit; no resampling.")
    print("- Actual KD labels come from UFCStats round-level kd totals by fight_id.")
    print("- Locked KD=80 reservoir mechanics are being validated, not re-fit here.")
    print("- KO/TKO mechanics remain disabled and out of scope.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare actual historical UFCStats KDs with the 300-bout KD=80 MC audit"
    )
    parser.add_argument("--mc-audit", type=Path, default=MC_AUDIT_PATH)
    parser.add_argument("--round-stats", type=Path, default=Path(ROUND_STATS_PATH))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[actual-vs-MC KD] loading MC audit: {args.mc_audit}", flush=True)
    mc = _load_mc_bouts(args.mc_audit)
    print(f"[actual-vs-MC KD] sampled bouts in MC artifact: {len(mc):,}", flush=True)

    print(f"[actual-vs-MC KD] loading raw round stats: {args.round_stats}", flush=True)
    actual = _load_actual_bouts(args.round_stats, set(mc["bout_id"]))
    print(f"[actual-vs-MC KD] matched actual bout IDs: {len(actual):,}", flush=True)

    merged = mc.merge(actual, on="bout_id", how="left", validate="one_to_one")
    missing = merged["actual_any_kd"].isna()
    if missing.any():
        missing_ids = merged.loc[missing, "bout_id"].astype(str).tolist()
        raise ValueError(
            f"Actual UFCStats KD labels missing for {len(missing_ids)} sampled bouts. "
            f"First missing IDs: {missing_ids[:10]}. Do not calibrate on a partial join."
        )

    merged["actual_any_kd"] = merged["actual_any_kd"].astype(int)
    merged["actual_total_kd"] = merged["actual_total_kd"].astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    _print_summary(merged)
    print(f"\n[actual-vs-MC KD] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
