"""Evaluate KO-related FSR signal on the modern, mature-fighter cohort.

This study intentionally targets the use case that matters for forward UFC
prediction rather than the older 300-bout research sample.

Cohort contract
---------------
- UFC bouts dated 2020-01-01 or later.
- Both fighters had at least 3 *prior* UFC fights before the bout.
- Leakage-safe pre-fight FSR snapshots only.
- Full eligible historical cohort, not the previous 300-bout sample.

The fixed learner and feature bundles are reused from
``fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv.py`` so the only substantive
change is the cohort. This is a diagnostic study; it changes no FSR values or
simulator constants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_kd_audit as hist
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv as joint


MASTER_PATH = Path("data/master/ufc_master.parquet")
FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature.parquet"
)
START_DATE = pd.Timestamp("2020-01-01")
MIN_PRIOR_UFC_FIGHTS = 3


def _resolve_date_column(frame: pd.DataFrame) -> str:
    for candidate in ("event_date", "date", "fight_date"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Could not resolve fight date column in ufc_master.parquet; "
        "expected one of event_date/date/fight_date."
    )


def _load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"UFC master artifact not found: {path}")

    frame = pd.read_parquet(path).copy()
    required = {"fight_id", "r_id", "b_id", "method", "finish_round"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"UFC master missing required columns: {missing}")

    date_col = _resolve_date_column(frame)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["r_id"] = frame["r_id"].astype(str)
    frame["b_id"] = frame["b_id"].astype(str)

    # Canonical master should already be one row per fight; dedupe defensively so
    # prior-fight counts cannot be inflated by accidental duplicate rows.
    frame = (
        frame.sort_values([date_col, "fight_id"])
        .drop_duplicates(subset=["fight_id"], keep="last")
        .reset_index(drop=True)
    )
    frame = frame.rename(columns={date_col: "event_date"})
    return frame


def _attach_prior_ufc_fight_counts(master: pd.DataFrame) -> pd.DataFrame:
    """Attach strict pre-date UFC appearance counts for both fighters."""
    red = master[["fight_id", "event_date", "r_id"]].rename(columns={"r_id": "fighter_id"})
    blue = master[["fight_id", "event_date", "b_id"]].rename(columns={"b_id": "fighter_id"})
    appearances = pd.concat([red, blue], ignore_index=True)

    # Count appearances by fighter/date, then cumulative appearances on dates
    # strictly earlier than the current fight date. This avoids same-date leakage.
    daily = (
        appearances.groupby(["fighter_id", "event_date"], as_index=False)
        .size()
        .rename(columns={"size": "fights_on_date"})
        .sort_values(["fighter_id", "event_date"])
    )
    daily["prior_ufc_fights"] = (
        daily.groupby("fighter_id")["fights_on_date"].cumsum() - daily["fights_on_date"]
    )

    prior_map = daily.set_index(["fighter_id", "event_date"])["prior_ufc_fights"]
    out = master.copy()
    out["r_prior_ufc_fights"] = [
        int(prior_map.loc[(fighter_id, event_date)])
        for fighter_id, event_date in zip(out["r_id"], out["event_date"])
    ]
    out["b_prior_ufc_fights"] = [
        int(prior_map.loc[(fighter_id, event_date)])
        for fighter_id, event_date in zip(out["b_id"], out["event_date"])
    ]
    return out


def _is_ko_tko(method: object) -> int:
    if pd.isna(method):
        return 0
    text = str(method).strip().lower()
    # UFCStats method labels include KO/TKO and TKO - Doctor's Stoppage.
    return int("ko" in text or "tko" in text)


def _build_outcome_cohort(master: pd.DataFrame) -> pd.DataFrame:
    frame = _attach_prior_ufc_fight_counts(master)
    frame = frame[
        frame["event_date"].ge(START_DATE)
        & frame["r_prior_ufc_fights"].ge(MIN_PRIOR_UFC_FIGHTS)
        & frame["b_prior_ufc_fights"].ge(MIN_PRIOR_UFC_FIGHTS)
    ].copy()

    frame["actual_ko_tko"] = frame["method"].map(_is_ko_tko).astype(int)
    frame["actual_finish_round"] = pd.to_numeric(frame["finish_round"], errors="coerce")
    frame["actual_r1_ko"] = (
        frame["actual_ko_tko"].eq(1) & frame["actual_finish_round"].eq(1)
    ).astype(int)

    return frame.rename(columns={"fight_id": "bout_id"})[
        [
            "bout_id",
            "event_date",
            "r_id",
            "b_id",
            "r_prior_ufc_fights",
            "b_prior_ufc_fights",
            "actual_ko_tko",
            "actual_finish_round",
            "actual_r1_ko",
        ]
    ]


def _load_fsr_pairs_for_cohort(
    fsr_path: Path,
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.Series, pd.Series]]]:
    if not fsr_path.exists():
        raise FileNotFoundError(f"FSR artifact not found: {fsr_path}")

    fsr = pd.read_parquet(fsr_path)
    bout_key = hist._resolve_bout_key(fsr, None)
    fsr[bout_key] = fsr[bout_key].astype(str)

    requested = set(cohort["bout_id"].astype(str))
    available = set(fsr.loc[fsr[bout_key].isin(requested), bout_key].astype(str))
    matched_ids = requested & available

    cohort = cohort[cohort["bout_id"].astype(str).isin(matched_ids)].copy()
    bouts, _ = hist._prepare_historical_bouts(
        fsr[fsr[bout_key].isin(matched_ids)].copy(),
        bout_key=bout_key,
    )
    pairs = {str(bout_id): (red, blue) for bout_id, red, blue in bouts}
    cohort = cohort[cohort["bout_id"].astype(str).isin(pairs)].copy()

    if cohort.empty:
        raise ValueError("No 2020+ mature-fighter bouts matched the leakage-safe FSR artifact.")
    return cohort, pairs


def _build_joint_frame(
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
) -> pd.DataFrame:
    # Reuse the validated symmetric feature construction. mc_p_ko_tko is a
    # placeholder because this full cohort was not run through the 300-bout MC.
    validation_like = cohort[
        ["bout_id", "actual_ko_tko", "actual_r1_ko"]
    ].copy()
    validation_like["mc_p_ko_tko"] = np.nan

    frame = joint._build_features(validation_like, pairs)
    metadata = cohort[
        [
            "bout_id",
            "event_date",
            "r_prior_ufc_fights",
            "b_prior_ufc_fights",
            "actual_finish_round",
        ]
    ].copy()
    frame = frame.merge(metadata, on="bout_id", how="left", validate="one_to_one")
    return frame


def _power_default_summary(
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
) -> None:
    rows: list[dict[str, object]] = []
    for _, bout in cohort.iterrows():
        red, blue = pairs[str(bout["bout_id"])]
        rows.append(
            {
                "year": int(pd.Timestamp(bout["event_date"]).year),
                "red_power": joint._numeric(red, "striking_power"),
                "blue_power": joint._numeric(blue, "striking_power"),
            }
        )

    side = pd.DataFrame(rows)
    yearly_rows: list[dict[str, object]] = []
    for year, g in side.groupby("year"):
        values = pd.concat([g["red_power"], g["blue_power"]], ignore_index=True)
        yearly_rows.append(
            {
                "year": int(year),
                "bouts": len(g),
                "fighter_sides": len(values),
                "power_exact_50": int(values.eq(50.0).sum()),
                "power_exact_50_rate": float(values.eq(50.0).mean()),
                "power_mean": float(values.mean()),
                "power_std": float(values.std(ddof=1)),
            }
        )

    print("\nSTRIKING-POWER FSR MATURITY BY YEAR")
    print(
        pd.DataFrame(yearly_rows).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )


def _print_results(results: pd.DataFrame, frame: pd.DataFrame) -> None:
    print("\n" + "=" * 128)
    print("2020+ MATURE-FIGHTER JOINT KO-RELEVANT FSR SIGNAL")
    print("=" * 128)
    print(f"bouts: {len(frame):,}")
    print(f"date range: {frame['event_date'].min().date()} -> {frame['event_date'].max().date()}")
    print(
        f"minimum prior UFC fights: both fighters >= {MIN_PRIOR_UFC_FIGHTS} "
        "before the bout"
    )
    print(f"actual KO/TKO: {int(frame['actual_ko_tko'].sum()):,} ({frame['actual_ko_tko'].mean():.2%})")
    print(f"actual R1 KO/TKO: {int(frame['actual_r1_ko'].sum()):,} ({frame['actual_r1_ko'].mean():.2%})")
    print("fixed learner: standardized regularized logistic regression; no hyperparameter tuning")
    print("all reported probabilities are repeated out-of-fold predictions")

    display = [
        "target", "bundle", "features", "positive_bouts", "prevalence",
        "oof_auc", "oof_average_precision", "oof_brier", "fold_auc_mean", "fold_auc_std",
    ]
    print("\nCROSS-VALIDATED SIGNAL BY FEATURE BUNDLE")
    print(results[display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDECISION GUIDE")
    print("- This is the primary cohort for modern forward-looking evaluation: 2020+ and both fighters with >=3 prior UFC fights.")
    print("- A/B measure whether power/KD-resistance/durability are enough by themselves.")
    print("- C measures distance-striking matchup signal.")
    print("- D combines all finish matchup edges.")
    print("- E adds raw-trait context and simple interactions; if it degrades, the extra dimensions are mostly noise at this sample size.")
    print("- No simulator constants or FSR values are changed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate KO-related FSR signal on 2020+ bouts with both fighters >=3 prior UFC fights"
    )
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--splits", type=int, default=joint.DEFAULT_SPLITS)
    parser.add_argument("--repeats", type=int, default=joint.DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=joint.DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    master = _load_master(args.master)
    candidate = _build_outcome_cohort(master)
    print(
        f"[2020+ mature FSR CV] master eligible before FSR match={len(candidate):,}",
        flush=True,
    )

    cohort, pairs = _load_fsr_pairs_for_cohort(args.fsr_path, candidate)
    print(
        f"[2020+ mature FSR CV] matched leakage-safe FSR pairs={len(cohort):,}",
        flush=True,
    )
    print(
        f"[2020+ mature FSR CV] date range={cohort['event_date'].min().date()} -> "
        f"{cohort['event_date'].max().date()}",
        flush=True,
    )

    frame = _build_joint_frame(cohort, pairs)
    bundles = joint._feature_bundles(frame)

    result_rows: list[dict[str, object]] = []
    prediction_columns: dict[str, np.ndarray] = {}
    for target_col in ("actual_ko_tko", "actual_r1_ko"):
        for bundle_index, (bundle_name, feature_cols) in enumerate(bundles.items()):
            row, pred = joint._evaluate_bundle(
                frame,
                feature_cols,
                bundle_name=bundle_name,
                target_col=target_col,
                n_splits=args.splits,
                n_repeats=args.repeats,
                seed=args.seed + bundle_index + (1000 if target_col == "actual_r1_ko" else 0),
            )
            result_rows.append(row)
            prediction_columns[f"oof_{target_col}__{bundle_name}"] = pred

    for col, values in prediction_columns.items():
        frame[col] = values

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    results = pd.DataFrame(result_rows)
    _print_results(results, frame)
    _power_default_summary(cohort, pairs)
    print(f"\n[2020+ mature FSR CV] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
