"""Leakage-safe FSR-32 trajectory forecast audit.

Builds two shadow simulator-facing alternatives from the existing FSR-32
prefight snapshot history without changing the canonical FSR artifact or MC:

1. raw trend forecast
   - fit a linear trend to the last N strictly-prior prefight snapshots
   - extrapolate that trend to the target fight date
2. shrunk trend forecast
   - blend the raw trend forecast back toward the canonical target prefight FSR
   - lambda = n / (n + SHRINK_K)

The canonical target prefight row is always retained as the control.  The raw
forecast never sees the target row.  The shrunk variant is a pre-fight
trend-adjusted state: both the canonical target prefight FSR and all historical
trend inputs are available before the target fight.

Research/shadow only.  No production artifact is modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH


FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)
OUTPUT_DIR = Path("data/experimental/fsr_trajectory_forecast")
RAW_PATH = OUTPUT_DIR / "fsr32_forecast_raw.parquet"
SHRUNK_PATH = OUTPUT_DIR / "fsr32_forecast_shrunk.parquet"
AUDIT_PATH = OUTPUT_DIR / "fsr32_trajectory_audit.csv"
PAIR_AUDIT_PATH = OUTPUT_DIR / "fsr32_trajectory_pair_audit.csv"
TRAIT_SUMMARY_PATH = OUTPUT_DIR / "fsr32_trajectory_trait_summary.csv"
BUCKET_SUMMARY_PATH = OUTPUT_DIR / "fsr32_trajectory_bucket_summary.csv"

WINDOW = 4
MIN_HISTORY = 2
SHRINK_K = 3.0
DAYS_PER_YEAR = 365.2425
START_DATE = pd.Timestamp("2020-01-01")
MIN_PRIOR_UFC_FIGHTS = 3

META_COLUMNS = {
    "fight_id",
    "date",
    "fighter_id",
    "fighter_name",
    "prior_ufc_fights",
}

# This is constant simulator plumbing, not a learned fighter trajectory.
NO_TREND_COLUMNS = {"stamina_capacity"}


def _trait_columns(fsr: pd.DataFrame) -> list[str]:
    traits: list[str] = []
    for col in fsr.columns:
        if col in META_COLUMNS:
            continue
        if col.endswith("_updates") or col.endswith("_evidence_score"):
            continue
        if pd.api.types.is_numeric_dtype(fsr[col]):
            traits.append(col)
    return traits


def _bounds(trait: str) -> tuple[float, float]:
    if trait == "striking_power":
        return 35.0, 90.0
    if trait == "stamina_capacity":
        return 1.0, np.inf
    return 10.0, 90.0


def _fit_forecast(
    hist: pd.DataFrame,
    trait: str,
    target_date: pd.Timestamp,
) -> tuple[float | None, float, int, float | None]:
    """Return raw forecast, slope rating-points/year, n, and R^2."""
    h = hist[["date", trait]].copy()
    h[trait] = pd.to_numeric(h[trait], errors="coerce")
    h = h.dropna().sort_values("date").tail(WINDOW)
    h = h.drop_duplicates(subset=["date"], keep="last")
    n = len(h)
    if n < MIN_HISTORY:
        return None, 0.0, n, None

    # Center x at the target date so the intercept itself is the target forecast.
    x = (h["date"] - target_date).dt.total_seconds().to_numpy(dtype=float)
    x = x / (DAYS_PER_YEAR * 24.0 * 60.0 * 60.0)
    y = h[trait].to_numpy(dtype=float)

    if np.ptp(x) <= 0.0:
        return None, 0.0, n, None

    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = None if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return float(intercept), float(slope), n, r2


def build_forecasts(
    fsr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    frame = fsr.copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    traits = _trait_columns(frame)
    if len(traits) != 32:
        raise RuntimeError(
            f"Expected 32 FSR-32 rating/parameter columns, found {len(traits)}: {traits}"
        )

    raw = frame.copy()
    shrunk = frame.copy()
    audit_rows: list[dict[str, object]] = []

    for fighter_id, group in frame.groupby("fighter_id", sort=False):
        group = group.sort_values(["date", "fight_id"])
        prior = group.iloc[0:0].copy()

        for idx, row in group.iterrows():
            target_date = pd.Timestamp(row["date"])
            # Strictly earlier dates only. This is the leakage barrier.
            hist = prior.loc[prior["date"] < target_date]

            for trait in traits:
                current = float(row[trait])

                if trait in NO_TREND_COLUMNS:
                    raw_value = current
                    shrunk_value = current
                    slope = 0.0
                    history_n = len(hist.tail(WINDOW))
                    r2 = None
                    lam = 0.0
                else:
                    forecast, slope, history_n, r2 = _fit_forecast(
                        hist,
                        trait,
                        target_date,
                    )
                    if forecast is None:
                        raw_value = current
                        shrunk_value = current
                        lam = 0.0
                    else:
                        low, high = _bounds(trait)
                        raw_value = float(np.clip(forecast, low, high))
                        lam = float(history_n / (history_n + SHRINK_K))
                        shrunk_value = current + lam * (raw_value - current)
                        shrunk_value = float(np.clip(shrunk_value, low, high))

                raw.at[idx, trait] = raw_value
                shrunk.at[idx, trait] = shrunk_value

                audit_rows.append(
                    {
                        "fight_id": str(row["fight_id"]),
                        "date": target_date,
                        "fighter_id": str(fighter_id),
                        "fighter_name": row.get("fighter_name"),
                        "prior_ufc_fights": row.get("prior_ufc_fights"),
                        "trait": trait,
                        "current_fsr": current,
                        "raw_forecast_fsr": raw_value,
                        "shrunk_forecast_fsr": shrunk_value,
                        "raw_delta": raw_value - current,
                        "shrunk_delta": shrunk_value - current,
                        "trend_slope_per_year": slope,
                        "history_n": int(history_n),
                        "trend_r2": r2,
                        "shrink_lambda": lam,
                    }
                )

            # The current row becomes historical only after all target forecasts
            # are produced.  Same-date rows are never admitted by the date filter.
            prior = pd.concat([prior, group.loc[[idx]]], ignore_index=False)

    raw = raw.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)
    shrunk = shrunk.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return raw, shrunk, audit, traits


def _load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master not found: {MASTER_PATH}")
    master = pd.read_parquet(MASTER_PATH).copy()
    required = {
        "fight_id",
        "date",
        "r_id",
        "b_id",
        "winner_id",
        "r_name",
        "b_name",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise RuntimeError(f"Master missing trajectory-audit columns: {missing}")
    master["fight_id"] = master["fight_id"].astype(str)
    master["r_id"] = master["r_id"].astype(str)
    master["b_id"] = master["b_id"].astype(str)
    master["winner_id"] = master["winner_id"].astype("string")
    master["date"] = pd.to_datetime(master["date"], errors="coerce")
    master = master.dropna(subset=["date"]).copy()
    return master.sort_values(["date", "fight_id"]).drop_duplicates("fight_id", keep="last")


def _fighter_trajectory(audit: pd.DataFrame) -> pd.DataFrame:
    usable = audit.loc[~audit["trait"].isin(NO_TREND_COLUMNS)].copy()
    grouped = usable.groupby(["fight_id", "fighter_id"], as_index=False).agg(
        date=("date", "first"),
        fighter_name=("fighter_name", "first"),
        prior_ufc_fights=("prior_ufc_fights", "first"),
        raw_trajectory_score=("raw_delta", "mean"),
        shrunk_trajectory_score=("shrunk_delta", "mean"),
        mean_slope_per_year=("trend_slope_per_year", "mean"),
        median_slope_per_year=("trend_slope_per_year", "median"),
        traits_with_history=("history_n", lambda x: int((x >= MIN_HISTORY).sum())),
        mean_trend_r2=("trend_r2", "mean"),
    )
    return grouped


def build_pair_audit(audit: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    fighter = _fighter_trajectory(audit)

    r = fighter.add_prefix("r_").rename(columns={"r_fight_id": "fight_id"})
    b = fighter.add_prefix("b_").rename(columns={"b_fight_id": "fight_id"})

    pairs = master.merge(
        r,
        left_on=["fight_id", "r_id"],
        right_on=["fight_id", "r_fighter_id"],
        how="inner",
        validate="one_to_one",
    ).merge(
        b,
        left_on=["fight_id", "b_id"],
        right_on=["fight_id", "b_fighter_id"],
        how="inner",
        validate="one_to_one",
    )

    winner = pairs["winner_id"].astype("string")
    valid = winner.eq(pairs["r_id"]) | winner.eq(pairs["b_id"])
    pairs = pairs.loc[valid].copy()
    pairs["actual_red_win"] = pairs["winner_id"].astype(str).eq(pairs["r_id"]).astype(int)

    pairs["raw_trend_advantage"] = (
        pairs["r_raw_trajectory_score"] - pairs["b_raw_trajectory_score"]
    )
    pairs["shrunk_trend_advantage"] = (
        pairs["r_shrunk_trajectory_score"] - pairs["b_shrunk_trajectory_score"]
    )

    for variant in ("raw", "shrunk"):
        advantage = pairs[f"{variant}_trend_advantage"]
        pairs[f"{variant}_trend_pick_red"] = advantage.gt(0)
        pairs[f"{variant}_trend_pick_correct"] = np.where(
            advantage.eq(0),
            np.nan,
            pairs[f"{variant}_trend_pick_red"].eq(pairs["actual_red_win"].eq(1)).astype(float),
        )

    pairs["mature_2020plus"] = (
        pairs["date"].ge(START_DATE)
        & pd.to_numeric(pairs["r_prior_ufc_fights"], errors="coerce").ge(MIN_PRIOR_UFC_FIGHTS)
        & pd.to_numeric(pairs["b_prior_ufc_fights"], errors="coerce").ge(MIN_PRIOR_UFC_FIGHTS)
    )
    return pairs


def trait_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trait, g in audit.groupby("trait", sort=True):
        valid = g["history_n"].ge(MIN_HISTORY)
        gv = g.loc[valid]
        rows.append(
            {
                "trait": trait,
                "rows": len(g),
                "forecastable_rows": int(valid.sum()),
                "forecastable_rate": float(valid.mean()),
                "mean_raw_delta": float(gv["raw_delta"].mean()) if len(gv) else np.nan,
                "mean_abs_raw_delta": float(gv["raw_delta"].abs().mean()) if len(gv) else np.nan,
                "mean_abs_shrunk_delta": float(gv["shrunk_delta"].abs().mean()) if len(gv) else np.nan,
                "mean_slope_per_year": float(gv["trend_slope_per_year"].mean()) if len(gv) else np.nan,
                "median_slope_per_year": float(gv["trend_slope_per_year"].median()) if len(gv) else np.nan,
                "mean_r2": float(gv["trend_r2"].mean()) if len(gv) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bucket_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort_name, cohort in (
        ("all_valid", pairs),
        ("mature_2020plus", pairs.loc[pairs["mature_2020plus"]]),
    ):
        for variant in ("raw", "shrunk"):
            col = f"{variant}_trend_advantage"
            g = cohort.loc[cohort[col].notna()].copy()
            if len(g) < 5:
                continue
            # Duplicate edges are possible when many fights have exactly-zero
            # trajectory; rank() guarantees deterministic five equally-sized bins.
            g["trend_quintile"] = pd.qcut(
                g[col].rank(method="first"),
                q=5,
                labels=["Q1 strong blue", "Q2", "Q3", "Q4", "Q5 strong red"],
            )
            for bucket, b in g.groupby("trend_quintile", observed=True):
                rows.append(
                    {
                        "cohort": cohort_name,
                        "variant": variant,
                        "bucket": str(bucket),
                        "fights": len(b),
                        "mean_trend_advantage": float(b[col].mean()),
                        "red_win_rate": float(b["actual_red_win"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _print_pair_summary(pairs: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("FSR-32 TRAJECTORY WINNER AUDIT")
    print("=" * 112)
    for label, g in (
        ("ALL VALID", pairs),
        ("MATURE 2020+", pairs.loc[pairs["mature_2020plus"]]),
    ):
        print(f"\n{label}: {len(g):,} fights")
        for variant in ("raw", "shrunk"):
            correct = pd.to_numeric(g[f"{variant}_trend_pick_correct"], errors="coerce").dropna()
            advantage = pd.to_numeric(g[f"{variant}_trend_advantage"], errors="coerce")
            winner_signed = np.where(g["actual_red_win"].eq(1), 1.0, -1.0)
            directional_edge = advantage.to_numpy(dtype=float) * winner_signed
            print(
                f"  {variant:6s} higher-trend fighter win rate: "
                f"{correct.mean():.2%} ({int(correct.sum()):,}/{len(correct):,}) | "
                f"mean winner-aligned advantage={np.nanmean(directional_edge):+.3f} rating pts"
            )


def main() -> None:
    if not FSR_PATH.exists():
        raise FileNotFoundError(f"FSR-32 artifact not found: {FSR_PATH}")

    print(f"Loading FSR-32: {FSR_PATH}")
    fsr = pd.read_parquet(FSR_PATH)
    print(f"Rows: {len(fsr):,}")

    raw, shrunk, audit, traits = build_forecasts(fsr)
    master = _load_master()
    pairs = build_pair_audit(audit, master)
    trait = trait_summary(audit)
    buckets = bucket_summary(pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_PATH, index=False)
    shrunk.to_parquet(SHRUNK_PATH, index=False)
    audit.to_csv(AUDIT_PATH, index=False)
    pairs.to_csv(PAIR_AUDIT_PATH, index=False)
    trait.to_csv(TRAIT_SUMMARY_PATH, index=False)
    buckets.to_csv(BUCKET_SUMMARY_PATH, index=False)

    print(f"\nFSR traits: {len(traits)}")
    print(f"Window: last {WINDOW} strictly-prior snapshots; minimum {MIN_HISTORY}")
    print(f"Shrinkage: lambda = n / (n + {SHRINK_K:g})")
    print("Raw forecast is contract-clipped but otherwise unshrunk.")

    _print_pair_summary(pairs)

    print("\nMATURE 2020+ TREND QUINTILES")
    display = buckets.loc[buckets["cohort"].eq("mature_2020plus")].copy()
    if len(display):
        display["red_win_rate"] = display["red_win_rate"].map(lambda x: f"{x:.2%}")
        print(display.to_string(index=False))

    print("\nWrote:")
    for path in (
        RAW_PATH,
        SHRUNK_PATH,
        AUDIT_PATH,
        PAIR_AUDIT_PATH,
        TRAIT_SUMMARY_PATH,
        BUCKET_SUMMARY_PATH,
    ):
        print(" ", path)


if __name__ == "__main__":
    main()
