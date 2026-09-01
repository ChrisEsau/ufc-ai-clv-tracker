from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/research/prop_mispricing")
WINNERS = ROOT / "ko_method_route_reallocation_winners.csv"
PERIODS = ROOT / "ko_method_route_reallocation_group_periods.csv"
BINS = ROOT / "ko_method_route_reallocation_feature_bins.csv"
THRESHOLDS = ROOT / "ko_method_route_reallocation_thresholds.csv"

BASELINE_ADJUSTED_OUT = ROOT / "ko_method_route_reallocation_baseline_adjusted_groups.csv"
MARKET_CORRECT_OUT = ROOT / "ko_method_route_reallocation_market_correct_winner_groups.csv"
EXTREME_DELTAS_OUT = ROOT / "ko_method_route_reallocation_feature_extreme_deltas.csv"

DEV_START = pd.Timestamp("2021-01-01")
DEV_END = pd.Timestamp("2024-12-31")
VAL_START = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-12-31")


def _route_stats(g: pd.DataFrame) -> dict:
    if g.empty:
        return {
            "n": 0,
            "mean_market_ml_probability": np.nan,
            "actual_ko_share": np.nan,
            "market_ko_share": np.nan,
            "ko_residual": np.nan,
            "actual_sub_share": np.nan,
            "market_sub_share": np.nan,
            "sub_residual": np.nan,
            "actual_dec_share": np.nan,
            "market_dec_share": np.nan,
            "dec_residual": np.nan,
        }
    actual_ko = float(g["actual_ko"].mean())
    actual_sub = float(g["actual_sub"].mean())
    actual_dec = float(g["actual_dec"].mean())
    market_ko = float(g["market_conditional_ko_given_win"].mean())
    market_sub = float(g["market_conditional_sub_given_win"].mean())
    market_dec = float(g["market_conditional_dec_given_win"].mean())
    return {
        "n": int(len(g)),
        "mean_market_ml_probability": float(g["market_ml_probability"].mean()),
        "actual_ko_share": actual_ko,
        "market_ko_share": market_ko,
        "ko_residual": actual_ko - market_ko,
        "actual_sub_share": actual_sub,
        "market_sub_share": market_sub,
        "sub_residual": actual_sub - market_sub,
        "actual_dec_share": actual_dec,
        "market_dec_share": market_dec,
        "dec_residual": actual_dec - market_dec,
    }


def _period_masks(w: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("dev_2021_2024", w["date"].between(DEV_START, DEV_END)),
        ("2021", w["date"].dt.year.eq(2021)),
        ("2022", w["date"].dt.year.eq(2022)),
        ("2023", w["date"].dt.year.eq(2023)),
        ("2024", w["date"].dt.year.eq(2024)),
        ("validation_2025", w["date"].between(VAL_START, VAL_END)),
    ]


def _baseline_residuals(w: pd.DataFrame, market_correct_only: bool = False) -> dict[str, float]:
    z = w[w["betting_eligible"].astype(bool)].copy()
    if market_correct_only:
        z = z[pd.to_numeric(z["market_ml_probability"], errors="coerce").ge(0.50)]
    out = {}
    for period, mask in _period_masks(z):
        g = z[mask]
        out[period] = float((g["actual_ko"] - g["market_conditional_ko_given_win"]).mean()) if len(g) else np.nan
    return out


def build_baseline_adjusted(w: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    baseline = _baseline_residuals(w, market_correct_only=False)
    out = periods.copy()
    out["all_winner_baseline_ko_residual"] = out["period"].map(baseline)
    out["ko_excess_vs_all_winners"] = out["ko_residual"] - out["all_winner_baseline_ko_residual"]
    return out


def build_market_correct_groups(w: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    t = thresholds.set_index("feature")

    def q40(feature: str) -> float:
        return float(t.loc[feature, "q40"])

    def q60(feature: str) -> float:
        return float(t.loc[feature, "q60"])

    z = w[w["betting_eligible"].astype(bool)].copy()
    z = z[pd.to_numeric(z["market_ml_probability"], errors="coerce").ge(0.50)].copy()

    z["high_accuracy"] = pd.to_numeric(z["ewm_str_acc_diff"], errors="coerce").ge(q60("ewm_str_acc_diff"))
    z["height_advantage"] = pd.to_numeric(z["height_diff"], errors="coerce").ge(q60("height_diff"))
    z["reach_advantage"] = pd.to_numeric(z["reach_diff"], errors="coerce").ge(q60("reach_diff"))
    z["low_td"] = pd.to_numeric(z["fighter_ewm_td_avg"], errors="coerce").le(q40("fighter_ewm_td_avg"))
    z["high_td"] = pd.to_numeric(z["fighter_ewm_td_avg"], errors="coerce").ge(q60("fighter_ewm_td_avg"))
    z["low_control"] = pd.to_numeric(z["fighter_ewm_ctrl_per_min"], errors="coerce").le(q40("fighter_ewm_ctrl_per_min"))
    z["high_control"] = pd.to_numeric(z["fighter_ewm_ctrl_per_min"], errors="coerce").ge(q60("fighter_ewm_ctrl_per_min"))
    z["low_submission_activity"] = pd.to_numeric(z["fighter_ewm_sub_avg"], errors="coerce").le(q40("fighter_ewm_sub_avg"))
    z["high_submission_activity"] = pd.to_numeric(z["fighter_ewm_sub_avg"], errors="coerce").ge(q60("fighter_ewm_sub_avg"))
    z["low_historical_ko_rate"] = pd.to_numeric(z["fighter_pre_ko_rate"], errors="coerce").le(q40("fighter_pre_ko_rate"))
    z["high_historical_ko_rate"] = pd.to_numeric(z["fighter_pre_ko_rate"], errors="coerce").ge(q60("fighter_pre_ko_rate"))
    z["no_recent5_r1_ko"] = pd.to_numeric(z["recent5_r1_ko_wins"], errors="coerce").eq(0)
    z["any_recent5_r1_ko"] = pd.to_numeric(z["recent5_r1_ko_wins"], errors="coerce").ge(1)

    masks = {
        "all_market_correct_winners": pd.Series(True, index=z.index),
        "high_accuracy": z["high_accuracy"],
        "height_advantage": z["height_advantage"],
        "reach_advantage": z["reach_advantage"],
        "height_adv__high_accuracy": z["height_advantage"] & z["high_accuracy"],
        "reach_adv__high_accuracy": z["reach_advantage"] & z["high_accuracy"],
        "high_accuracy__low_td": z["high_accuracy"] & z["low_td"],
        "high_accuracy__high_td": z["high_accuracy"] & z["high_td"],
        "high_accuracy__low_control": z["high_accuracy"] & z["low_control"],
        "high_accuracy__high_control": z["high_accuracy"] & z["high_control"],
        "high_accuracy__low_sub_activity": z["high_accuracy"] & z["low_submission_activity"],
        "high_accuracy__high_sub_activity": z["high_accuracy"] & z["high_submission_activity"],
        "height_adv__high_accuracy__low_td": z["height_advantage"] & z["high_accuracy"] & z["low_td"],
        "height_adv__high_accuracy__high_td": z["height_advantage"] & z["high_accuracy"] & z["high_td"],
        "reach_adv__high_accuracy__low_td": z["reach_advantage"] & z["high_accuracy"] & z["low_td"],
        "reach_adv__high_accuracy__high_td": z["reach_advantage"] & z["high_accuracy"] & z["high_td"],
        "high_accuracy__low_historical_ko_rate": z["high_accuracy"] & z["low_historical_ko_rate"],
        "high_accuracy__high_historical_ko_rate": z["high_accuracy"] & z["high_historical_ko_rate"],
        "high_accuracy__no_recent5_r1_ko": z["high_accuracy"] & z["no_recent5_r1_ko"],
        "high_accuracy__any_recent5_r1_ko": z["high_accuracy"] & z["any_recent5_r1_ko"],
    }

    baseline = _baseline_residuals(w, market_correct_only=True)
    rows = []
    for group, group_mask in masks.items():
        for period, period_mask in _period_masks(z):
            g = z[group_mask & period_mask]
            stats = _route_stats(g)
            rows.append(
                {
                    "group": group,
                    "period": period,
                    **stats,
                    "market_correct_winner_baseline_ko_residual": baseline[period],
                    "ko_excess_vs_market_correct_winners": stats["ko_residual"] - baseline[period]
                    if pd.notna(stats["ko_residual"]) and pd.notna(baseline[period])
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_extreme_deltas(bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, feature), g in bins.groupby(["period", "feature"], sort=False):
        x = g.set_index("bucket")
        if "Q1" not in x.index or "Q5" not in x.index:
            continue
        q1 = x.loc["Q1"]
        q5 = x.loc["Q5"]
        rows.append(
            {
                "period": period,
                "feature": feature,
                "q1_n": int(q1["n"]),
                "q5_n": int(q5["n"]),
                "q1_feature_mean": q1["feature_mean"],
                "q5_feature_mean": q5["feature_mean"],
                "actual_ko_q5_minus_q1": q5["actual_ko_share"] - q1["actual_ko_share"],
                "market_ko_q5_minus_q1": q5["market_ko_share"] - q1["market_ko_share"],
                "ko_residual_q5_minus_q1": q5["ko_residual"] - q1["ko_residual"],
                "actual_sub_q5_minus_q1": q5["actual_sub_share"] - q1["actual_sub_share"],
                "market_sub_q5_minus_q1": q5["market_sub_share"] - q1["market_sub_share"],
                "sub_residual_q5_minus_q1": q5["sub_residual"] - q1["sub_residual"],
                "actual_dec_q5_minus_q1": q5["actual_dec_share"] - q1["actual_dec_share"],
                "market_dec_q5_minus_q1": q5["market_dec_share"] - q1["market_dec_share"],
                "dec_residual_q5_minus_q1": q5["dec_residual"] - q1["dec_residual"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    w = pd.read_csv(WINNERS, low_memory=False)
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    if w["date"].isna().any() or (w["date"] >= "2026-01-01").any():
        raise RuntimeError("invalid or 2026+ winner row entered postprocess")
    if not w[["actual_ko", "actual_sub", "actual_dec"]].sum(axis=1).eq(1).all():
        raise RuntimeError("winner method target is not one-hot")

    periods = pd.read_csv(PERIODS)
    bins = pd.read_csv(BINS)
    thresholds = pd.read_csv(THRESHOLDS)

    adjusted = build_baseline_adjusted(w, periods)
    market_correct = build_market_correct_groups(w, thresholds)
    extreme = build_extreme_deltas(bins)

    adjusted.to_csv(BASELINE_ADJUSTED_OUT, index=False)
    market_correct.to_csv(MARKET_CORRECT_OUT, index=False)
    extreme.to_csv(EXTREME_DELTAS_OUT, index=False)

    print("baseline-adjusted rows", len(adjusted))
    print("market-correct winner rows", len(market_correct))
    print("feature extreme delta rows", len(extreme))


if __name__ == "__main__":
    main()
