"""Leakage-safe population backtest of fighter-specific next-FSR trajectory forecasts.

Research/shadow only. This experiment deliberately does NOT use the age-modifier
layer and does not modify stored FSR or the Monte Carlo simulator.

Question
--------
Can a fighter's own prior canonical FSR trajectory predict the fighter's next
prefight FSR better than simply carrying forward the latest demonstrated FSR?

For each canonical trait and each historical target observation:

1. use only that fighter's FSR observations strictly before the target fight;
2. fit candidate polynomials on UFC observation sequence (1, 2, 3, ...);
3. forecast the next sequence point;
4. optionally shrink the raw forecast movement back toward the latest FSR;
5. clamp the forecast to the canonical 10-90 FSR range;
6. compare with the actual target prefight FSR.

Candidate methods
-----------------
- latest: next FSR = latest demonstrated FSR
- linear with shrink alpha in {0.25, 0.50, 0.75, 1.00}
- quadratic with the same shrink grid when >=3 prior observations
- cubic with the same shrink grid when >=4 prior observations

Population selection is chronological. Earlier target fights (80% of distinct
target dates) select the lowest-RMSE method separately for each
``trait x prior-history bucket``. The selected rule is then evaluated on the
later untouched 20% holdout. History buckets are 2, 3, 4, 5, and 6+ prior FSR
observations.

This answers two separate questions without age double counting:
- which trajectory family/shrinkage works for a trait;
- how much fighter history is required before that trajectory is useful.

Outputs
-------
data/experimental/fsr_individual_trajectory_backtest/
  all_forecasts.parquet
  method_metrics_train.csv
  selected_rules.csv
  selected_rules_holdout.csv
  trait_holdout_summary.csv

No simulator/config files are changed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import build_fsr_canonical_database as canonical


OUTPUT_DIR = Path("data/experimental/fsr_individual_trajectory_backtest")
ALL_FORECASTS_PATH = OUTPUT_DIR / "all_forecasts.parquet"
TRAIN_METRICS_PATH = OUTPUT_DIR / "method_metrics_train.csv"
SELECTED_RULES_PATH = OUTPUT_DIR / "selected_rules.csv"
HOLDOUT_RULES_PATH = OUTPUT_DIR / "selected_rules_holdout.csv"
TRAIT_SUMMARY_PATH = OUTPUT_DIR / "trait_holdout_summary.csv"

FSR_MIN = 10.0
FSR_MAX = 90.0
TRAIN_FRACTION = 0.80
SHRINK_ALPHAS = (0.25, 0.50, 0.75, 1.00)
MIN_PRIOR_OBSERVATIONS = 2


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _history_bucket(n: int) -> str:
    if n >= 6:
        return "6+"
    return str(int(n))


def _load_fsr() -> pd.DataFrame:
    path = fsr32.OUTPUT_PATH
    print(f"[trajectory] loading FSR snapshots: {path}", flush=True)
    df = pd.read_parquet(path).copy()
    required = {"fight_id", "fighter_id", "fighter_name", "date", *canonical.CANONICAL_RATINGS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"FSR artifact missing required columns: {missing}")

    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)
    if df.duplicated(["fighter_id", "fight_id"]).any():
        raise RuntimeError("FSR artifact violates fighter-fight grain")
    print(
        f"[trajectory] rows={len(df):,} | fighters={df['fighter_id'].nunique():,} | "
        f"traits={len(canonical.CANONICAL_RATINGS)}",
        flush=True,
    )
    return df


def _poly_raw_forecast(values: np.ndarray, degree: int) -> float:
    n = len(values)
    if n < degree + 1:
        raise ValueError(f"degree {degree} requires >= {degree + 1} observations, got {n}")
    x = np.arange(1, n + 1, dtype=float)
    # Center/scale sequence to reduce polynomial conditioning problems while
    # retaining the exact same forecast geometry.
    center = float(x.mean())
    scale = max(float(x.std(ddof=0)), 1.0)
    z = (x - center) / scale
    target_z = ((n + 1.0) - center) / scale
    coeff = np.polyfit(z, values.astype(float), degree)
    return float(np.polyval(coeff, target_z))


def _candidate_predictions(history: np.ndarray) -> dict[str, float]:
    latest = float(history[-1])
    out = {"latest": latest}
    max_degree = min(3, len(history) - 1)
    for degree in range(1, max_degree + 1):
        raw = _poly_raw_forecast(history, degree)
        label = {1: "linear", 2: "quadratic", 3: "cubic"}[degree]
        for alpha in SHRINK_ALPHAS:
            pred = latest + float(alpha) * (raw - latest)
            pred = float(np.clip(pred, FSR_MIN, FSR_MAX))
            out[f"{label}_a{alpha:.2f}"] = pred
    return out


def _build_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    grouped = list(df.groupby("fighter_id", sort=False))
    total = len(grouped)
    print("[trajectory] building leakage-safe fighter-specific forecasts...", flush=True)

    for fighter_i, (fighter_id, fighter) in enumerate(grouped, start=1):
        fighter = fighter.sort_values(["date", "fight_id"]).reset_index(drop=True)
        name = str(fighter["fighter_name"].iloc[-1])
        for target_idx in range(MIN_PRIOR_OBSERVATIONS, len(fighter)):
            target = fighter.iloc[target_idx]
            prior_n = target_idx
            bucket = _history_bucket(prior_n)
            for trait in canonical.CANONICAL_RATINGS:
                history = pd.to_numeric(fighter.loc[: target_idx - 1, trait], errors="coerce").to_numpy(float)
                actual = pd.to_numeric(pd.Series([target[trait]]), errors="coerce").iloc[0]
                if pd.isna(actual) or not np.isfinite(history).all():
                    continue
                preds = _candidate_predictions(history)
                latest = float(history[-1])
                for method, prediction in preds.items():
                    rows.append(
                        {
                            "fighter_id": str(fighter_id),
                            "fighter_name": name,
                            "target_fight_id": str(target["fight_id"]),
                            "target_date": pd.Timestamp(target["date"]),
                            "trait": trait,
                            "prior_observations": int(prior_n),
                            "history_bucket": bucket,
                            "method": method,
                            "latest_fsr": latest,
                            "prediction": float(prediction),
                            "actual_next_fsr": float(actual),
                            "error": float(prediction - actual),
                            "abs_error": float(abs(prediction - actual)),
                            "sq_error": float((prediction - actual) ** 2),
                            "forecast_change": float(prediction - latest),
                        }
                    )

        if fighter_i % 250 == 0 or fighter_i == total:
            print(
                f"[trajectory] fighters {fighter_i:,}/{total:,} | forecasts={len(rows):,} | elapsed={_elapsed(start)}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("trajectory backtest produced no forecasts")
    return out


def _split_date(forecasts: pd.DataFrame) -> pd.Timestamp:
    dates = np.sort(forecasts["target_date"].dropna().unique())
    if len(dates) < 5:
        raise RuntimeError("not enough distinct target dates for chronological split")
    idx = max(1, int(len(dates) * TRAIN_FRACTION)) - 1
    return pd.Timestamp(dates[idx])


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": int(len(frame)),
        "rmse": float(np.sqrt(frame["sq_error"].mean())),
        "mae": float(frame["abs_error"].mean()),
        "mean_error": float(frame["error"].mean()),
        "mean_abs_forecast_change": float(frame["forecast_change"].abs().mean()),
    }


def _select_rules(forecasts: pd.DataFrame, split_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = forecasts.loc[forecasts["target_date"] <= split_date].copy()
    holdout = forecasts.loc[forecasts["target_date"] > split_date].copy()
    print(
        f"[trajectory] chronological split={split_date.date()} | "
        f"train forecast rows={len(train):,} | holdout forecast rows={len(holdout):,}",
        flush=True,
    )

    metric_rows = []
    for (trait, bucket, method), frame in train.groupby(["trait", "history_bucket", "method"], sort=False):
        row = {"trait": trait, "history_bucket": bucket, "method": method}
        row.update(_metrics(frame))
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    # Lowest training RMSE wins. Exact/near ties favor the structurally simpler
    # method: latest, then linear, quadratic, cubic; within a family, stronger
    # shrinkage (smaller alpha) is simpler/safer.
    def complexity(method: str) -> tuple[int, float]:
        if method == "latest":
            return (0, 0.0)
        family = method.split("_a", 1)[0]
        alpha = float(method.split("_a", 1)[1])
        order = {"linear": 1, "quadratic": 2, "cubic": 3}[family]
        return (order, alpha)

    selected_rows = []
    for (trait, bucket), frame in metrics.groupby(["trait", "history_bucket"], sort=False):
        frame = frame.copy()
        best_rmse = float(frame["rmse"].min())
        # Within 0.25% of the best RMSE, prefer the simpler forecast. This avoids
        # selecting a higher-order curve for numerically trivial gains.
        eligible = frame.loc[frame["rmse"] <= best_rmse * 1.0025].copy()
        eligible["complexity"] = eligible["method"].map(complexity)
        winner = eligible.sort_values("complexity").iloc[0]
        baseline = frame.loc[frame["method"].eq("latest")].iloc[0]
        selected_rows.append(
            {
                "trait": trait,
                "history_bucket": bucket,
                "selected_method": winner["method"],
                "train_rows": int(winner["rows"]),
                "train_rmse": float(winner["rmse"]),
                "train_mae": float(winner["mae"]),
                "baseline_train_rmse": float(baseline["rmse"]),
                "train_rmse_improvement": float(baseline["rmse"] - winner["rmse"]),
                "train_relative_rmse_improvement_pct": float(
                    100.0 * (baseline["rmse"] - winner["rmse"]) / baseline["rmse"]
                ) if baseline["rmse"] > 0 else 0.0,
            }
        )

    selected = pd.DataFrame(selected_rows)

    holdout_rows = []
    for rule in selected.itertuples(index=False):
        selected_frame = holdout.loc[
            holdout["trait"].eq(rule.trait)
            & holdout["history_bucket"].eq(rule.history_bucket)
            & holdout["method"].eq(rule.selected_method)
        ].copy()
        baseline_frame = holdout.loc[
            holdout["trait"].eq(rule.trait)
            & holdout["history_bucket"].eq(rule.history_bucket)
            & holdout["method"].eq("latest")
        ].copy()
        if selected_frame.empty or baseline_frame.empty:
            continue
        sm = _metrics(selected_frame)
        bm = _metrics(baseline_frame)
        holdout_rows.append(
            {
                "trait": rule.trait,
                "history_bucket": rule.history_bucket,
                "selected_method": rule.selected_method,
                "holdout_rows": sm["rows"],
                "holdout_rmse": sm["rmse"],
                "baseline_holdout_rmse": bm["rmse"],
                "holdout_rmse_improvement": float(bm["rmse"] - sm["rmse"]),
                "holdout_relative_rmse_improvement_pct": float(
                    100.0 * (bm["rmse"] - sm["rmse"]) / bm["rmse"]
                ) if bm["rmse"] > 0 else 0.0,
                "holdout_mae": sm["mae"],
                "baseline_holdout_mae": bm["mae"],
                "holdout_mae_improvement": float(bm["mae"] - sm["mae"]),
                "mean_abs_forecast_change": sm["mean_abs_forecast_change"],
            }
        )
    return metrics, selected.merge(pd.DataFrame(holdout_rows), on=["trait", "history_bucket", "selected_method"], how="left")


def _trait_summary(rule_holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trait, frame in rule_holdout.groupby("trait", sort=False):
        valid = frame.dropna(subset=["holdout_rows", "holdout_rmse", "baseline_holdout_rmse"]).copy()
        if valid.empty:
            continue
        weights = valid["holdout_rows"].to_numpy(float)
        # Pool squared errors from bucket RMSEs using row counts.
        selected_sse = np.sum(weights * valid["holdout_rmse"].to_numpy(float) ** 2)
        baseline_sse = np.sum(weights * valid["baseline_holdout_rmse"].to_numpy(float) ** 2)
        n = float(weights.sum())
        selected_rmse = float(np.sqrt(selected_sse / n))
        baseline_rmse = float(np.sqrt(baseline_sse / n))
        rows.append(
            {
                "trait": trait,
                "holdout_targets": int(n),
                "selected_policy_rmse": selected_rmse,
                "latest_baseline_rmse": baseline_rmse,
                "rmse_improvement": float(baseline_rmse - selected_rmse),
                "relative_rmse_improvement_pct": float(
                    100.0 * (baseline_rmse - selected_rmse) / baseline_rmse
                ) if baseline_rmse > 0 else 0.0,
                "buckets_beating_baseline": int((valid["holdout_rmse_improvement"] > 0).sum()),
                "buckets_tested": int(len(valid)),
                "selected_methods": "; ".join(
                    f"{r.history_bucket}:{r.selected_method}" for r in valid.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["relative_rmse_improvement_pct", "trait"], ascending=[False, True]
    ).reset_index(drop=True)


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_fsr()
    forecasts = _build_forecasts(df)
    split = _split_date(forecasts)

    print(f"[trajectory] writing all forecasts: {ALL_FORECASTS_PATH}", flush=True)
    forecasts.to_parquet(ALL_FORECASTS_PATH, index=False)

    metrics, rule_holdout = _select_rules(forecasts, split)
    selected = rule_holdout[
        [
            "trait", "history_bucket", "selected_method", "train_rows", "train_rmse", "train_mae",
            "baseline_train_rmse", "train_rmse_improvement", "train_relative_rmse_improvement_pct",
        ]
    ].copy()
    metrics.to_csv(TRAIN_METRICS_PATH, index=False)
    selected.to_csv(SELECTED_RULES_PATH, index=False)
    rule_holdout.to_csv(HOLDOUT_RULES_PATH, index=False)

    trait_summary = _trait_summary(rule_holdout)
    trait_summary.to_csv(TRAIT_SUMMARY_PATH, index=False)

    print("\n" + "=" * 170)
    print("INDIVIDUAL FSR TRAJECTORY — HOLDOUT SUMMARY")
    print("=" * 170)
    display = trait_summary.copy()
    for col in ("selected_policy_rmse", "latest_baseline_rmse", "rmse_improvement", "relative_rmse_improvement_pct"):
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    print(display.to_string(index=False), flush=True)

    print("\nSELECTED RULES BY TRAIT x HISTORY BUCKET")
    cols = [
        "trait", "history_bucket", "selected_method", "holdout_rows",
        "holdout_rmse", "baseline_holdout_rmse", "holdout_relative_rmse_improvement_pct",
    ]
    print(rule_holdout[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    improved = int((trait_summary["rmse_improvement"] > 0).sum())
    print(
        f"\ntraits improving over latest-FSR baseline on holdout: "
        f"{improved}/{len(trait_summary)}",
        flush=True,
    )
    print(f"split date: {split.date()}", flush=True)
    print(f"elapsed: {_elapsed(start)}", flush=True)
    print(f"wrote: {ALL_FORECASTS_PATH}", flush=True)
    print(f"wrote: {TRAIN_METRICS_PATH}", flush=True)
    print(f"wrote: {SELECTED_RULES_PATH}", flush=True)
    print(f"wrote: {HOLDOUT_RULES_PATH}", flush=True)
    print(f"wrote: {TRAIT_SUMMARY_PATH}", flush=True)
    print("Research only. No age modifiers, stored FSR, or simulator state were changed.", flush=True)


if __name__ == "__main__":
    main()
