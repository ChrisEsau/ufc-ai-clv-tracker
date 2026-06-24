from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_IMPLIED_BUCKETS = [
    {"name": "0_10", "min": 0.00, "max": 0.10},
    {"name": "10_20", "min": 0.10, "max": 0.20},
    {"name": "20_30", "min": 0.20, "max": 0.30},
    {"name": "30_40", "min": 0.30, "max": 0.40},
    {"name": "40_50", "min": 0.40, "max": 0.50},
    {"name": "50_60", "min": 0.50, "max": 0.60},
    {"name": "60_70", "min": 0.60, "max": 0.70},
    {"name": "70_80", "min": 0.70, "max": 0.80},
    {"name": "80_90", "min": 0.80, "max": 0.90},
    {"name": "90_100", "min": 0.90, "max": 1.00},
]


@dataclass(frozen=True)
class ImpliedBucketCalibrationResult:
    rows: pd.DataFrame
    report: pd.DataFrame
    config: dict[str, Any]


def apply_implied_bucket_delta_calibration(joined: pd.DataFrame, *, config: dict[str, Any]) -> ImpliedBucketCalibrationResult:
    calibration_config = (config.get("post_market_calibration") or {}) or {}
    if not calibration_config.get("enabled", False):
        return ImpliedBucketCalibrationResult(rows=joined.copy(), report=pd.DataFrame(), config=calibration_config)

    method = str(calibration_config.get("method", "implied_bucket_delta")).strip().lower()
    if method not in {"implied_bucket_delta", "market_bucket_delta"}:
        raise ValueError(f"Unsupported post-market calibration method: {method}")

    out = joined.copy()
    required = ["model_probability", "implied_probability", "won", "date"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"Implied bucket calibration missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["raw_model_probability"] = pd.to_numeric(out["model_probability"], errors="coerce")
    out["implied_probability"] = pd.to_numeric(out["implied_probability"], errors="coerce")
    out["won_numeric"] = out["won"].astype("boolean").astype("Int64").astype(float)

    split_config = (config.get("split") or {}) or {}
    train_end_date = pd.to_datetime(split_config.get("train_end_date"), errors="coerce")
    calibration_end_date = pd.to_datetime(split_config.get("calibration_end_date"), errors="coerce")
    if pd.isna(train_end_date) or pd.isna(calibration_end_date):
        raise ValueError("post-market calibration requires split.train_end_date and split.calibration_end_date")

    calibration_rows = out[
        (out["date"] > train_end_date)
        & (out["date"] <= calibration_end_date)
        & out["raw_model_probability"].notna()
        & out["implied_probability"].notna()
        & out["won_numeric"].notna()
    ].copy()
    if calibration_rows.empty:
        raise ValueError("No rows available for implied bucket calibration")

    buckets = calibration_config.get("buckets") or DEFAULT_IMPLIED_BUCKETS
    min_rows = int(calibration_config.get("min_rows_per_bucket", 50))
    smoothing_config = calibration_config.get("smoothing") or {}
    smoothing_enabled = bool(smoothing_config.get("enabled", True))
    previous_weight = float(smoothing_config.get("previous_weight", 0.25))
    current_weight = float(smoothing_config.get("current_weight", 0.50))
    next_weight = float(smoothing_config.get("next_weight", 0.25))

    global_delta = float(calibration_rows["raw_model_probability"].mean() - calibration_rows["won_numeric"].mean())
    bucket_stats: list[dict[str, Any]] = []

    for bucket in buckets:
        name = str(bucket["name"])
        mask = _bucket_mask(calibration_rows["implied_probability"].to_numpy(dtype=float), bucket)
        rows = int(mask.sum())
        bucket_df = calibration_rows.loc[mask]
        use_bucket = rows >= min_rows and bucket_df["won_numeric"].nunique(dropna=True) > 1
        avg_model = _safe_mean(bucket_df["raw_model_probability"])
        avg_implied = _safe_mean(bucket_df["implied_probability"])
        actual_rate = _safe_mean(bucket_df["won_numeric"])
        raw_delta = (float(avg_model) - float(actual_rate)) if use_bucket and avg_model is not None and actual_rate is not None else global_delta
        bucket_stats.append(
            {
                "bucket": bucket,
                "name": name,
                "rows": rows,
                "use_bucket": use_bucket,
                "avg_model_probability": avg_model,
                "avg_implied_probability": avg_implied,
                "actual_win_rate": actual_rate,
                "raw_delta": raw_delta,
            }
        )

    smoothed_deltas: dict[str, float] = {}
    for idx, item in enumerate(bucket_stats):
        name = item["name"]
        if not item["use_bucket"]:
            smoothed_deltas[name] = global_delta
            continue
        if not smoothing_enabled:
            smoothed_deltas[name] = float(item["raw_delta"])
            continue
        weighted_sum = current_weight * float(item["raw_delta"])
        total_weight = current_weight
        if idx > 0 and bucket_stats[idx - 1]["use_bucket"]:
            weighted_sum += previous_weight * float(bucket_stats[idx - 1]["raw_delta"])
            total_weight += previous_weight
        if idx < len(bucket_stats) - 1 and bucket_stats[idx + 1]["use_bucket"]:
            weighted_sum += next_weight * float(bucket_stats[idx + 1]["raw_delta"])
            total_weight += next_weight
        smoothed_deltas[name] = weighted_sum / total_weight if total_weight else float(item["raw_delta"])

    applied_delta = np.full(shape=len(out), fill_value=global_delta, dtype=float)
    implied_values = out["implied_probability"].to_numpy(dtype=float)
    for bucket in buckets:
        name = str(bucket["name"])
        delta = smoothed_deltas.get(name)
        if delta is None:
            continue
        mask = _bucket_mask(implied_values, bucket)
        applied_delta[mask] = float(delta)

    out["post_market_calibration_method"] = method
    out["post_market_calibration_delta"] = applied_delta
    out["calibrated_model_probability"] = np.clip(out["raw_model_probability"] - applied_delta, 0.0, 1.0)
    out["model_probability"] = out["calibrated_model_probability"]
    out = out.drop(columns=["won_numeric"])

    report_rows = []
    for item in bucket_stats:
        bucket = item["bucket"]
        delta = smoothed_deltas[item["name"]]
        calibrated_avg = None
        if item["rows"] > 0 and item["avg_model_probability"] is not None:
            calibrated_avg = float(np.clip(float(item["avg_model_probability"]) - delta, 0.0, 1.0))
        report_rows.append(
            {
                "name": item["name"],
                "min_implied_probability": float(bucket["min"]),
                "max_implied_probability": float(bucket["max"]),
                "validation_rows": int(item["rows"]),
                "avg_implied_probability": item["avg_implied_probability"],
                "avg_model_probability": item["avg_model_probability"],
                "actual_win_rate": item["actual_win_rate"],
                "raw_delta": float(item["raw_delta"]),
                "smoothed_delta": float(delta),
                "calibrated_avg_probability": calibrated_avg,
                "used_bucket_calibrator": bool(item["use_bucket"]),
                "fallback_used": not bool(item["use_bucket"]),
            }
        )

    return ImpliedBucketCalibrationResult(rows=out, report=pd.DataFrame(report_rows), config=calibration_config)


def _bucket_mask(values: np.ndarray, bucket: dict[str, Any]) -> np.ndarray:
    lower = float(bucket["min"])
    upper = float(bucket["max"])
    if upper >= 1.0:
        return (values >= lower) & (values <= upper)
    return (values >= lower) & (values < upper)


def _safe_mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    result = pd.to_numeric(values, errors="coerce").mean()
    return None if pd.isna(result) else float(result)
