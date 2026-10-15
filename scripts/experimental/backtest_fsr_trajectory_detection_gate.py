"""Backtest gated fighter-specific FSR trajectory forecasts.

Research-only and intentionally independent of age modifiers.

Question
--------
Can we detect, using only a fighter's FSR history available before a target fight,
when recent directional movement is coherent enough that a conservative trajectory
forecast beats simply carrying the latest FSR forward?

Design
------
For every canonical trait and every target observation with >=3 prior FSR points:

1. Baseline forecast = latest observed FSR.
2. Compute pre-target trajectory features from prior observations only:
   - recent 3-point linear slope;
   - full-history linear slope;
   - recent-step directional agreement;
   - slope signal/noise ratio;
   - recent absolute movement magnitude.
3. Candidate gates decide whether to activate a trajectory forecast.
4. When a gate fires, forecast = latest + alpha * selected slope, with conservative
   alpha values. Otherwise forecast = latest.
5. Earlier chronological targets select the best gate per trait. The later 20%
   of target dates are untouched holdout evaluation.

No age input is used. No stored FSR or simulator state is changed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import build_fsr_32_database as fsr32


OUTPUT_DIR = Path("data/experimental/fsr_trajectory_detection_gate")
TARGETS_PATH = OUTPUT_DIR / "trajectory_gate_targets.parquet"
TRAIN_METRICS_PATH = OUTPUT_DIR / "trajectory_gate_train_metrics.csv"
SELECTED_PATH = OUTPUT_DIR / "trajectory_gate_selected_rules.csv"
HOLDOUT_PATH = OUTPUT_DIR / "trajectory_gate_holdout_summary.csv"

ALPHAS = (0.25, 0.50, 0.75, 1.00)
AGREEMENT_THRESHOLDS = (0.67, 1.00)
SNR_THRESHOLDS = (0.50, 1.00, 1.50, 2.00)
MAG_THRESHOLDS = (0.25, 0.50, 1.00)
MIN_PRIOR_POINTS = 3


def _slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _trajectory_features(history: np.ndarray) -> dict[str, float]:
    y = np.asarray(history, dtype=float)
    recent = y[-3:] if len(y) >= 3 else y
    recent_slope = _slope(recent)
    full_slope = _slope(y)

    steps = np.diff(recent)
    nonzero = steps[np.abs(steps) > 1e-12]
    if len(nonzero) == 0 or abs(recent_slope) <= 1e-12:
        agreement = 0.0
    else:
        direction = np.sign(recent_slope)
        agreement = float(np.mean(np.sign(nonzero) == direction))

    # Residual noise around the recent linear fit. Small epsilon makes a perfectly
    # monotonic three-point line high-signal without division errors.
    x = np.arange(len(recent), dtype=float)
    if len(recent) >= 2:
        coeff = np.polyfit(x, recent, 1)
        residual = recent - np.polyval(coeff, x)
        noise = float(np.sqrt(np.mean(residual ** 2)))
    else:
        noise = 0.0
    snr = float(abs(recent_slope) / max(noise, 0.10))
    recent_magnitude = float(np.mean(np.abs(steps))) if len(steps) else 0.0

    return {
        "recent_slope": recent_slope,
        "full_slope": full_slope,
        "agreement": agreement,
        "snr": snr,
        "recent_magnitude": recent_magnitude,
    }


def _load_targets() -> pd.DataFrame:
    print(f"[trajectory-gate] loading FSR snapshots: {fsr32.OUTPUT_PATH}", flush=True)
    df = pd.read_parquet(fsr32.OUTPUT_PATH).copy()
    required = {"fighter_id", "fighter_name", "fight_id", "date", *canonical.CANONICAL_RATINGS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"FSR snapshots missing required columns: {missing}")

    df["fighter_id"] = df["fighter_id"].astype(str)
    df["fight_id"] = df["fight_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    fighters = list(df.groupby("fighter_id", sort=False))
    for i, (fighter_id, group) in enumerate(fighters, start=1):
        group = group.sort_values(["date", "fight_id"]).reset_index(drop=True)
        if len(group) <= MIN_PRIOR_POINTS:
            continue
        for target_idx in range(MIN_PRIOR_POINTS, len(group)):
            target = group.iloc[target_idx]
            prior = group.iloc[:target_idx]
            for trait in canonical.CANONICAL_RATINGS:
                hist = pd.to_numeric(prior[trait], errors="coerce").to_numpy(float)
                actual = pd.to_numeric(pd.Series([target[trait]]), errors="coerce").iloc[0]
                if len(hist) < MIN_PRIOR_POINTS or not np.all(np.isfinite(hist)) or pd.isna(actual):
                    continue
                feat = _trajectory_features(hist)
                rows.append({
                    "fighter_id": fighter_id,
                    "fighter_name": target.get("fighter_name"),
                    "fight_id": str(target["fight_id"]),
                    "date": target["date"],
                    "trait": trait,
                    "prior_points": int(len(hist)),
                    "latest": float(hist[-1]),
                    "actual": float(actual),
                    **feat,
                })
        if i % 250 == 0 or i == len(fighters):
            print(
                f"[trajectory-gate] fighters {i:,}/{len(fighters):,} | targets={len(rows):,} | "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    out = pd.DataFrame(rows)
    print(
        f"[trajectory-gate] target rows={len(out):,} | unique target fighter-fights="
        f"{out[['fighter_id','fight_id']].drop_duplicates().shape[0]:,}",
        flush=True,
    )
    return out


def _candidate_rules() -> list[dict[str, object]]:
    rules: list[dict[str, object]] = [{"name": "latest", "slope": "none", "alpha": 0.0}]
    for slope_name in ("recent_slope", "full_slope"):
        for alpha in ALPHAS:
            # Ungated shrunk slope remains a competitor.
            rules.append({
                "name": f"{slope_name}_a{alpha:.2f}",
                "slope": slope_name,
                "alpha": alpha,
                "agreement_min": 0.0,
                "snr_min": 0.0,
                "magnitude_min": 0.0,
            })
            for agreement in AGREEMENT_THRESHOLDS:
                for snr in SNR_THRESHOLDS:
                    for mag in MAG_THRESHOLDS:
                        rules.append({
                            "name": (
                                f"gate_{slope_name}_a{alpha:.2f}_agree{agreement:.2f}_"
                                f"snr{snr:.2f}_mag{mag:.2f}"
                            ),
                            "slope": slope_name,
                            "alpha": alpha,
                            "agreement_min": agreement,
                            "snr_min": snr,
                            "magnitude_min": mag,
                        })
    return rules


def _predict(frame: pd.DataFrame, rule: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    latest = frame["latest"].to_numpy(float)
    if rule["name"] == "latest":
        return latest.copy(), np.zeros(len(frame), dtype=bool)
    slope = frame[str(rule["slope"])].to_numpy(float)
    active = (
        (frame["agreement"].to_numpy(float) >= float(rule.get("agreement_min", 0.0)))
        & (frame["snr"].to_numpy(float) >= float(rule.get("snr_min", 0.0)))
        & (frame["recent_magnitude"].to_numpy(float) >= float(rule.get("magnitude_min", 0.0)))
    )
    pred = latest.copy()
    pred[active] = latest[active] + float(rule["alpha"]) * slope[active]
    return pred, active


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _chronological_split(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.sort(targets["date"].dropna().unique())
    idx = max(1, int(len(dates) * 0.80)) - 1
    split = pd.Timestamp(dates[idx])
    train = targets.loc[targets["date"] <= split].copy()
    holdout = targets.loc[targets["date"] > split].copy()
    if train.empty or holdout.empty:
        raise RuntimeError("trajectory-gate chronological split produced empty side")
    return train, holdout, split


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = _load_targets()
    targets.to_parquet(TARGETS_PATH, index=False)
    train, holdout, split = _chronological_split(targets)
    rules = _candidate_rules()
    print(
        f"[trajectory-gate] chronological split={split.date()} | train rows={len(train):,} | "
        f"holdout rows={len(holdout):,} | candidate rules={len(rules):,}",
        flush=True,
    )

    metric_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []

    for i, trait in enumerate(canonical.CANONICAL_RATINGS, start=1):
        tr = train.loc[train["trait"].eq(trait)].copy()
        ho = holdout.loc[holdout["trait"].eq(trait)].copy()
        if tr.empty or ho.empty:
            continue
        actual_tr = tr["actual"].to_numpy(float)
        actual_ho = ho["actual"].to_numpy(float)
        baseline_ho = _rmse(actual_ho, ho["latest"].to_numpy(float))

        trait_metrics = []
        for rule in rules:
            pred, active = _predict(tr, rule)
            rmse = _rmse(actual_tr, pred)
            row = {
                "trait": trait,
                "rule": rule["name"],
                "train_rows": len(tr),
                "train_rmse": rmse,
                "gate_rate": float(np.mean(active)),
                "slope": rule.get("slope"),
                "alpha": rule.get("alpha"),
                "agreement_min": rule.get("agreement_min", np.nan),
                "snr_min": rule.get("snr_min", np.nan),
                "magnitude_min": rule.get("magnitude_min", np.nan),
            }
            trait_metrics.append(row)
            metric_rows.append(row)

        # Select by train RMSE, but require a real improvement over latest. Ties favor
        # latest and then lower gate rate / smaller alpha to remain conservative.
        metrics_df = pd.DataFrame(trait_metrics)
        baseline_train = float(metrics_df.loc[metrics_df["rule"].eq("latest"), "train_rmse"].iloc[0])
        eligible = metrics_df.loc[metrics_df["train_rmse"] < baseline_train - 1e-12].copy()
        if eligible.empty:
            chosen = metrics_df.loc[metrics_df["rule"].eq("latest")].iloc[0]
        else:
            eligible["complexity"] = (
                eligible["gate_rate"].fillna(0.0)
                + 0.01 * pd.to_numeric(eligible["alpha"], errors="coerce").fillna(0.0)
            )
            chosen = eligible.sort_values(["train_rmse", "complexity", "rule"]).iloc[0]

        selected_rule = next(rule for rule in rules if rule["name"] == chosen["rule"])
        pred_ho, active_ho = _predict(ho, selected_rule)
        selected_rmse = _rmse(actual_ho, pred_ho)
        improvement = baseline_ho - selected_rmse
        relative = 100.0 * improvement / baseline_ho if baseline_ho > 0 else 0.0

        selected_rows.append({
            "trait": trait,
            "selected_rule": chosen["rule"],
            "train_rmse": float(chosen["train_rmse"]),
            "baseline_train_rmse": baseline_train,
            "train_gate_rate": float(chosen["gate_rate"]),
        })
        holdout_rows.append({
            "trait": trait,
            "holdout_targets": len(ho),
            "selected_rule": chosen["rule"],
            "holdout_gate_rate": float(np.mean(active_ho)),
            "selected_policy_rmse": selected_rmse,
            "latest_baseline_rmse": baseline_ho,
            "rmse_improvement": improvement,
            "relative_rmse_improvement_pct": relative,
            "beats_baseline": int(improvement > 0.0),
        })
        print(
            f"[trajectory-gate] {i:02d}/25 {trait}: {chosen['rule']} | "
            f"holdout {baseline_ho:.4f} -> {selected_rmse:.4f} "
            f"({relative:+.2f}%) | gate={np.mean(active_ho):.1%}",
            flush=True,
        )

    metrics_all = pd.DataFrame(metric_rows)
    selected = pd.DataFrame(selected_rows)
    summary = pd.DataFrame(holdout_rows).sort_values(
        ["relative_rmse_improvement_pct", "trait"], ascending=[False, True]
    )
    metrics_all.to_csv(TRAIN_METRICS_PATH, index=False)
    selected.to_csv(SELECTED_PATH, index=False)
    summary.to_csv(HOLDOUT_PATH, index=False)

    print("\n" + "=" * 150)
    print("FSR TRAJECTORY-DETECTION GATE — UNTOUCHED HOLDOUT")
    print("=" * 150)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(
        f"\ntraits beating latest-FSR baseline: {int(summary['beats_baseline'].sum())}/{len(summary)}",
        flush=True,
    )
    gated = summary.loc[~summary["selected_rule"].eq("latest")]
    print(f"traits selecting a trajectory rule: {len(gated)}/{len(summary)}", flush=True)
    print(f"split date: {split.date()}", flush=True)
    print(f"elapsed: {time.perf_counter() - start:.1f}s", flush=True)
    print(f"wrote: {TARGETS_PATH}", flush=True)
    print(f"wrote: {TRAIN_METRICS_PATH}", flush=True)
    print(f"wrote: {SELECTED_PATH}", flush=True)
    print(f"wrote: {HOLDOUT_PATH}", flush=True)
    print("Research only. No age modifiers, stored FSR, or simulator state were changed.", flush=True)


if __name__ == "__main__":
    main()
