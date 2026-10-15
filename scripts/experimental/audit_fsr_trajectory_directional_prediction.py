"""Directional audit for fighter-specific next-FSR trajectory forecasts.

Research/shadow only. No age modifier is applied and no simulator/config state is changed.

Question:
When a trait actually changes meaningfully at the next observed fight, can the
fighter's prior FSR trajectory correctly predict the direction of that change?

This reuses the leakage-safe candidate forecasts from
``backtest_fsr_individual_trajectory_forecasts.py`` and evaluates three groups:
- all fighters
- target age < 29
- target age >= 29

For each trait, the earlier chronological sample determines a trait-specific
"meaningful move" threshold from the median non-zero absolute next-FSR change.
Candidate direction families are latest/no-call, linear, quadratic and cubic.
Shrink alpha does not change sign, so one representative (a0.25) is used per
family. Earlier data selects the family maximizing directional edge:

    edge = (correct_calls - incorrect_calls) / meaningful_targets

This rewards both accuracy and coverage and prevents a nearly-always-abstain
method from winning on tiny samples. The selected family is then frozen and
evaluated on the later untouched 20% holdout.

Outputs:
data/experimental/fsr_trajectory_directional_audit/
  train_direction_metrics.csv
  selected_direction_rules.csv
  holdout_direction_summary.csv
  holdout_direction_by_trait.csv

No stored FSR, age modifier, or Monte Carlo state is changed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import backtest_fsr_individual_trajectory_by_age29 as age29

FORECASTS_PATH = Path("data/experimental/fsr_individual_trajectory_backtest/all_forecasts.parquet")
OUTPUT_DIR = Path("data/experimental/fsr_trajectory_directional_audit")
TRAIN_METRICS_PATH = OUTPUT_DIR / "train_direction_metrics.csv"
SELECTED_RULES_PATH = OUTPUT_DIR / "selected_direction_rules.csv"
HOLDOUT_SUMMARY_PATH = OUTPUT_DIR / "holdout_direction_summary.csv"
HOLDOUT_TRAIT_PATH = OUTPUT_DIR / "holdout_direction_by_trait.csv"
TRAIN_FRACTION = 0.80
REPRESENTATIVE_METHODS = ("linear_a0.25", "quadratic_a0.25", "cubic_a0.25")
MIN_MEANINGFUL_TRAIN = 30


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _load() -> pd.DataFrame:
    if not FORECASTS_PATH.exists():
        raise FileNotFoundError(f"missing {FORECASTS_PATH}; run the trajectory backtest first")
    print(f"[direction] loading forecasts: {FORECASTS_PATH}", flush=True)
    fc = pd.read_parquet(FORECASTS_PATH).copy()
    fc["fighter_id"] = fc["fighter_id"].astype(str)
    fc["target_fight_id"] = fc["target_fight_id"].astype(str)
    fc["target_date"] = pd.to_datetime(fc["target_date"], errors="coerce")
    fc = fc.dropna(subset=["target_date"]).copy()

    # Keep one row per target/trait/family. Shrink alpha does not affect sign.
    fc = fc.loc[fc["method"].isin(REPRESENTATIVE_METHODS)].copy()
    fc["family"] = fc["method"].str.split("_a", n=1).str[0]

    ages = age29._load_ages().rename(columns={"fight_id": "target_fight_id"})
    fc = fc.merge(
        ages[["target_fight_id", "fighter_id", "target_age"]],
        on=["target_fight_id", "fighter_id"],
        how="left",
        validate="many_to_one",
    )
    fc["actual_change"] = fc["actual_next_fsr"] - fc["latest_fsr"]
    fc["predicted_change"] = fc["prediction"] - fc["latest_fsr"]
    print(
        f"[direction] rows={len(fc):,} | target fighter-fights="
        f"{fc[['fighter_id','target_fight_id']].drop_duplicates().shape[0]:,}",
        flush=True,
    )
    return fc


def _split_date(fc: pd.DataFrame) -> pd.Timestamp:
    dates = np.sort(fc["target_date"].dropna().unique())
    idx = max(1, int(len(dates) * TRAIN_FRACTION)) - 1
    return pd.Timestamp(dates[idx])


def _subgroups(fc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all": fc.copy(),
        "age_lt_29": fc.loc[fc["target_age"] < 29.0].copy(),
        "age_ge_29": fc.loc[fc["target_age"] >= 29.0].copy(),
    }


def _meaningful_threshold(train_trait: pd.DataFrame) -> float:
    # actual_change is duplicated across families; dedupe target rows first.
    base = train_trait.drop_duplicates(["fighter_id", "target_fight_id", "trait"])
    vals = base["actual_change"].abs()
    vals = vals.loc[vals > 1e-12]
    if vals.empty:
        return float("inf")
    return float(vals.median())


def _direction_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    meaningful = frame.loc[frame["actual_change"].abs() >= threshold].copy()
    n = len(meaningful)
    if n == 0:
        return {
            "meaningful_targets": 0,
            "calls": 0,
            "coverage": 0.0,
            "direction_accuracy": np.nan,
            "directional_edge": 0.0,
            "up_accuracy": np.nan,
            "down_accuracy": np.nan,
        }
    actual_sign = np.sign(meaningful["actual_change"].to_numpy(float))
    pred_sign = np.sign(meaningful["predicted_change"].to_numpy(float))
    called = pred_sign != 0
    calls = int(called.sum())
    correct = int(((pred_sign == actual_sign) & called).sum())
    incorrect = int(((pred_sign != actual_sign) & called).sum())
    up = actual_sign > 0
    down = actual_sign < 0
    up_called = called & up
    down_called = called & down
    return {
        "meaningful_targets": n,
        "calls": calls,
        "coverage": calls / n,
        "direction_accuracy": correct / calls if calls else np.nan,
        "directional_edge": (correct - incorrect) / n,
        "up_accuracy": float((pred_sign[up_called] == actual_sign[up_called]).mean()) if up_called.any() else np.nan,
        "down_accuracy": float((pred_sign[down_called] == actual_sign[down_called]).mean()) if down_called.any() else np.nan,
    }


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fc = _load()
    split = _split_date(fc)
    print(f"[direction] chronological split={split.date()}", flush=True)

    train_metric_rows = []
    selected_rows = []
    holdout_rows = []

    for subgroup, sub in _subgroups(fc).items():
        train = sub.loc[sub["target_date"] <= split].copy()
        holdout = sub.loc[sub["target_date"] > split].copy()
        print(
            f"[direction] {subgroup}: train={len(train):,} holdout={len(holdout):,}",
            flush=True,
        )

        traits = sorted(train["trait"].unique())
        for i, trait in enumerate(traits, start=1):
            tr_trait = train.loc[train["trait"].eq(trait)].copy()
            ho_trait = holdout.loc[holdout["trait"].eq(trait)].copy()
            threshold = _meaningful_threshold(tr_trait)

            family_rows = []
            for family, part in tr_trait.groupby("family", sort=False):
                metrics = _direction_metrics(part, threshold)
                row = {
                    "subgroup": subgroup,
                    "trait": trait,
                    "family": family,
                    "meaningful_threshold": threshold,
                    **metrics,
                }
                family_rows.append(row)
                train_metric_rows.append(row)

            eligible = [r for r in family_rows if r["meaningful_targets"] >= MIN_MEANINGFUL_TRAIN]
            if not eligible:
                winner = {"family": "none", "directional_edge": 0.0}
            else:
                # Highest edge; then highest coverage; then simpler family.
                order = {"linear": 0, "quadratic": 1, "cubic": 2}
                winner = sorted(
                    eligible,
                    key=lambda r: (-float(r["directional_edge"]), -float(r["coverage"]), order.get(str(r["family"]), 9)),
                )[0]

            selected_rows.append({
                "subgroup": subgroup,
                "trait": trait,
                "selected_family": winner["family"],
                "meaningful_threshold": threshold,
                "train_directional_edge": float(winner.get("directional_edge", 0.0)),
                "train_direction_accuracy": float(winner.get("direction_accuracy", np.nan)),
                "train_coverage": float(winner.get("coverage", 0.0)),
            })

            if winner["family"] == "none":
                hm = {
                    "meaningful_targets": int(
                        ho_trait.drop_duplicates(["fighter_id", "target_fight_id", "trait"])
                        .loc[lambda x: x["actual_change"].abs() >= threshold].shape[0]
                    ),
                    "calls": 0,
                    "coverage": 0.0,
                    "direction_accuracy": np.nan,
                    "directional_edge": 0.0,
                    "up_accuracy": np.nan,
                    "down_accuracy": np.nan,
                }
            else:
                hm = _direction_metrics(
                    ho_trait.loc[ho_trait["family"].eq(winner["family"])], threshold
                )
            holdout_rows.append({
                "subgroup": subgroup,
                "trait": trait,
                "selected_family": winner["family"],
                "meaningful_threshold": threshold,
                **hm,
            })

        print(f"[direction] {subgroup}: completed {len(traits)} traits", flush=True)

    train_metrics = pd.DataFrame(train_metric_rows)
    selected = pd.DataFrame(selected_rows)
    holdout = pd.DataFrame(holdout_rows)

    summary_rows = []
    for subgroup, part in holdout.groupby("subgroup", sort=False):
        meaningful = int(part["meaningful_targets"].sum())
        calls = int(part["calls"].sum())
        # Aggregate weighted edge/correctness from per-trait metrics.
        weighted_edge = float(
            np.average(part["directional_edge"], weights=part["meaningful_targets"])
        ) if meaningful else 0.0
        called_parts = part.loc[part["calls"] > 0]
        weighted_acc = float(
            np.average(called_parts["direction_accuracy"], weights=called_parts["calls"])
        ) if not called_parts.empty else np.nan
        summary_rows.append({
            "subgroup": subgroup,
            "meaningful_trait_targets": meaningful,
            "direction_calls": calls,
            "coverage": calls / meaningful if meaningful else 0.0,
            "direction_accuracy": weighted_acc,
            "directional_edge": weighted_edge,
            "traits_positive_edge": int((part["directional_edge"] > 0).sum()),
            "traits_negative_edge": int((part["directional_edge"] < 0).sum()),
        })
    summary = pd.DataFrame(summary_rows)

    train_metrics.to_csv(TRAIN_METRICS_PATH, index=False)
    selected.to_csv(SELECTED_RULES_PATH, index=False)
    holdout.to_csv(HOLDOUT_TRAIT_PATH, index=False)
    summary.to_csv(HOLDOUT_SUMMARY_PATH, index=False)

    print("\n" + "=" * 120)
    print("FSR TRAJECTORY DIRECTION — UNTOUCHED HOLDOUT")
    print("=" * 120)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    print("\nTOP POSITIVE HOLDOUT TRAITS")
    show = holdout.sort_values(["subgroup", "directional_edge"], ascending=[True, False])
    print(
        show[[
            "subgroup", "trait", "selected_family", "meaningful_threshold",
            "meaningful_targets", "coverage", "direction_accuracy", "directional_edge",
            "up_accuracy", "down_accuracy",
        ]].groupby("subgroup", group_keys=False).head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        flush=True,
    )

    print(f"\nsplit date: {split.date()}")
    print(f"elapsed: {_elapsed(start)}")
    print(f"wrote: {TRAIN_METRICS_PATH}")
    print(f"wrote: {SELECTED_RULES_PATH}")
    print(f"wrote: {HOLDOUT_TRAIT_PATH}")
    print(f"wrote: {HOLDOUT_SUMMARY_PATH}")
    print("Research only. No age modifier, stored FSR, or simulator state was changed.")


if __name__ == "__main__":
    main()
