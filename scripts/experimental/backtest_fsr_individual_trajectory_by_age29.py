"""Age-subgroup audit of fighter-specific next-FSR trajectory forecasts.

Research/shadow only. Age is used ONLY as a subgroup filter; no age modifier is
added to FSR and no simulator/config state is changed.

Hypothesis
----------
Fighter-specific trajectory forecasts may be more useful for developing fighters
under age 29 than for older/more established fighters.

This script reuses the leakage-safe forecasts already created by
``backtest_fsr_individual_trajectory_forecasts.py``. It does not rebuild the
2.5M+ candidate forecasts. Target-fight age is attached from the master fight
artifact, then candidate-rule selection is performed independently within:

- age_lt_29
- age_ge_29

For each subgroup, earlier chronological targets select the best method for each
``trait x history_bucket`` and later targets evaluate that frozen rule. The
latest-FSR carry-forward remains the baseline in every bucket.

Outputs
-------
data/experimental/fsr_individual_trajectory_age29/
  subgroup_method_metrics_train.csv
  subgroup_selected_rules.csv
  subgroup_selected_rules_holdout.csv
  subgroup_trait_holdout_summary.csv

No age modifier is applied. No stored FSR or simulator state is changed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_age_adjustment_kd_durability_controlled_2020plus_mature as age_study


FORECASTS_PATH = Path(
    "data/experimental/fsr_individual_trajectory_backtest/all_forecasts.parquet"
)
MASTER_PATH = age_study.MASTER_PATH
OUTPUT_DIR = Path("data/experimental/fsr_individual_trajectory_age29")
TRAIN_METRICS_PATH = OUTPUT_DIR / "subgroup_method_metrics_train.csv"
SELECTED_RULES_PATH = OUTPUT_DIR / "subgroup_selected_rules.csv"
HOLDOUT_RULES_PATH = OUTPUT_DIR / "subgroup_selected_rules_holdout.csv"
SUMMARY_PATH = OUTPUT_DIR / "subgroup_trait_holdout_summary.csv"

TRAIN_FRACTION = 0.80
AGE_CUTOFF = 29.0


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _load_ages() -> pd.DataFrame:
    print(f"[trajectory-age29] loading master ages: {MASTER_PATH}", flush=True)
    master = pd.read_parquet(MASTER_PATH).copy()
    date_col = age_study.modern._resolve_date_column(master)
    master[date_col] = pd.to_datetime(master[date_col], errors="coerce")
    master = master.dropna(subset=[date_col]).copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["r_id"] = master["r_id"].astype(str)
    master["b_id"] = master["b_id"].astype(str)
    master["r_age_calc"] = age_study._resolve_corner_age(master, "r")
    master["b_age_calc"] = age_study._resolve_corner_age(master, "b")

    red = master[["fight_id", "r_id", "r_age_calc"]].rename(
        columns={"r_id": "fighter_id", "r_age_calc": "target_age"}
    )
    blue = master[["fight_id", "b_id", "b_age_calc"]].rename(
        columns={"b_id": "fighter_id", "b_age_calc": "target_age"}
    )
    ages = pd.concat([red, blue], ignore_index=True)
    ages["fighter_id"] = ages["fighter_id"].astype(str)
    ages["fight_id"] = ages["fight_id"].astype(str)
    ages["target_age"] = pd.to_numeric(ages["target_age"], errors="coerce")
    ages = ages.dropna(subset=["target_age"]).drop_duplicates(
        ["fight_id", "fighter_id"], keep="last"
    )
    return ages


def _load_forecasts() -> pd.DataFrame:
    if not FORECASTS_PATH.exists():
        raise FileNotFoundError(
            f"missing {FORECASTS_PATH}; run backtest_fsr_individual_trajectory_forecasts.py first"
        )
    print(f"[trajectory-age29] loading existing forecasts: {FORECASTS_PATH}", flush=True)
    fc = pd.read_parquet(FORECASTS_PATH).copy()
    fc["fighter_id"] = fc["fighter_id"].astype(str)
    fc["target_fight_id"] = fc["target_fight_id"].astype(str)
    fc["target_date"] = pd.to_datetime(fc["target_date"], errors="coerce")
    fc = fc.dropna(subset=["target_date"]).copy()
    print(
        f"[trajectory-age29] forecast rows={len(fc):,} | "
        f"target fighter-fights={fc[['fighter_id','target_fight_id']].drop_duplicates().shape[0]:,}",
        flush=True,
    )
    return fc


def _global_split_date(fc: pd.DataFrame) -> pd.Timestamp:
    dates = np.sort(fc["target_date"].dropna().unique())
    if len(dates) < 5:
        raise RuntimeError("not enough target dates for chronological split")
    idx = max(1, int(len(dates) * TRAIN_FRACTION)) - 1
    return pd.Timestamp(dates[idx])


def _method_complexity(method: str) -> tuple[int, float]:
    if method == "latest":
        return (0, 0.0)
    family = method.split("_a", 1)[0]
    alpha = float(method.split("_a", 1)[1])
    order = {"linear": 1, "quadratic": 2, "cubic": 3}[family]
    return (order, alpha)


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": int(len(frame)),
        "rmse": float(np.sqrt(frame["sq_error"].mean())),
        "mae": float(frame["abs_error"].mean()),
        "mean_error": float(frame["error"].mean()),
        "mean_abs_forecast_change": float(frame["forecast_change"].abs().mean()),
    }


def _analyze_subgroup(
    frame: pd.DataFrame,
    subgroup: str,
    split_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["target_date"] <= split_date].copy()
    holdout = frame.loc[frame["target_date"] > split_date].copy()
    print(
        f"[trajectory-age29] {subgroup}: train rows={len(train):,} | "
        f"holdout rows={len(holdout):,} | target fighter-fights="
        f"{frame[['fighter_id','target_fight_id']].drop_duplicates().shape[0]:,}",
        flush=True,
    )

    metric_rows = []
    for (trait, bucket, method), part in train.groupby(
        ["trait", "history_bucket", "method"], sort=False
    ):
        row = {
            "subgroup": subgroup,
            "trait": trait,
            "history_bucket": bucket,
            "method": method,
        }
        row.update(_metrics(part))
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    selected_rows = []
    for (trait, bucket), part in metrics.groupby(["trait", "history_bucket"], sort=False):
        latest = part.loc[part["method"].eq("latest")]
        if latest.empty:
            continue
        best_rmse = float(part["rmse"].min())
        eligible = part.loc[part["rmse"] <= best_rmse * 1.0025].copy()
        eligible["complexity"] = eligible["method"].map(_method_complexity)
        winner = eligible.sort_values("complexity").iloc[0]
        baseline = latest.iloc[0]
        selected_rows.append({
            "subgroup": subgroup,
            "trait": trait,
            "history_bucket": bucket,
            "selected_method": str(winner["method"]),
            "train_rows": int(winner["rows"]),
            "train_rmse": float(winner["rmse"]),
            "baseline_train_rmse": float(baseline["rmse"]),
            "train_relative_rmse_improvement_pct": (
                100.0 * (float(baseline["rmse"]) - float(winner["rmse"])) / float(baseline["rmse"])
                if float(baseline["rmse"]) > 0 else 0.0
            ),
        })
    selected = pd.DataFrame(selected_rows)

    holdout_rows = []
    for rule in selected.itertuples(index=False):
        cell = holdout.loc[
            holdout["trait"].eq(rule.trait)
            & holdout["history_bucket"].eq(rule.history_bucket)
        ]
        chosen = cell.loc[cell["method"].eq(rule.selected_method)]
        baseline = cell.loc[cell["method"].eq("latest")]
        if chosen.empty or baseline.empty:
            continue
        cm = _metrics(chosen)
        bm = _metrics(baseline)
        holdout_rows.append({
            "subgroup": subgroup,
            "trait": rule.trait,
            "history_bucket": rule.history_bucket,
            "selected_method": rule.selected_method,
            "holdout_rows": int(cm["rows"]),
            "holdout_rmse": float(cm["rmse"]),
            "baseline_holdout_rmse": float(bm["rmse"]),
            "holdout_relative_rmse_improvement_pct": (
                100.0 * (float(bm["rmse"]) - float(cm["rmse"])) / float(bm["rmse"])
                if float(bm["rmse"]) > 0 else 0.0
            ),
            "beats_baseline": int(float(cm["rmse"]) < float(bm["rmse"])),
        })
    holdout_rules = pd.DataFrame(holdout_rows)

    summary_rows = []
    for trait, rules in holdout_rules.groupby("trait", sort=False):
        chosen_parts = []
        base_parts = []
        for rule in rules.itertuples(index=False):
            cell = holdout.loc[
                holdout["trait"].eq(trait)
                & holdout["history_bucket"].eq(rule.history_bucket)
            ]
            chosen_parts.append(cell.loc[cell["method"].eq(rule.selected_method)])
            base_parts.append(cell.loc[cell["method"].eq("latest")])
        chosen = pd.concat(chosen_parts, ignore_index=True)
        baseline = pd.concat(base_parts, ignore_index=True)
        cm = _metrics(chosen)
        bm = _metrics(baseline)
        methods = "; ".join(
            f"{r.history_bucket}:{r.selected_method}" for r in rules.itertuples(index=False)
        )
        summary_rows.append({
            "subgroup": subgroup,
            "trait": trait,
            "holdout_targets": int(cm["rows"]),
            "selected_policy_rmse": float(cm["rmse"]),
            "latest_baseline_rmse": float(bm["rmse"]),
            "rmse_improvement": float(bm["rmse"] - cm["rmse"]),
            "relative_rmse_improvement_pct": (
                100.0 * (float(bm["rmse"]) - float(cm["rmse"])) / float(bm["rmse"])
                if float(bm["rmse"]) > 0 else 0.0
            ),
            "buckets_beating_baseline": int(rules["beats_baseline"].sum()),
            "buckets_tested": int(len(rules)),
            "selected_methods": methods,
            "beats_baseline": int(float(cm["rmse"]) < float(bm["rmse"])),
        })
    summary = pd.DataFrame(summary_rows)
    return metrics, selected, holdout_rules, summary


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fc = _load_forecasts()
    ages = _load_ages()
    fc = fc.merge(
        ages.rename(columns={"fight_id": "target_fight_id"}),
        on=["target_fight_id", "fighter_id"],
        how="inner",
        validate="many_to_one",
    )
    if fc.empty:
        raise RuntimeError("age merge produced no forecast rows")
    fc["age_subgroup"] = np.where(fc["target_age"] < AGE_CUTOFF, "age_lt_29", "age_ge_29")

    split_date = _global_split_date(fc)
    print(
        f"[trajectory-age29] fixed cutoff={AGE_CUTOFF:.0f} | global split={split_date.date()} | "
        "age is FILTER ONLY; no age FSR adjustment",
        flush=True,
    )

    metric_parts = []
    selected_parts = []
    holdout_parts = []
    summary_parts = []
    for subgroup in ("age_lt_29", "age_ge_29"):
        part = fc.loc[fc["age_subgroup"].eq(subgroup)].copy()
        metrics, selected, holdout, summary = _analyze_subgroup(part, subgroup, split_date)
        metric_parts.append(metrics)
        selected_parts.append(selected)
        holdout_parts.append(holdout)
        summary_parts.append(summary)

    metrics_all = pd.concat(metric_parts, ignore_index=True)
    selected_all = pd.concat(selected_parts, ignore_index=True)
    holdout_all = pd.concat(holdout_parts, ignore_index=True)
    summary_all = pd.concat(summary_parts, ignore_index=True)

    metrics_all.to_csv(TRAIN_METRICS_PATH, index=False)
    selected_all.to_csv(SELECTED_RULES_PATH, index=False)
    holdout_all.to_csv(HOLDOUT_RULES_PATH, index=False)
    summary_all.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 180)
    print("INDIVIDUAL FSR TRAJECTORY BY AGE <29 vs >=29 — UNTOUCHED HOLDOUT")
    print("=" * 180)
    display = summary_all.sort_values(
        ["subgroup", "relative_rmse_improvement_pct"], ascending=[True, False]
    )
    print(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    print("\nSUBGROUP SCORECARD")
    for subgroup, part in summary_all.groupby("subgroup", sort=False):
        improving = int(part["beats_baseline"].sum())
        mean_imp = float(part["relative_rmse_improvement_pct"].mean())
        median_imp = float(part["relative_rmse_improvement_pct"].median())
        targets = fc.loc[fc["age_subgroup"].eq(subgroup), ["fighter_id", "target_fight_id"]].drop_duplicates()
        print(
            f"{subgroup}: target fighter-fights={len(targets):,} | "
            f"traits beating baseline={improving}/{len(part)} | "
            f"mean trait improvement={mean_imp:+.3f}% | median={median_imp:+.3f}%",
            flush=True,
        )

    print(f"split date: {split_date.date()}")
    print(f"elapsed: {_elapsed(start)}")
    print(f"wrote: {TRAIN_METRICS_PATH}")
    print(f"wrote: {SELECTED_RULES_PATH}")
    print(f"wrote: {HOLDOUT_RULES_PATH}")
    print(f"wrote: {SUMMARY_PATH}")
    print("Research only. Age is a subgroup filter only; no age modifier was applied.")


if __name__ == "__main__":
    main()
