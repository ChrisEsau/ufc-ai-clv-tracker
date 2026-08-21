"""Measurement-only FSR V2 -> V3 / ECV1 -> ECV2 interface audit.

Compares deterministic pre-Monte-Carlo fighter inputs and direct predicted
budgets for one historical event. It does not execute or alter fight mechanics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.frozen_inference import predict_target
from pipeline.simulation.event_clock_mc_v2.inference import predict_target_v3

V1_BUNDLE = Path("data/models/event_clock_mc_v1/event_clock_frozen_bundle.joblib")
V2_BUNDLE = Path("data/models/event_clock_mc_v2/event_clock_v2_fsr_v3_bundle.joblib")
OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/interface_audit")

NATIVE_TRAITS = (
    "standing_striking_tendency", "standing_striking_suppression",
    "standing_striking_offense", "standing_striking_defense",
    "standing_accuracy_baseline", "takedown_tendency", "takedown_suppression",
    "takedown_offense", "takedown_defense", "takedown_completion_baseline",
    "ground_striking_tendency", "ground_striking_suppression",
    "ground_striking_offense", "ground_accuracy_baseline",
)
TRANSFORMS = (
    "effective_standing_rate", "standing_accuracy_matchup",
    "effective_td_rate", "td_completion_matchup", "successful_td_pressure",
    "control_pressure", "retention_mean_base", "effective_ground_rate",
    "ground_accuracy_matchup",
)
BUDGETS = (
    "pred_standing_attempted", "pred_standing_landed",
    "pred_distance_attempted", "pred_distance_landed",
    "pred_clinch_attempted", "pred_clinch_landed",
    "pred_td_attempted", "pred_td_landed",
    "pred_qualified_control_inflicted_seconds",
    "pred_ground_attempted", "pred_ground_landed",
    "pred_standing_rate_free_15m", "submission_clock_rate",
    "submission_conversion_probability",
)


def num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def pct(new, old):
    new, old = num(new), num(old)
    if not np.isfinite(new) or not np.isfinite(old) or abs(old) <= 1e-12:
        return np.nan
    return 100.0 * (new - old) / abs(old)


def pp(new, old):
    new, old = num(new), num(old)
    if not np.isfinite(new) or not np.isfinite(old):
        return np.nan
    return 100.0 * (new - old)


def load_context(path: Path, schema: int):
    if not path.exists():
        raise RuntimeError(f"Missing required bundle: {path}")
    payload = joblib.load(path)
    if payload.get("schema_version") != schema:
        raise RuntimeError(
            f"{path} schema={payload.get('schema_version')!r}; expected {schema}"
        )
    return payload["context"]


def event_master(event_date):
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="raise").dt.normalize()
    target = master[master["event_date"].eq(event_date)].copy()
    if target.empty:
        raise RuntimeError(f"No fights found for {event_date.date()}")
    return target.reset_index(drop=True)


def values(frame, lookup, keys, field):
    return [num(lookup.loc[(fid, side)].get(field)) for fid, side in frame[keys].itertuples(index=False, name=None)]


def build_detail(v1, v3):
    keys = ["fight_id", "side"]
    v1, v3 = v1.copy(), v3.copy()
    v1["fight_id"] = v1["fight_id"].astype(str)
    v3["fight_id"] = v3["fight_id"].astype(str)
    detail = v1[keys + ["fighter_name", "opponent_name"]].merge(
        v3[keys + ["fighter_name", "opponent_name"]],
        on=keys, suffixes=("_v1", "_v3"), validate="one_to_one",
    )
    if not (
        detail["fighter_name_v1"].eq(detail["fighter_name_v3"]).all()
        and detail["opponent_name_v1"].eq(detail["opponent_name_v3"]).all()
    ):
        raise RuntimeError("V1/V3 fighter identity mismatch")
    detail = detail.rename(columns={"fighter_name_v1": "fighter", "opponent_name_v1": "opponent"})
    detail = detail.drop(columns=["fighter_name_v3", "opponent_name_v3"])
    old, new = v1.set_index(keys), v3.set_index(keys)

    for trait in NATIVE_TRAITS:
        field = f"self_{trait}"
        detail[f"v1_{trait}"] = values(detail, old, keys, field)
        detail[f"v3_{trait}"] = values(detail, new, keys, field)
        detail[f"delta_pct_{trait}"] = [pct(n, o) for n, o in zip(detail[f"v3_{trait}"], detail[f"v1_{trait}"])]

    # V3 ground is structurally different: burst + own-control slope. Do not
    # force these into a fake apples-to-apples V1 rate comparison.
    for field in (
        "self_ground_striking_burst_baseline",
        "self_ground_striking_population_slope_15m",
        "ground_burst_attempts",
    ):
        detail[f"v3_{field}"] = values(detail, new, keys, field)
    detail["v1_ground_striking_defense"] = values(detail, old, keys, "self_ground_striking_defense")

    for field in TRANSFORMS + BUDGETS:
        detail[f"v1_{field}"] = values(detail, old, keys, field)
        detail[f"v3_{field}"] = values(detail, new, keys, field)
        detail[f"delta_pct_{field}"] = [pct(n, o) for n, o in zip(detail[f"v3_{field}"], detail[f"v1_{field}"])]

    for field in (
        "standing_accuracy_matchup", "td_completion_matchup",
        "ground_accuracy_matchup", "submission_conversion_probability",
    ):
        detail[f"delta_pp_{field}"] = [pp(n, o) for n, o in zip(detail[f"v3_{field}"], detail[f"v1_{field}"])]
    return detail


def metric_summary(detail):
    rows = []
    for field in TRANSFORMS + BUDGETS:
        old = pd.to_numeric(detail[f"v1_{field}"], errors="coerce")
        new = pd.to_numeric(detail[f"v3_{field}"], errors="coerce")
        d = pd.to_numeric(detail[f"delta_pct_{field}"], errors="coerce").abs()
        rows.append({
            "metric": field,
            "v1_mean": float(old.mean()),
            "v3_mean": float(new.mean()),
            "mean_abs_pct_change": float(d.mean()),
            "median_abs_pct_change": float(d.median()),
            "correlation_v1_v3": float(old.corr(new)),
        })
    return pd.DataFrame(rows)


def mean_abs_pct(detail, field):
    return float(pd.to_numeric(detail[f"delta_pct_{field}"], errors="coerce").abs().mean())


def mean_abs_pp(detail, field):
    return float(pd.to_numeric(detail[f"delta_pp_{field}"], errors="coerce").abs().mean())


def print_summary(detail):
    rows = [
        ("Standing", mean_abs_pct(detail, "effective_standing_rate"), mean_abs_pp(detail, "standing_accuracy_matchup"), mean_abs_pct(detail, "pred_standing_attempted"), mean_abs_pct(detail, "pred_standing_landed")),
        ("Takedown", mean_abs_pct(detail, "effective_td_rate"), mean_abs_pp(detail, "td_completion_matchup"), mean_abs_pct(detail, "pred_td_attempted"), mean_abs_pct(detail, "pred_td_landed")),
        ("Ground*", np.nan, mean_abs_pp(detail, "ground_accuracy_matchup"), mean_abs_pct(detail, "pred_ground_attempted"), mean_abs_pct(detail, "pred_ground_landed")),
    ]
    frame = pd.DataFrame(rows, columns=[
        "family", "mean_abs_rate_change_pct", "mean_abs_accuracy_change_pp",
        "mean_abs_attempt_budget_change_pct", "mean_abs_landed_budget_change_pct",
    ])
    print("=" * 118)
    print("FSR V2 -> V3 SIGNAL CHANGE VS ECV1 -> ECV2 BUDGET CHANGE")
    print("=" * 118)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    print("\n* Ground rates are not directly comparable: V1 subtractive rate vs V3 burst + own-control slope.")
    print(
        f"Control pressure mean abs change: {mean_abs_pct(detail, 'control_pressure'):.2f}% | "
        f"predicted control-seconds mean abs change: {mean_abs_pct(detail, 'pred_qualified_control_inflicted_seconds'):.2f}%"
    )

    fields = [
        "pred_standing_attempted", "pred_td_attempted",
        "pred_qualified_control_inflicted_seconds", "pred_ground_attempted",
    ]
    view = detail[["fighter", "opponent"]].copy()
    for field in fields:
        view[field] = detail[f"delta_pct_{field}"]
    view["max_abs_budget_change_pct"] = view[fields].abs().max(axis=1)
    view = view.sort_values("max_abs_budget_change_pct", ascending=False).head(12)
    print("\n" + "=" * 118)
    print("LARGEST FIGHTER-LEVEL PRE-MC BUDGET MOVES (% V3 vs V1)")
    print("=" * 118)
    print(view[["fighter", "opponent"] + fields].to_string(index=False, float_format=lambda x: f"{x:8.2f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-date", default="2026-07-25")
    parser.add_argument("--v1-bundle", type=Path, default=V1_BUNDLE)
    parser.add_argument("--v2-bundle", type=Path, default=V2_BUNDLE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    date = pd.Timestamp(args.event_date).normalize()
    target = event_master(date)
    v1 = load_context(args.v1_bundle, 2)
    v2 = load_context(args.v2_bundle, 3)
    fsr_v3 = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    fsr_v3["fight_id"] = fsr_v3["fight_id"].astype(str)
    fsr_v3["event_date"] = pd.to_datetime(fsr_v3["event_date"], errors="raise").dt.normalize()

    pred1, _ = predict_target(target, v1["fsr_all"], v1["inference_models"], v1["submission_scale"], v1["conversion_offset"])
    pred3, _ = predict_target_v3(target, fsr_v3, v2["inference_models"], v2["submission_scale"], v2["conversion_offset"])
    if pred1["fight_id"].nunique() != len(target) or pred3["fight_id"].nunique() != len(target):
        raise RuntimeError("Full event was not covered by both interfaces")

    detail = build_detail(pred1, pred3)
    summary = metric_summary(detail)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = date.strftime("%Y_%m_%d")
    detail_path = args.out_dir / f"{stem}_fsr_v2_v3_interface_detail.csv"
    summary_path = args.out_dir / f"{stem}_fsr_v2_v3_interface_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("=" * 118)
    print("EVENT CLOCK FSR V2 -> V3 INTERFACE AUDIT")
    print("=" * 118)
    print(f"event date: {date.date()} | fights: {len(target)} | fighter rows: {len(detail)}")
    print("fight mechanics executed: NO | Monte Carlo paths executed: NO")
    print_summary(detail)
    print(f"\ndetail CSV:  {detail_path}")
    print(f"summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
