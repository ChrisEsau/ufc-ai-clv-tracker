"""Standing-only Event Clock V2 interface experiment.

Three common-random-number arms:
A. current       — current ECV2 fitted standing rate + fitted landing mix.
B. literal_rate  — literal V3 distance rate + preserved fitted clinch rate;
                   current ECV2 aggregate standing landing probability.
C. literal_full  — literal V3 distance rate + preserved fitted clinch rate;
                   V3 distance accuracy + preserved fitted clinch accuracy.

Only the four Stage-9 standing budget fields are allowed to differ across arms.
For every path the current full budget is drawn once; alternate Stage-9 draws are
used only to replace red/blue standing attempted/landed. TD, ground, control,
submission inputs and every downstream fight mechanic are therefore identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import (
    DEFAULT_PATHS,
    SEED,
    select_target,
    simulate_detailed_path,
    summarize_fight,
)
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots
from pipeline.simulation.event_clock_mc_v2.inference import predict_target_v3
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET,
    _draw_budgets,
    _submission_inputs,
    load_frozen_context,
)

OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/standing_literal_interface")
ARMS = ("current", "literal_rate", "literal_full")
STANDING_BUDGET_FIELDS = (
    "red_standing_attempted",
    "red_standing_landed",
    "blue_standing_attempted",
    "blue_standing_landed",
)


def finite(value, default=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def standing_interface(pair: pd.DataFrame, pair_info: pd.Series) -> pd.DataFrame:
    """Build literal V3 distance + preserved fitted-clinch standing semantics."""
    duration = float(pair["duration"].iloc[0])
    predicted_control = float(np.clip(finite(pair_info.get("pred_total_control")), 0.0, duration))
    predicted_free = max(duration - predicted_control, 1.0)
    records = []

    for _, row in pair.iterrows():
        current_rate = max(finite(row.get("pred_standing_rate_free_15m")), 0.0)
        v3_distance_rate = max(finite(row.get("effective_standing_rate")), 0.0)

        clinch_attempted = max(finite(row.get("pred_clinch_attempted")), 0.0)
        clinch_landed = max(finite(row.get("pred_clinch_landed")), 0.0)
        clinch_rate = clinch_attempted * 900.0 / predicted_free
        clinch_accuracy = (
            float(np.clip(clinch_landed / clinch_attempted, 0.0, 1.0))
            if clinch_attempted > 1e-12 else 0.0
        )

        current_attempted = max(finite(row.get("pred_standing_attempted")), 0.0)
        current_landed = max(finite(row.get("pred_standing_landed")), 0.0)
        current_accuracy = (
            float(np.clip(current_landed / current_attempted, 0.0, 1.0))
            if current_attempted > 1e-12 else 0.0
        )
        v3_distance_accuracy = float(np.clip(finite(row.get("standing_accuracy_matchup")), 0.0, 1.0))

        literal_rate = v3_distance_rate + clinch_rate
        literal_accuracy = (
            (v3_distance_rate * v3_distance_accuracy + clinch_rate * clinch_accuracy) / literal_rate
            if literal_rate > 1e-12 else 0.0
        )
        records.append({
            "fight_id": str(row["fight_id"]),
            "side": str(row["side"]),
            "fighter": str(row["fighter_name"]),
            "opponent": str(row["opponent_name"]),
            "predicted_free_seconds": predicted_free,
            "current_standing_rate_free_15m": current_rate,
            "v3_literal_distance_rate_15m": v3_distance_rate,
            "preserved_clinch_rate_free_15m": clinch_rate,
            "literal_total_standing_rate_free_15m": literal_rate,
            "rate_change_pct": 100.0 * (literal_rate - current_rate) / max(current_rate, 1e-9),
            "current_standing_accuracy": current_accuracy,
            "v3_distance_accuracy": v3_distance_accuracy,
            "preserved_clinch_accuracy": clinch_accuracy,
            "literal_total_standing_accuracy": literal_accuracy,
            "accuracy_change_pp": 100.0 * (literal_accuracy - current_accuracy),
        })
    return pd.DataFrame(records)


def make_arm_pair(base: pd.DataFrame, interface: pd.DataFrame, arm: str) -> pd.DataFrame:
    out = base.copy()
    lookup = interface.set_index(["fight_id", "side"])
    for idx, row in out.iterrows():
        info = lookup.loc[(str(row["fight_id"]), str(row["side"]))]
        if arm in ("literal_rate", "literal_full"):
            out.at[idx, "pred_standing_rate_free_15m"] = float(info["literal_total_standing_rate_free_15m"])
        if arm == "literal_full":
            anchor = max(finite(row.get("pred_standing_attempted"), 1.0), 1e-9)
            out.at[idx, "pred_standing_attempted"] = anchor
            out.at[idx, "pred_standing_landed"] = anchor * float(info["literal_total_standing_accuracy"])
    return out


def standing_only_budget(base_budget: dict, candidate_budget: dict) -> dict:
    out = dict(base_budget)
    for key in STANDING_BUDGET_FIELDS:
        out[key] = candidate_budget[key]
    return out


def card_metrics(summary: pd.DataFrame) -> dict:
    red_actual = (summary["actual_winner"] == "red").astype(float).to_numpy()
    p_red = np.clip(summary["p_red_win"].to_numpy(float), 1e-9, 1.0 - 1e-9)
    p_actual = np.where(red_actual == 1.0, p_red, 1.0 - p_red)
    joint = [
        float(row[f"p_{row['actual_winner']}_{str(row['actual_method']).lower()}"])
        for _, row in summary.iterrows()
    ]
    return {
        "fights": len(summary),
        "ml_accuracy": float(summary["ml_correct"].mean()),
        "method_accuracy": float(summary["method_correct"].mean()),
        "winner_method_accuracy": float(summary["winner_method_correct"].mean()),
        "mean_p_actual_winner": float(np.mean(p_actual)),
        "mean_p_actual_joint": float(np.mean(joint)),
        "brier": float(np.mean((p_red - red_actual) ** 2)),
        "log_loss": float(-np.mean(red_actual * np.log(p_red) + (1.0 - red_actual) * np.log(1.0 - p_red))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-date", default="2026-07-25")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="raise").dt.normalize()

    class TargetArgs:
        event = None
        fight_id = None
        fighter = None
        opponent = None
        event_date = args.event_date

    target = select_target(master, TargetArgs())
    context = load_frozen_context(args.bundle)
    fsr_v3 = load_prefight_snapshots(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    mean_test, mean_pair = predict_target_v3(
        target,
        fsr_v3,
        context["inference_models"],
        context["submission_scale"],
        context["conversion_offset"],
    )
    pair_info_lookup = {str(row["fight_id"]): row for _, row in mean_pair.iterrows()}
    master_lookup = {str(row["fight_id"]): row for _, row in master.iterrows()}

    all_interface = []
    summary_rows = []
    budget_rows = []
    path_rows = []

    groups = list(mean_test.groupby("fight_id", sort=False))
    for fight_index, (fight_id, base_pair) in enumerate(groups):
        fight_id = str(fight_id)
        master_row = master_lookup[fight_id]
        pair_info = pair_info_lookup[fight_id]
        interface = standing_interface(base_pair, pair_info)
        all_interface.append(interface)
        arm_pairs = {arm: make_arm_pair(base_pair, interface, arm) for arm in ARMS}
        arm_results = {arm: [] for arm in ARMS}

        fight = _fight(master_row, context["fsr_all"])
        sub_rate, convert = _submission_inputs(base_pair)
        print(f"[{fight_index + 1}/{len(groups)}] {master_row['r_name']} vs {master_row['b_name']}")

        for path in range(args.paths):
            seed = args.seed + fight_index * 1_000_000 + path
            base_budget = _draw_budgets(
                arm_pairs["current"], pair_info, context, np.random.default_rng(seed)
            )
            budgets_by_arm = {"current": base_budget}
            for arm in ("literal_rate", "literal_full"):
                candidate = _draw_budgets(
                    arm_pairs[arm], pair_info, context, np.random.default_rng(seed)
                )
                budgets_by_arm[arm] = standing_only_budget(base_budget, candidate)

            for arm in ("literal_rate", "literal_full"):
                for key, value in base_budget.items():
                    if key in STANDING_BUDGET_FIELDS:
                        continue
                    other = budgets_by_arm[arm][key]
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        assert np.isclose(float(value), float(other), rtol=0.0, atol=0.0, equal_nan=True), key
                    else:
                        assert value == other, key

            for arm in ARMS:
                budgets = budgets_by_arm[arm]
                result = simulate_detailed_path(
                    fight,
                    budgets,
                    sub_rate,
                    convert,
                    context["judge_model"],
                    context["judge_features"],
                    seed + DETAILED_PATH_SEED_OFFSET,
                )
                result.update({"fight_id": fight_id, "path": path, "arm": arm})
                arm_results[arm].append(result)
                path_rows.append(result)
                budget_rows.append({
                    "fight_id": fight_id,
                    "arm": arm,
                    "red_standing_attempted": budgets["red_standing_attempted"],
                    "red_standing_landed": budgets["red_standing_landed"],
                    "blue_standing_attempted": budgets["blue_standing_attempted"],
                    "blue_standing_landed": budgets["blue_standing_landed"],
                    "free_seconds": budgets["free_seconds"],
                    "total_control": budgets["total_control"],
                })

        for arm in ARMS:
            row = summarize_fight(fight_id, arm_pairs[arm], arm_results[arm], master_row)
            row["arm"] = arm
            summary_rows.append(row)

    interface = pd.concat(all_interface, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    budget_paths = pd.DataFrame(budget_rows)
    budget_summary = budget_paths.groupby(["fight_id", "arm"], as_index=False).mean(numeric_only=True)

    metric_rows = [{"arm": arm, **card_metrics(summary[summary["arm"] == arm])} for arm in ARMS]
    metrics = pd.DataFrame(metric_rows)
    movement = summary.pivot(
        index=["fight_id", "red", "blue", "actual_winner", "actual_method"],
        columns="arm",
        values="p_red_win",
    ).reset_index()
    movement["literal_rate_delta_pp"] = 100.0 * (movement["literal_rate"] - movement["current"])
    movement["literal_full_delta_pp"] = 100.0 * (movement["literal_full"] - movement["current"])

    print("=" * 145)
    print("EVENT CLOCK V2 — LITERAL V3 STANDING INTERFACE EXPERIMENT")
    print("=" * 145)
    print(f"event date: {args.event_date} | fights: {len(groups)} | paths/fight/arm: {args.paths}")
    print("Only standing Stage-9 budget fields differ across arms: VERIFIED")
    print("V3 distance semantics + preserved ECV2 clinch component")
    print(
        f"mean abs pre-MC total standing-rate change: {interface['rate_change_pct'].abs().mean():.2f}% | "
        f"median: {interface['rate_change_pct'].abs().median():.2f}% | "
        f"mean abs accuracy change: {interface['accuracy_change_pp'].abs().mean():.2f} pp"
    )
    print("\nCARD METRICS")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFIGHT-LEVEL P(RED WIN)")
    print(movement[[
        "red", "blue", "actual_winner", "current", "literal_rate", "literal_full",
        "literal_rate_delta_pp", "literal_full_delta_pp",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    largest = interface.sort_values("rate_change_pct", key=lambda s: s.abs(), ascending=False).head(16)
    print("\nLARGEST PRE-MC STANDING RATE MOVES")
    print(largest[[
        "fighter", "opponent", "current_standing_rate_free_15m",
        "v3_literal_distance_rate_15m", "preserved_clinch_rate_free_15m",
        "literal_total_standing_rate_free_15m", "rate_change_pct",
        "current_standing_accuracy", "literal_total_standing_accuracy", "accuracy_change_pp",
    ]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = pd.Timestamp(args.event_date).strftime("%Y_%m_%d")
    outputs = {
        "interface": args.out_dir / f"{stem}_standing_interface.csv",
        "summary": args.out_dir / f"{stem}_standing_literal_{args.paths}paths_summary.csv",
        "budgets": args.out_dir / f"{stem}_standing_literal_{args.paths}paths_budget_summary.csv",
        "metrics": args.out_dir / f"{stem}_standing_literal_{args.paths}paths_card_metrics.csv",
        "movement": args.out_dir / f"{stem}_standing_literal_{args.paths}paths_ml_movement.csv",
        "paths": args.out_dir / f"{stem}_standing_literal_{args.paths}paths_paths.csv",
    }
    interface.to_csv(outputs["interface"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    budget_summary.to_csv(outputs["budgets"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    movement.to_csv(outputs["movement"], index=False)
    pd.DataFrame(path_rows).to_csv(outputs["paths"], index=False)
    print("\noutputs:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
