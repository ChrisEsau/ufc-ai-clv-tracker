"""Standing-only Event Clock V2 interface experiment.

Compare three arms with common random numbers while leaving TD, ground, control,
submissions, judge logic, damage/KO mechanics, stamina, and all other downstream
fight mechanics frozen:

A. current       — current ECV2 fitted standing free-time rate + fitted landing mix
B. literal_rate  — V3 distance matchup rate + unchanged fitted clinch component;
                   current ECV2 standing landing probability retained
C. literal_full  — V3 distance matchup rate + unchanged fitted clinch component;
                   V3 distance accuracy + fitted clinch accuracy

Important semantic guard:
FSR V3 standing tendency is a DISTANCE-strike rate, while Stage 9's standing
bucket is distance + clinch. Therefore the experiment never replaces the clinch
component with V3 distance semantics. It converts the existing predicted clinch
budget to a free-time rate and adds it to the literal V3 distance rate.
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


def _finite(value, default=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def _clip_probability(value):
    return float(np.clip(_finite(value, 0.0), 0.0, 1.0))


def _standing_interface(pair: pd.DataFrame, pair_info: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fighter-level interface diagnostics and an arm-ready base pair.

    V3 effective_standing_rate is the validated distance-attempt rate per 15m of
    standing/free exposure. Existing clinch prediction remains untouched and is
    converted to the same free-time rate scale using the deterministic predicted
    fight-level control total.
    """
    base = pair.copy()
    duration = float(base["duration"].iloc[0])
    predicted_total_control = float(np.clip(_finite(pair_info.get("pred_total_control"), 0.0), 0.0, duration))
    predicted_free = max(duration - predicted_total_control, 1.0)

    records = []
    for idx, row in base.iterrows():
        current_rate = max(_finite(row.get("pred_standing_rate_free_15m"), 0.0), 0.0)
        distance_rate = max(_finite(row.get("effective_standing_rate"), 0.0), 0.0)

        clinch_attempted = max(_finite(row.get("pred_clinch_attempted"), 0.0), 0.0)
        clinch_landed = max(_finite(row.get("pred_clinch_landed"), 0.0), 0.0)
        clinch_rate = clinch_attempted * 900.0 / predicted_free
        clinch_accuracy = (
            float(np.clip(clinch_landed / clinch_attempted, 0.0, 1.0))
            if clinch_attempted > 1e-12 else 0.0
        )

        v3_distance_accuracy = _clip_probability(row.get("standing_accuracy_matchup"))
        literal_rate = distance_rate + clinch_rate
        literal_accuracy = (
            (distance_rate * v3_distance_accuracy + clinch_rate * clinch_accuracy) / literal_rate
            if literal_rate > 1e-12 else 0.0
        )

        current_attempted = max(_finite(row.get("pred_standing_attempted"), 0.0), 0.0)
        current_landed = max(_finite(row.get("pred_standing_landed"), 0.0), 0.0)
        current_accuracy = (
            float(np.clip(current_landed / current_attempted, 0.0, 1.0))
            if current_attempted > 1e-12 else 0.0
        )

        records.append({
            "fight_id": str(row["fight_id"]),
            "side": str(row["side"]),
            "fighter": str(row["fighter_name"]),
            "opponent": str(row["opponent_name"]),
            "duration": duration,
            "predicted_total_control": predicted_total_control,
            "predicted_free_seconds": predicted_free,
            "current_standing_rate_free_15m": current_rate,
            "v3_literal_distance_rate_15m": distance_rate,
            "preserved_clinch_rate_free_15m": clinch_rate,
            "literal_total_standing_rate_free_15m": literal_rate,
            "rate_change_pct": 100.0 * (literal_rate - current_rate) / max(current_rate, 1e-9),
            "current_standing_accuracy": current_accuracy,
            "v3_distance_accuracy": v3_distance_accuracy,
            "preserved_clinch_accuracy": clinch_accuracy,
            "literal_total_standing_accuracy": literal_accuracy,
            "accuracy_change_pp": 100.0 * (literal_accuracy - current_accuracy),
        })

    return pd.DataFrame(records), base


def _arm_pair(base: pd.DataFrame, interface: pd.DataFrame, arm: str) -> pd.DataFrame:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    out = base.copy()
    lookup = interface.set_index(["fight_id", "side"])

    for idx, row in out.iterrows():
        key = (str(row["fight_id"]), str(row["side"]))
        info = lookup.loc[key]
        if arm in ("literal_rate", "literal_full"):
            out.at[idx, "pred_standing_rate_free_15m"] = float(info["literal_total_standing_rate_free_15m"])
        if arm == "literal_full":
            # Stage 9 consumes only the landed/attempted ratio for standing
            # accuracy. Preserve the attempt anchor and replace that ratio.
            attempt_anchor = max(_finite(row.get("pred_standing_attempted"), 1.0), 1e-9)
            out.at[idx, "pred_standing_attempted"] = attempt_anchor
            out.at[idx, "pred_standing_landed"] = attempt_anchor * float(info["literal_total_standing_accuracy"])
    return out


def _card_metrics(summary: pd.DataFrame) -> dict:
    red_actual = (summary["actual_winner"] == "red").astype(float).to_numpy()
    p_red = np.clip(summary["p_red_win"].to_numpy(float), 1e-9, 1.0 - 1e-9)
    p_actual_winner = np.where(red_actual == 1.0, p_red, 1.0 - p_red)

    joint = []
    for _, row in summary.iterrows():
        joint.append(float(row[f"p_{row['actual_winner']}_{str(row['actual_method']).lower()}"]))

    return {
        "fights": int(len(summary)),
        "ml_accuracy": float(summary["ml_correct"].mean()),
        "method_accuracy": float(summary["method_correct"].mean()),
        "winner_method_accuracy": float(summary["winner_method_correct"].mean()),
        "mean_p_actual_winner": float(np.mean(p_actual_winner)),
        "mean_p_actual_joint": float(np.mean(joint)),
        "brier": float(np.mean((p_red - red_actual) ** 2)),
        "log_loss": float(-np.mean(red_actual * np.log(p_red) + (1.0 - red_actual) * np.log(1.0 - p_red))),
    }


def _print_results(interface: pd.DataFrame, summary: pd.DataFrame, budget_summary: pd.DataFrame):
    print("=" * 145)
    print("EVENT CLOCK V2 — LITERAL V3 STANDING INTERFACE EXPERIMENT")
    print("=" * 145)
    print("Changed subsystem: standing interface ONLY")
    print("V3 distance rate replaces fitted distance translation; fitted clinch component preserved.")
    print("TD / ground / control / submissions / judging / damage / KO / stamina mechanics: frozen")

    print("\n" + "=" * 145)
    print("PRE-MC STANDING INTERFACE MOVEMENT")
    print("=" * 145)
    print(
        f"mean abs total-rate change: {interface['rate_change_pct'].abs().mean():.2f}% | "
        f"median abs: {interface['rate_change_pct'].abs().median():.2f}% | "
        f"mean abs accuracy change: {interface['accuracy_change_pp'].abs().mean():.2f} pp"
    )
    view = interface.sort_values("rate_change_pct", key=lambda x: x.abs(), ascending=False).head(16)
    print(view[[
        "fighter", "opponent", "current_standing_rate_free_15m",
        "v3_literal_distance_rate_15m", "preserved_clinch_rate_free_15m",
        "literal_total_standing_rate_free_15m", "rate_change_pct",
        "current_standing_accuracy", "literal_total_standing_accuracy", "accuracy_change_pp",
    ]].to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    rows = []
    for arm in ARMS:
        part = summary[summary["arm"] == arm]
        rows.append({"arm": arm, **_card_metrics(part)})
    metrics = pd.DataFrame(rows)
    print("\n" + "=" * 145)
    print("CARD OUTCOMES — 2000-PATH COMMON-RANDOM-NUMBER COMPARISON")
    print("=" * 145)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pivot = summary.pivot(index=["fight_id", "red", "blue", "actual_winner", "actual_method"], columns="arm", values="p_red_win").reset_index()
    pivot["literal_rate_delta_pp"] = 100.0 * (pivot["literal_rate"] - pivot["current"])
    pivot["literal_full_delta_pp"] = 100.0 * (pivot["literal_full"] - pivot["current"])
    print("\n" + "=" * 145)
    print("FIGHT-LEVEL RED WIN PROBABILITY MOVEMENT")
    print("=" * 145)
    print(pivot[[
        "red", "blue", "actual_winner", "current", "literal_rate", "literal_full",
        "literal_rate_delta_pp", "literal_full_delta_pp",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 145)
    print("REALIZED STAGE-9 FULL-HORIZON STANDING BUDGETS")
    print("=" * 145)
    print(budget_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return metrics, pivot


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
    mean_pair_lookup = {str(row["fight_id"]): row for _, row in mean_pair.iterrows()}
    master_lookup = {str(row["fight_id"]): row for _, row in master.iterrows()}

    interface_parts = []
    summary_rows = []
    budget_rows = []
    path_rows = []

    groups = list(mean_test.groupby("fight_id", sort=False))
    for fight_index, (fight_id, mean_fighter_pair) in enumerate(groups):
        fight_id = str(fight_id)
        master_row = master_lookup[fight_id]
        pair_info = mean_pair_lookup[fight_id]
        interface, base_pair = _standing_interface(mean_fighter_pair, pair_info)
        interface_parts.append(interface)

        fight = _fight(master_row, context["fsr_all"])
        sub_rate, convert = _submission_inputs(mean_fighter_pair)
        print(f"[{fight_index + 1}/{len(groups)}] {master_row['r_name']} vs {master_row['b_name']}")

        for arm in ARMS:
            pair_arm = _arm_pair(base_pair, interface, arm)
            fight_rows = []
            arm_budget_rows = []
            for path in range(args.paths):
                seed = args.seed + fight_index * 1_000_000 + path
                budgets = _draw_budgets(pair_arm, pair_info, context, np.random.default_rng(seed))
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
                fight_rows.append(result)
                path_rows.append(result)
                arm_budget_rows.append({
                    "fight_id": fight_id,
                    "arm": arm,
                    "red_standing_attempted_budget": budgets["red_standing_attempted"],
                    "red_standing_landed_budget": budgets["red_standing_landed"],
                    "blue_standing_attempted_budget": budgets["blue_standing_attempted"],
                    "blue_standing_landed_budget": budgets["blue_standing_landed"],
                    "free_seconds": budgets["free_seconds"],
                    "total_control": budgets["total_control"],
                })

            s = summarize_fight(fight_id, pair_arm, fight_rows, master_row)
            s["arm"] = arm
            summary_rows.append(s)
            budget_rows.extend(arm_budget_rows)

    interface = pd.concat(interface_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    budget_paths = pd.DataFrame(budget_rows)
    budget_summary = budget_paths.groupby(["fight_id", "arm"], as_index=False).agg(
        red_standing_attempted_budget=("red_standing_attempted_budget", "mean"),
        red_standing_landed_budget=("red_standing_landed_budget", "mean"),
        blue_standing_attempted_budget=("blue_standing_attempted_budget", "mean"),
        blue_standing_landed_budget=("blue_standing_landed_budget", "mean"),
        free_seconds=("free_seconds", "mean"),
        total_control=("total_control", "mean"),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = pd.Timestamp(args.event_date).strftime("%Y_%m_%d")
    interface_path = args.out_dir / f"{stem}_standing_interface.csv"
    summary_path = args.out_dir / f"{stem}_standing_literal_{args.paths}paths_summary.csv"
    budget_path = args.out_dir / f"{stem}_standing_literal_{args.paths}paths_budget_summary.csv"
    paths_path = args.out_dir / f"{stem}_standing_literal_{args.paths}paths_paths.csv"
    interface.to_csv(interface_path, index=False)
    summary.to_csv(summary_path, index=False)
    budget_summary.to_csv(budget_path, index=False)
    pd.DataFrame(path_rows).to_csv(paths_path, index=False)

    metrics, pivot = _print_results(interface, summary, budget_summary)
    metrics_path = args.out_dir / f"{stem}_standing_literal_{args.paths}paths_card_metrics.csv"
    movement_path = args.out_dir / f"{stem}_standing_literal_{args.paths}paths_ml_movement.csv"
    metrics.to_csv(metrics_path, index=False)
    pivot.to_csv(movement_path, index=False)

    print("\noutputs:")
    for path in (interface_path, summary_path, budget_path, metrics_path, movement_path, paths_path):
        print(path)


if __name__ == "__main__":
    main()
