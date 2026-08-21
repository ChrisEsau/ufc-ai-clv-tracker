"""Ground-only Event Clock V2 literal FSR V3 interface experiment.

Three arms with common random numbers and frozen non-ground budgets:
A. current      — exact current ECV2 Stage-9 budgets.
B. literal_mean — native V3 ground NB2 count with mean
                  burst + own_control/900 * (attacker slope * defender suppression),
                  retaining current ECV2 ground landing probability.
C. literal_full — same V3 ground count plus validated attacker-only V3 ground accuracy.

Control, TD, standing, submissions, damage/KO, stamina and judging remain frozen.
The literal count uses the exact chronological ground-suppression observation
alpha published in the rebuilt V3 history for the target fight.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import (
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
    GROUND_SUPPRESSION_HISTORY_PATH,
)
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
from pipeline.simulation.event_clock_mc_v2.diagnostics.td_literal_control_interface_experiment import (
    _card_metrics,
)

OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/ground_literal_interface")
ARMS = ("current", "literal_mean", "literal_full")
GROUND_COUNT_SEED_OFFSET = 150_000_000
GROUND_LAND_SEED_OFFSET = 160_000_000


def _finite(value, default=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def _nb2_count(mean: float, alpha: float, rng: np.random.Generator) -> int:
    mean = max(float(mean), 0.0)
    if mean <= 0.0:
        return 0
    if alpha <= 1e-12:
        frailty = 1.0
    else:
        frailty = float(rng.gamma(shape=1.0 / alpha, scale=alpha))
    return int(rng.poisson(mean * frailty))


def _ground_alpha_by_fight() -> dict[str, float]:
    hist = pd.read_parquet(GROUND_SUPPRESSION_HISTORY_PATH).copy()
    hist["fight_id"] = hist["fight_id"].astype(str)
    hist["observation_alpha"] = pd.to_numeric(hist["observation_alpha"], errors="raise")
    spread = hist.groupby("fight_id")["observation_alpha"].agg(lambda x: float(x.max() - x.min()))
    if (spread > 1e-10).any():
        raise AssertionError("ground suppression alpha differs within a fight")
    return hist.groupby("fight_id")["observation_alpha"].first().astype(float).to_dict()


def _literal_ground_draws(pair: pd.DataFrame, current: dict, alpha: float, seed: int) -> dict:
    rows = {str(row["side"]): row for _, row in pair.iterrows()}
    out = {}
    for side, side_offset in (("red", 0), ("blue", 1)):
        row = rows[side]
        own_control = max(float(current[f"{side}_control"]), 0.0)
        burst = max(_finite(row.get("ground_burst_attempts"), 0.0), 0.0)
        slope = max(_finite(row.get("effective_ground_rate"), 0.0), 0.0)
        mean = burst + own_control / 900.0 * slope
        count_rng = np.random.default_rng(seed + GROUND_COUNT_SEED_OFFSET + side_offset)
        attempts = _nb2_count(mean, alpha, count_rng)

        pred_a = max(_finite(row.get("pred_ground_attempted"), 0.0), 0.0)
        pred_l = max(_finite(row.get("pred_ground_landed"), 0.0), 0.0)
        current_p = float(np.clip(pred_l / pred_a, 0.0, 1.0)) if pred_a > 1e-12 else 0.0
        v3_p = float(np.clip(_finite(row.get("ground_accuracy_matchup"), 0.0), 0.0, 1.0))

        land_rng = np.random.default_rng(seed + GROUND_LAND_SEED_OFFSET + side_offset)
        uniforms = land_rng.random(attempts) if attempts > 0 else np.empty(0, dtype=float)
        out[side] = {
            "own_control": own_control,
            "burst": burst,
            "slope": slope,
            "mean": mean,
            "attempts": attempts,
            "landed_current_p": int(np.sum(uniforms < current_p)),
            "landed_v3_p": int(np.sum(uniforms < v3_p)),
            "current_p": current_p,
            "v3_p": v3_p,
        }
    return out


def _arm_budgets(current: dict, pair: pd.DataFrame, alpha: float, seed: int):
    literal = _literal_ground_draws(pair, current, alpha, seed)
    mean_arm = dict(current)
    full_arm = dict(current)
    for side in ("red", "blue"):
        mean_arm[f"{side}_ground_attempted"] = literal[side]["attempts"]
        mean_arm[f"{side}_ground_landed"] = literal[side]["landed_current_p"]
        full_arm[f"{side}_ground_attempted"] = literal[side]["attempts"]
        full_arm[f"{side}_ground_landed"] = literal[side]["landed_v3_p"]

    arms = {"current": dict(current), "literal_mean": mean_arm, "literal_full": full_arm}
    allowed = {"red_ground_attempted", "red_ground_landed", "blue_ground_attempted", "blue_ground_landed"}
    for arm, budget in arms.items():
        if arm == "current":
            continue
        for key, value in current.items():
            if key not in allowed and budget[key] != value:
                raise AssertionError(f"{arm} unexpectedly changed frozen budget field {key}")
    return arms, literal


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
    pair_rows, pair_control = predict_target_v3(
        target,
        fsr_v3,
        context["inference_models"],
        context["submission_scale"],
        context["conversion_offset"],
    )
    pair_info_lookup = {str(row["fight_id"]): row for _, row in pair_control.iterrows()}
    master_lookup = {str(row["fight_id"]): row for _, row in master.iterrows()}
    alpha_lookup = _ground_alpha_by_fight()

    summary_rows, budget_rows = [], []
    groups = list(pair_rows.groupby("fight_id", sort=False))
    for fight_index, (fight_id, pair) in enumerate(groups):
        fight_id = str(fight_id)
        if fight_id not in alpha_lookup:
            raise RuntimeError(f"missing chronological ground alpha for fight {fight_id}")
        alpha = float(alpha_lookup[fight_id])
        master_row = master_lookup[fight_id]
        pair_info = pair_info_lookup[fight_id]
        fight = _fight(master_row, context["fsr_all"])
        sub_rate, convert = _submission_inputs(pair)
        print(f"[{fight_index + 1}/{len(groups)}] {master_row['r_name']} vs {master_row['b_name']} | ground alpha={alpha:.4f}")
        arm_paths = {arm: [] for arm in ARMS}

        for path in range(args.paths):
            seed = args.seed + fight_index * 1_000_000 + path
            current = _draw_budgets(pair, pair_info, context, np.random.default_rng(seed))
            arms, literal = _arm_budgets(current, pair, alpha, seed)
            for arm in ARMS:
                result = simulate_detailed_path(
                    fight,
                    arms[arm],
                    sub_rate,
                    convert,
                    context["judge_model"],
                    context["judge_features"],
                    seed + DETAILED_PATH_SEED_OFFSET,
                )
                arm_paths[arm].append(result)
                budget_rows.append({
                    "fight_id": fight_id,
                    "path": path,
                    "arm": arm,
                    "red_ground_attempted": arms[arm]["red_ground_attempted"],
                    "red_ground_landed": arms[arm]["red_ground_landed"],
                    "blue_ground_attempted": arms[arm]["blue_ground_attempted"],
                    "blue_ground_landed": arms[arm]["blue_ground_landed"],
                    "red_control": arms[arm]["red_control"],
                    "blue_control": arms[arm]["blue_control"],
                    "red_literal_mean": literal["red"]["mean"],
                    "blue_literal_mean": literal["blue"]["mean"],
                    "ground_alpha": alpha,
                })

        for arm in ARMS:
            s = summarize_fight(fight_id, pair, arm_paths[arm], master_row)
            s["arm"] = arm
            summary_rows.append(s)

    summary = pd.DataFrame(summary_rows)
    budgets = pd.DataFrame(budget_rows)
    metrics = pd.DataFrame([
        {"arm": arm, **_card_metrics(summary[summary["arm"] == arm])}
        for arm in ARMS
    ])
    pivot = summary.pivot(
        index=["fight_id", "red", "blue", "actual_winner", "actual_method"],
        columns="arm",
        values="p_red_win",
    ).reset_index()
    pivot["literal_mean_delta_pp"] = 100.0 * (pivot["literal_mean"] - pivot["current"])
    pivot["literal_full_delta_pp"] = 100.0 * (pivot["literal_full"] - pivot["current"])

    budget_summary = budgets.groupby(["fight_id", "arm"], as_index=False).agg(
        red_ground_attempted=("red_ground_attempted", "mean"),
        red_ground_landed=("red_ground_landed", "mean"),
        blue_ground_attempted=("blue_ground_attempted", "mean"),
        blue_ground_landed=("blue_ground_landed", "mean"),
        red_control=("red_control", "mean"),
        blue_control=("blue_control", "mean"),
        red_literal_mean=("red_literal_mean", "mean"),
        blue_literal_mean=("blue_literal_mean", "mean"),
        ground_alpha=("ground_alpha", "first"),
    )

    print("=" * 150)
    print("EVENT CLOCK V2 — LITERAL V3 GROUND INTERFACE EXPERIMENT")
    print("=" * 150)
    print(f"event date: {args.event_date} | fights: {len(groups)} | paths/fight/arm: {args.paths}")
    print("Only ground attempt/landing budgets differ across arms: VERIFIED")
    print("\nCARD METRICS")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nFIGHT-LEVEL P(RED WIN)")
    print(pivot[[
        "red", "blue", "actual_winner", "current", "literal_mean", "literal_full",
        "literal_mean_delta_pp", "literal_full_delta_pp",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = pd.Timestamp(args.event_date).strftime("%Y_%m_%d")
    summary_path = args.out_dir / f"{stem}_ground_literal_{args.paths}paths_summary.csv"
    metrics_path = args.out_dir / f"{stem}_ground_literal_{args.paths}paths_metrics.csv"
    movement_path = args.out_dir / f"{stem}_ground_literal_{args.paths}paths_ml_movement.csv"
    budget_path = args.out_dir / f"{stem}_ground_literal_{args.paths}paths_budget_summary.csv"
    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pivot.to_csv(movement_path, index=False)
    budget_summary.to_csv(budget_path, index=False)
    print("\noutputs:")
    print(summary_path)
    print(metrics_path)
    print(movement_path)
    print(budget_path)


if __name__ == "__main__":
    main()
