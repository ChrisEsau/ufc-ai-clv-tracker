"""Takedown/control-only Event Clock V2 interface experiment.

Five arms isolate whether the validated FSR V3 takedown matchup should enter
Stage 9 more literally while leaving standing, ground, submissions, damage/KO,
stamina, judging, and total-control generation frozen:

A. current                — exact current ECV2 Stage-9 budgets.
B. current_resplit        — current TD budgets, but control ownership is redrawn
                            with a dedicated common-random-number stream. This is
                            a sanity arm for the ownership comparison.
C. literal_rate           — V3 matchup TD rate on each path's realized
                            non-opponent-control exposure; current ECV2 TD
                            completion retained; current control allocation frozen.
D. literal_full           — same literal V3 TD attempts plus V3 matchup TD
                            completion; current control allocation frozen.
E. literal_full_control   — same literal V3 TD attempts/completion, then redraw
                            only the red/blue split of the already-frozen total
                            control using the exact Stage-9 control ownership model.

Important semantic guards:
- V3 takedown tendency is per non-opponent-control exposure, so literal expected
  attempts use duration - opponent_control from the already-drawn current path.
- The total-control draw is never replaced because FSR V3 does not contain a
  separately validated literal total-control trait.
- Defender-adjusted literal attempts use the validated TD suppression NB2 alpha
  because effective_td_rate = attacker tendency * defender suppression.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage9_final_flow import (
    draw_stage9_control_split,
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

OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/td_literal_control_interface")
ARMS = (
    "current",
    "current_resplit",
    "literal_rate",
    "literal_full",
    "literal_full_control",
)
TD_DRAW_SEED_OFFSET = 120_000_000
TD_LAND_SEED_OFFSET = 130_000_000
CONTROL_SPLIT_SEED_OFFSET = 140_000_000
CFG = FSRV3Config()


def _finite(value, default=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def _clip_probability(value):
    return float(np.clip(_finite(value, 0.0), 0.0, 1.0))


def _nb2_count(mean: float, alpha: float, rng: np.random.Generator) -> int:
    mean = max(float(mean), 0.0)
    if mean <= 0.0:
        return 0
    if alpha <= 1e-12:
        frailty = 1.0
    else:
        frailty = float(rng.gamma(shape=1.0 / alpha, scale=alpha))
    return int(rng.poisson(mean * frailty))


def _td_interface(pair: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in pair.iterrows():
        pred_attempted = max(_finite(row.get("pred_td_attempted"), 0.0), 0.0)
        pred_landed = max(_finite(row.get("pred_td_landed"), 0.0), 0.0)
        current_completion = (
            float(np.clip(pred_landed / pred_attempted, 0.0, 1.0))
            if pred_attempted > 1e-12 else 0.0
        )
        records.append(
            {
                "fight_id": str(row["fight_id"]),
                "side": str(row["side"]),
                "fighter": str(row["fighter_name"]),
                "opponent": str(row["opponent_name"]),
                "duration": float(row["duration"]),
                "current_pred_td_attempted": pred_attempted,
                "current_td_completion": current_completion,
                "v3_literal_td_rate_15m": max(_finite(row.get("effective_td_rate"), 0.0), 0.0),
                "v3_td_completion": _clip_probability(row.get("td_completion_matchup")),
            }
        )
    return pd.DataFrame(records)


def _literal_td_draws(
    pair: pd.DataFrame,
    current_budget: dict,
    seed: int,
) -> dict:
    """Draw one literal V3 TD attempt budget and coupled landing outcomes."""
    rows = {str(row["side"]): row for _, row in pair.iterrows()}
    duration = float(pair["duration"].iloc[0])
    output = {}

    for side, opponent_side, side_offset in (("red", "blue", 0), ("blue", "red", 1)):
        row = rows[side]
        exposure = max(duration - float(current_budget[f"{opponent_side}_control"]), 0.0)
        rate_15m = max(_finite(row.get("effective_td_rate"), 0.0), 0.0)
        expected_attempts = rate_15m * exposure / 900.0

        attempts_rng = np.random.default_rng(seed + TD_DRAW_SEED_OFFSET + side_offset)
        attempts = _nb2_count(
            expected_attempts,
            CFG.takedown_suppression_initial_alpha,
            attempts_rng,
        )

        current_pred_attempted = max(_finite(row.get("pred_td_attempted"), 0.0), 0.0)
        current_pred_landed = max(_finite(row.get("pred_td_landed"), 0.0), 0.0)
        current_p = (
            float(np.clip(current_pred_landed / current_pred_attempted, 0.0, 1.0))
            if current_pred_attempted > 1e-12 else 0.0
        )
        v3_p = _clip_probability(row.get("td_completion_matchup"))

        land_rng = np.random.default_rng(seed + TD_LAND_SEED_OFFSET + side_offset)
        uniforms = land_rng.random(attempts) if attempts > 0 else np.empty(0, dtype=float)
        landed_current_p = int(np.sum(uniforms < current_p))
        landed_v3_p = int(np.sum(uniforms < v3_p))

        output[side] = {
            "exposure": exposure,
            "expected_attempts": expected_attempts,
            "attempts": attempts,
            "landed_current_p": landed_current_p,
            "landed_v3_p": landed_v3_p,
            "current_p": current_p,
            "v3_p": v3_p,
        }
    return output


def _resplit_control(
    budget: dict,
    pair_info: pd.Series,
    red_td_landed: int,
    blue_td_landed: int,
    context: dict,
    seed: int,
) -> dict:
    out = dict(budget)
    red_control, blue_control = draw_stage9_control_split(
        float(budget["total_control"]),
        pair_info,
        int(red_td_landed),
        int(blue_td_landed),
        context["td_control_beta"],
        context["dominance_kappa"],
        context["minority_classifier"],
        context["minority_share_model"],
        context["minority_residual_sigma"],
        np.random.default_rng(seed + CONTROL_SPLIT_SEED_OFFSET),
    )
    out["red_control"] = red_control
    out["blue_control"] = blue_control
    # Total control and free time are deliberately frozen.
    if abs((red_control + blue_control) - float(budget["total_control"])) > 1e-9:
        raise AssertionError("control resplit changed total control")
    return out


def _arm_budgets(
    current: dict,
    pair: pd.DataFrame,
    pair_info: pd.Series,
    context: dict,
    seed: int,
) -> dict[str, dict]:
    literal = _literal_td_draws(pair, current, seed)

    current_resplit = _resplit_control(
        current,
        pair_info,
        int(current["red_td_landed"]),
        int(current["blue_td_landed"]),
        context,
        seed,
    )

    literal_rate = dict(current)
    literal_full = dict(current)
    for side in ("red", "blue"):
        literal_rate[f"{side}_td_attempted"] = literal[side]["attempts"]
        literal_rate[f"{side}_td_landed"] = literal[side]["landed_current_p"]
        literal_full[f"{side}_td_attempted"] = literal[side]["attempts"]
        literal_full[f"{side}_td_landed"] = literal[side]["landed_v3_p"]

    literal_full_control = _resplit_control(
        literal_full,
        pair_info,
        int(literal_full["red_td_landed"]),
        int(literal_full["blue_td_landed"]),
        context,
        seed,
    )

    arms = {
        "current": dict(current),
        "current_resplit": current_resplit,
        "literal_rate": literal_rate,
        "literal_full": literal_full,
        "literal_full_control": literal_full_control,
    }

    # Hard isolation checks.
    td_fields = {
        "red_td_attempted", "red_td_landed", "blue_td_attempted", "blue_td_landed"
    }
    control_split_fields = {"red_control", "blue_control"}
    for arm, budget in arms.items():
        for key, value in current.items():
            if arm == "current":
                continue
            allowed = set()
            if arm == "current_resplit":
                allowed = control_split_fields
            elif arm in ("literal_rate", "literal_full"):
                allowed = td_fields
            elif arm == "literal_full_control":
                allowed = td_fields | control_split_fields
            if key not in allowed and budget[key] != value:
                raise AssertionError(f"{arm} unexpectedly changed frozen budget field {key}")
        if budget["total_control"] != current["total_control"]:
            raise AssertionError(f"{arm} changed total control")
        if budget["free_seconds"] != current["free_seconds"]:
            raise AssertionError(f"{arm} changed free time")

    return arms, literal


def _card_metrics(summary: pd.DataFrame) -> dict:
    red_actual = (summary["actual_winner"] == "red").astype(float).to_numpy()
    p_red = np.clip(summary["p_red_win"].to_numpy(float), 1e-9, 1.0 - 1e-9)
    p_actual = np.where(red_actual == 1.0, p_red, 1.0 - p_red)
    joint = [
        float(row[f"p_{row['actual_winner']}_{str(row['actual_method']).lower()}"])
        for _, row in summary.iterrows()
    ]
    return {
        "fights": int(len(summary)),
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
    pair_lookup = {str(row["fight_id"]): row for _, row in mean_pair.iterrows()}
    master_lookup = {str(row["fight_id"]): row for _, row in master.iterrows()}

    interface_parts = []
    summary_rows = []
    budget_rows = []
    path_rows = []

    groups = list(mean_test.groupby("fight_id", sort=False))
    for fight_index, (fight_id, pair) in enumerate(groups):
        fight_id = str(fight_id)
        master_row = master_lookup[fight_id]
        pair_info = pair_lookup[fight_id]
        interface_parts.append(_td_interface(pair))
        fight = _fight(master_row, context["fsr_all"])
        sub_rate, convert = _submission_inputs(pair)
        print(f"[{fight_index + 1}/{len(groups)}] {master_row['r_name']} vs {master_row['b_name']}")

        arm_paths = {arm: [] for arm in ARMS}
        for path in range(args.paths):
            seed = args.seed + fight_index * 1_000_000 + path
            current = _draw_budgets(pair, pair_info, context, np.random.default_rng(seed))
            arms, literal = _arm_budgets(current, pair, pair_info, context, seed)

            for arm in ARMS:
                budget = arms[arm]
                result = simulate_detailed_path(
                    fight,
                    budget,
                    sub_rate,
                    convert,
                    context["judge_model"],
                    context["judge_features"],
                    seed + DETAILED_PATH_SEED_OFFSET,
                )
                result.update({"fight_id": fight_id, "path": path, "arm": arm})
                arm_paths[arm].append(result)
                path_rows.append(result)
                budget_rows.append(
                    {
                        "fight_id": fight_id,
                        "arm": arm,
                        "red_td_attempted": budget["red_td_attempted"],
                        "red_td_landed": budget["red_td_landed"],
                        "blue_td_attempted": budget["blue_td_attempted"],
                        "blue_td_landed": budget["blue_td_landed"],
                        "red_control": budget["red_control"],
                        "blue_control": budget["blue_control"],
                        "total_control": budget["total_control"],
                        "free_seconds": budget["free_seconds"],
                        "red_literal_td_exposure": literal["red"]["exposure"],
                        "blue_literal_td_exposure": literal["blue"]["exposure"],
                        "red_literal_td_expected": literal["red"]["expected_attempts"],
                        "blue_literal_td_expected": literal["blue"]["expected_attempts"],
                    }
                )

        for arm in ARMS:
            s = summarize_fight(fight_id, pair, arm_paths[arm], master_row)
            s["arm"] = arm
            summary_rows.append(s)

    interface = pd.concat(interface_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    budget_paths = pd.DataFrame(budget_rows)
    budget_summary = budget_paths.groupby(["fight_id", "arm"], as_index=False).agg(
        red_td_attempted=("red_td_attempted", "mean"),
        red_td_landed=("red_td_landed", "mean"),
        blue_td_attempted=("blue_td_attempted", "mean"),
        blue_td_landed=("blue_td_landed", "mean"),
        red_control=("red_control", "mean"),
        blue_control=("blue_control", "mean"),
        total_control=("total_control", "mean"),
        free_seconds=("free_seconds", "mean"),
        red_literal_td_exposure=("red_literal_td_exposure", "mean"),
        blue_literal_td_exposure=("blue_literal_td_exposure", "mean"),
        red_literal_td_expected=("red_literal_td_expected", "mean"),
        blue_literal_td_expected=("blue_literal_td_expected", "mean"),
    )

    metrics = pd.DataFrame(
        [{"arm": arm, **_card_metrics(summary[summary["arm"] == arm])} for arm in ARMS]
    )
    pivot = summary.pivot(
        index=["fight_id", "red", "blue", "actual_winner", "actual_method"],
        columns="arm",
        values="p_red_win",
    ).reset_index()
    for arm in ARMS[1:]:
        pivot[f"{arm}_delta_pp"] = 100.0 * (pivot[arm] - pivot["current"])

    print("=" * 150)
    print("EVENT CLOCK V2 — LITERAL V3 TAKEDOWN / CONTROL INTERFACE EXPERIMENT")
    print("=" * 150)
    print(f"event date: {args.event_date} | fights: {len(groups)} | paths/fight/arm: {args.paths}")
    print("Standing / ground / total-control / submissions / damage / stamina / judging: frozen")
    print("Only TD budgets differ in literal_rate/full; literal_full_control additionally resplits fixed total control")
    print(f"literal defender-adjusted TD NB2 alpha: {CFG.takedown_suppression_initial_alpha:.4f}")

    print("\nCARD METRICS")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nFIGHT-LEVEL P(RED WIN)")
    cols = ["red", "blue", "actual_winner"] + list(ARMS) + [f"{arm}_delta_pp" for arm in ARMS[1:]]
    print(pivot[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    literal_budget = budget_summary[budget_summary["arm"] == "literal_full"]
    current_budget = budget_summary[budget_summary["arm"] == "current"]
    merged = current_budget.merge(literal_budget, on="fight_id", suffixes=("_current", "_literal"))
    for side in ("red", "blue"):
        merged[f"{side}_td_attempt_change_pct"] = 100.0 * (
            merged[f"{side}_td_attempted_literal"] - merged[f"{side}_td_attempted_current"]
        ) / np.maximum(merged[f"{side}_td_attempted_current"], 1e-9)
    attempt_changes = np.concatenate(
        [merged["red_td_attempt_change_pct"].to_numpy(float), merged["blue_td_attempt_change_pct"].to_numpy(float)]
    )
    print("\nREALIZED TD BUDGET MOVEMENT")
    print(f"mean abs fighter-level TD attempt change: {np.mean(np.abs(attempt_changes)):.2f}%")
    print(
        f"mean abs V3-vs-current completion change: "
        f"{100.0 * np.mean(np.abs(interface['v3_td_completion'] - interface['current_td_completion'])):.2f} pp"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = pd.Timestamp(args.event_date).strftime("%Y_%m_%d")
    outputs = {
        "interface": args.out_dir / f"{stem}_td_interface.csv",
        "summary": args.out_dir / f"{stem}_td_literal_control_{args.paths}paths_summary.csv",
        "budgets": args.out_dir / f"{stem}_td_literal_control_{args.paths}paths_budget_summary.csv",
        "metrics": args.out_dir / f"{stem}_td_literal_control_{args.paths}paths_card_metrics.csv",
        "movement": args.out_dir / f"{stem}_td_literal_control_{args.paths}paths_ml_movement.csv",
        "paths": args.out_dir / f"{stem}_td_literal_control_{args.paths}paths_paths.csv",
    }
    interface.to_csv(outputs["interface"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    budget_summary.to_csv(outputs["budgets"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    pivot.to_csv(outputs["movement"], index=False)
    pd.DataFrame(path_rows).to_csv(outputs["paths"], index=False)
    print("\noutputs:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
