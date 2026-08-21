"""Fresh held-out validation of current ECV2 vs literal FSR V3 takedown rate.

Measurement only. Uses the frozen Event Clock V2 direct bundle and exact frozen
V1 detailed mechanics. For each path the current Stage-9 budget is drawn once;
the literal arm replaces only TD attempts/landings using the validated V3 TD
matchup rate on realized non-opponent-control exposure while retaining the
current ECV2 TD completion probability. All standing, ground, control, free-time,
submission, damage/KO, stamina, and judging inputs remain identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import (
    SEED,
    simulate_detailed_path,
    summarize_fight,
)
from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import (
    select_fresh_cohort,
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
    _arm_budgets,
    _card_metrics,
)

OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/td_literal_rate_fresh100")
ARMS = ("current", "literal_rate")
# Intentionally narrow: this validation does not test V3 TD completion or control ownership.


def _actual_winner_probability(frame: pd.DataFrame) -> np.ndarray:
    red_actual = frame["actual_winner"].eq("red").to_numpy()
    p_red = frame["p_red_win"].to_numpy(float)
    return np.where(red_actual, p_red, 1.0 - p_red)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fights", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    cohort, _, metadata = select_fresh_cohort(args.fights, offset=args.offset)
    cohort = cohort.copy()
    cohort["fight_id"] = cohort["fight_id"].astype(str)
    cohort["event_date"] = pd.to_datetime(cohort["event_date"], errors="raise").dt.normalize()

    context = load_frozen_context(args.bundle)
    fsr_v3 = load_prefight_snapshots(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    mean_test, mean_pair = predict_target_v3(
        cohort,
        fsr_v3,
        context["inference_models"],
        context["submission_scale"],
        context["conversion_offset"],
    )
    built = int(mean_test["fight_id"].nunique())
    if built != args.fights:
        raise RuntimeError(f"Expected {args.fights} V3 target fights, built {built}")

    pair_lookup = {str(row["fight_id"]): row for _, row in mean_pair.iterrows()}
    master_lookup = {str(row["fight_id"]): row for _, row in cohort.iterrows()}
    summary_rows = []

    groups = list(mean_test.groupby("fight_id", sort=False))
    for fight_index, (fight_id, pair) in enumerate(groups):
        fight_id = str(fight_id)
        master_row = master_lookup[fight_id]
        pair_info = pair_lookup[fight_id]
        fight = _fight(master_row, context["fsr_all"])
        sub_rate, convert = _submission_inputs(pair)
        arm_paths = {arm: [] for arm in ARMS}

        if fight_index % 10 == 0:
            print(f"[{fight_index + 1}/{len(groups)}] {master_row['r_name']} vs {master_row['b_name']}")

        for path in range(args.paths):
            seed = args.seed + fight_index * 1_000_000 + path
            current = _draw_budgets(pair, pair_info, context, np.random.default_rng(seed))
            arms, _ = _arm_budgets(current, pair, pair_info, context, seed)

            literal = arms["literal_rate"]
            allowed = {"red_td_attempted", "red_td_landed", "blue_td_attempted", "blue_td_landed"}
            for key, value in current.items():
                if key not in allowed and literal[key] != value:
                    raise AssertionError(f"literal_rate changed frozen budget field {key}")

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

        for arm in ARMS:
            s = summarize_fight(fight_id, pair, arm_paths[arm], master_row)
            s["arm"] = arm
            summary_rows.append(s)

    summary = pd.DataFrame(summary_rows)
    metrics_rows = []
    for arm in ARMS:
        metrics_rows.append({"arm": arm, **_card_metrics(summary[summary["arm"] == arm])})
    metrics = pd.DataFrame(metrics_rows)

    current = summary[summary["arm"] == "current"].set_index("fight_id").sort_index()
    literal = summary[summary["arm"] == "literal_rate"].set_index("fight_id").sort_index()
    if not current.index.equals(literal.index):
        raise AssertionError("arm fight sets differ")

    movement = current[["red", "blue", "actual_winner", "actual_method", "p_red_win", "ml_correct"]].copy()
    movement = movement.rename(columns={"p_red_win": "current_p_red", "ml_correct": "current_ml_correct"})
    movement["literal_p_red"] = literal["p_red_win"]
    movement["literal_ml_correct"] = literal["ml_correct"]
    movement["red_delta_pp"] = 100.0 * (movement["literal_p_red"] - movement["current_p_red"])
    movement["current_p_actual_winner"] = _actual_winner_probability(current.reset_index())
    movement["literal_p_actual_winner"] = _actual_winner_probability(literal.reset_index())
    movement["actual_winner_delta_pp"] = 100.0 * (
        movement["literal_p_actual_winner"] - movement["current_p_actual_winner"]
    )
    movement["flip_to_correct"] = (
        movement["current_ml_correct"].eq(0) & movement["literal_ml_correct"].eq(1)
    )
    movement["flip_to_wrong"] = (
        movement["current_ml_correct"].eq(1) & movement["literal_ml_correct"].eq(0)
    )
    movement = movement.reset_index()

    improved = int((movement["actual_winner_delta_pp"] > 1e-12).sum())
    worsened = int((movement["actual_winner_delta_pp"] < -1e-12).sum())
    tied = int(len(movement) - improved - worsened)

    print("=" * 150)
    print("EVENT CLOCK V2 — FRESH HELD-OUT LITERAL V3 TD RATE VALIDATION")
    print("=" * 150)
    print(
        f"fights: {args.fights} | offset: {args.offset} | paths/fight/arm: {args.paths} | "
        f"dates: {metadata['first_event_date']} through {metadata['last_event_date']}"
    )
    print("Only TD attempts/landings differ; all other Stage-9 budgets and detailed mechanics are frozen.")
    print("\nCARD/COHORT METRICS")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPAIRED PROBABILITY MOVEMENT")
    print(f"mean actual-winner delta: {movement['actual_winner_delta_pp'].mean():+.3f} pp")
    print(f"median actual-winner delta: {movement['actual_winner_delta_pp'].median():+.3f} pp")
    print(f"mean abs red-win movement: {movement['red_delta_pp'].abs().mean():.3f} pp")
    print(f"actual-winner probability improved/worsened/tied: {improved}/{worsened}/{tied}")
    print(f"classification flips to correct: {int(movement['flip_to_correct'].sum())}")
    print(f"classification flips to wrong:   {int(movement['flip_to_wrong'].sum())}")
    print("\nLARGEST ABSOLUTE MONEYLINE MOVES")
    print(
        movement.sort_values("red_delta_pp", key=lambda x: x.abs(), ascending=False)
        .head(20)[[
            "red", "blue", "actual_winner", "current_p_red", "literal_p_red",
            "red_delta_pp", "actual_winner_delta_pp", "current_ml_correct", "literal_ml_correct",
        ]]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fresh{args.fights}_offset{args.offset}_{args.paths}paths"
    summary_path = args.out_dir / f"{stem}_summary.csv"
    metrics_path = args.out_dir / f"{stem}_metrics.csv"
    movement_path = args.out_dir / f"{stem}_movement.csv"
    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    movement.to_csv(movement_path, index=False)
    print("\noutputs:")
    print(summary_path)
    print(metrics_path)
    print(movement_path)


if __name__ == "__main__":
    main()
