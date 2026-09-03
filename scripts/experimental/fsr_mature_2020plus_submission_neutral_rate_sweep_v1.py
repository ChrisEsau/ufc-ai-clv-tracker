"""Fixed-seed sweep of global neutral P(SUB | attempt) for the mature 2020+ cohort.

Research-only. Submission attempt generation, FSR traits, KO/damage/stamina physics,
phase behavior, and the seed matrix are held fixed. The only swept parameter is
``SUBMISSION_NEUTRAL_FINISH_PROBABILITY_PER_ATTEMPT``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_full_cohort_submission_validation_v1 as validation
from scripts.experimental import fsr_static_mc_ko_sub_v1 as combined

DEFAULT_PATHS = 10
DEFAULT_SEED = 20260811
DEFAULT_ROUNDS = 3
CANDIDATES = (0.18, 0.22, 0.26, 0.30, 0.34)
OUTPUT_DIR = Path("data/experimental/submission_neutral_rate_sweep_v1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep neutral submission finish probability on fixed mature-cohort paths"
    )
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def _historical_targets(cohort: pd.DataFrame) -> tuple[float, dict[int, float]]:
    overall = float(cohort["actual_submission_within_horizon"].mean())
    round_rates: dict[int, float] = {}
    for r in range(1, DEFAULT_ROUNDS + 1):
        reached = int((cohort["actual_finish_round_resolved"].fillna(DEFAULT_ROUNDS) >= r).sum())
        subs = int((
            cohort["actual_submission_within_horizon"].eq(1)
            & cohort["actual_finish_round_resolved"].eq(float(r))
        ).sum())
        round_rates[r] = subs / reached if reached else np.nan
    return overall, round_rates


def _prepare_cohort() -> tuple[pd.DataFrame, dict[str, tuple[pd.Series, pd.Series]]]:
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    cohort = cohort.merge(
        validation._master_metadata(),
        on="bout_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_master"),
    )
    cohort["actual_submission"] = cohort["method"].map(validation._is_submission_method).astype(int)
    cohort["actual_finish_round_resolved"] = cohort.apply(validation._finish_round, axis=1)
    cohort["actual_submission_within_horizon"] = (
        cohort["actual_submission"].eq(1)
        & cohort["actual_finish_round_resolved"].notna()
        & cohort["actual_finish_round_resolved"].le(DEFAULT_ROUNDS)
    ).astype(int)
    return cohort.reset_index(drop=True), pairs


def _run_candidate(
    neutral_rate: float,
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    seed_matrix: np.ndarray,
) -> dict[str, float | int]:
    combined.SUBMISSION_NEUTRAL_FINISH_PROBABILITY_PER_ATTEMPT = float(neutral_rate)
    combined.configure_current_finish_candidate()

    total_subs = 0
    red_subs = 0
    blue_subs = 0
    total_attempts = 0
    paths_with_attempt = 0
    sub_without_attempt = 0
    round_reached = {r: 0 for r in range(1, DEFAULT_ROUNDS + 1)}
    round_subs = {r: 0 for r in range(1, DEFAULT_ROUNDS + 1)}

    # Direction is measured from bout-level red/blue SUB probabilities using the
    # same fixed 10-path sample for every candidate.
    direction_rows: list[dict[str, object]] = []

    for bout_index, (_, bout) in enumerate(cohort.iterrows()):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        red_age = validation._age(bout, "r_age")
        blue_age = validation._age(bout, "b_age")
        bout_red_sub = 0
        bout_blue_sub = 0

        for seed in seed_matrix[bout_index]:
            sim = combined.StaticFSRMCKOSUBV1(
                red,
                blue,
                rounds=DEFAULT_ROUNDS,
                seed=int(seed),
                red_age=red_age,
                blue_age=blue_age,
            )
            path = sim.run()

            attempts = int(sim.stats[0].sub_att) + int(sim.stats[1].sub_att)
            total_attempts += attempts
            paths_with_attempt += int(attempts > 0)

            finish_round = 0
            method = "NONE"
            winner = -1
            if path.finish is not None:
                finish_round = int(path.finish.round or 0)
                method = str(path.finish.method)
                winner = int(path.finish.winner)

            for r in range(1, DEFAULT_ROUNDS + 1):
                if finish_round == 0 or finish_round >= r:
                    round_reached[r] += 1

            if method == "SUB":
                total_subs += 1
                if winner == 0:
                    red_subs += 1
                    bout_red_sub += 1
                elif winner == 1:
                    blue_subs += 1
                    bout_blue_sub += 1
                if finish_round in round_subs:
                    round_subs[finish_round] += 1
                if attempts <= 0:
                    sub_without_attempt += 1

        winner_id = str(bout.get("winner_id", ""))
        r_id = str(bout.get("r_id", red.get("fighter_id", "")))
        b_id = str(bout.get("b_id", blue.get("fighter_id", "")))
        actual_side = "red" if winner_id == r_id else "blue" if winner_id == b_id else ""
        predicted_side = "red" if bout_red_sub > bout_blue_sub else "blue" if bout_blue_sub > bout_red_sub else "tie"
        if int(bout["actual_submission_within_horizon"]) == 1:
            direction_rows.append({
                "predicted_side": predicted_side,
                "actual_side": actual_side,
                "hit": int(predicted_side == actual_side) if predicted_side != "tie" else np.nan,
            })

        if (bout_index + 1) % 250 == 0 or bout_index + 1 == len(cohort):
            print(
                f"  neutral={neutral_rate:.0%}: bouts {bout_index + 1:,}/{len(cohort):,}",
                flush=True,
            )

    total_paths = len(cohort) * seed_matrix.shape[1]
    direction = pd.DataFrame(direction_rows)
    non_tie = direction.loc[direction["predicted_side"].ne("tie")]
    ties = int(direction["predicted_side"].eq("tie").sum())

    row: dict[str, float | int] = {
        "neutral_rate": float(neutral_rate),
        "simulated_submission_rate": total_subs / total_paths,
        "simulated_attempts_per_path": total_attempts / total_paths,
        "simulated_paths_with_attempt_rate": paths_with_attempt / total_paths,
        "red_submission_rate": red_subs / total_paths,
        "blue_submission_rate": blue_subs / total_paths,
        "sub_finish_without_attempt": sub_without_attempt,
        "non_tie_direction_calls": len(non_tie),
        "tie_direction_calls": ties,
        "direction_accuracy_non_tie": float(non_tie["hit"].mean()) if len(non_tie) else np.nan,
    }
    for r in range(1, DEFAULT_ROUNDS + 1):
        row[f"sim_r{r}_sub_rate_conditional"] = (
            round_subs[r] / round_reached[r] if round_reached[r] else np.nan
        )
    return row


def main() -> None:
    args = parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = _prepare_cohort()
    hist_overall, hist_round = _historical_targets(cohort)

    seed_rng = np.random.default_rng(args.seed)
    seed_matrix = seed_rng.integers(
        1,
        np.iinfo(np.int32).max,
        size=(len(cohort), args.paths),
        dtype=np.int64,
    )

    rows = []
    for candidate in CANDIDATES:
        row = _run_candidate(candidate, cohort, pairs, seed_matrix)
        row["historical_submission_rate"] = hist_overall
        row["overall_error_pp"] = 100.0 * (
            float(row["simulated_submission_rate"]) - hist_overall
        )
        for r in range(1, DEFAULT_ROUNDS + 1):
            row[f"historical_r{r}_sub_rate_conditional"] = hist_round[r]
            row[f"r{r}_error_pp"] = 100.0 * (
                float(row[f"sim_r{r}_sub_rate_conditional"]) - hist_round[r]
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    result["abs_overall_error_pp"] = result["overall_error_pp"].abs()
    result = result.sort_values(["abs_overall_error_pp", "neutral_rate"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "neutral_rate_sweep.csv"
    result.to_csv(output_path, index=False)

    print("\n" + "=" * 122)
    print("MATURE 2020+ SUBMISSION NEUTRAL-RATE SWEEP — FIXED COHORT / FIXED SEEDS")
    print("=" * 122)
    print(f"cohort bouts: {len(cohort):,}")
    print(f"paths/bout: {args.paths:,}")
    print(f"historical R1-R3 SUB rate: {hist_overall:.2%}")
    print("historical conditional round rates: " + ", ".join(
        f"R{r}={hist_round[r]:.2%}" for r in range(1, DEFAULT_ROUNDS + 1)
    ))
    print("\nRESULTS — SORTED BY ABSOLUTE OVERALL SUB-RATE ERROR")
    display_cols = [
        "neutral_rate",
        "simulated_submission_rate",
        "overall_error_pp",
        "sim_r1_sub_rate_conditional",
        "r1_error_pp",
        "sim_r2_sub_rate_conditional",
        "r2_error_pp",
        "sim_r3_sub_rate_conditional",
        "r3_error_pp",
        "direction_accuracy_non_tie",
        "non_tie_direction_calls",
        "tie_direction_calls",
        "simulated_attempts_per_path",
        "simulated_paths_with_attempt_rate",
    ]
    formatters = {
        "neutral_rate": lambda x: f"{x:.0%}",
        "simulated_submission_rate": lambda x: f"{x:.2%}",
        "overall_error_pp": lambda x: f"{x:+.2f}",
        "sim_r1_sub_rate_conditional": lambda x: f"{x:.2%}",
        "r1_error_pp": lambda x: f"{x:+.2f}",
        "sim_r2_sub_rate_conditional": lambda x: f"{x:.2%}",
        "r2_error_pp": lambda x: f"{x:+.2f}",
        "sim_r3_sub_rate_conditional": lambda x: f"{x:.2%}",
        "r3_error_pp": lambda x: f"{x:+.2f}",
        "direction_accuracy_non_tie": lambda x: f"{x:.2%}",
        "simulated_attempts_per_path": lambda x: f"{x:.4f}",
        "simulated_paths_with_attempt_rate": lambda x: f"{x:.2%}",
    }
    print(result[display_cols].to_string(index=False, formatters=formatters))

    best = result.iloc[0]
    print("\nBEST OVERALL-RATE CANDIDATE")
    print(
        f"neutral={best['neutral_rate']:.0%} -> simulated SUB={best['simulated_submission_rate']:.2%}, "
        f"error={best['overall_error_pp']:+.2f} pp"
    )
    print(f"\nsaved: {output_path}")
    print("Research-only; no production artifacts or frozen KO benchmark outputs are modified.")


if __name__ == "__main__":
    main()
