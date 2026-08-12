"""Full mature 2020+ cohort validation for the first submission-finish layer.

Research-only. The current KO/damage/stamina/phase configuration is preserved.
Submission attempts are generated exactly as before; this audit only evaluates
new probabilistic submission finishes after those attempts.

Default run: all aligned mature-cohort bouts x 10 paths, 3-round horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as population
from scripts.experimental import fsr_static_mc_ko_sub_v1 as combined
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_PATHS = 10
DEFAULT_SEED = 20260811
DEFAULT_ROUNDS = 3
OUTPUT_DIR = Path("data/experimental/full_cohort_submission_validation_v1")
RFS_HISTORY_PATH = Path("data/features/round_fighter_state_history.parquet")
SUB_ATTEMPT_COLUMN = "rfs_finish_state_fight_submission_attempts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate submission finishes on the full mature 2020+ FSR-32 cohort"
    )
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def _master_metadata() -> pd.DataFrame:
    raw = pd.read_parquet(population.modern.MASTER_PATH).copy()
    raw["fight_id"] = raw["fight_id"].astype(str)
    date_col = population.modern._resolve_date_column(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.sort_values([date_col, "fight_id"]).drop_duplicates("fight_id", keep="last")

    keep = ["fight_id"]
    for col in (
        "winner_id", "method", "finish_round", "round", "last_round",
        "r_name", "b_name", "r_id", "b_id",
    ):
        if col in raw.columns and col not in keep:
            keep.append(col)
    return raw[keep].rename(columns={"fight_id": "bout_id"})


def _is_submission_method(value: object) -> bool:
    text = str(value or "").strip().upper()
    return "SUB" in text


def _finish_round(row: pd.Series) -> float:
    for col in ("actual_finish_round", "finish_round", "round", "last_round"):
        if col not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return np.nan


def _historical_submission_attempts(bout_ids: set[str]) -> pd.DataFrame:
    hist = pd.read_parquet(RFS_HISTORY_PATH).copy()
    id_col = "fight_id" if "fight_id" in hist.columns else "bout_id"
    if id_col not in hist.columns:
        raise RuntimeError("RFS history missing fight_id/bout_id")
    if SUB_ATTEMPT_COLUMN not in hist.columns:
        raise RuntimeError(f"RFS history missing {SUB_ATTEMPT_COLUMN}")

    hist[id_col] = hist[id_col].astype(str)
    hist = hist.loc[hist[id_col].isin(bout_ids), [id_col, SUB_ATTEMPT_COLUMN]].copy()
    hist[SUB_ATTEMPT_COLUMN] = pd.to_numeric(
        hist[SUB_ATTEMPT_COLUMN], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)

    out = hist.groupby(id_col, as_index=False)[SUB_ATTEMPT_COLUMN].sum()
    out = out.rename(columns={id_col: "bout_id", SUB_ATTEMPT_COLUMN: "historical_sub_attempts"})
    return out


def _age(row: pd.Series, col: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def main() -> None:
    args = parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    combined.configure_current_finish_candidate()

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    cohort = cohort.merge(_master_metadata(), on="bout_id", how="left", validate="one_to_one", suffixes=("", "_master"))
    cohort = cohort.merge(
        _historical_submission_attempts(set(cohort["bout_id"])),
        on="bout_id",
        how="left",
        validate="one_to_one",
    )
    cohort["historical_sub_attempts"] = cohort["historical_sub_attempts"].fillna(0.0)
    cohort["actual_submission"] = cohort["method"].map(_is_submission_method).astype(int)
    cohort["actual_finish_round_resolved"] = cohort.apply(_finish_round, axis=1)
    cohort["actual_submission_within_horizon"] = (
        cohort["actual_submission"].eq(1)
        & cohort["actual_finish_round_resolved"].notna()
        & cohort["actual_finish_round_resolved"].le(DEFAULT_ROUNDS)
    ).astype(int)

    seed_rng = np.random.default_rng(args.seed)
    seed_matrix = seed_rng.integers(
        1,
        np.iinfo(np.int32).max,
        size=(len(cohort), args.paths),
        dtype=np.int64,
    )

    rows: list[dict[str, object]] = []
    sim_round = {r: {"reached": 0, "sub": 0} for r in range(1, DEFAULT_ROUNDS + 1)}
    total_sim_sub_attempts = 0
    sim_paths_with_attempt = 0
    sub_without_attempt = 0

    for bout_index, (_, bout) in enumerate(cohort.iterrows()):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        red_age = _age(bout, "r_age")
        blue_age = _age(bout, "b_age")

        red_sub = 0
        blue_sub = 0
        any_sub = 0
        ko_count = 0
        none_count = 0
        attempt_total = 0
        attempt_paths = 0
        finish_round_counts = {1: 0, 2: 0, 3: 0}

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

            red_attempts = int(sim.stats[0].sub_att)
            blue_attempts = int(sim.stats[1].sub_att)
            attempts = red_attempts + blue_attempts
            attempt_total += attempts
            total_sim_sub_attempts += attempts
            if attempts > 0:
                attempt_paths += 1
                sim_paths_with_attempt += 1

            finish_method = "NONE"
            finish_round = 0
            winner = -1
            if path.finish is not None:
                finish_method = str(path.finish.method)
                finish_round = int(path.finish.round or 0)
                winner = int(path.finish.winner)

            # Conditional round denominators: path reached round r if it either
            # survived beyond r or finished during r.
            for r in range(1, DEFAULT_ROUNDS + 1):
                if finish_round == 0 or finish_round >= r:
                    sim_round[r]["reached"] += 1

            if finish_method == "SUB":
                any_sub += 1
                if winner == 0:
                    red_sub += 1
                elif winner == 1:
                    blue_sub += 1
                if finish_round in finish_round_counts:
                    finish_round_counts[finish_round] += 1
                    sim_round[finish_round]["sub"] += 1
                if attempts <= 0:
                    sub_without_attempt += 1
            elif finish_method == "KO/TKO":
                ko_count += 1
            else:
                none_count += 1

        n = float(args.paths)
        winner_id = str(bout.get("winner_id", ""))
        r_id = str(bout.get("r_id", red.get("fighter_id", "")))
        b_id = str(bout.get("b_id", blue.get("fighter_id", "")))
        actual_winner_side = ""
        if winner_id == r_id:
            actual_winner_side = "red"
        elif winner_id == b_id:
            actual_winner_side = "blue"

        predicted_side = "tie"
        if red_sub > blue_sub:
            predicted_side = "red"
        elif blue_sub > red_sub:
            predicted_side = "blue"

        rows.append({
            "bout_id": bout_id,
            "r_name": base._display_name(red),
            "b_name": base._display_name(blue),
            "actual_method": str(bout.get("method", "")),
            "actual_finish_round": bout["actual_finish_round_resolved"],
            "actual_submission_within_horizon": int(bout["actual_submission_within_horizon"]),
            "actual_submission_winner_side": actual_winner_side if int(bout["actual_submission_within_horizon"]) else "",
            "historical_sub_attempts": float(bout["historical_sub_attempts"]),
            "p_red_sub": red_sub / n,
            "p_blue_sub": blue_sub / n,
            "p_any_sub": any_sub / n,
            "p_ko_tko": ko_count / n,
            "p_none": none_count / n,
            "sim_sub_attempts_per_path": attempt_total / n,
            "sim_paths_with_sub_attempt": attempt_paths / n,
            "predicted_sub_side": predicted_side,
            "sub_direction_hit": (
                int(predicted_side == actual_winner_side)
                if int(bout["actual_submission_within_horizon"]) and predicted_side != "tie"
                else np.nan
            ),
            "sim_r1_sub": finish_round_counts[1] / n,
            "sim_r2_sub": finish_round_counts[2] / n,
            "sim_r3_sub": finish_round_counts[3] / n,
        })

        if (bout_index + 1) % 100 == 0 or bout_index + 1 == len(cohort):
            print(f"bouts {bout_index + 1:,}/{len(cohort):,}", flush=True)

    result = pd.DataFrame(rows)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "bout_level.csv", index=False)

    hist_round_rows = []
    for r in range(1, DEFAULT_ROUNDS + 1):
        reached = int((cohort["actual_finish_round_resolved"].fillna(DEFAULT_ROUNDS) >= r).sum())
        sub_count = int((
            cohort["actual_submission_within_horizon"].eq(1)
            & cohort["actual_finish_round_resolved"].eq(float(r))
        ).sum())
        sim_reached = sim_round[r]["reached"]
        sim_sub = sim_round[r]["sub"]
        hist_round_rows.append({
            "round": r,
            "historical_fights_reached": reached,
            "historical_sub_finishes": sub_count,
            "historical_sub_rate_conditional": sub_count / reached if reached else np.nan,
            "sim_path_rounds_reached": sim_reached,
            "sim_sub_finishes": sim_sub,
            "sim_sub_rate_conditional": sim_sub / sim_reached if sim_reached else np.nan,
        })
    round_df = pd.DataFrame(hist_round_rows)
    round_df.to_csv(out_dir / "round_comparison.csv", index=False)

    historical_subs = int(cohort["actual_submission_within_horizon"].sum())
    historical_sub_rate = historical_subs / len(cohort)
    simulated_sub_rate = float(result["p_any_sub"].mean())
    historical_attempts_per_fight = float(cohort["historical_sub_attempts"].mean())
    historical_attempt_fight_rate = float(cohort["historical_sub_attempts"].gt(0).mean())
    total_paths = len(cohort) * args.paths
    simulated_attempts_per_path = total_sim_sub_attempts / total_paths
    simulated_attempt_path_rate = sim_paths_with_attempt / total_paths

    actual_sub_rows = result.loc[result["actual_submission_within_horizon"].eq(1)].copy()
    non_tie = actual_sub_rows.loc[actual_sub_rows["predicted_sub_side"].ne("tie")].copy()
    ties = int(actual_sub_rows["predicted_sub_side"].eq("tie").sum())
    direction_accuracy = float(non_tie["sub_direction_hit"].mean()) if len(non_tie) else np.nan

    metrics = pd.DataFrame([{
        "cohort_bouts": len(cohort),
        "paths_per_bout": args.paths,
        "neutral_sub_finish_probability_per_attempt": combined.SUBMISSION_NEUTRAL_FINISH_PROBABILITY_PER_ATTEMPT,
        "historical_submissions_r1_r3": historical_subs,
        "historical_submission_rate": historical_sub_rate,
        "simulated_submission_rate": simulated_sub_rate,
        "submission_rate_error_pp": 100.0 * (simulated_sub_rate - historical_sub_rate),
        "historical_sub_attempts_per_fight": historical_attempts_per_fight,
        "simulated_sub_attempts_per_path": simulated_attempts_per_path,
        "historical_fights_with_sub_attempt_rate": historical_attempt_fight_rate,
        "simulated_paths_with_sub_attempt_rate": simulated_attempt_path_rate,
        "historical_submission_fights_for_direction": len(actual_sub_rows),
        "non_tie_direction_calls": len(non_tie),
        "tie_direction_calls": ties,
        "direction_accuracy_non_tie": direction_accuracy,
        "submission_finishes_without_attempt": sub_without_attempt,
    }])
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    actual_sub_rows.sort_values(["sub_direction_hit", "p_any_sub"], ascending=[True, False], na_position="last").to_csv(
        out_dir / "actual_submission_fights.csv", index=False
    )

    print("\n" + "=" * 112)
    print("FULL MATURE 2020+ SUBMISSION VALIDATION — PROVISIONAL V1")
    print("=" * 112)
    print(f"cohort bouts: {len(cohort):,}")
    print(f"paths/bout: {args.paths:,}")
    print(f"neutral P(SUB|attempt): {combined.SUBMISSION_NEUTRAL_FINISH_PROBABILITY_PER_ATTEMPT:.2%}")
    print("\nOVERALL SUBMISSION FINISH RATE")
    print(f"historical R1-R3 SUB: {historical_sub_rate:.2%} ({historical_subs}/{len(cohort)})")
    print(f"simulated R1-R3 SUB:  {simulated_sub_rate:.2%}")
    print(f"error:                 {100.0 * (simulated_sub_rate - historical_sub_rate):+.2f} pp")

    print("\nSUBMISSION ATTEMPT FREQUENCY")
    print(f"historical attempts/fight:       {historical_attempts_per_fight:.4f}")
    print(f"simulated attempts/path:         {simulated_attempts_per_path:.4f}")
    print(f"historical fights with attempt:  {historical_attempt_fight_rate:.2%}")
    print(f"simulated paths with attempt:    {simulated_attempt_path_rate:.2%}")
    print(f"SUB finishes without attempt:    {sub_without_attempt}")

    print("\nROUND-BY-ROUND — CONDITIONAL ON REACHING ROUND")
    print(round_df.to_string(index=False, formatters={
        "historical_sub_rate_conditional": lambda x: f"{x:.2%}",
        "sim_sub_rate_conditional": lambda x: f"{x:.2%}",
    }))

    print("\nSUBMISSION WINNER DIRECTION — HISTORICAL R1-R3 SUB FIGHTS")
    print(f"historical SUB fights: {len(actual_sub_rows)}")
    print(f"non-tie calls:         {len(non_tie)}")
    print(f"ties:                  {ties}")
    if len(non_tie):
        print(f"direction accuracy:    {direction_accuracy:.2%}")
    else:
        print("direction accuracy:    n/a")

    print("\nSaved:")
    print(f"  {out_dir / 'metrics.csv'}")
    print(f"  {out_dir / 'round_comparison.csv'}")
    print(f"  {out_dir / 'bout_level.csv'}")
    print(f"  {out_dir / 'actual_submission_fights.csv'}")
    print("Research-only; no production artifacts or frozen KO benchmark outputs are modified.")


if __name__ == "__main__":
    main()
