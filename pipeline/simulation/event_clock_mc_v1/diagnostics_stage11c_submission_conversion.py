from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.paths import FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11_submission_attempts import (
    build_submission_targets,
)

FIGHTS = 500
PATHS = 20

STAGE11B_PATHS = Path(
    "data/diagnostics/event_clock_mc_v1/stage11b_submission_clock_paths_500x20.csv"
)
OUT = Path(
    "data/diagnostics/event_clock_mc_v1/stage11c_submission_conversion_500x20.csv"
)
PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/stage11c_submission_conversion_paths_500x20.csv"
)


def clip_probability(x):
    return np.clip(np.asarray(x, dtype=float), 1e-8, 1.0 - 1e-8)


def logistic(x):
    x = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(x):
    p = clip_probability(x)
    return np.log(p / (1.0 - p))


def load_submission_baseline() -> pd.DataFrame:
    snapshots = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
        columns=["fight_id", "submission_conversion_baseline"],
    ).copy()
    snapshots["fight_id"] = snapshots["fight_id"].astype(str)
    snapshots["submission_conversion_baseline"] = pd.to_numeric(
        snapshots["submission_conversion_baseline"], errors="coerce"
    )
    if snapshots["submission_conversion_baseline"].isna().any():
        raise RuntimeError("FSR V2 submission_conversion_baseline contains nulls.")

    check = snapshots.groupby("fight_id")["submission_conversion_baseline"].agg(
        baseline_min="min",
        baseline_max="max",
        submission_conversion_baseline="median",
    ).reset_index()
    check["baseline_within_fight_range"] = (
        check["baseline_max"] - check["baseline_min"]
    )
    max_range = float(check["baseline_within_fight_range"].max())
    if max_range > 1e-10:
        raise RuntimeError(
            "submission_conversion_baseline differs within a fight; "
            f"max range={max_range:.12g}"
        )
    return check[["fight_id", "submission_conversion_baseline"]]


def add_conversion(frame: pd.DataFrame, baseline_by_fight: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(
        baseline_by_fight,
        on="fight_id",
        how="left",
        validate="many_to_one",
    )
    required = (
        "submission_conversion_baseline",
        "self_submission_offense",
        "opp_submission_defense",
    )
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise RuntimeError(f"Missing/non-numeric canonical submission input: {col}")

    baseline = clip_probability(out["submission_conversion_baseline"])
    offense = out["self_submission_offense"].to_numpy(float)
    defense = out["opp_submission_defense"].to_numpy(float)

    out["p_submission_convert_baseline"] = baseline
    out["p_submission_convert_fsr"] = logistic(logit(baseline) + offense - defense)
    out["submission_offense_edge"] = offense - defense
    return out


def at_least_one_success(per_attempt_probability, attempts):
    p = np.clip(np.asarray(per_attempt_probability, dtype=float), 0.0, 1.0)
    n = np.maximum(np.asarray(attempts, dtype=float), 0.0)
    return 1.0 - np.power(1.0 - p, n)


def binary_metrics(actual, probability):
    y = np.asarray(actual, dtype=int)
    p = clip_probability(probability)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    brier = brier_score_loss(y, p)
    ll = log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])
    return float(auc), float(brier), float(ll)


def correct_sub_side(frame: pd.DataFrame, probability_col: str):
    correct = 0.0
    total = 0
    for _, group in frame.groupby("fight_id", sort=False):
        if int(group["submission_win"].sum()) != 1 or len(group) != 2:
            continue
        winner = group.loc[group["submission_win"].eq(1)].iloc[0]
        loser = group.loc[group["submission_win"].eq(0)].iloc[0]
        wp = float(winner[probability_col])
        lp = float(loser[probability_col])
        correct += 1.0 if wp > lp else 0.5 if wp == lp else 0.0
        total += 1
    return (correct / total if total else np.nan), total


def print_conversion_model(frame, probability_col, label):
    y = frame["submission_win"].to_numpy(int)
    p = frame[probability_col].to_numpy(float)
    auc, brier, ll = binary_metrics(y, p)
    side, n = correct_sub_side(frame, probability_col)
    winner_mean = float(frame.loc[frame["submission_win"].eq(1), probability_col].mean())
    nonwinner_mean = float(frame.loc[frame["submission_win"].eq(0), probability_col].mean())

    active = frame[frame["submission_attempted"] > 0]
    active_auc, active_brier, active_ll = binary_metrics(
        active["submission_win"], active[probability_col]
    )

    print("\n" + label)
    print("-" * 140)
    print(
        f"actual fighter SUB-win rate={np.mean(y):.2%} | "
        f"predicted={np.mean(p):.2%} | AUC={auc:.4f} | "
        f"Brier={brier:.4f} | log loss={ll:.4f}"
    )
    print(
        f"actual SUB-winner mean P={winner_mean:.4f} | "
        f"non-winner mean P={nonwinner_mean:.4f} | "
        f"correct-side among SUB fights={side:.2%} (N={n})"
    )
    print(
        f"attempt-positive rows only: N={len(active)} | AUC={active_auc:.4f} | "
        f"Brier={active_brier:.4f} | log loss={active_ll:.4f}"
    )


def print_per_attempt_audit(frame):
    attempts = frame["submission_attempted"].to_numpy(float)
    wins = frame["submission_win"].to_numpy(float)
    total_attempts = float(attempts.sum())
    total_wins = float(wins.sum())
    raw = total_wins / total_attempts if total_attempts > 0 else np.nan

    baseline_weighted = float(
        np.sum(attempts * frame["p_submission_convert_baseline"].to_numpy(float))
        / max(total_attempts, 1.0)
    )
    fsr_weighted = float(
        np.sum(attempts * frame["p_submission_convert_fsr"].to_numpy(float))
        / max(total_attempts, 1.0)
    )

    print("\n" + "=" * 140)
    print("PER-ATTEMPT CONVERSION ANCHOR — FRESH 500")
    print("=" * 140)
    print(f"historical submission wins:      {total_wins:.0f}")
    print(f"historical recorded attempts:   {total_attempts:.0f}")
    print(f"raw SUB wins / attempts:        {raw:.2%}")
    print(f"attempt-weighted pop baseline:  {baseline_weighted:.2%}")
    print(f"attempt-weighted FSR matchup P: {fsr_weighted:.2%}")
    print(
        "NOTE: wins/attempts is a calibration anchor, not an exact iid per-attempt "
        "MLE because the fight terminates on a successful submission."
    )


def fight_level_overlay(path_frame: pd.DataFrame, probability_col: str):
    pair = path_frame.pivot_table(
        index=["fight_id", "path"],
        columns="side",
        values=probability_col,
        aggfunc="first",
    ).reset_index()
    if "red" not in pair or "blue" not in pair:
        raise RuntimeError("Stage 11C path frame lost red/blue pair rows.")
    red = np.clip(pair["red"].to_numpy(float), 0.0, 1.0)
    blue = np.clip(pair["blue"].to_numpy(float), 0.0, 1.0)
    pair["independent_any_sub_probability"] = 1.0 - (1.0 - red) * (1.0 - blue)
    pair["independent_double_success_probability"] = red * blue
    return pair


def main():
    print("=" * 150)
    print("EVENT CLOCK MC — STAGE 11C SUBMISSION CONVERSION DIAGNOSTIC")
    print("=" * 150)
    print(
        "Stage-11B attempt clock -> canonical FSR V2 population conversion baseline "
        "+ attacker submission offense - defender submission defense"
    )
    print("No position bonus, no age adjustment, no terminal fight ordering in this stage.")

    train, test = prepare_direct_predictions()
    for frame in (train, test):
        frame["fight_id"] = frame["fight_id"].astype(str)

    targets = build_submission_targets()[
        ["fight_id", "side", "submission_attempted", "submission_win"]
    ].copy()
    targets["fight_id"] = targets["fight_id"].astype(str)

    test = test.merge(
        targets,
        on=["fight_id", "side"],
        how="inner",
        validate="one_to_one",
    )
    if test["fight_id"].nunique() != FIGHTS:
        raise RuntimeError("Submission target join lost fresh fights.")

    baseline_by_fight = load_submission_baseline()
    test = add_conversion(test, baseline_by_fight)

    attempts = test["submission_attempted"].to_numpy(float)
    test["pred_sub_actual_attempts_baseline"] = at_least_one_success(
        test["p_submission_convert_baseline"], attempts
    )
    test["pred_sub_actual_attempts_fsr"] = at_least_one_success(
        test["p_submission_convert_fsr"], attempts
    )

    print_per_attempt_audit(test)

    print("\n" + "=" * 150)
    print("CONVERSION WITH ACTUAL RECORDED ATTEMPTS — FRESH 500")
    print("=" * 150)
    print_conversion_model(
        test,
        "pred_sub_actual_attempts_baseline",
        "POPULATION BASELINE ONLY",
    )
    print_conversion_model(
        test,
        "pred_sub_actual_attempts_fsr",
        "CANONICAL FSR OFFENSE-vs-DEFENSE",
    )

    if not STAGE11B_PATHS.exists():
        raise RuntimeError(f"Stage-11B path file not found: {STAGE11B_PATHS}")

    path_frame = pd.read_csv(STAGE11B_PATHS, low_memory=False)
    path_frame["fight_id"] = path_frame["fight_id"].astype(str)
    if path_frame["fight_id"].nunique() != FIGHTS:
        raise RuntimeError(f"Expected {FIGHTS} fights in Stage-11B path file.")
    if "sim_submission_attempted" not in path_frame:
        raise RuntimeError("Stage-11B path output is missing sim_submission_attempted.")

    static = test[
        [
            "fight_id",
            "side",
            "fighter_name",
            "opponent_name",
            "submission_win",
            "submission_attempted",
            "submission_conversion_baseline",
            "self_submission_offense",
            "opp_submission_defense",
            "p_submission_convert_baseline",
            "p_submission_convert_fsr",
            "submission_offense_edge",
        ]
    ].copy()

    path_frame = path_frame.merge(
        static,
        on=["fight_id", "side"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_stage11c"),
    )

    sim_attempts = pd.to_numeric(
        path_frame["sim_submission_attempted"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    path_frame["pred_sub_sim_attempts_baseline"] = at_least_one_success(
        path_frame["p_submission_convert_baseline"], sim_attempts
    )
    path_frame["pred_sub_sim_attempts_fsr"] = at_least_one_success(
        path_frame["p_submission_convert_fsr"], sim_attempts
    )

    sim_mean = path_frame.groupby(["fight_id", "side"], as_index=False).agg(
        pred_sub_sim_attempts_baseline=("pred_sub_sim_attempts_baseline", "mean"),
        pred_sub_sim_attempts_fsr=("pred_sub_sim_attempts_fsr", "mean"),
        sim_submission_attempted=("sim_submission_attempted", "mean"),
    )

    result = test.merge(
        sim_mean,
        on=["fight_id", "side"],
        how="left",
        validate="one_to_one",
    )

    print("\n" + "=" * 150)
    print("STAGE-11B SIMULATED ATTEMPTS + CONVERSION — FRESH 500")
    print("=" * 150)
    print_conversion_model(
        result,
        "pred_sub_sim_attempts_baseline",
        "SIM ATTEMPTS + POPULATION BASELINE",
    )
    print_conversion_model(
        result,
        "pred_sub_sim_attempts_fsr",
        "SIM ATTEMPTS + CANONICAL FSR OFFENSE-vs-DEFENSE",
    )

    historical_sub_share = float(
        result.groupby("fight_id")["submission_win"].max().mean()
    )
    base_pair = fight_level_overlay(path_frame, "pred_sub_sim_attempts_baseline")
    fsr_pair = fight_level_overlay(path_frame, "pred_sub_sim_attempts_fsr")

    print("\n" + "=" * 150)
    print("FIGHT-LEVEL SUBMISSION SHARE — INDEPENDENT OVERLAY DIAGNOSTIC")
    print("=" * 150)
    print(f"historical SUB fight share:                     {historical_sub_share:.2%}")
    print(
        "sim attempts + baseline implied SUB share:       "
        f"{base_pair['independent_any_sub_probability'].mean():.2%}"
    )
    print(
        "sim attempts + FSR implied SUB share:            "
        f"{fsr_pair['independent_any_sub_probability'].mean():.2%}"
    )
    print(
        "FSR independent double-success overlap:          "
        f"{fsr_pair['independent_double_success_probability'].mean():.3%}"
    )
    print(
        "NOTE: fight-level overlay is diagnostic only. A real terminal SUB must be "
        "ordered on the fight timeline and cannot allow both fighters to finish the same path."
    )

    print("\n" + "=" * 150)
    print("LOWEST-PROBABILITY ACTUAL SUBMISSION WINNERS — SIM ATTEMPTS + FSR")
    print("=" * 150)
    columns = [
        "fight_id",
        "fighter_name",
        "opponent_name",
        "submission_attempted",
        "sim_submission_attempted",
        "p_submission_convert_fsr",
        "submission_offense_edge",
        "pred_sub_sim_attempts_fsr",
    ]
    print(
        result.loc[result["submission_win"].eq(1), columns]
        .sort_values("pred_sub_sim_attempts_fsr", ascending=True)
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 150)
    print("HIGHEST-PROBABILITY NON-SUBMISSION FIGHTERS — SIM ATTEMPTS + FSR")
    print("=" * 150)
    print(
        result.loc[result["submission_win"].eq(0), columns]
        .sort_values("pred_sub_sim_attempts_fsr", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    path_frame.to_csv(PATH_OUT, index=False)
    print(f"\nwrote: {OUT}")
    print(f"wrote: {PATH_OUT}")
    print(
        "\nNOTE: Stage 11C measures conversion only. It does not yet terminate paths. "
        "If conversion calibration/discrimination passes, the next stage should place "
        "successful attempts onto the event timeline and enforce first-terminal-event wins."
    )


if __name__ == "__main__":
    main()
