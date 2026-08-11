"""Full mature-cohort validation for the locked R3 damage/stamina recovery candidate.

Locked research candidate
-------------------------
- R2 entry (after R1): damage recovery 20% of missing, stamina recovery 40%.
- R3 entry (after R2): damage recovery 60% of missing, stamina recovery 0%.
- fatigue exponent 2.0.
- KD base -8.80, collapse scale 2.0, curvature 16.0.
- all other current shadow simulator / FSR settings unchanged.

Purpose
-------
1. Run the full aligned 2020+ mature FSR-32 cohort at 10 paths/bout.
2. Compare historical vs simulated R1/R2/R3 significant strikes, KDs, and KO/TKO rates.
3. Evaluate KO/TKO occurrence prediction within the 3-round simulation horizon.
4. Evaluate KO-winner direction on historical KO/TKO fights within that horizon.
5. Persist bout-level audit rows and explicit KO/error lists for later fight-by-fight review.

Important scope
---------------
This engine has no judging/decision winner layer. Therefore this script does NOT
report all-bout winner accuracy. "Winner accuracy" here means only: when a
historical fight ended by KO/TKO within R1-R3, did the simulator assign the
higher KO probability to the fighter who actually scored the KO/TKO?

Historical R4/R5 KO/TKO finishes are retained in the bout-level file but excluded
from 3-round KO occurrence/direction metrics because this validation horizon cannot
simulate those rounds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as population
from scripts.experimental import fsr_mature_2020plus_r3_recovery_compare_curve16_exp2_200 as recovery
from scripts.experimental import historical_sigstr_kd_ko_exposure_2020plus_mature as hist_rounds
from scripts.experimental import fsr_static_mc_ko_tko_v3_1_rolling_fsr as v31
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
DEFAULT_ROUNDS = 3
KO_CLASSIFICATION_THRESHOLD = 0.50

LOCKED = recovery.RecoveryCandidate("locked_r3_d60_s0", 0.60, 0.00)

OUTPUT_DIR = Path("data/experimental/full_cohort_ko_validation_r3_d60_s0")
BOUTS_PATH = OUTPUT_DIR / "bout_level.csv"
ACTUAL_KO_PATH = OUTPUT_DIR / "actual_ko_fights.csv"
ERRORS_PATH = OUTPUT_DIR / "prediction_errors.csv"
ROUND_COMPARE_PATH = OUTPUT_DIR / "round_comparison.csv"
METRICS_PATH = OUTPUT_DIR / "ko_prediction_metrics.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full mature-cohort KO validation for locked R3 60% damage / 0% stamina recovery")
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def _configure_locked_candidate() -> None:
    recovery._configure_locked_candidate()
    v31.FATIGUE_CURVE_EXPONENT = recovery.FATIGUE_EXPONENT


def _master_extra_metadata() -> pd.DataFrame:
    raw = pd.read_parquet(population.modern.MASTER_PATH).copy()
    date_col = population.modern._resolve_date_column(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    raw["fight_id"] = raw["fight_id"].astype(str)
    for col in ("r_id", "b_id", "winner_id"):
        if col in raw.columns:
            raw[col] = raw[col].astype(str)
    raw = raw.sort_values(["event_date", "fight_id"]).drop_duplicates("fight_id", keep="last")

    keep = ["fight_id"]
    for col in ("winner_id", "method", "r_name", "b_name"):
        if col in raw.columns:
            keep.append(col)
    out = raw[keep].rename(columns={"fight_id": "bout_id"})
    if "winner_id" not in out.columns:
        out["winner_id"] = ""
    if "method" not in out.columns:
        out["method"] = ""
    return out


def _build_full_cohort():
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.merge(_master_extra_metadata(), on="bout_id", how="left", validate="one_to_one")
    return cohort.reset_index(drop=True), pairs


def _fighter_name(profile: pd.Series) -> str:
    return base._display_name(profile)


def _run_prefix(red, blue, *, rounds: int, seed: int, red_age, blue_age):
    return recovery._run_prefix(
        red,
        blue,
        candidate=LOCKED,
        rounds=rounds,
        seed=seed,
        red_age=red_age,
        blue_age=blue_age,
    )


def _simulate_bout(bout: pd.Series, pair, seeds: np.ndarray) -> tuple[dict[str, object], dict[int, dict[str, int]]]:
    red, blue = pair
    r_id = str(bout["r_id"])
    b_id = str(bout["b_id"])
    winner_id = str(bout.get("winner_id", ""))
    r_age = float(bout["r_age"]) if pd.notna(bout.get("r_age")) else None
    b_age = float(bout["b_age"]) if pd.notna(bout.get("b_age")) else None

    path_count = len(seeds)
    ko_count = 0
    r_ko = 0
    b_ko = 0
    finish_round_counts = {1: 0, 2: 0, 3: 0}
    terminal_count = 0
    direct_count = 0

    # Conditional round aggregates: a path contributes to a round only if it
    # survived to the start of that round, matching historical fight-round grain.
    round_totals = {
        1: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
        2: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
        3: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
    }

    for seed in seeds:
        sim1, path1, kd1, fr1 = _run_prefix(red, blue, rounds=1, seed=int(seed), red_age=r_age, blue_age=b_age)
        sim2, path2, kd2, fr2 = _run_prefix(red, blue, rounds=2, seed=int(seed), red_age=r_age, blue_age=b_age)
        sim3, path3, kd3, fr3 = _run_prefix(red, blue, rounds=3, seed=int(seed), red_age=r_age, blue_age=b_age)

        sig1 = int(sim1.stats[0].sig_landed) + int(sim1.stats[1].sig_landed)
        sig2 = int(sim2.stats[0].sig_landed) + int(sim2.stats[1].sig_landed)
        sig3 = int(sim3.stats[0].sig_landed) + int(sim3.stats[1].sig_landed)

        round_totals[1]["reached"] += 1
        round_totals[1]["sig"] += sig1
        round_totals[1]["kd"] += kd1
        round_totals[1]["ko"] += int(path1.finish is not None and fr1 == 1)

        if path1.finish is None:
            round_totals[2]["reached"] += 1
            round_totals[2]["sig"] += max(0, sig2 - sig1)
            round_totals[2]["kd"] += max(0, kd2 - kd1)
            round_totals[2]["ko"] += int(path2.finish is not None and fr2 == 2)

        if path2.finish is None:
            round_totals[3]["reached"] += 1
            round_totals[3]["sig"] += max(0, sig3 - sig2)
            round_totals[3]["kd"] += max(0, kd3 - kd2)
            round_totals[3]["ko"] += int(path3.finish is not None and fr3 == 3)

        if path3.finish is not None:
            ko_count += 1
            rnd = int(fr3)
            if rnd in finish_round_counts:
                finish_round_counts[rnd] += 1
            winner = int(path3.finish.winner)
            if winner == 0:
                r_ko += 1
            elif winner == 1:
                b_ko += 1
            terminal_count += int(sim3.terminal_collapse_finishes > 0)
            direct_count += int(sim3.direct_strike_finishes > 0)

    p_r_ko = r_ko / path_count
    p_b_ko = b_ko / path_count
    p_any_ko = ko_count / path_count
    predicted_side = "tie"
    predicted_ko_winner_id = ""
    if p_r_ko > p_b_ko:
        predicted_side = "red"
        predicted_ko_winner_id = r_id
    elif p_b_ko > p_r_ko:
        predicted_side = "blue"
        predicted_ko_winner_id = b_id

    actual_ko = int(bout["actual_ko_tko"])
    actual_round = pd.to_numeric(pd.Series([bout["actual_finish_round"]]), errors="coerce").iloc[0]
    actual_ko_within_horizon = int(actual_ko == 1 and pd.notna(actual_round) and int(actual_round) <= DEFAULT_ROUNDS)
    actual_ko_after_horizon = int(actual_ko == 1 and pd.notna(actual_round) and int(actual_round) > DEFAULT_ROUNDS)

    actual_ko_winner_side = ""
    if actual_ko == 1 and winner_id == r_id:
        actual_ko_winner_side = "red"
    elif actual_ko == 1 and winner_id == b_id:
        actual_ko_winner_side = "blue"

    predicted_method_ko = int(p_any_ko >= KO_CLASSIFICATION_THRESHOLD)
    direction_hit = np.nan
    if actual_ko_within_horizon and predicted_ko_winner_id:
        direction_hit = int(predicted_ko_winner_id == winner_id)

    method_hit = int(predicted_method_ko == actual_ko_within_horizon)
    combined_ko_winner_method_hit = np.nan
    if actual_ko_within_horizon:
        combined_ko_winner_method_hit = int(predicted_method_ko == 1 and predicted_ko_winner_id == winner_id)

    row = {
        "bout_id": str(bout["bout_id"]),
        "event_date": bout["event_date"],
        "r_id": r_id,
        "b_id": b_id,
        "red_name": _fighter_name(red),
        "blue_name": _fighter_name(blue),
        "winner_id": winner_id,
        "actual_method": str(bout.get("method", "")),
        "actual_ko_tko": actual_ko,
        "actual_finish_round": bout["actual_finish_round"],
        "actual_ko_within_r3": actual_ko_within_horizon,
        "actual_ko_after_r3": actual_ko_after_horizon,
        "actual_ko_winner_side": actual_ko_winner_side,
        "paths": path_count,
        "p_any_ko": p_any_ko,
        "p_r_ko": p_r_ko,
        "p_b_ko": p_b_ko,
        "p_r1_ko": finish_round_counts[1] / path_count,
        "p_r2_ko": finish_round_counts[2] / path_count,
        "p_r3_ko": finish_round_counts[3] / path_count,
        "predicted_method_ko_at_50": predicted_method_ko,
        "predicted_ko_winner_side": predicted_side,
        "predicted_ko_winner_id": predicted_ko_winner_id,
        "ko_method_classification_hit": method_hit,
        "ko_winner_direction_hit": direction_hit,
        "combined_ko_winner_method_hit": combined_ko_winner_method_hit,
        "terminal_collapse_path_count": terminal_count,
        "direct_strike_path_count": direct_count,
    }
    return row, round_totals


def _historical_round_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    rs = hist_rounds._load_round_stats(hist_rounds.ROUND_STATS_PATH)
    rounds = hist_rounds._build_fight_rounds(rs, cohort)
    rows = []
    for rnd in (1, 2, 3):
        g = rounds.loc[rounds["round"].eq(rnd)]
        s = hist_rounds._summarize(g)
        rows.append({
            "round": rnd,
            "historical_fight_rounds": int(s["fight_rounds"]),
            "historical_sig_mean": float(s["mean_sig_str_landed"]),
            "historical_kd_mean": float(s["mean_kd_per_round"]),
            "historical_ko_rate": float(s["p_ko_tko"]),
            "historical_any_kd_rate": float(s["p_any_kd"]),
        })
    return pd.DataFrame(rows)


def _sim_round_summary(global_round_totals: dict[int, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for rnd in (1, 2, 3):
        g = global_round_totals[rnd]
        reached = g["reached"]
        rows.append({
            "round": rnd,
            "sim_path_rounds": reached,
            "sim_sig_mean": g["sig"] / reached if reached else np.nan,
            "sim_kd_mean": g["kd"] / reached if reached else np.nan,
            "sim_ko_rate": g["ko"] / reached if reached else np.nan,
        })
    return pd.DataFrame(rows)


def _binary_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    # R4/R5 KO/TKO fights are excluded from binary accuracy because a 3-round
    # simulator cannot represent the historical event that occurred after R3.
    work = frame.loc[frame["actual_ko_after_r3"].eq(0)].copy()
    y = work["actual_ko_within_r3"].astype(int)
    p = work["p_any_ko"].astype(float).clip(1e-6, 1 - 1e-6)
    pred = work["predicted_method_ko_at_50"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)

    metrics = {
        "eligible_bouts": int(len(work)),
        "actual_ko_rate": float(y.mean()),
        "mean_predicted_ko_probability": float(p.mean()),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if y.nunique() == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "classification_threshold": KO_CLASSIFICATION_THRESHOLD,
        "classification_accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return pd.DataFrame([metrics])


def _prediction_errors(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["error_type"] = ""
    fp = work["actual_ko_after_r3"].eq(0) & work["actual_ko_within_r3"].eq(0) & work["predicted_method_ko_at_50"].eq(1)
    fn = work["actual_ko_within_r3"].eq(1) & work["predicted_method_ko_at_50"].eq(0)
    wrong_side = work["actual_ko_within_r3"].eq(1) & work["predicted_ko_winner_id"].ne("") & work["predicted_ko_winner_id"].ne(work["winner_id"])
    tie_side = work["actual_ko_within_r3"].eq(1) & work["predicted_ko_winner_id"].eq("")
    after = work["actual_ko_after_r3"].eq(1)

    work.loc[fp, "error_type"] = "false_positive_ko"
    work.loc[fn, "error_type"] = "false_negative_ko"
    work.loc[wrong_side, "error_type"] = work.loc[wrong_side, "error_type"].replace("", "wrong_ko_winner").where(work.loc[wrong_side, "error_type"].eq(""), work.loc[wrong_side, "error_type"] + "+wrong_ko_winner")
    work.loc[tie_side, "error_type"] = work.loc[tie_side, "error_type"].replace("", "ko_winner_tie").where(work.loc[tie_side, "error_type"].eq(""), work.loc[tie_side, "error_type"] + "+ko_winner_tie")
    work.loc[after, "error_type"] = "historical_ko_after_r3_horizon"
    return work.loc[work["error_type"].ne("")].sort_values(["error_type", "event_date", "bout_id"]).reset_index(drop=True)


def _print_results(bouts: pd.DataFrame, round_compare: pd.DataFrame, metrics: pd.DataFrame) -> None:
    print("\n" + "=" * 170)
    print("FULL MATURE 2020+ KO/TKO VALIDATION — LOCKED R3 RECOVERY 60% DAMAGE / 0% STAMINA")
    print("=" * 170)
    print(f"bouts: {len(bouts):,}")
    print(f"paths/bout: {int(bouts['paths'].iloc[0]) if len(bouts) else 0}")
    print("R2 entry recovery: damage 20%, stamina 40%")
    print("R3 entry recovery: damage 60%, stamina 0%")
    print("No judging layer: all-bout winner accuracy is intentionally not reported")

    print("\nROUND COMPARISON")
    print(round_compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nKO/TKO OCCURRENCE PREDICTION — 3-ROUND HORIZON")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    ko = bouts.loc[bouts["actual_ko_within_r3"].eq(1)].copy()
    directional = ko.loc[ko["ko_winner_direction_hit"].notna()]
    print("\nKO/TKO WINNER DIRECTION — ACTUAL KO/TKO FIGHTS WITHIN R1-R3")
    print(f"actual KO/TKO fights within R3: {len(ko):,}")
    print(f"non-tie KO-side calls:          {len(directional):,}")
    print(f"tie KO-side calls:              {len(ko) - len(directional):,} ({(len(ko)-len(directional))/len(ko):.2%})" if len(ko) else "tie KO-side calls: 0")
    if len(directional):
        print(f"KO winner direction accuracy:   {directional['ko_winner_direction_hit'].mean():.2%}")
    print(f"mean P(actual KO winner KO):     {np.mean([row.p_r_ko if row.winner_id == row.r_id else row.p_b_ko for row in ko.itertuples()]):.2%}" if len(ko) else "mean P(actual KO winner KO): n/a")
    print(f"historical KO/TKO after R3 excluded from horizon metrics: {int(bouts['actual_ko_after_r3'].sum()):,}")


def main() -> None:
    args = parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    _configure_locked_candidate()
    cohort, pairs = _build_full_cohort()
    print(f"full aligned mature cohort: {len(cohort):,} bouts")
    print(f"paths per bout: {args.paths}")
    print(f"total 3-round paths: {len(cohort) * args.paths:,}")
    print("locked R3 recovery: damage=60% of missing; stamina=0% of missing")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    global_round_totals = {
        1: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
        2: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
        3: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
    }

    total_paths = len(cohort) * args.paths
    completed_paths = 0
    for bout_no, (_, bout) in enumerate(cohort.iterrows(), start=1):
        seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        row, round_totals = _simulate_bout(bout, pairs[str(bout["bout_id"])], seeds)
        rows.append(row)
        for rnd in (1, 2, 3):
            for key in ("reached", "sig", "kd", "ko"):
                global_round_totals[rnd][key] += round_totals[rnd][key]

        completed_paths += args.paths
        if completed_paths % 1000 == 0 or bout_no == len(cohort):
            print(f"paths {completed_paths:,}/{total_paths:,}; bouts {bout_no:,}/{len(cohort):,}", flush=True)

    bouts = pd.DataFrame(rows)
    historical = _historical_round_summary(cohort)
    simulated = _sim_round_summary(global_round_totals)
    round_compare = historical.merge(simulated, on="round", how="inner", validate="one_to_one")
    round_compare["sig_error_pct"] = (round_compare["sim_sig_mean"] / round_compare["historical_sig_mean"] - 1.0) * 100.0
    round_compare["kd_error_pct"] = (round_compare["sim_kd_mean"] / round_compare["historical_kd_mean"] - 1.0) * 100.0
    round_compare["ko_error_pp"] = (round_compare["sim_ko_rate"] - round_compare["historical_ko_rate"]) * 100.0

    metrics = _binary_metrics(bouts)
    errors = _prediction_errors(bouts)
    actual_ko_fights = bouts.loc[bouts["actual_ko_tko"].eq(1)].copy().sort_values(["event_date", "bout_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bouts.to_csv(args.output_dir / BOUTS_PATH.name, index=False)
    actual_ko_fights.to_csv(args.output_dir / ACTUAL_KO_PATH.name, index=False)
    errors.to_csv(args.output_dir / ERRORS_PATH.name, index=False)
    round_compare.to_csv(args.output_dir / ROUND_COMPARE_PATH.name, index=False)
    metrics.to_csv(args.output_dir / METRICS_PATH.name, index=False)

    _print_results(bouts, round_compare, metrics)

    print("\nOUTPUTS")
    for path in (
        args.output_dir / BOUTS_PATH.name,
        args.output_dir / ACTUAL_KO_PATH.name,
        args.output_dir / ERRORS_PATH.name,
        args.output_dir / ROUND_COMPARE_PATH.name,
        args.output_dir / METRICS_PATH.name,
    ):
        print(path)
    print("Research-only validation; stored FSR values and production simulator remain unchanged.")


if __name__ == "__main__":
    main()
