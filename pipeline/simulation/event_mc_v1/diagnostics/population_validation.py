"""Compact, leakage-safe Phase 6 historical population validation harness."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.common.fight_time import elapsed_fight_time_seconds

from ..components.profiles import FighterProfile, MatchupProfiles
from ..flow_stats import FlowStatsSink
from ..single_fight import HistoricalFight, build_engine
from .distance_parity import FSR_32_PATH


METHODS = ("KO_TKO", "SUB", "DEC")


def normalize_method(value: object) -> str | None:
    text = str(value).upper()
    if "SUBMISSION" in text:
        return "SUB"
    if "KO" in text or "TKO" in text:
        return "KO_TKO"
    if "DECISION" in text:
        return "DEC"
    return None


def observed_duration_seconds(row, *, match_time_semantics="elapsed") -> float:
    """Return historical exposure under an explicit master-time contract.

    Authoritative master ``match_time_sec`` is already total elapsed fight
    time. Legacy final-round-clock values are supported only when callers
    explicitly identify that older semantic; they are never guessed from the
    numeric value.
    """
    if match_time_semantics == "elapsed":
        return float(row["match_time_sec"])
    if match_time_semantics == "legacy_final_round":
        return elapsed_fight_time_seconds(row["finish_round"], row["match_time_sec"])
    raise ValueError(f"unsupported match_time_semantics: {match_time_semantics}")


def build_cohort(start_year=2020, limit=None, weight_class=None):
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["event_date"] = pd.to_datetime(master["date"])
    master = master[(master["winner"].notna()) & (master["event_date"].dt.year >= start_year)]
    if weight_class:
        master = master[master["division"] == weight_class]
    fsr = pd.read_parquet(FSR_32_PATH).copy()
    fsr["event_date"] = pd.to_datetime(fsr["date"])
    eligible_ids = set(fsr.loc[fsr["prior_ufc_fights"] >= 3].groupby("fight_id").filter(lambda x: len(x) >= 2)["fight_id"].astype(str))
    master = master[master["fight_id"].astype(str).isin(eligible_ids)].sort_values(["event_date", "fight_id"])
    if limit:
        master = master.head(limit)
    return master.reset_index(drop=True), fsr


def _fight(row, fsr):
    snapshots = fsr[fsr["fight_id"].astype(str) == str(row["fight_id"])].set_index("fighter_name")
    def profile(name):
        values = snapshots.loc[name].to_dict(); values["fighter_name"] = name
        return FighterProfile.from_mapping(values)
    return HistoricalFight(str(row["fight_id"]), row["event_date"].date().isoformat(), str(row["r_name"]), str(row["b_name"]), str(row["division"]), int(row["total_rounds"]), MatchupProfiles(profile(row["r_name"]), profile(row["b_name"])))


def simulate_fight(row, fsr, paths, seed):
    fight = _fight(row, fsr); results = []
    for index in range(paths):
        result = build_engine(fight, seed + index, FlowStatsSink())[0].run()
        results.append(result)
    methods = Counter(r.state.finish_method for r in results)
    winners = Counter(r.state.winner for r in results)
    kds = [sum(x.knockdown for x in r.sink_result["physiology"]) for r in results]
    sub_attempts = [sum(r.sink_result["attempts"][s].get("submission_attempt", 0) for s in ("red", "blue")) for r in results]
    finish_times = [r.state.fight_time_seconds for r in results if r.state.finish_method != "DEC"]
    finish_rounds = Counter(int(max(t - 1e-12, 0) // 300) + 1 for t in finish_times)
    total_exposure = sum(r.state.fight_time_seconds for r in results)
    actual_red = int(str(row["winner"]) == str(row["r_name"]))
    output = {
        "fight_id": fight.fight_id, "event_date": fight.date, "year": row["event_date"].year,
        "red_fighter": fight.red_name, "blue_fighter": fight.blue_name, "division": fight.division,
        "scheduled_rounds": fight.rounds, "actual_red_win": actual_red, "actual_method": normalize_method(row["method"]),
        "actual_finish_round": int(row["finish_round"]), "actual_duration_seconds": observed_duration_seconds(row),
        "historical_kd": float(row["r_kd"] + row["b_kd"]), "historical_sub_attempts": float(row["r_sub_att"] + row["b_sub_att"]),
        "red_win_probability": winners["red"] / paths, "blue_win_probability": winners["blue"] / paths,
        "average_simulated_finish_time": float(np.mean(finish_times)) if finish_times else np.nan,
        "simulated_nondecision_paths": len(finish_times), "simulated_finish_time_sum_seconds": float(sum(finish_times)),
        "simulated_total_kd": int(sum(kds)), "simulated_total_exposure_seconds": float(total_exposure),
        "simulated_total_submission_attempts": int(sum(sub_attempts)),
        "simulated_paths_with_submission_attempt": int(sum(x >= 1 for x in sub_attempts)),
        "simulated_total_path_count": paths,
        "kd_per_path": float(np.mean(kds)), "zero_kd_share": float(np.mean(np.array(kds) == 0)), "multi_kd_share": float(np.mean(np.array(kds) >= 2)),
        "submission_attempts_per_path": float(np.mean(sub_attempts)),
    }
    for method in METHODS:
        output[f"{method.lower()}_probability"] = methods[method] / paths
        for side in ("red", "blue"):
            output[f"{side}_{method.lower()}_probability"] = sum(r.state.finish_method == method and r.state.winner == side for r in results) / paths
    for round_no in range(1, 6):
        output[f"sim_finish_r{round_no}_share"] = finish_rounds[round_no] / max(len(finish_times), 1)
        output[f"sim_finish_r{round_no}_count"] = finish_rounds[round_no]
    for side in ("red", "blue"):
        attempts = [r.sink_result["attempts"][side] for r in results]
        outcomes = [r.sink_result["outcomes"][side] for r in results]
        output[f"{side}_td_attempts"] = float(np.mean([x.get("takedown", 0) + x.get("clinch_takedown", 0) for x in attempts]))
        output[f"{side}_td_completions"] = float(np.mean([x.get("takedown_landed", 0) + x.get("clinch_takedown_landed", 0) for x in outcomes]))
        output[f"{side}_strike_attempts"] = float(np.mean([sum(v for k, v in x.items() if "strike" in k) for x in attempts]))
        output[f"{side}_strikes_landed"] = float(np.mean([sum(v for k, v in x.items() if "strike_landed" in k) for x in outcomes]))
    return output


def compute_metrics(rows: pd.DataFrame):
    if rows.empty:
        return {}
    y = rows["actual_red_win"].to_numpy(float); p = rows["red_win_probability"].to_numpy(float)
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    bins = pd.cut(p, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = rows.assign(_bin=bins).groupby("_bin", observed=False).agg(fights=("fight_id", "size"), mean_predicted_red=("red_win_probability", "mean"), actual_red_rate=("actual_red_win", "mean")).reset_index()
    historical_methods = rows["actual_method"].value_counts(normalize=True)
    historical_finishes = rows[rows["actual_method"] != "DEC"]
    total_nondecision = rows["simulated_nondecision_paths"].sum() if "simulated_nondecision_paths" in rows else 0
    simulated_finish_rounds = {
        str(round_no): float(rows[f"sim_finish_r{round_no}_count"].sum() / max(total_nondecision, 1))
        for round_no in range(1, 6)
    } if "sim_finish_r1_count" in rows else {}
    total_simulated_kd = rows["simulated_total_kd"].sum() if "simulated_total_kd" in rows else np.nan
    total_simulated_exposure = rows["simulated_total_exposure_seconds"].sum() if "simulated_total_exposure_seconds" in rows else np.nan
    return {
        "fights": len(rows), "winner_accuracy": float(np.mean((p >= .5) == y)),
        "brier_score": float(np.mean((p - y) ** 2)), "log_loss": float(-np.mean(y * np.log(safe) + (1-y) * np.log(1-safe))),
        "mean_probability_actual_winner": float(np.mean(np.where(y == 1, p, 1-p))),
        "historical_method_shares": {m: float(historical_methods.get(m, 0)) for m in METHODS},
        "simulated_method_shares": {m: float(rows[f"{m.lower()}_probability"].mean()) for m in METHODS},
        "historical_kd_per_fight": float(rows["historical_kd"].mean()), "simulated_kd_per_path": float(rows["kd_per_path"].mean()),
        "historical_kd_per_15_minutes": float(rows["historical_kd"].sum() / rows["actual_duration_seconds"].sum() * 900),
        "simulated_kd_per_15_minutes": float(total_simulated_kd / total_simulated_exposure * 900),
        "historical_zero_kd_share": float((rows["historical_kd"] == 0).mean()), "historical_multi_kd_share": float((rows["historical_kd"] >= 2).mean()),
        "simulated_zero_kd_share": float(rows["zero_kd_share"].mean()), "simulated_multi_kd_share": float(rows["multi_kd_share"].mean()),
        "historical_submission_attempts_per_fight": float(rows["historical_sub_attempts"].mean()),
        "historical_share_with_submission_attempt": float((rows["historical_sub_attempts"] > 0).mean()),
        "simulated_submission_attempts_per_path": float(rows["submission_attempts_per_path"].mean()),
        "simulated_share_with_submission_attempt": float(rows["simulated_paths_with_submission_attempt"].sum() / rows["simulated_total_path_count"].sum()),
        "historical_finish_round_shares": {str(round_no): float((historical_finishes["actual_finish_round"] == round_no).mean()) for round_no in range(1, 6)},
        "simulated_finish_round_shares": simulated_finish_rounds,
        "historical_finish_time_mean": float(historical_finishes["actual_duration_seconds"].mean()),
        "historical_finish_time_median": float(historical_finishes["actual_duration_seconds"].median()),
        "simulated_finish_time_mean": float(rows["simulated_finish_time_sum_seconds"].sum() / max(total_nondecision, 1)),
        "calibration_bins": calibration.assign(_bin=calibration["_bin"].astype(str)).rename(columns={"_bin":"bin"}).to_dict("records"),
    }


def breakdown(rows, column):
    output = {}
    for key, group in rows.groupby(column):
        output[str(key)] = {"fights": len(group), "historical": {m: float((group.actual_method == m).mean()) for m in METHODS}, "simulated": {m: float(group[f"{m.lower()}_probability"].mean()) for m in METHODS}}
    return output


def run(paths=50, start_year=2020, limit=None, seed=20260813, output_dir=Path("data/diagnostics/event_mc_v1_population"), weight_class=None):
    cohort, fsr = build_cohort(start_year, limit, weight_class); started = time.perf_counter(); rows=[]
    for index, (_, row) in enumerate(cohort.iterrows()):
        rows.append(simulate_fight(row, fsr, paths, seed + index * 100000))
    elapsed = time.perf_counter() - started; frame = pd.DataFrame(rows); summary = compute_metrics(frame)
    summary.update({"paths_per_fight": paths, "total_paths": len(frame)*paths, "runtime_seconds": elapsed, "paths_per_second": len(frame)*paths/elapsed if elapsed else 0, "fights_per_second": len(frame)/elapsed if elapsed else 0, "by_scheduled_rounds": breakdown(frame, "scheduled_rounds"), "by_weight_class": breakdown(frame, "division")})
    output_dir.mkdir(parents=True, exist_ok=True); frame.to_csv(output_dir / "fight_level.csv", index=False); (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True)); return frame, summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--paths",type=int,default=50); parser.add_argument("--start-year",type=int,default=2020); parser.add_argument("--limit",type=int); parser.add_argument("--seed",type=int,default=20260813); parser.add_argument("--output-dir",type=Path,default=Path("data/diagnostics/event_mc_v1_population")); parser.add_argument("--weight-class")
    args=parser.parse_args(); run(**vars(args))


if __name__ == "__main__": main()
