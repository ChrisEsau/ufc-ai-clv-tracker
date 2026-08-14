"""Leakage-safe predictive replay on the first 100 eligible post-2025-03-22 fights."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH

from ..calibration import DEFAULT_CALIBRATION
from ..flow_stats import FlowStatsSink
from ..single_fight import build_engine
from .distance_parity import FSR_32_PATH
from .phase7b_kd_calibration import temporal_cohorts
from .phase7m_td_success_calibration import (
    current_calibration_values,
    global_comparison,
    render_comparison,
)
from .population_validation import METHODS, _fight, normalize_method, observed_duration_seconds

CUTOFF = pd.Timestamp("2025-03-22")
EXPECTED_FSR_SHA256 = "621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a"
JOINT_CLASSES = tuple(f"{side}_{method}" for side in ("red", "blue") for method in METHODS)
STRIKE_FAMILIES = {"distance": "strike", "clinch": "clinch_strike", "ground": "ground_strike"}


def fsr_sha256(path: Path = FSR_32_PATH) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_fresh_cohort(limit: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select chronologically without inspecting outcome class beyond eligibility."""
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["event_date"] = pd.to_datetime(master["date"])
    candidates = master[master.event_date > CUTOFF].sort_values(["event_date", "fight_id"])
    fsr = pd.read_parquet(FSR_32_PATH).copy()
    valid_fsr = fsr[fsr.prior_ufc_fights >= 3].groupby(fsr.fight_id.astype(str)).size()
    valid_fsr_ids = set(valid_fsr[valid_fsr >= 2].index)
    train, holdout, _ = temporal_cohorts(100, 50)
    calibration_ids = set(train.fight_id.astype(str)) | set(holdout.fight_id.astype(str))

    selected, missing_fsr, unsupported = [], 0, 0
    for _, row in candidates.iterrows():
        fight_id = str(row.fight_id)
        method = normalize_method(row.method)
        supported = (
            row.winner in {row.r_name, row.b_name}
            and method in METHODS
            and int(row.total_rounds) in {3, 5}
            and pd.notna(row.match_time_sec)
        )
        if not supported:
            unsupported += 1
            continue
        if fight_id in calibration_ids:
            continue
        if fight_id not in valid_fsr_ids:
            missing_fsr += 1
            continue
        selected.append(row)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise RuntimeError(f"required {limit} eligible fresh fights, found {len(selected)}")
    cohort = pd.DataFrame(selected).reset_index(drop=True)
    metadata = {
        "cutoff_exclusive": str(CUTOFF.date()),
        "first_event_date": str(cohort.event_date.min().date()),
        "last_event_date": str(cohort.event_date.max().date()),
        "bout_ids": cohort.fight_id.astype(str).tolist(),
        "excluded_missing_fsr_before_cohort_completed": missing_fsr,
        "excluded_unsupported_or_incomplete_before_cohort_completed": unsupported,
        "calibration_overlap_count": int(cohort.fight_id.astype(str).isin(calibration_ids).sum()),
    }
    return cohort, fsr, metadata


def build_simulation_inputs(cohort: pd.DataFrame, fsr: pd.DataFrame):
    """Build simulator objects before actual outcomes are attached for scoring."""
    return [_fight(row, fsr) for _, row in cohort.iterrows()]


def _simulate_one(args):
    fight_index, fight, paths, seed = args
    joint, strike_attempts, strike_landed = Counter(), Counter(), Counter()
    td_attempts = []; td_landed = []; kds = []; subs = []; exposures = []; nondecision = []
    phase_seconds = Counter()
    for path_index in range(paths):
        path_seed = seed + fight_index * 100000 + path_index
        result = build_engine(fight, path_seed, FlowStatsSink())[0].run()
        stats = result.sink_result
        phase_seconds.update(stats["phase_seconds"])
        winner, method = result.state.winner, result.state.finish_method
        joint[f"{winner}_{method}"] += 1
        exposures.append(float(result.state.fight_time_seconds))
        if method != "DEC": nondecision.append(float(result.state.fight_time_seconds))
        for phase, family in STRIKE_FAMILIES.items():
            strike_attempts[phase] += sum(stats["attempts"][side].get(family, 0) for side in ("red", "blue"))
            strike_landed[phase] += sum(stats["outcomes"][side].get(f"{family}_landed", 0) for side in ("red", "blue"))
        path_td_attempts = sum(stats["attempts"][side].get(family, 0) for side in ("red", "blue") for family in ("takedown", "clinch_takedown"))
        path_td_landed = sum(stats["outcomes"][side].get(f"{family}_landed", 0) for side in ("red", "blue") for family in ("takedown", "clinch_takedown"))
        td_attempts.append(path_td_attempts); td_landed.append(path_td_landed)
        kds.append(sum(int(item.knockdown) for item in stats["physiology"]))
        subs.append(sum(stats["attempts"][side].get("submission_attempt", 0) for side in ("red", "blue")))
    return {
        "fight_index": fight_index, "joint_counts": dict(joint), "paths": paths,
        "mean_elapsed": float(np.mean(exposures)), "exposures": exposures,
        "nondecision": nondecision, "strike_attempts": dict(strike_attempts),
        "strike_landed": dict(strike_landed), "td_attempts": td_attempts,
        "td_landed": td_landed, "kds": kds, "sub_attempts": subs,
        "phase_seconds": dict(phase_seconds),
    }


def probabilities_from_counts(counts: dict, paths: int) -> dict:
    joint = {key: counts.get(key, 0) / paths for key in JOINT_CLASSES}
    red = sum(joint[f"red_{method}"] for method in METHODS)
    methods = {method: joint[f"red_{method}"] + joint[f"blue_{method}"] for method in METHODS}
    return {"joint": joint, "red": red, "blue": 1.0 - red, "methods": methods}


def _safe_log(value):
    return math.log(max(float(value), 1e-12))


def _confidence_buckets(rows: pd.DataFrame) -> list[dict]:
    bounds = ((.50, .55), (.55, .60), (.60, .65), (.65, .70), (.70, .75),
              (.75, .80), (.80, .90), (.90, 1.0000001))
    output = []
    for low, high in bounds:
        mask = (rows.predicted_winner_probability >= low) & (rows.predicted_winner_probability < high)
        group = rows[mask]
        output.append({"bucket": f"{low:.0%}-{min(high,1):.0%}", "fights": len(group),
                       "average_confidence": float(group.predicted_winner_probability.mean()) if len(group) else None,
                       "accuracy": float(group.winner_correct.mean()) if len(group) else None})
    return output


def score_rows(rows: pd.DataFrame) -> dict:
    actual_red = (rows.actual_side == "red").astype(float).to_numpy()
    pred_red = rows.P_red_win.to_numpy(float)
    winner_log_loss = -float(np.mean(actual_red * np.log(np.clip(pred_red, 1e-12, 1)) + (1-actual_red) * np.log(np.clip(1-pred_red, 1e-12, 1))))
    method_log_loss = -float(np.mean([_safe_log(row[f"P_{row.actual_method}"]) for _, row in rows.iterrows()]))
    method_brier = float(np.mean([sum((row[f"P_{m}"] - (m == row.actual_method)) ** 2 for m in METHODS) for _, row in rows.iterrows()]))
    joint_log_loss = -float(np.mean([_safe_log(row[f"P_{row.actual_side}_{row.actual_method}"]) for _, row in rows.iterrows()]))
    confusion = pd.crosstab(rows.actual_method, rows.predicted_method).reindex(index=METHODS, columns=METHODS, fill_value=0)
    joint_actual = rows.actual_side + "_" + rows.actual_method
    joint_confusion = pd.crosstab(joint_actual, rows.predicted_winner_method).reindex(index=JOINT_CLASSES, columns=JOINT_CLASSES, fill_value=0)
    return {
        "winner": {"correct": int(rows.winner_correct.sum()), "accuracy": float(rows.winner_correct.mean()),
                   "brier": float(np.mean((pred_red-actual_red)**2)), "log_loss": winner_log_loss,
                   "actual_red_win_rate": float(actual_red.mean()), "mean_predicted_red_probability": float(pred_red.mean()),
                   "red_prediction_accuracy": float(rows.loc[rows.predicted_side == "red", "winner_correct"].mean()) if (rows.predicted_side == "red").any() else None,
                   "blue_prediction_accuracy": float(rows.loc[rows.predicted_side == "blue", "winner_correct"].mean()) if (rows.predicted_side == "blue").any() else None,
                   "confidence_buckets": _confidence_buckets(rows),
                   "thresholds": {str(x): {"fights": int((rows.predicted_winner_probability >= x).sum()), "accuracy": float(rows.loc[rows.predicted_winner_probability >= x, "winner_correct"].mean()) if (rows.predicted_winner_probability >= x).any() else None} for x in (.55,.60,.65,.70,.75,.80)}},
        "method": {"correct": int(rows.method_correct.sum()), "accuracy": float(rows.method_correct.mean()),
                   "brier": method_brier, "log_loss": method_log_loss, "confusion": confusion.to_dict(),
                   "actual_shares": {m: float((rows.actual_method == m).mean()) for m in METHODS},
                   "mean_probabilities": {m: float(rows[f"P_{m}"].mean()) for m in METHODS},
                   "predicted_counts": {m: int((rows.predicted_method == m).sum()) for m in METHODS},
                   "recall": {m: float(rows.loc[rows.actual_method == m, "method_correct"].mean()) for m in METHODS}},
        "joint": {"correct": int(rows.winner_method_correct.sum()), "accuracy": float(rows.winner_method_correct.mean()),
                  "log_loss": joint_log_loss, "mean_probability_actual": float(rows.joint_probability_assigned_to_actual_result.mean()),
                  "confusion": joint_confusion.to_dict(),
                  "actual_shares": {c: float((joint_actual == c).mean()) for c in JOINT_CLASSES},
                  "mean_probabilities": {c: float(rows[f"P_{c}"].mean()) for c in JOINT_CLASSES}},
        "timing": {"historical_mean_duration": float(rows.actual_elapsed_seconds.mean()),
                   "simulated_mean_duration": float(rows.simulated_mean_elapsed_seconds.mean()),
                   "expected_duration_mae": float((rows.simulated_mean_elapsed_seconds-rows.actual_elapsed_seconds).abs().mean()),
                   "historical_mean_nondecision_finish": float(rows.loc[rows.actual_method != "DEC", "actual_elapsed_seconds"].mean()),
                   "simulated_mean_nondecision_finish": float(rows.simulated_nondecision_sum_seconds.sum() / rows.simulated_nondecision_count.sum())},
    }


def _simulated_global(raw: list[dict]) -> dict:
    exposures = np.array([x for item in raw for x in item["exposures"]], float)
    td_a = pd.Series([x for item in raw for x in item["td_attempts"]], dtype=float)
    td_l = pd.Series([x for item in raw for x in item["td_landed"]], dtype=float)
    kds = np.array([x for item in raw for x in item["kds"]], float)
    subs = np.array([x for item in raw for x in item["sub_attempts"]], float)
    joint = Counter(); attempts = Counter(); landed = Counter(); phases = Counter(); nondec = []
    for item in raw:
        joint.update(item["joint_counts"]); attempts.update(item["strike_attempts"]); landed.update(item["strike_landed"]); phases.update(item["phase_seconds"]); nondec.extend(item["nondecision"])
    paths, exposure = len(exposures), float(exposures.sum())
    total_a, total_l = sum(attempts.values()), sum(landed.values())
    td_attempt_total, td_landed_total = float(td_a.sum()), float(td_l.sum())
    ratio = lambda n,d: float(n/d) if d else 0.0
    return {"paths": paths, "exposure_seconds": exposure,
        "total": {"attempts_per_path": ratio(td_attempt_total, paths), "landed_per_path": ratio(td_landed_total, paths),
                  "attempts_per_15min": ratio(td_attempt_total*900, exposure), "landed_per_15min": ratio(td_landed_total*900, exposure),
                  "success_percentage": ratio(td_landed_total, td_attempt_total), "paths_with_attempt_share": float((td_a>=1).mean()),
                  "paths_with_landed_share": float((td_l>=1).mean()), "zero_attempt_share": float((td_a==0).mean()),
                  "multi_attempt_share": float((td_a>=2).mean()), "attempt_quantiles": {str(q):float(td_a.quantile(q)) for q in (.25,.5,.75)},
                  "landed_quantiles": {str(q):float(td_l.quantile(q)) for q in (.25,.5,.75)}},
        "guardrails": {"strike_attempts_per_path":ratio(total_a,paths), "strike_attempts_per_15min":ratio(total_a*900,exposure),
            "strike_landed_per_path":ratio(total_l,paths), "strike_landed_per_15min":ratio(total_l*900,exposure), "strike_landing_percentage":ratio(total_l,total_a),
            "strike_phase": {p:{"attempts_per_path":ratio(attempts[p],paths), "attempts_per_15min":ratio(attempts[p]*900,exposure),
                "landed_per_path":ratio(landed[p],paths), "landed_per_15min":ratio(landed[p]*900,exposure), "accuracy":ratio(landed[p],attempts[p]),
                "attempt_share":ratio(attempts[p],total_a), "landed_share":ratio(landed[p],total_l)} for p in STRIKE_FAMILIES},
            "method_shares": {m:ratio(joint[f"red_{m}"]+joint[f"blue_{m}"],paths) for m in METHODS},
            "kd_per_path":float(kds.mean()), "kd_per_15min":ratio(kds.sum()*900,exposure), "kd_per_100_landed":ratio(kds.sum()*100,total_l),
            "zero_kd_share":float((kds==0).mean()), "multi_kd_share":float((kds>=2).mean()),
            "submission_attempts_per_path":float(subs.mean()), "submission_attempts_per_15min":ratio(subs.sum()*900,exposure),
            "paths_with_submission_attempt_share":float((subs>0).mean()), "p_sub_given_attempt":ratio(sum(joint[f"{s}_SUB"] for s in ("red","blue")),subs.sum()),
            "mean_fight_duration":float(exposures.mean()), "mean_nondecision_finish_time":float(np.mean(nondec)),
            **{f"{p}_seconds_per_path": ratio(phases[p], paths) for p in ("distance","clinch","ground")}}}


def run(paths=250, seed=20260813, workers=2, output=Path("/tmp/event_mc_fresh_100_replay.json"), csv=Path("/tmp/event_mc_fresh_100_replay.csv")):
    if fsr_sha256() != EXPECTED_FSR_SHA256:
        raise RuntimeError("frozen FSR-32 SHA-256 mismatch")
    cohort, fsr, selection = select_fresh_cohort(100)
    fights = build_simulation_inputs(cohort, fsr)
    print(json.dumps(selection, indent=2)); started = time.perf_counter()
    raw = [None] * len(fights)
    tasks = [(i, fight, paths, seed) for i, fight in enumerate(fights)]
    if workers == 1:
        for task in tasks: raw[task[0]] = _simulate_one(task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_simulate_one, task): task[0] for task in tasks}
            for future in as_completed(futures): raw[futures[future]] = future.result()

    rows = []
    for index, (_, actual) in enumerate(cohort.iterrows()):
        probs = probabilities_from_counts(raw[index]["joint_counts"], paths)
        actual_side = "red" if actual.winner == actual.r_name else "blue"
        predicted_side = "red" if probs["red"] >= probs["blue"] else "blue"
        predicted_method = max(METHODS, key=probs["methods"].get)
        predicted_joint = max(JOINT_CLASSES, key=probs["joint"].get)
        actual_method = normalize_method(actual.method)
        row = {"event_date":str(actual.event_date.date()), "bout_id":str(actual.fight_id), "red_fighter":str(actual.r_name), "blue_fighter":str(actual.b_name),
            "P_red_win":probs["red"], "P_blue_win":probs["blue"], "predicted_winner":actual.r_name if predicted_side=="red" else actual.b_name,
            "predicted_winner_probability":probs[predicted_side], "predicted_side":predicted_side, **{f"P_{m}":probs["methods"][m] for m in METHODS},
            "predicted_method":predicted_method, "predicted_method_probability":probs["methods"][predicted_method],
            **{f"P_{c}":probs["joint"][c] for c in JOINT_CLASSES}, "predicted_winner_method":predicted_joint,
            "predicted_joint_probability":probs["joint"][predicted_joint], "actual_winner":str(actual.winner), "actual_side":actual_side,
            "actual_method":actual_method, "actual_finish_round":int(actual.finish_round), "actual_elapsed_seconds":observed_duration_seconds(actual),
            "simulated_mean_elapsed_seconds":raw[index]["mean_elapsed"], "simulated_nondecision_mean_seconds":float(np.mean(raw[index]["nondecision"])) if raw[index]["nondecision"] else np.nan,
            "simulated_nondecision_sum_seconds":float(sum(raw[index]["nondecision"])), "simulated_nondecision_count":len(raw[index]["nondecision"])}
        row.update({"winner_correct":row["predicted_winner"]==row["actual_winner"], "method_correct":predicted_method==actual_method,
                    "winner_method_correct":predicted_joint==f"{actual_side}_{actual_method}",
                    "winner_probability_assigned_to_actual":probs[actual_side], "method_probability_assigned_to_actual":probs["methods"][actual_method],
                    "joint_probability_assigned_to_actual_result":probs["joint"][f"{actual_side}_{actual_method}"]})
        rows.append(row)
    frame = pd.DataFrame(rows); metrics = score_rows(frame)
    misses = frame[~frame.winner_correct].sort_values("predicted_winner_probability", ascending=False)
    method_misses = frame[(~frame.method_correct)&(frame.predicted_method_probability>=.60)]
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    sanity = global_comparison(cohort, rounds, _simulated_global(raw))
    report = {"selection":selection, "paths_per_fight":paths, "seed":seed, "runtime_seconds":time.perf_counter()-started,
              "fsr_sha256":EXPECTED_FSR_SHA256, "calibration":current_calibration_values(DEFAULT_CALIBRATION),
              "fight_predictions":frame.replace({np.nan:None}).to_dict("records"), "performance":metrics,
              "winner_misses":misses.replace({np.nan:None}).to_dict("records"),
              "winner_misses_by_threshold": {str(x): misses[misses.predicted_winner_probability>=x].replace({np.nan:None}).to_dict("records") for x in (.60,.65,.70,.75,.80)},
              "winner_misses_ge_70":misses[misses.predicted_winner_probability>=.70].replace({np.nan:None}).to_dict("records"), "winner_misses_ge_80":misses[misses.predicted_winner_probability>=.80].replace({np.nan:None}).to_dict("records"),
              "method_misses_ge_60":method_misses.replace({np.nan:None}).to_dict("records"), "global_sanity":sanity}
    output.write_text(json.dumps(report, indent=2, sort_keys=True)); frame.to_csv(csv,index=False)
    display = frame.drop(columns=["actual_side","predicted_side","simulated_nondecision_mean_seconds","simulated_nondecision_sum_seconds","simulated_nondecision_count"]).copy()
    probability_columns = [c for c in display if c.startswith("P_") or c.endswith("_probability") or "probability_assigned" in c]
    display[probability_columns] = display[probability_columns].map(lambda x:f"{x:.1%}")
    print(display.to_string(index=False)); print(json.dumps(metrics,indent=2,sort_keys=True)); print(render_comparison("fresh_100",sanity))
    print("WINNER MISSES\n"+misses[["event_date","bout_id","actual_winner","predicted_winner","predicted_winner_probability","actual_method","predicted_method"]].to_string(index=False))
    miss_columns = ["event_date","bout_id","actual_winner","predicted_winner","predicted_winner_probability","actual_method","predicted_method"]
    print("WINNER MISSES >=70%\n" + misses[misses.predicted_winner_probability>=.70][miss_columns].to_string(index=False))
    print("WINNER MISSES >=80%\n" + misses[misses.predicted_winner_probability>=.80][miss_columns].to_string(index=False))
    print("METHOD MISSES >=60%\n" + method_misses[["event_date","bout_id","actual_method","predicted_method","predicted_method_probability"]].to_string(index=False))
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--paths",type=int,default=250); parser.add_argument("--seed",type=int,default=20260813); parser.add_argument("--workers",type=int,default=2); parser.add_argument("--output",type=Path,default=Path("/tmp/event_mc_fresh_100_replay.json")); parser.add_argument("--csv",type=Path,default=Path("/tmp/event_mc_fresh_100_replay.csv")); run(**vars(parser.parse_args()))
