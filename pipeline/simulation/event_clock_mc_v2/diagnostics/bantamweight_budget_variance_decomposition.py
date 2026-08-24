"""Research-only decomposition of bantamweight favorite compression from budget variance.

Posterior-mean FSR, locked i10_b0 mechanics, and historical offered/legacy-consensus
moneylines.  The fitted budget generator is never refit.  Variance arms average
independent draws from that same generator, preserving expected budgets while
reducing budget-level path variance.  Detailed fight mechanics remain frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, MARKET_PATH
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    _draw_budgets, _submission_inputs, load_frozen_context, DETAILED_PATH_SEED_OFFSET,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots, historical_fighter_rows, initialize_path_matchup
from pipeline.simulation.event_clock_mc_v2.canonical_c import load_kd_resistance_history, historical_kd_resistance_row, fight_with_kd_resistance
from pipeline.simulation.event_clock_mc_v2.inference import load_submission_baseline_v3, predict_target_v3, predict_feature_frame_v3
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

DIVISION = "bantamweight"
ARMS = {"current": 1, "half_sd": 4, "quarter_sd": 16, "near_deterministic_budget": 256}
COUNT_SUFFIXES = ("attempted", "landed")


def install_i10_b0():
    seq.INTERCEPT = 10.0; seq.DENOMINATOR = 12.0; seq.LOWER_CAP = -40.0; seq.UPPER_CAP = 10.0
    seq.ARMS = {"i10_b0": None}; seq._MODE = "i10_b0"
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path


def averaged_budget(pair, info, context, seed: int, n: int):
    if n == 1:
        return _draw_budgets(pair, info, context, np.random.default_rng(seed))
    draws = [_draw_budgets(pair, info, context, np.random.default_rng(seed + 100003 * j)) for j in range(n)]
    out = dict(draws[0])
    for k, v in list(out.items()):
        vals = [d.get(k) for d in draws]
        if all(isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x)) for x in vals):
            m = float(np.mean(vals))
            if k.endswith(COUNT_SUFFIXES):
                m = float(int(np.rint(max(0.0, m))))
            out[k] = m
    # Preserve obvious count constraints after rounding.
    for side in ("red", "blue"):
        for phase in ("standing", "ground"):
            a = f"{side}_{phase}_attempted"; l = f"{side}_{phase}_landed"
            if a in out and l in out: out[l] = min(float(out[l]), float(out[a]))
        a = f"{side}_td_attempted"; l = f"{side}_td_landed"
        if a in out and l in out: out[l] = min(float(out[l]), float(out[a]))
    return out


def method_key(result):
    m = str(result.get("method", ""))
    return "DEC" if m == "DEC" else ("KO" if m == "KO_TKO" else ("SUB" if m == "SUB" else m))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    install_i10_b0()
    context = load_frozen_context(V2_BUNDLE_PATH)
    fsr = load_prefight_snapshots(canonical.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    kd_history = load_kd_resistance_history(); sub_baseline = load_submission_baseline_v3()
    cohort, _ = wc_audit.select_cohort(DIVISION, 100)
    cohort["fight_id"] = cohort["fight_id"].astype(str)
    market = build_two_way_market(MARKET_PATH)
    market = market[market["fight_id"].isin(set(cohort["fight_id"]))].copy()
    cohort = cohort[cohort["fight_id"].isin(set(market["fight_id"]))].reset_index(drop=True)
    market = market.set_index("fight_id")
    if len(cohort) < 30:
        raise RuntimeError(f"unexpectedly small priced bantamweight cohort: {len(cohort)}")

    mean_test, mean_control = predict_target_v3(cohort, fsr, context["inference_models"], context["submission_scale"], context["conversion_offset"])
    pair_groups = {str(fid): g.copy() for fid, g in mean_test.groupby("fight_id", sort=False)}
    info_lookup = {str(r["fight_id"]): r for _, r in mean_control.iterrows()}

    fight_rows = []
    for fi, mr in cohort.iterrows():
        fid = str(mr["fight_id"]); mkt = market.loc[fid]
        fav_id = str(mkt["favorite_id"]); fav_side = "red" if fav_id == str(mr["r_id"]) else "blue"
        dog_side = "blue" if fav_side == "red" else "red"
        event_date = pd.Timestamp(mr["event_date"]).normalize()
        red_row, blue_row = historical_fighter_rows(fsr, event_date=event_date, fight_id=fid, fighter_ids=(str(mr["r_id"]), str(mr["b_id"])))
        matchup = initialize_path_matchup(red_row, blue_row, None, None, rng=np.random.default_rng(args.seed + fi), sample_epistemic=False)
        features = build_sampled_fight_feature_rows_v3(mr, red_record=red_row.to_dict(), blue_record=blue_row.to_dict(), red_traits=matchup.red, blue_traits=matchup.blue)
        pair, ctrl = predict_feature_frame_v3(features, context["inference_models"], context["submission_scale"], context["conversion_offset"], submission_baseline=sub_baseline)
        info = ctrl.iloc[0]; sub_rates, conv = _submission_inputs(pair)
        red_kd = historical_kd_resistance_row(kd_history, event_date=event_date, fight_id=fid, fighter_id=str(mr["r_id"]))
        blue_kd = historical_kd_resistance_row(kd_history, event_date=event_date, fight_id=fid, fighter_id=str(mr["b_id"]))
        path_fight = fight_with_kd_resistance(_fight(mr, context["fsr_all"]), red_native_resistance=float(red_kd["pre_rating"]), blue_native_resistance=float(blue_kd["pre_rating"]))

        for arm, ndraw in ARMS.items():
            counts = {"fav_win":0,"fav_dec":0,"fav_ko":0,"fav_sub":0,"dog_dec":0,"dog_ko":0,"dog_sub":0}
            budget_samples = []
            for p in range(args.paths):
                seed = args.seed + fi * 1_000_000 + p
                budgets = averaged_budget(pair, info, context, seed, ndraw)
                budget_samples.append(budgets)
                result = canonical.simulate_detailed_path(path_fight, budgets, sub_rates, conv, context["judge_model"], context["judge_features"], seed + DETAILED_PATH_SEED_OFFSET)
                winner = str(result["winner"]); mk = method_key(result)
                if winner == fav_side:
                    counts["fav_win"] += 1
                    if mk in ("DEC","KO","SUB"): counts[f"fav_{mk.lower()}"] += 1
                elif winner == dog_side and mk in ("DEC","KO","SUB"):
                    counts[f"dog_{mk.lower()}"] += 1
            rec = {"fight_id":fid,"arm":arm,"budget_draws_averaged":ndraw,"market_favorite_fair_p":float(mkt["market_favorite_fair_p"]),"favorite_won":float(mkt["favorite_won"]),"favorite_side":fav_side}
            rec.update({k:v/args.paths for k,v in counts.items()})
            for key in ("red_standing_attempted","blue_standing_attempted","red_standing_landed","blue_standing_landed","red_td_attempted","blue_td_attempted","red_control","blue_control"):
                vals = [float(b[key]) for b in budget_samples if key in b]
                if vals:
                    rec[f"mean_{key}"] = float(np.mean(vals)); rec[f"sd_{key}"] = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
            fight_rows.append(rec)

    out = pd.DataFrame(fight_rows)
    out["bucket"] = pd.cut(out["market_favorite_fair_p"], [0.5,0.6,0.7,0.8,0.9,1.01], labels=["50-60","60-70","70-80","80-90","90+"], right=False)
    bucket = out.groupby(["arm","bucket"], observed=True).agg(fights=("fight_id","size"),market_p=("market_favorite_fair_p","mean"),fav_win=("fav_win","mean"),actual_fav_win=("favorite_won","mean"),dog_dec=("dog_dec","mean"),dog_ko=("dog_ko","mean"),dog_sub=("dog_sub","mean")).reset_index()
    bucket["compression_pp"] = 100*(bucket["market_p"]-bucket["fav_win"])
    overall = out.groupby("arm").agg(fights=("fight_id","size"),market_p=("market_favorite_fair_p","mean"),fav_win=("fav_win","mean"),actual_fav_win=("favorite_won","mean"),dog_dec=("dog_dec","mean"),dog_ko=("dog_ko","mean"),dog_sub=("dog_sub","mean")).reset_index()
    overall["compression_pp"] = 100*(overall["market_p"]-overall["fav_win"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir/"fight_arm_results.csv", index=False); bucket.to_csv(args.out_dir/"market_bucket_results.csv", index=False); overall.to_csv(args.out_dir/"overall_results.csv", index=False)
    print("BANTAMWEIGHT BUDGET VARIANCE DECOMPOSITION")
    print(f"priced fights={out['fight_id'].nunique()} paths/fight/arm={args.paths} | posterior means | i10_b0 | sequence off")
    print("\nOVERALL"); print(overall.to_string(index=False, float_format=lambda x:f"{x:.4f}"))
    print("\nMARKET BUCKETS"); print(bucket.to_string(index=False, float_format=lambda x:f"{x:.4f}"))

if __name__ == "__main__": main()
