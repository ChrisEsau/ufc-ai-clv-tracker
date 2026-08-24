"""Research-only path-level audit of inputs feeding the frozen decision judge.

Uses the exact 41 priced men's bantamweight fights, posterior-mean FSR traits,
locked i10_b0 mechanics, and the frozen judge. No fitting or tuning.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    _draw_budgets, _submission_inputs, load_frozen_context,
    DETAILED_PATH_SEED_OFFSET, EPISTEMIC_SEED_OFFSET,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    load_prefight_snapshots, historical_fighter_rows, historical_uncertainty_rows,
    initialize_path_matchup,
)
from pipeline.simulation.event_clock_mc_v2.canonical_c import (
    load_kd_resistance_history, historical_kd_resistance_row, fight_with_kd_resistance,
)
from pipeline.simulation.event_clock_mc_v2.inference import load_submission_baseline_v3, predict_feature_frame_v3
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

DIVISION = "bantamweight"
SEED = 20260823


def install_i10_b0():
    seq.INTERCEPT = 10.0
    seq.DENOMINATOR = 12.0
    seq.LOWER_CAP = -40.0
    seq.UPPER_CAP = 10.0
    seq.ARMS = {"i10_b0": None}
    seq._MODE = "i10_b0"
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path


class RecordingJudge:
    def __init__(self, base):
        self.base = base
        self.last_row = None
        self.last_p_red = None
    def predict_proba(self, frame):
        out = self.base.predict_proba(frame)
        self.last_row = frame.iloc[0].to_dict()
        self.last_p_red = float(out[0, 1])
        return out


def fav_orient(value, fav_side):
    value = float(value)
    return value if fav_side == "red" else -value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priced-fights-path", type=Path, required=True)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    market = pd.read_csv(args.priced_fights_path).drop_duplicates("fight_id").copy()
    market["fight_id"] = market["fight_id"].astype(str)
    market_lookup = market.set_index("fight_id")

    cohort, _ = wc_audit.select_cohort(DIVISION, 100)
    cohort["fight_id"] = cohort["fight_id"].astype(str)
    cohort = cohort[cohort["fight_id"].isin(set(market["fight_id"]))].reset_index(drop=True)
    if len(cohort) != len(market):
        raise RuntimeError(f"priced cohort mismatch cohort={len(cohort)} market={len(market)}")

    install_i10_b0()
    context = load_frozen_context(V2_BUNDLE_PATH)
    judge = RecordingJudge(context["judge_model"])
    fsr = load_prefight_snapshots(canonical.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    uncertainty = canonical._load_core_uncertainty()
    kd_history = load_kd_resistance_history()
    sub_baseline = load_submission_baseline_v3()

    rows = []
    for fi, master_row in cohort.iterrows():
        fight_id = str(master_row["fight_id"])
        mkt = market_lookup.loc[fight_id]
        fav_side = str(mkt["favorite_side"])
        base_fight = _fight(master_row, context["fsr_all"])
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        red_row, blue_row = historical_fighter_rows(
            fsr, event_date=event_date, fight_id=fight_id,
            fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])))
        red_unc = historical_uncertainty_rows(uncertainty, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["r_id"]))
        blue_unc = historical_uncertainty_rows(uncertainty, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["b_id"]))
        red_kd = historical_kd_resistance_row(kd_history, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["r_id"]))
        blue_kd = historical_kd_resistance_row(kd_history, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["b_id"]))

        for path in range(args.paths):
            seed = args.seed + fi * 1_000_000 + path
            matchup = initialize_path_matchup(red_row, blue_row, red_unc, blue_unc,
                rng=np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET), sample_epistemic=False)
            path_fight = fight_with_kd_resistance(base_fight,
                red_native_resistance=float(red_kd["pre_rating"]),
                blue_native_resistance=float(blue_kd["pre_rating"]))
            features = build_sampled_fight_feature_rows_v3(master_row,
                red_record=red_row.to_dict(), blue_record=blue_row.to_dict(),
                red_traits=matchup.red, blue_traits=matchup.blue)
            pair, control = predict_feature_frame_v3(features, context["inference_models"],
                context["submission_scale"], context["conversion_offset"], submission_baseline=sub_baseline)
            info = control.iloc[0]
            sub_rates, conv = _submission_inputs(pair)
            budgets = _draw_budgets(pair, info, context, np.random.default_rng(seed))

            judge.last_row = None
            judge.last_p_red = None
            result = canonical.simulate_detailed_path(path_fight, budgets, sub_rates, conv,
                judge, context["judge_features"], seed + DETAILED_PATH_SEED_OFFSET)
            if result["method"] != "DEC":
                continue
            if judge.last_row is None:
                raise RuntimeError("decision path did not call judge")

            jr = judge.last_row
            p_fav = judge.last_p_red if fav_side == "red" else 1.0 - judge.last_p_red
            fav_won_dec = int((result["winner"] == "red") == (fav_side == "red"))
            rec = {
                "fight_id": fight_id, "path": path,
                "favorite": mkt["favorite"], "underdog": mkt["underdog"],
                "favorite_side": fav_side,
                "market_favorite_fair_p": float(mkt["market_favorite_fair_p"]),
                "favorite_won_actual": int(mkt["favorite_won"]),
                "judge_p_favorite": float(p_fav),
                "favorite_won_decision_path": fav_won_dec,
                "fav_sig_diff": fav_orient(jr["sig_diff"], fav_side),
                "fav_kd_diff": fav_orient(jr["kd_diff"], fav_side),
                "fav_td_diff": fav_orient(jr["td_diff"], fav_side),
                "fav_sub_diff": fav_orient(jr["sub_diff"], fav_side),
                "fav_ctrl_diff": fav_orient(jr["ctrl_diff"], fav_side),
            }
            fav_budget_sig = (budgets["red_standing_landed"] + budgets["red_ground_landed"])
            dog_budget_sig = (budgets["blue_standing_landed"] + budgets["blue_ground_landed"])
            if fav_side == "blue": fav_budget_sig, dog_budget_sig = dog_budget_sig, fav_budget_sig
            rec["budget_sig_land_share_fav"] = fav_budget_sig / max(fav_budget_sig + dog_budget_sig, 1e-9)
            rows.append(rec)

    paths = pd.DataFrame(rows)
    if paths.empty:
        raise RuntimeError("no decision paths captured")
    paths["bucket"] = pd.cut(paths["market_favorite_fair_p"], [0.5,0.6,0.7,0.8,0.9,1.01], labels=["50-60","60-70","70-80","80-90","90+"], right=False)
    paths["judge_favors_favorite"] = (paths["judge_p_favorite"] > 0.5).astype(int)
    paths["near_even_judge"] = paths["judge_p_favorite"].between(0.4, 0.6).astype(int)
    paths["dog_favored_judge"] = (paths["judge_p_favorite"] < 0.5).astype(int)

    agg = paths.groupby("bucket", observed=True).agg(
        decision_paths=("fight_id","size"), fights=("fight_id","nunique"),
        mean_market_favorite_p=("market_favorite_fair_p","mean"),
        mean_judge_p_favorite=("judge_p_favorite","mean"),
        favorite_decision_win_rate=("favorite_won_decision_path","mean"),
        judge_favors_favorite_rate=("judge_favors_favorite","mean"),
        dog_favored_judge_rate=("dog_favored_judge","mean"),
        near_even_judge_rate=("near_even_judge","mean"),
        mean_fav_sig_diff=("fav_sig_diff","mean"), sd_fav_sig_diff=("fav_sig_diff","std"),
        mean_fav_kd_diff=("fav_kd_diff","mean"), mean_fav_td_diff=("fav_td_diff","mean"),
        mean_fav_sub_diff=("fav_sub_diff","mean"), mean_fav_ctrl_diff=("fav_ctrl_diff","mean"),
        mean_budget_sig_land_share_fav=("budget_sig_land_share_fav","mean"),
        sd_budget_sig_land_share_fav=("budget_sig_land_share_fav","std"),
    ).reset_index()

    fight = paths.groupby(["fight_id","favorite","underdog","market_favorite_fair_p"], as_index=False).agg(
        decision_paths=("path","size"), mean_judge_p_favorite=("judge_p_favorite","mean"),
        dog_favored_judge_rate=("dog_favored_judge","mean"), near_even_judge_rate=("near_even_judge","mean"),
        mean_fav_sig_diff=("fav_sig_diff","mean"), sd_fav_sig_diff=("fav_sig_diff","std"),
        mean_fav_ctrl_diff=("fav_ctrl_diff","mean"), sd_budget_sig_land_share_fav=("budget_sig_land_share_fav","std"))

    corr_cols = ["fav_sig_diff","fav_kd_diff","fav_td_diff","fav_sub_diff","fav_ctrl_diff","budget_sig_land_share_fav"]
    corrs = pd.DataFrame([{ "metric": c,
        "corr_with_judge_p_favorite": paths[c].corr(paths["judge_p_favorite"]),
        "corr_with_market_favorite_p": paths[c].corr(paths["market_favorite_fair_p"])} for c in corr_cols])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths.to_csv(args.out_dir / "decision_path_level.csv", index=False)
    agg.to_csv(args.out_dir / "decision_path_market_buckets.csv", index=False)
    fight.to_csv(args.out_dir / "decision_path_fight_summary.csv", index=False)
    corrs.to_csv(args.out_dir / "decision_path_correlations.csv", index=False)

    print("BANTAMWEIGHT PATH-LEVEL DECISION INPUT AUDIT")
    print(f"priced fights={len(cohort)} paths/fight={args.paths} decision paths={len(paths)}")
    print("\nMARKET-STRENGTH BUCKETS")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCORRELATIONS")
    print(corrs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMOST DOG-FAVORED DECISION DISTRIBUTIONS")
    print(fight.sort_values(["dog_favored_judge_rate","mean_judge_p_favorite"], ascending=[False,True]).head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
