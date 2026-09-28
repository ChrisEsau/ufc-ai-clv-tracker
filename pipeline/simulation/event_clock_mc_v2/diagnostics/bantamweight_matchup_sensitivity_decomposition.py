"""Trace bantamweight favorite-vs-dog separation through Event Clock V2.

Research-only measurement. Uses posterior-mean FSR profiles, locked i10_b0
mechanics, and the exact 41 priced bantamweight fights. Market data is used only
after simulation to orient favorite vs underdog; it never enters path generation.
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
from pipeline.simulation.event_clock_mc_v2.inference import (
    load_submission_baseline_v3, predict_target_v3, predict_feature_frame_v3,
)
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

DIVISION = "bantamweight"
INTERCEPT = 10.0
DENOMINATOR = 12.0
LOWER_CAP = -40.0


def install_i10_b0() -> None:
    seq.INTERCEPT = INTERCEPT
    seq.DENOMINATOR = DENOMINATOR
    seq.LOWER_CAP = LOWER_CAP
    seq.UPPER_CAP = INTERCEPT
    seq.ARMS = {"i10_b0": None}
    seq._MODE = "i10_b0"
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path


def orient(vr, vb, fav_side):
    return (float(vr), float(vb)) if fav_side == "red" else (float(vb), float(vr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priced-fights-path", type=Path, required=True)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260823)
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
    fsr = load_prefight_snapshots(canonical.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    uncertainty = canonical._load_core_uncertainty()
    kd_history = load_kd_resistance_history()
    sub_baseline = load_submission_baseline_v3()

    mean_test, mean_control = predict_target_v3(
        cohort, fsr, context["inference_models"], context["submission_scale"], context["conversion_offset"]
    )
    groups = {str(fid): g.copy() for fid, g in mean_test.groupby("fight_id", sort=False)}
    infos = {str(r["fight_id"]): r for _, r in mean_control.iterrows()}

    rows = []
    for fi, master_row in cohort.iterrows():
        fight_id = str(master_row["fight_id"])
        mkt = market_lookup.loc[fight_id]
        fav_side = str(mkt["favorite_side"])
        mean_pair = groups[fight_id]
        base_fight = _fight(master_row, context["fsr_all"])
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        red_row, blue_row = historical_fighter_rows(
            fsr, event_date=event_date, fight_id=fight_id,
            fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])),
        )
        red_unc = historical_uncertainty_rows(uncertainty, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["r_id"]))
        blue_unc = historical_uncertainty_rows(uncertainty, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["b_id"]))
        red_kd = historical_kd_resistance_row(kd_history, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["r_id"]))
        blue_kd = historical_kd_resistance_row(kd_history, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["b_id"]))

        acc = []
        for path in range(args.paths):
            seed = args.seed + fi * 1_000_000 + path
            epi_rng = np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET)
            matchup = initialize_path_matchup(red_row, blue_row, red_unc, blue_unc, rng=epi_rng, sample_epistemic=False)
            path_fight = fight_with_kd_resistance(
                base_fight,
                red_native_resistance=float(red_kd["pre_rating"]),
                blue_native_resistance=float(blue_kd["pre_rating"]),
            )
            features = build_sampled_fight_feature_rows_v3(
                master_row, red_record=red_row.to_dict(), blue_record=blue_row.to_dict(),
                red_traits=matchup.red, blue_traits=matchup.blue,
            )
            pair, control = predict_feature_frame_v3(
                features, context["inference_models"], context["submission_scale"], context["conversion_offset"],
                submission_baseline=sub_baseline,
            )
            info = control.iloc[0]
            sub_rates, conv = _submission_inputs(pair)
            budgets = _draw_budgets(pair, info, context, np.random.default_rng(seed))
            result = canonical.simulate_detailed_path(
                path_fight, budgets, sub_rates, conv, context["judge_model"], context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            fr, dr = orient(budgets["red_standing_attempted"] + budgets["red_ground_attempted"], budgets["blue_standing_attempted"] + budgets["blue_ground_attempted"], fav_side)
            fl, dl = orient(budgets["red_standing_landed"] + budgets["red_ground_landed"], budgets["blue_standing_landed"] + budgets["blue_ground_landed"], fav_side)
            ftd, dtd = orient(budgets["red_td_attempted"], budgets["blue_td_attempted"], fav_side)
            fctl, dctl = orient(budgets["red_control"], budgets["blue_control"], fav_side)
            fsa, dsa = orient(sub_rates["red"], sub_rates["blue"], fav_side)
            rs = "red" if fav_side == "red" else "blue"
            ds = "blue" if fav_side == "red" else "red"
            acc.append({
                "fav_budget_sig_attempt": fr, "dog_budget_sig_attempt": dr,
                "fav_budget_sig_land": fl, "dog_budget_sig_land": dl,
                "fav_budget_td_attempt": ftd, "dog_budget_td_attempt": dtd,
                "fav_budget_control": fctl, "dog_budget_control": dctl,
                "fav_sub_rate": fsa, "dog_sub_rate": dsa,
                "fav_sim_sig_attempt": result[f"{rs}_sig_attempted"], "dog_sim_sig_attempt": result[f"{ds}_sig_attempted"],
                "fav_sim_sig_land": result[f"{rs}_sig_landed"], "dog_sim_sig_land": result[f"{ds}_sig_landed"],
                "fav_sim_td_land": result[f"{rs}_td_landed"], "dog_sim_td_land": result[f"{ds}_td_landed"],
                "fav_sim_control": result[f"{rs}_control_seconds"], "dog_sim_control": result[f"{ds}_control_seconds"],
                "fav_win": int(result["winner"] == rs),
                "fav_dec": int(result["winner"] == rs and result["method"] == "DEC"),
                "fav_ko": int(result["winner"] == rs and result["method"] == "KO_TKO"),
                "fav_sub": int(result["winner"] == rs and result["method"] == "SUB"),
                "dog_dec": int(result["winner"] == ds and result["method"] == "DEC"),
                "dog_ko": int(result["winner"] == ds and result["method"] == "KO_TKO"),
                "dog_sub": int(result["winner"] == ds and result["method"] == "SUB"),
            })
        a = pd.DataFrame(acc).mean(numeric_only=True)
        rec = {
            "fight_id": fight_id,
            "favorite": mkt["favorite"], "underdog": mkt["underdog"], "favorite_side": fav_side,
            "market_favorite_fair_p": float(mkt["market_favorite_fair_p"]),
            "favorite_won": int(mkt["favorite_won"]),
            "min_prior_ufc_fights": int(min(mkt["red_prior_ufc_fights"], mkt["blue_prior_ufc_fights"])),
        }
        rec.update(a.to_dict())
        rec["compression_pp"] = 100.0 * (rec["market_favorite_fair_p"] - rec["fav_win"])
        rows.append(rec)

    out = pd.DataFrame(rows)
    for stem in ["budget_sig_attempt","budget_sig_land","budget_td_attempt","budget_control","sub_rate","sim_sig_attempt","sim_sig_land","sim_td_land","sim_control"]:
        out[f"share_{stem}"] = out[f"fav_{stem}"] / (out[f"fav_{stem}"] + out[f"dog_{stem}"]).replace(0, np.nan)

    out["bucket"] = pd.cut(out["market_favorite_fair_p"], [0.5,0.6,0.7,0.8,0.9,1.01], labels=["50-60","60-70","70-80","80-90","90+"], right=False)
    share_cols = [c for c in out.columns if c.startswith("share_")]
    agg_cols = ["market_favorite_fair_p","fav_win","compression_pp","fav_dec","fav_ko","fav_sub","dog_dec","dog_ko","dog_sub"] + share_cols
    bucket = out.groupby("bucket", observed=True)[agg_cols].mean().reset_index()

    corr_rows = []
    for c in share_cols:
        corr_rows.append({
            "stage_metric": c,
            "corr_with_market_favorite_p": out[c].corr(out["market_favorite_fair_p"]),
            "corr_with_mc_favorite_p": out[c].corr(out["fav_win"]),
            "corr_with_compression": out[c].corr(out["compression_pp"]),
        })
    corrs = pd.DataFrame(corr_rows).sort_values("corr_with_market_favorite_p", ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir/"fight_level_stage_decomposition.csv", index=False)
    bucket.to_csv(args.out_dir/"market_strength_stage_buckets.csv", index=False)
    corrs.to_csv(args.out_dir/"stage_correlations.csv", index=False)

    print("BANTAMWEIGHT MATCHUP SENSITIVITY DECOMPOSITION")
    print(f"fights={len(out)} paths/fight={args.paths} | posterior means | i10_b0")
    print("\nMARKET-STRENGTH STAGE BUCKETS")
    print(bucket.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSTAGE CORRELATIONS")
    print(corrs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOUTPUT: {args.out_dir}")

if __name__ == "__main__":
    main()
