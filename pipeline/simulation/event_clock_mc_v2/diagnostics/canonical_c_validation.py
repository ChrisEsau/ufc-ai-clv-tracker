"""Held-out validation for canonical Event Clock C only.

C is now the standard model:
  * canonical FSR V3 posterior means;
  * validated path-level epistemic draws for standing/TD core traits;
  * validated path-level V3 KD-resistance draw;
  * frozen Event Clock detailed mechanics.

A and B are intentionally not simulated.  The same fresh 150-fight stratified
method-market cohort is retained so the new standard remains comparable to
previously recorded C checkpoints without spending paths on obsolete arms.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH, FSR_V3_PREFIGHT_UNCERTAINTY_PATH
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import SEED, simulate_detailed_path, summarize_fight
from pipeline.simulation.event_clock_mc_v2.canonical_c import (
    fight_with_kd_resistance,
    historical_kd_resistance_row,
    load_kd_resistance_history,
    sample_kd_resistance_latent,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.method_market_abc_validation import (
    _aggregate_fight_scores,
    _bet_metrics,
    _edge_bins,
    _fight_scores,
    _fighter_profile_metrics,
    _outcome_metrics,
    _outcome_rows,
    _prepare_market,
    _select_cohort,
)
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    historical_uncertainty_rows,
    initialize_path_matchup,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.inference import (
    load_submission_baseline_v3,
    predict_feature_frame_v3,
    predict_target_v3,
)
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET,
    EPISTEMIC_SEED_OFFSET,
    _draw_budgets,
    _submission_inputs,
    load_frozen_context,
)
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/canonical_c")
ARM = "C_v3_canonical"


def _load_core_uncertainty() -> pd.DataFrame:
    frame = pd.read_parquet(FSR_V3_PREFIGHT_UNCERTAINTY_PATH).copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    # KD resistance is sampled at the detailed-physics boundary, not by the
    # positive-trait adapter used for standing/TD flow.
    return frame[~frame["trait"].eq("knockdown_resistance_v3")].copy()


def _simulate_c(target: pd.DataFrame, paths: int, seed0: int) -> pd.DataFrame:
    context = load_frozen_context(V2_BUNDLE_PATH)
    fsr_v3 = load_prefight_snapshots(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    uncertainty = _load_core_uncertainty()
    kd_history = load_kd_resistance_history()
    submission_baseline = load_submission_baseline_v3()

    mean_test, mean_control = predict_target_v3(
        target,
        fsr_v3,
        context["inference_models"],
        context["submission_scale"],
        context["conversion_offset"],
    )
    groups = {str(fid): g.copy() for fid, g in mean_test.groupby("fight_id", sort=False)}
    infos = {str(row["fight_id"]): row for _, row in mean_control.iterrows()}
    master_lookup = {str(row["fight_id"]): row for _, row in target.iterrows()}

    summaries: list[dict] = []
    fight_ids = target["fight_id"].astype(str).tolist()
    for fight_index, fight_id in enumerate(fight_ids):
        master_row = master_lookup[fight_id]
        mean_pair = groups[fight_id]
        base_fight = _fight(master_row, context["fsr_all"])
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        red_row, blue_row = historical_fighter_rows(
            fsr_v3,
            event_date=event_date,
            fight_id=fight_id,
            fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])),
        )
        red_unc = historical_uncertainty_rows(
            uncertainty,
            event_date=event_date,
            fight_id=fight_id,
            fighter_id=str(master_row["r_id"]),
        )
        blue_unc = historical_uncertainty_rows(
            uncertainty,
            event_date=event_date,
            fight_id=fight_id,
            fighter_id=str(master_row["b_id"]),
        )
        red_kd = historical_kd_resistance_row(
            kd_history,
            event_date=event_date,
            fight_id=fight_id,
            fighter_id=str(master_row["r_id"]),
        )
        blue_kd = historical_kd_resistance_row(
            kd_history,
            event_date=event_date,
            fight_id=fight_id,
            fighter_id=str(master_row["b_id"]),
        )

        if fight_index % 10 == 0:
            print(f"[{fight_index + 1}/{len(fight_ids)}] {master_row['r_name']} vs {master_row['b_name']}")

        path_results = []
        for path in range(paths):
            seed = seed0 + fight_index * 1_000_000 + path
            epistemic_rng = np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET)
            matchup = initialize_path_matchup(
                red_row,
                blue_row,
                red_unc,
                blue_unc,
                rng=epistemic_rng,
                sample_epistemic=True,
            )
            red_kd_draw = sample_kd_resistance_latent(red_kd, epistemic_rng)
            blue_kd_draw = sample_kd_resistance_latent(blue_kd, epistemic_rng)
            path_fight = fight_with_kd_resistance(
                base_fight,
                red_native_resistance=red_kd_draw,
                blue_native_resistance=blue_kd_draw,
            )

            features = build_sampled_fight_feature_rows_v3(
                master_row,
                red_record=red_row.to_dict(),
                blue_record=blue_row.to_dict(),
                red_traits=matchup.red,
                blue_traits=matchup.blue,
            )
            pair_c, control_c = predict_feature_frame_v3(
                features,
                context["inference_models"],
                context["submission_scale"],
                context["conversion_offset"],
                submission_baseline=submission_baseline,
            )
            info_c = control_c.iloc[0]
            sub_c, conv_c = _submission_inputs(pair_c)
            budgets = _draw_budgets(pair_c, info_c, context, np.random.default_rng(seed))
            result = simulate_detailed_path(
                path_fight,
                budgets,
                sub_c,
                conv_c,
                context["judge_model"],
                context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            path_results.append(result)

        summary = summarize_fight(fight_id, mean_pair, path_results, master_row)
        summary["arm"] = ARM
        summary["event_date"] = master_row["event_date"]
        summary["red_id"] = str(master_row["r_id"])
        summary["blue_id"] = str(master_row["b_id"])
        summary["red_prior_ufc_fights"] = int(master_row["red_prior_ufc_fights"])
        summary["blue_prior_ufc_fights"] = int(master_row["blue_prior_ufc_fights"])
        summary["fight_evidence_bucket"] = master_row["fight_evidence_bucket"]
        summaries.append(summary)
    return pd.DataFrame(summaries)


def _method_share(summary: pd.DataFrame) -> pd.DataFrame:
    actual = summary["actual_method"].value_counts(normalize=True)
    return pd.DataFrame(
        [
            {
                "method": "DEC",
                "actual_share": float(actual.get("DEC", 0.0)),
                "simulated_share": float((summary["p_red_dec"] + summary["p_blue_dec"]).mean()),
            },
            {
                "method": "KO_TKO",
                "actual_share": float(actual.get("KO_TKO", 0.0)),
                "simulated_share": float((summary["p_red_ko_tko"] + summary["p_blue_ko_tko"]).mean()),
            },
            {
                "method": "SUB",
                "actual_share": float(actual.get("SUB", 0.0)),
                "simulated_share": float((summary["p_red_sub"] + summary["p_blue_sub"]).mean()),
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--quota-per-bucket", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="coerce").dt.normalize()
    market, market_ids = _prepare_market()
    cohort = _select_cohort(master, market_ids, args.quota_per_bucket)

    print("=" * 160)
    print("EVENT CLOCK MC V2 — CANONICAL C HELD-OUT VALIDATION")
    print("=" * 160)
    print(f"selected fights: {len(cohort)} | paths/fight: {args.paths}")
    print("arms simulated: C ONLY")
    print("C = final FSR V3 + validated epistemic sampling + frozen Event Clock mechanics")
    print(cohort["fight_evidence_bucket"].value_counts().to_string())

    summary = _simulate_c(cohort, args.paths, args.seed)
    market_selected = market[market["fight_id"].isin(set(cohort["fight_id"].astype(str)))].copy()
    scores = _fight_scores(summary, market_selected)
    outcomes = _outcome_rows(summary, market_selected)
    fight_metrics = _aggregate_fight_scores(scores)
    outcome_metrics = _outcome_metrics(outcomes)
    bet_metrics = _bet_metrics(outcomes)
    edge_bins = _edge_bins(outcomes)
    fighter_profiles = _fighter_profile_metrics(outcomes)
    method_share = _method_share(summary)

    print("\nCANONICAL C HEADLINE")
    print(f"ML accuracy: {summary['ml_correct'].mean():.2%}")
    print(f"Method accuracy: {summary['method_correct'].mean():.2%}")
    print(f"Winner+method accuracy: {summary['winner_method_correct'].mean():.2%}")
    print("\nSIX-WAY METRICS")
    print(fight_metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nMETHOD SHARES")
    print(method_share.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"c_{len(cohort)}f_{args.paths}paths"
    files = {
        "cohort": cohort,
        "summary": summary,
        "fight_scores": scores,
        "fight_metrics": fight_metrics,
        "outcomes": outcomes,
        "outcome_metrics": outcome_metrics,
        "bet_metrics": bet_metrics,
        "edge_bins": edge_bins,
        "fighter_profiles": fighter_profiles,
        "method_share": method_share,
    }
    for name, frame in files.items():
        path = args.out_dir / f"{stem}_{name}.csv"
        frame.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
