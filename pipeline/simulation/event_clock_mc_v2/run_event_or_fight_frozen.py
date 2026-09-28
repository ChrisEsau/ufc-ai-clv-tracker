"""Run frozen Event Clock MC V2 with FSR V3 means or validated uncertainty."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH, FSR_V3_PREFIGHT_UNCERTAINTY_PATH
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage9_final_flow import simulate_stage9_path
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import (
    DEFAULT_PATHS, SEED, _slug, select_target, simulate_detailed_path, summarize_fight,
)
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows, historical_uncertainty_rows, initialize_path_matchup,
    load_prefight_snapshots, load_prefight_uncertainty,
)
from pipeline.simulation.event_clock_mc_v2.inference import (
    load_submission_baseline_v3, predict_feature_frame_v3, predict_target_v3,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH

OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/event_predictions")
EPISTEMIC_SEED_OFFSET = 80_000_000
DETAILED_PATH_SEED_OFFSET = 50_000_000


def load_frozen_context(bundle_path):
    if not bundle_path.exists():
        raise RuntimeError(
            f"Frozen Event Clock V2 bundle not found: {bundle_path}\n"
            "Build it with: PYTHONPATH=. python -m pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle"
        )
    payload = joblib.load(bundle_path)
    if payload.get("schema_version") != 3:
        raise RuntimeError(f"Expected Event Clock V2 bundle schema 3, got {payload.get('schema_version')!r}")
    context = payload["context"]
    if context.get("inference_models", {}).get("schema") != "event_clock_mc_v2_fsr_v3_direct_v1":
        raise RuntimeError("Unexpected V2 direct-inference schema")
    if "fsr_all" not in context:
        raise RuntimeError("V2 bundle lost the parent V1 mechanics-profile snapshots")
    return context


def _submission_inputs(pair):
    rates, conversion = {}, None
    for side in ("red", "blue"):
        row = pair[pair["side"] == side].iloc[0]
        rates[side] = float(row["submission_clock_rate"])
        if conversion is None:
            conversion = float(row["submission_conversion_probability"])
    if conversion is None:
        raise RuntimeError("missing submission conversion probability")
    return rates, conversion


def _draw_budgets(pair, pair_info, context, rng):
    return simulate_stage9_path(
        pair, pair_info, context["hurdle_alpha"], context["control_alpha"],
        context["dominance_kappa"], context["td_control_beta"], context["standing_alpha"],
        context["minority_classifier"], context["minority_share_model"],
        context["minority_residual_sigma"], rng,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--event")
    group.add_argument("--event-date")
    group.add_argument("--fight-id")
    group.add_argument("--fighter")
    parser.add_argument("--opponent")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--epistemic", choices=("off", "validated"), default="off")
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    sample_epistemic = args.epistemic == "validated"
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="raise").dt.normalize()
    target = select_target(master, args)

    context = load_frozen_context(args.bundle)
    fsr_v3 = load_prefight_snapshots(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    uncertainty = load_prefight_uncertainty(FSR_V3_PREFIGHT_UNCERTAINTY_PATH) if sample_epistemic else None
    submission_baseline = load_submission_baseline_v3()

    mean_test, mean_pair = predict_target_v3(
        target, fsr_v3, context["inference_models"],
        context["submission_scale"], context["conversion_offset"],
    )
    mean_pair_lookup = {str(row["fight_id"]): row for _, row in mean_pair.iterrows()}
    metadata = context.get("bundle_metadata", {})

    print("=" * 150)
    print("EVENT CLOCK MC V2 — FSR V3 FROZEN PREDICTION")
    print("=" * 150)
    print(f"target fights: {target['fight_id'].nunique()} | paths/fight: {args.paths}")
    print(f"FSR epistemic mode: {args.epistemic}")
    print("direct flow: FSR V3 | detailed mechanics: exact frozen V1 context")
    print(f"V2 bundle git SHA: {metadata.get('git_sha', 'unknown')}")
    print(f"parent V1 mechanics git SHA: {metadata.get('parent_v1_git_sha', 'unknown')}")

    master_lookup = {str(row["fight_id"]): row for _, row in master.iterrows()}
    all_path_rows, summary_rows = [], []
    groups = list(mean_test.groupby("fight_id", sort=False))

    for fight_index, (fight_id, mean_fighter_pair) in enumerate(groups):
        fight_id = str(fight_id)
        master_row = master_lookup[fight_id]
        # The detailed path receives the exact V1 profile object. FSR V3 has
        # already resolved all new flow semantics into path budgets upstream.
        fight = _fight(master_row, context["fsr_all"])
        mean_pair_info = mean_pair_lookup[fight_id]
        mean_sub_rate, mean_convert = _submission_inputs(mean_fighter_pair)
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        red_row, blue_row = historical_fighter_rows(
            fsr_v3, event_date=event_date, fight_id=fight_id,
            fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])),
        )
        if sample_epistemic:
            red_unc = historical_uncertainty_rows(
                uncertainty, event_date=event_date, fight_id=fight_id,
                fighter_id=str(master_row["r_id"]),
            )
            blue_unc = historical_uncertainty_rows(
                uncertainty, event_date=event_date, fight_id=fight_id,
                fighter_id=str(master_row["b_id"]),
            )
        else:
            red_unc = blue_unc = None

        print(f"[{fight_index + 1}/{len(groups)}] {master_row['r_name']} vs {master_row['b_name']}")
        fight_rows = []
        for path in range(args.paths):
            seed = args.seed + fight_index * 1_000_000 + path
            if sample_epistemic:
                path_matchup = initialize_path_matchup(
                    red_row, blue_row, red_unc, blue_unc,
                    rng=np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET),
                    sample_epistemic=True,
                )
                path_features = build_sampled_fight_feature_rows_v3(
                    master_row, red_record=red_row.to_dict(), blue_record=blue_row.to_dict(),
                    red_traits=path_matchup.red, blue_traits=path_matchup.blue,
                )
                path_pair, path_control = predict_feature_frame_v3(
                    path_features, context["inference_models"], context["submission_scale"],
                    context["conversion_offset"], submission_baseline=submission_baseline,
                )
                pair_info = path_control.iloc[0]
                sub_rate, convert = _submission_inputs(path_pair)
                draws = {
                    "red_td_tendency_draw": path_matchup.red["takedown_tendency"],
                    "red_td_suppression_draw": path_matchup.red["takedown_suppression"],
                    "red_standing_tendency_draw": path_matchup.red["standing_striking_tendency"],
                    "red_standing_suppression_draw": path_matchup.red["standing_striking_suppression"],
                    "blue_td_tendency_draw": path_matchup.blue["takedown_tendency"],
                    "blue_td_suppression_draw": path_matchup.blue["takedown_suppression"],
                    "blue_standing_tendency_draw": path_matchup.blue["standing_striking_tendency"],
                    "blue_standing_suppression_draw": path_matchup.blue["standing_striking_suppression"],
                }
            else:
                path_pair, pair_info = mean_fighter_pair, mean_pair_info
                sub_rate, convert, draws = mean_sub_rate, mean_convert, {}

            budgets = _draw_budgets(path_pair, pair_info, context, np.random.default_rng(seed))
            result = simulate_detailed_path(
                fight, budgets, sub_rate, convert,
                context["judge_model"], context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            result.update({"fight_id": fight_id, "path": path, "epistemic_mode": args.epistemic, **draws})
            fight_rows.append(result)
            all_path_rows.append(result)

        summary = summarize_fight(fight_id, mean_fighter_pair, fight_rows, master_row)
        summary["epistemic_mode"] = args.epistemic
        summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    paths = pd.DataFrame(all_path_rows)
    suffix = "means" if not sample_epistemic else "epistemic"
    prefix = args.output_prefix or f"{_slug(target.attrs.get('label', 'prediction'))}_{suffix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"{prefix}_{args.paths}paths_summary.csv"
    paths_path = OUT_DIR / f"{prefix}_{args.paths}paths_paths.csv"
    summary.to_csv(summary_path, index=False)
    paths.to_csv(paths_path, index=False)

    cols = [
        "red", "blue", "actual_winner", "actual_method", "p_red_win", "p_blue_win",
        "p_red_dec", "p_red_ko_tko", "p_red_sub", "p_blue_dec", "p_blue_ko_tko",
        "p_blue_sub", "ml_correct", "method_correct", "winner_method_correct",
    ]
    print()
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"ML accuracy: {summary['ml_correct'].mean():.2%}")
    print(f"Method accuracy: {summary['method_correct'].mean():.2%}")
    print(f"Winner+method accuracy: {summary['winner_method_correct'].mean():.2%}")
    print(f"summary CSV: {summary_path}")
    print(f"path CSV:    {paths_path}")


if __name__ == "__main__":
    main()
