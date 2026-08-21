"""Run frozen Event Clock MC V2 with FSR V3 means or validated uncertainty.

Fight mechanics are imported unchanged from Event Clock V1.  V2 changes only
FSR input semantics, the V3-fitted direct inference bundle, and (optionally)
one path-initialization draw for the four validated epistemic traits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import (
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
    FSR_V3_PREFIGHT_UNCERTAINTY_PATH,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage9_final_flow import (
    simulate_stage9_path,
)
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import (
    DEFAULT_PATHS,
    SEED,
    _slug,
    select_target,
    simulate_detailed_path,
    summarize_fight,
)
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight
from pipeline.simulation.event_clock_mc_v2.feature_builder import (
    build_sampled_fight_feature_rows_v3,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    historical_uncertainty_rows,
    initialize_path_matchup,
    load_prefight_snapshots,
    load_prefight_uncertainty,
)
from pipeline.simulation.event_clock_mc_v2.inference import (
    load_submission_baseline_v3,
    predict_feature_frame_v3,
    predict_target_v3,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import (
    DEFAULT_BUNDLE_PATH,
)


OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/event_predictions")
EPISTEMIC_SEED_OFFSET = 80_000_000
DETAILED_PATH_SEED_OFFSET = 50_000_000


def load_frozen_context(bundle_path: Path) -> dict:
    if not bundle_path.exists():
        raise RuntimeError(
            f"Frozen Event Clock V2 bundle not found: {bundle_path}\n"
            "Build it once with:\n"
            "  PYTHONPATH=. python -m pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle"
        )
    payload = joblib.load(bundle_path)
    if payload.get("schema_version") != 3:
        raise RuntimeError(
            f"Expected Event Clock V2 bundle schema 3, got {payload.get('schema_version')!r}"
        )
    context = payload["context"]
    inference_schema = context.get("inference_models", {}).get("schema")
    if inference_schema != "event_clock_mc_v2_fsr_v3_direct_v1":
        raise RuntimeError(f"Unexpected V2 direct-inference schema: {inference_schema!r}")
    return context


def _compatibility_fsr_for_frozen_physics(fsr_v3: pd.DataFrame) -> pd.DataFrame:
    """Provide the rejected ground-defense field only to the legacy profile parser.

    The frozen detailed-path mechanics use this fight object for physical/stamina
    and KO/KD profiles after path budgets have already been generated.  Ground
    accuracy is never recomputed from this compatibility field.  Zero is
    therefore a neutral parser shim, not a V3 fighter trait.
    """
    compat = fsr_v3.copy()
    if "ground_striking_defense" in compat.columns:
        raise RuntimeError("canonical FSR V3 unexpectedly contains ground_striking_defense")
    compat["ground_striking_defense"] = 0.0
    return compat


def _submission_inputs(pair: pd.DataFrame) -> tuple[dict[str, float], float]:
    rates: dict[str, float] = {}
    conversion = None
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
        pair,
        pair_info,
        context["hurdle_alpha"],
        context["control_alpha"],
        context["dominance_kappa"],
        context["td_control_beta"],
        context["standing_alpha"],
        context["minority_classifier"],
        context["minority_share_model"],
        context["minority_residual_sigma"],
        rng,
    )


def main() -> None:
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
    parser.add_argument(
        "--epistemic",
        choices=("off", "validated"),
        default="off",
        help="off = posterior means only; validated = sample four validated V3 traits once/path",
    )
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    sample_epistemic = args.epistemic == "validated"
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="raise").dt.normalize()
    target = select_target(master, args)
    target_ids = set(target["fight_id"].astype(str))

    context = load_frozen_context(args.bundle)
    fsr_v3 = load_prefight_snapshots(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    uncertainty = (
        load_prefight_uncertainty(FSR_V3_PREFIGHT_UNCERTAINTY_PATH)
        if sample_epistemic
        else None
    )
    compat_fsr = _compatibility_fsr_for_frozen_physics(fsr_v3)
    submission_baseline = load_submission_baseline_v3()

    # Mean prediction is built once even in variance mode.  It is the B-arm
    # reference and supplies a stable summary feature frame; each C-arm path
    # rebuilds its own direct predictions from independently sampled FSR state.
    mean_test, mean_pair = predict_target_v3(
        target,
        fsr_v3,
        context["inference_models"],
        context["submission_scale"],
        context["conversion_offset"],
    )
    mean_pair_lookup = {
        str(row["fight_id"]): row for _, row in mean_pair.iterrows()
    }

    metadata = context.get("bundle_metadata", {})
    print("=" * 150)
    print("EVENT CLOCK MC V2 — FSR V3 FROZEN EVENT / FIGHT PREDICTION")
    print("=" * 150)
    print(f"target fights: {len(target_ids)} | paths/fight: {args.paths}")
    print("training refit during prediction: NO")
    print(f"FSR epistemic mode: {args.epistemic}")
    print("fight mechanics: frozen Event Clock V1")
    print(f"bundle: {args.bundle}")
    print(f"V2 bundle git SHA: {metadata.get('git_sha', 'unknown')}")
    print(f"parent V1 mechanics git SHA: {metadata.get('parent_v1_git_sha', 'unknown')}")

    master_lookup = {str(row["fight_id"]): row for _, row in master.iterrows()}
    all_path_rows: list[dict] = []
    summary_rows: list[dict] = []

    groups = list(mean_test.groupby("fight_id", sort=False))
    for fight_index, (fight_id, mean_fighter_pair) in enumerate(groups):
        fight_id = str(fight_id)
        master_row = master_lookup[fight_id]
        fight = _fight(master_row, compat_fsr)
        mean_pair_info = mean_pair_lookup[fight_id]
        mean_sub_rate, mean_convert = _submission_inputs(mean_fighter_pair)

        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        red_row, blue_row = historical_fighter_rows(
            fsr_v3,
            event_date=event_date,
            fight_id=fight_id,
            fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])),
        )
        if sample_epistemic:
            red_uncertainty = historical_uncertainty_rows(
                uncertainty,
                event_date=event_date,
                fight_id=fight_id,
                fighter_id=str(master_row["r_id"]),
            )
            blue_uncertainty = historical_uncertainty_rows(
                uncertainty,
                event_date=event_date,
                fight_id=fight_id,
                fighter_id=str(master_row["b_id"]),
            )
        else:
            red_uncertainty = blue_uncertainty = None

        fight_rows: list[dict] = []
        print(
            f"[{fight_index + 1}/{len(groups)}] "
            f"{master_row['r_name']} vs {master_row['b_name']}"
        )

        for path in range(args.paths):
            seed = args.seed + fight_index * 1_000_000 + path

            if sample_epistemic:
                # Separate RNG preserves the Stage-9 budget RNG stream between
                # the B and C arms.  Epistemic draws therefore do not merely
                # shift the random-number sequence used by fight mechanics.
                epistemic_rng = np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET)
                path_matchup = initialize_path_matchup(
                    red_row,
                    blue_row,
                    red_uncertainty,
                    blue_uncertainty,
                    rng=epistemic_rng,
                    sample_epistemic=True,
                )
                path_features = build_sampled_fight_feature_rows_v3(
                    master_row,
                    red_record=red_row.to_dict(),
                    blue_record=blue_row.to_dict(),
                    red_traits=path_matchup.red,
                    blue_traits=path_matchup.blue,
                )
                path_fighter_pair, path_pair = predict_feature_frame_v3(
                    path_features,
                    context["inference_models"],
                    context["submission_scale"],
                    context["conversion_offset"],
                    submission_baseline=submission_baseline,
                )
                path_pair_info = path_pair.iloc[0]
                sub_rate, convert = _submission_inputs(path_fighter_pair)
                epistemic_values = {
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
                path_fighter_pair = mean_fighter_pair
                path_pair_info = mean_pair_info
                sub_rate, convert = mean_sub_rate, mean_convert
                epistemic_values = {}

            budget_rng = np.random.default_rng(seed)
            budgets = _draw_budgets(
                path_fighter_pair,
                path_pair_info,
                context,
                budget_rng,
            )
            result = simulate_detailed_path(
                fight,
                budgets,
                sub_rate,
                convert,
                context["judge_model"],
                context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            result.update(
                {
                    "fight_id": fight_id,
                    "path": path,
                    "epistemic_mode": args.epistemic,
                    **epistemic_values,
                }
            )
            fight_rows.append(result)
            all_path_rows.append(result)

        summary = summarize_fight(
            fight_id,
            mean_fighter_pair,
            fight_rows,
            master_row,
        )
        summary["epistemic_mode"] = args.epistemic
        summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    paths = pd.DataFrame(all_path_rows)
    label = target.attrs.get("label", "prediction")
    mode_suffix = "means" if not sample_epistemic else "epistemic"
    prefix = args.output_prefix or f"{_slug(label)}_{mode_suffix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"{prefix}_{args.paths}paths_summary.csv"
    paths_path = OUT_DIR / f"{prefix}_{args.paths}paths_paths.csv"
    summary.to_csv(summary_path, index=False)
    paths.to_csv(paths_path, index=False)

    display_cols = [
        "red", "blue", "actual_winner", "actual_method",
        "p_red_win", "p_blue_win", "p_red_dec", "p_red_ko_tko", "p_red_sub",
        "p_blue_dec", "p_blue_ko_tko", "p_blue_sub",
        "ml_correct", "method_correct", "winner_method_correct",
    ]
    print()
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"ML accuracy:              {summary['ml_correct'].mean():.2%}")
    print(f"Method accuracy:          {summary['method_correct'].mean():.2%}")
    print(f"Winner+method accuracy:   {summary['winner_method_correct'].mean():.2%}")
    print(f"summary CSV: {summary_path}")
    print(f"path CSV:    {paths_path}")


if __name__ == "__main__":
    main()
