from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage9_final_flow import simulate_stage9_path
from pipeline.simulation.event_clock_mc_v1.frozen_inference import predict_target
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import DEFAULT_PATHS, OUT_DIR, SEED, _slug, select_target, simulate_detailed_path, summarize_fight
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

DEFAULT_BUNDLE_PATH = Path("data/models/event_clock_mc_v1/event_clock_frozen_bundle.joblib")


def load_frozen_context(bundle_path: Path):
    if not bundle_path.exists():
        raise RuntimeError(
            f"Frozen Event Clock bundle not found: {bundle_path}\n"
            "Build it once with:\n  PYTHONPATH=. python -m pipeline.simulation.event_clock_mc_v1.fit_event_clock_bundle"
        )
    payload = joblib.load(bundle_path)
    if payload.get("schema_version") != 2:
        raise RuntimeError(
            f"Frozen bundle schema {payload.get('schema_version')!r} is obsolete. "
            "Rebuild it once with fit_event_clock_bundle."
        )
    return payload["context"]


def main():
    parser = argparse.ArgumentParser(description="Run Event Clock MC from frozen target-independent fitted models; no training refit during prediction.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--event")
    group.add_argument("--event-date")
    group.add_argument("--fight-id")
    group.add_argument("--fighter")
    parser.add_argument("--opponent")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="raise").dt.normalize()
    target = select_target(master, args)
    target_ids = set(target["fight_id"].astype(str))

    context = load_frozen_context(args.bundle)
    test, target_pair = predict_target(
        target,
        context["fsr_all"],
        context["inference_models"],
        context["submission_scale"],
        context["conversion_offset"],
    )
    pair_lookup = {str(row["fight_id"]): row for _, row in target_pair.iterrows()}
    metadata = context.get("bundle_metadata", {})

    print("=" * 150)
    print("EVENT CLOCK MC — FROZEN EVENT / FIGHT PREDICTION")
    print("=" * 150)
    print(f"target fights: {len(target_ids)} | paths/fight: {args.paths}")
    print("training refit: NO")
    print("target universe: ARBITRARY ELIGIBLE HISTORICAL FIGHT")
    print(f"bundle: {args.bundle}")
    if metadata:
        print(f"bundle git SHA: {metadata.get('git_sha', 'unknown')}")
        print(
            "bundle validation dates: "
            f"{metadata.get('validation_first_event_date')} through "
            f"{metadata.get('validation_last_event_date')}"
        )
    print("KO/KD calibration: validated empirical default")

    master_lookup = {str(row["fight_id"]): row for _, row in master.iterrows()}
    all_path_rows, summary_rows = [], []
    groups = list(test.groupby("fight_id", sort=False))

    for fight_index, (fight_id, pair) in enumerate(groups):
        fight_id = str(fight_id)
        master_row = master_lookup[fight_id]
        fight = _fight(master_row, context["fsr_all"])
        pair_info = pair_lookup[fight_id]

        sub_rate, convert = {}, None
        for side in ("red", "blue"):
            row = pair[pair["side"] == side].iloc[0]
            sub_rate[side] = float(row["submission_clock_rate"])
            if convert is None:
                convert = float(row["submission_conversion_probability"])

        fight_rows = []
        print(f"[{fight_index + 1}/{len(groups)}] {master_row['r_name']} vs {master_row['b_name']}")
        for path in range(args.paths):
            seed = args.seed + fight_index * 1000000 + path
            rng = np.random.default_rng(seed)
            budgets = simulate_stage9_path(
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
            result = simulate_detailed_path(
                fight, budgets, sub_rate, convert,
                context["judge_model"], context["judge_features"],
                seed + 50000000,
            )
            result.update({"fight_id": fight_id, "path": path})
            fight_rows.append(result)
            all_path_rows.append(result)
        summary_rows.append(summarize_fight(fight_id, pair, fight_rows, master_row))

    summary = pd.DataFrame(summary_rows)
    paths = pd.DataFrame(all_path_rows)
    prefix = args.output_prefix or _slug(target.attrs.get("label", "prediction"))
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
