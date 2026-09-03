from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import build_context
from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import prepare_direct_predictions
from pipeline.simulation.event_clock_mc_v1.frozen_inference import fit_inference_models
from pipeline.simulation.event_clock_mc_v1.diagnostics_full_fight_predictive_replay_shadow_ko_kd import base_submission_rate
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11c_submission_conversion import clip_probability, logit
from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import select_fresh_cohort

DEFAULT_BUNDLE_PATH = Path("data/models/event_clock_mc_v1/event_clock_frozen_bundle.joblib")
DEFAULT_MANIFEST_PATH = Path("data/models/event_clock_mc_v1/event_clock_frozen_bundle_manifest.json")


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Fit Event Clock once and persist target-independent frozen inference models.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--cohort-fights", type=int, default=500)
    args = parser.parse_args()

    cohort, _, selection = select_fresh_cohort(args.cohort_fights, offset=0)
    fight_ids = set(cohort["fight_id"].astype(str))

    print("=" * 120)
    print("EVENT CLOCK MC — FIT TARGET-INDEPENDENT FROZEN BUNDLE")
    print("=" * 120)
    print(f"validation universe: {len(fight_ids)} fights")
    print(f"validation dates: {selection['first_event_date']} through {selection['last_event_date']}")
    print("fitting training-dependent components once...")

    # Validated calibration context. This preserves every previously frozen
    # Stage-9/KO-KD/SUB/judge parameter and the 500-fight reproduction frame.
    context = build_context(fight_ids)

    # Fit reusable target-inference objects on the identical training cohort.
    # This is done only while building the bundle, never during card prediction.
    train, _ = prepare_direct_predictions()
    train["fight_id"] = train["fight_id"].astype(str)
    context["inference_models"] = fit_inference_models(train)

    # Recover the already-validated submission scalar/offset from the frozen
    # 500-fight frame, rather than recalibrating them differently.
    test = context["test"].copy()
    raw_rate = base_submission_rate(test).to_numpy(float)
    clock_rate = test["submission_clock_rate"].to_numpy(float)
    active = raw_rate > 1e-12
    context["submission_scale"] = float(np.median(clock_rate[active] / raw_rate[active]))
    context["conversion_offset"] = float(np.median(
        logit(clip_probability(test["submission_conversion_probability"]))
        - logit(clip_probability(test["submission_conversion_baseline"]))
    ))

    context["bundle_metadata"] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "validation_fight_count": len(fight_ids),
        "validation_first_event_date": selection["first_event_date"],
        "validation_last_event_date": selection["last_event_date"],
        "target_scope": "arbitrary eligible historical fights with FSR V2 prefight snapshots",
        "ko_kd_calibration": "validated_empirical_default",
    }

    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 2, "context": context}, args.bundle, compress=3)
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(context["bundle_metadata"], f, indent=2, sort_keys=True)
        f.write("\n")

    print()
    print(f"bundle:   {args.bundle}")
    print(f"manifest: {args.manifest}")
    print(f"git SHA:  {context['bundle_metadata']['git_sha']}")
    print("target scope: arbitrary eligible historical fights; no fixed 500-fight prediction window")
    print("DONE — card/fight prediction does not refit training models.")


if __name__ == "__main__":
    main()
