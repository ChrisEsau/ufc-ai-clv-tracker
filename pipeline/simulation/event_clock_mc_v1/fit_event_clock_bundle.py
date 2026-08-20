from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib

from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import build_context
from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import (
    select_fresh_cohort,
)


DEFAULT_BUNDLE_PATH = Path(
    "data/models/event_clock_mc_v1/event_clock_frozen_bundle.joblib"
)
DEFAULT_MANIFEST_PATH = Path(
    "data/models/event_clock_mc_v1/event_clock_frozen_bundle_manifest.json"
)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit the current Event Clock statistical stack once and persist "
            "a frozen prediction bundle for repeated event/fight simulation."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--cohort-fights", type=int, default=500)
    args = parser.parse_args()

    cohort, _, selection = select_fresh_cohort(
        args.cohort_fights,
        offset=0,
    )
    fight_ids = set(cohort["fight_id"].astype(str))

    print("=" * 120)
    print("EVENT CLOCK MC — FIT FROZEN BUNDLE")
    print("=" * 120)
    print(f"prediction universe: {len(fight_ids)} fights")
    print(
        f"dates: {selection['first_event_date']} through "
        f"{selection['last_event_date']}"
    )
    print("fitting training-dependent components once...")

    context = build_context(fight_ids)

    # Keep provenance inside the serialized artifact so a loaded bundle can
    # identify exactly which fitted universe it represents.
    context["bundle_metadata"] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "prediction_fight_count": len(fight_ids),
        "prediction_first_event_date": selection["first_event_date"],
        "prediction_last_event_date": selection["last_event_date"],
        "prediction_fight_ids": sorted(fight_ids),
        "ko_kd_calibration": "validated_empirical_default",
    }

    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "schema_version": 1,
            "context": context,
        },
        args.bundle,
        compress=3,
    )

    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(context["bundle_metadata"], f, indent=2, sort_keys=True)
        f.write("\n")

    print()
    print(f"bundle:   {args.bundle}")
    print(f"manifest: {args.manifest}")
    print(f"git SHA:  {context['bundle_metadata']['git_sha']}")
    print("DONE — subsequent frozen runner calls do not refit training models.")


if __name__ == "__main__":
    main()
