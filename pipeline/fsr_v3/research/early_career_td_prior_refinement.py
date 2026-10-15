"""Fine-grid measurement-only refinement of the FSR V3 takedown tendency prior.

No production configuration is changed.  This follows the coarse early-career
study by resolving K between 0.4x and 1.25x of the locked 468.48-second prior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.rate_families import build_rate_fighter_fights, replay_tendency, takedown_spec
from pipeline.fsr_v3.research.early_career_prior_strength_study import (
    EVAL_START,
    _annual,
    _bootstrap,
    _candidate_rows,
    _summaries,
)

OUT_DIR = Path("data/diagnostics/fsr_v3_early_career_td_prior_refinement")
MULTIPLIERS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.25)


def main() -> None:
    spec = takedown_spec(FSRV3Config())
    fights = build_rate_fighter_fights(spec)
    baseline = replay_tendency(fights, spec)
    rows = _candidate_rows(baseline, spec, multipliers=MULTIPLIERS)
    summary = _summaries(rows)
    annual = _annual(rows)
    bootstrap = _bootstrap(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "td_prior_refinement_row_scores.csv", index=False)
    summary.to_csv(OUT_DIR / "td_prior_refinement_summary.csv", index=False)
    annual.to_csv(OUT_DIR / "td_prior_refinement_annual_one_prior.csv", index=False)
    bootstrap.to_csv(OUT_DIR / "td_prior_refinement_bootstrap_one_prior.csv", index=False)

    one = summary[(summary["family"] == "takedown") & (summary["prior_bucket"] == "1")].copy()
    b = bootstrap[bootstrap["family"] == "takedown"].copy()
    view = one.merge(
        b[[
            "k_multiplier", "predictive_ci_2_5", "predictive_ci_97_5",
            "plugin_ci_2_5", "plugin_ci_97_5",
        ]],
        on="k_multiplier",
        how="left",
        validate="one_to_one",
    ).sort_values("k_multiplier")

    print("=" * 154)
    print("FSR V3 TAKEDOWN TENDENCY PRIOR — EARLY-CAREER FINE GRID (MEASUREMENT ONLY)")
    print("=" * 154)
    print(f"evaluation start: {EVAL_START.date()}")
    print(view[[
        "k_multiplier", "k_seconds", "rows",
        "predictive_delta_vs_current", "predictive_ci_2_5", "predictive_ci_97_5",
        "plugin_delta_vs_current", "plugin_ci_2_5", "plugin_ci_97_5",
        "predictive_gain_vs_population", "plugin_gain_vs_population", "mean_posterior_sd",
    ]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    both = view[
        (view["predictive_delta_vs_current"] >= 0)
        & (view["plugin_delta_vs_current"] >= 0)
    ].copy()
    print()
    if both.empty:
        print("No tested K improves both one-prior posterior-predictive and posterior-mean plug-in LL versus current.")
    else:
        best = both.sort_values(["plugin_delta_vs_current", "predictive_delta_vs_current"], ascending=False).iloc[0]
        print(
            "Best tested K improving both metrics: "
            f"{best['k_seconds']:.2f}s ({best['k_multiplier']:.2f}x) | "
            f"predictive delta={best['predictive_delta_vs_current']:+.3f} | "
            f"plugin delta={best['plugin_delta_vs_current']:+.3f}"
        )

    print("DONE — production FSR and Event Clock settings unchanged.")


if __name__ == "__main__":
    main()
