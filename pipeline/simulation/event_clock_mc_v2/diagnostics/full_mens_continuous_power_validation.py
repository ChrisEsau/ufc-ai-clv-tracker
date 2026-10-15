"""Full men's validation of the selected continuous consequence-side power curve.

Research-only candidate:

    power_offset(t) = clip(35 - t / 12, -40, 35)

No FSR refit, no strike-budget change, no event-timing change, and no frozen V1
source modification.  The candidate is injected only at diagnostic module
boundaries.

The validation uses all eligible post-cutoff fights from the eight audited men's
divisions (up to target_n per division). It records predictive calibration,
method shares, historical-vs-simulated duration, cumulative nondecision finish
timing, division-level results, and R1-R3 KD/KO lethality.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import EventClockShadowKOKDModel as FrozenShadow
from pipeline.simulation.event_clock_mc_v1 import run_event_or_fight as frozen_runner
from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import lethality_round_bucket as bucket
from pipeline.simulation.event_clock_mc_v2.diagnostics import shared_power_decay_grid as shared
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit
from pipeline.simulation.event_mc_v1.components.profiles import MatchupProfiles, Side


INTERCEPT = 35.0
DENOMINATOR = 12.0
LOWER_CAP = -40.0
UPPER_CAP = 35.0


class SelectedContinuousPowerShadow(FrozenShadow):
    """Frozen shadow KO/KD mechanics with the selected elapsed-time power offset."""

    @staticmethod
    def _offset(state) -> float:
        t = float(state.fight_time_seconds)
        return float(np.clip(INTERCEPT - t / DENOMINATOR, LOWER_CAP, UPPER_CAP))

    def resolve_landed_strike(self, *, state, attacker, prior_defender_kds, rng):
        offset = self._offset(state)

        def shifted(side: Side):
            profile = self.profiles.fighter(side)
            return replace(profile, striking_power=float(profile.striking_power) + offset)

        model = FrozenShadow(
            profiles=MatchupProfiles(red=shifted(Side.RED), blue=shifted(Side.BLUE)),
            calibration=self.calibration,
        )
        return model.resolve_landed_strike(
            state=state,
            attacker=attacker,
            prior_defender_kds=prior_defender_kds,
            rng=rng,
        )


def _patch_diagnostic_boundaries() -> None:
    # simulate_detailed_path resolves this global from its defining V1 module.
    frozen_runner.EventClockShadowKOKDModel = SelectedContinuousPowerShadow
    # Round-bucket instrumentation has its own constructor reference.
    bucket.EventClockShadowKOKDModel = SelectedContinuousPowerShadow


def _finish_timing(summary: pd.DataFrame, cohort: pd.DataFrame, paths: int) -> pd.DataFrame:
    # Historical match_time_sec is already total elapsed fight seconds in the
    # canonical master semantics used by the corrected weight-class audit.
    hist_elapsed = pd.to_numeric(cohort["match_time_sec"], errors="raise").to_numpy(float)
    hist_method = cohort["method"].map(lambda x: wc_audit.normalize_method(x)).to_numpy()

    rows = []
    for threshold in (300.0, 600.0, 900.0):
        hist = float(np.mean((hist_method != "DEC") & (hist_elapsed <= threshold)))
        col = f"p_nondec_by_{int(threshold)}"
        sim = float(summary[col].mean())
        rows.append({
            "threshold_seconds": int(threshold),
            "historical_nondecision_finish_share": hist,
            "simulated_nondecision_finish_share": sim,
            "bias_sim_minus_actual": sim - hist,
            "paths_per_fight": int(paths),
        })
    return pd.DataFrame(rows)


def _install_summary_wrapper() -> None:
    original = canonical.summarize_fight

    def wrapped(fight_id, pair, rows, master_row):
        out = original(fight_id, pair, rows, master_row)
        paths = pd.DataFrame(rows)
        nondec = paths["method"].ne("DEC")
        for threshold in (300.0, 600.0, 900.0):
            out[f"p_nondec_by_{int(threshold)}"] = float(
                (nondec & paths["elapsed"].le(threshold)).mean()
            )
        return out

    canonical.summarize_fight = wrapped


def _predictive_validation(target_n: int, paths: int, seed: int):
    all_cohorts = []
    all_summaries = []
    division_outputs = []

    for i, division in enumerate(shared.MEN_DIVISIONS):
        cohort, eligible = wc_audit.select_cohort(division, target_n)
        print("=" * 120)
        print(f"PREDICTIVE | {division} | eligible={eligible} selected={len(cohort)} paths={paths}")
        summary = canonical._simulate_c(cohort, paths, seed + i * 100_000_000)
        summary["division"] = division
        cohort = cohort.copy()
        cohort["division_audit"] = division

        outputs = wc_audit.audit(summary, cohort, division, paths)
        headline = outputs["headline"].copy()
        method_share = outputs["method_share"].copy()
        method_share["division"] = division
        mechanics = outputs["mechanics"].copy()
        mechanics["division"] = division
        rates = outputs["mechanics_rates"].copy()
        rates["division"] = division

        division_outputs.append((headline, method_share, mechanics, rates))
        all_cohorts.append(cohort)
        all_summaries.append(summary)

    cohort_all = pd.concat(all_cohorts, ignore_index=True)
    summary_all = pd.concat(all_summaries, ignore_index=True)

    # Overall predictive metrics across all men's fights.
    y = summary_all["actual_winner"].eq("red").astype(float).to_numpy()
    p = summary_all["p_red_win"].to_numpy(float)
    winner_p = np.where(y > 0.5, p, 1.0 - p)
    method_cols = {"DEC": "p_fight_dec", "KO_TKO": "p_fight_ko_tko", "SUB": "p_fight_sub"}

    overall = pd.DataFrame([{
        "n_fights": len(summary_all),
        "paths_per_fight": int(paths),
        "ml_accuracy": float(summary_all["ml_correct"].mean()),
        "ml_brier": float(np.mean((p - y) ** 2)),
        "ml_logloss": float(-np.mean(np.log(np.clip(winner_p, 1e-9, 1.0)))),
        "method_accuracy": float(summary_all["method_correct"].mean()),
        "historical_mean_elapsed": float(pd.to_numeric(cohort_all["match_time_sec"], errors="raise").mean()),
        "simulated_mean_elapsed": float(summary_all["sim_mean_elapsed"].mean()),
        "duration_relative_bias": float(
            summary_all["sim_mean_elapsed"].mean()
            / pd.to_numeric(cohort_all["match_time_sec"], errors="raise").mean()
            - 1.0
        ),
    }])

    method_overall = pd.DataFrame([
        {
            "method": method,
            "actual_share": float(summary_all["actual_method"].eq(method).mean()),
            "simulated_share": float(summary_all[col].mean()),
            "bias_sim_minus_actual": float(summary_all[col].mean() - summary_all["actual_method"].eq(method).mean()),
        }
        for method, col in method_cols.items()
    ])
    timing = _finish_timing(summary_all, cohort_all, paths)

    headline_by_division = pd.concat([x[0] for x in division_outputs], ignore_index=True)
    method_by_division = pd.concat([x[1] for x in division_outputs], ignore_index=True)
    mechanics_by_division = pd.concat([x[2] for x in division_outputs], ignore_index=True)
    rates_by_division = pd.concat([x[3] for x in division_outputs], ignore_index=True)

    return {
        "cohort": cohort_all,
        "summary": summary_all,
        "overall": overall,
        "method_overall": method_overall,
        "finish_timing": timing,
        "headline_by_division": headline_by_division,
        "method_by_division": method_by_division,
        "mechanics_by_division": mechanics_by_division,
        "rates_by_division": rates_by_division,
    }


def _round_validation(target_n: int, paths: int, seed: int):
    frames = []
    for i, division in enumerate(shared.MEN_DIVISIONS):
        cohort, _ = wc_audit.select_cohort(division, target_n)
        print(f"ROUND BUCKET | {division} | fights={len(cohort)} paths={paths}")
        frames.append(
            bucket.simulate_round_rows(
                cohort,
                division,
                paths,
                seed + i * 100_000_000,
            )
        )
    sim_rows = pd.concat(frames, ignore_index=True)
    _, hist_summary = shared._historical_men(target_n)
    sim_summary = shared._aggregate(sim_rows, "simulated", 0.0, 0.0, "selected_continuous_full")
    score = shared._score(sim_summary, hist_summary)
    sim_summary["score"] = score
    return hist_summary, sim_summary, sim_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--predictive-paths", type=int, default=20)
    parser.add_argument("--round-paths", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/full_mens_continuous_power_validation"),
    )
    args = parser.parse_args()

    _patch_diagnostic_boundaries()
    _install_summary_wrapper()

    predictive = _predictive_validation(args.target_n, args.predictive_paths, args.seed)
    hist_round, sim_round, sim_round_rows = _round_validation(
        args.target_n,
        args.round_paths,
        args.seed + 5_000_000_000,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in predictive.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    hist_round.to_csv(args.out_dir / "historical_round_targets.csv", index=False)
    sim_round.to_csv(args.out_dir / "simulated_round_metrics.csv", index=False)
    sim_round_rows.to_csv(args.out_dir / "simulated_round_rows.csv", index=False)

    print("\nSELECTED CURVE")
    print(f"offset(t) = clip({INTERCEPT} - t/{DENOMINATOR}, {LOWER_CAP}, {UPPER_CAP})")
    print("\nOVERALL PREDICTIVE")
    print(predictive["overall"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nOVERALL METHOD")
    print(predictive["method_overall"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nCUMULATIVE NONDECISION FINISH TIMING")
    print(predictive["finish_timing"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nDIVISION HEADLINES")
    print(predictive["headline_by_division"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nHISTORICAL ROUND TARGETS")
    print(hist_round.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nSIMULATED ROUND METRICS")
    print(sim_round.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
