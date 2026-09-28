"""Research-only shared effective-power decay grid for Event Clock V2.

Purpose
-------
Test whether a higher fresh effective-power level plus a common round-by-round
power decline can recover the historical men's R1/R2/R3 KD and KO lethality
shape without changing strike budgets, event timing, FSR ratings, or frozen V1
code.

Experimental translation only:

    effective_power = persisted_power + fresh_boost - round_decay * round_index

where round_index is 0, 1, 2 for R1, R2, R3. Fighter-specific stamina
modulation is intentionally zero because the historical interaction study did
not validate either inherited stamina trait as a modifier of decay slope.

The script uses common random seeds across arms. It first runs a coarse grid at
low paths, ranks arms against aggregate historical men's KD/100 landed and
KO/100 landed by round, then re-runs the best candidates at higher paths.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.round_stats.build_round_fighter_state import (
    read_round_stats,
    standardize_round_stats_input,
    validate_round_stats_input,
)
from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import (
    EventClockShadowKOKDModel as FrozenShadow,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics import lethality_round_bucket as bucket
from pipeline.simulation.event_clock_mc_v2.diagnostics.pace_survival_decomposition import DIVISIONS
from pipeline.simulation.event_clock_mc_v2.diagnostics.weight_class_audit import select_cohort
from pipeline.simulation.event_mc_v1.components.profiles import MatchupProfiles, Side

MEN_DIVISIONS = tuple(d for d in DIVISIONS if not d.startswith("women's "))

_CURRENT_FRESH_BOOST = 0.0
_CURRENT_ROUND_DECAY = 0.0


class SharedRoundPowerShadow(FrozenShadow):
    """Frozen shadow mechanics with research-only round-dependent power profiles."""

    def _round_profiles(self, state) -> MatchupProfiles:
        round_index = min(max(int(float(state.fight_time_seconds) // 300.0), 0), 2)
        offset = _CURRENT_FRESH_BOOST - _CURRENT_ROUND_DECAY * float(round_index)

        def shifted(side: Side):
            profile = self.profiles.fighter(side)
            return replace(
                profile,
                striking_power=float(profile.striking_power) + offset,
            )

        return MatchupProfiles(
            red=shifted(Side.RED),
            blue=shifted(Side.BLUE),
        )

    def resolve_landed_strike(self, *, state, attacker, prior_defender_kds, rng):
        model = FrozenShadow(
            profiles=self._round_profiles(state),
            calibration=self.calibration,
        )
        return model.resolve_landed_strike(
            state=state,
            attacker=attacker,
            prior_defender_kds=prior_defender_kds,
            rng=rng,
        )


def _set_arm(fresh_boost: float, round_decay: float) -> None:
    global _CURRENT_FRESH_BOOST, _CURRENT_ROUND_DECAY
    _CURRENT_FRESH_BOOST = float(fresh_boost)
    _CURRENT_ROUND_DECAY = float(round_decay)


def _aggregate(rows: pd.DataFrame, source: str, fresh_boost: float, round_decay: float, stage: str) -> pd.DataFrame:
    out = []
    for round_no, g in rows.groupby("round", sort=True):
        landed = float(g["sig_landed"].sum())
        kd = float(g["kd"].sum())
        ko = float(g["ko_finish"].sum())
        exposure_min = float(g["round_exposure_seconds"].sum()) / 60.0
        out.append({
            "stage": stage,
            "source": source,
            "fresh_boost": float(fresh_boost),
            "round_decay": float(round_decay),
            "round": int(round_no),
            "at_risk_rounds": int(len(g)),
            "exposure_minutes": exposure_min,
            "sig_attempts_per_min": float(g["sig_attempted"].sum()) / exposure_min if exposure_min > 0 else np.nan,
            "sig_landed_per_min": landed / exposure_min if exposure_min > 0 else np.nan,
            "kd_per_100_sig_landed": kd / landed * 100.0 if landed > 0 else np.nan,
            "ko_finishes_per_100_sig_landed": ko / landed * 100.0 if landed > 0 else np.nan,
            "ko_finish_hazard": ko / float(len(g)) if len(g) else np.nan,
        })
    return pd.DataFrame(out)


def _score(sim_summary: pd.DataFrame, hist_summary: pd.DataFrame) -> float:
    merged = sim_summary.merge(
        hist_summary[["round", "kd_per_100_sig_landed", "ko_finishes_per_100_sig_landed"]],
        on="round",
        suffixes=("_sim", "_hist"),
        validate="one_to_one",
    )
    terms = []
    for metric, weight in (
        ("kd_per_100_sig_landed", 1.0),
        ("ko_finishes_per_100_sig_landed", 2.0),
    ):
        sim = np.clip(merged[f"{metric}_sim"].to_numpy(float), 1e-5, None)
        hist = np.clip(merged[f"{metric}_hist"].to_numpy(float), 1e-5, None)
        terms.extend((weight * (np.log(sim / hist) ** 2)).tolist())
    return float(np.mean(terms))


def _historical_men(target_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_round_stats()
    raw = standardize_round_stats_input(raw)
    validate_round_stats_input(raw)
    frames = []
    for division in MEN_DIVISIONS:
        cohort, _ = select_cohort(division, target_n)
        frames.append(bucket.historical_round_rows(cohort, division, raw))
    rows = pd.concat(frames, ignore_index=True)
    return rows, _aggregate(rows, "historical", 0.0, 0.0, "historical")


def _simulate_men(target_n: int, paths: int, seed: int, fresh_boost: float, round_decay: float) -> pd.DataFrame:
    _set_arm(fresh_boost, round_decay)
    frames = []
    for i, division in enumerate(MEN_DIVISIONS):
        cohort, _ = select_cohort(division, target_n)
        print(
            f"ARM fresh={fresh_boost:.1f} decay={round_decay:.1f} | "
            f"{division} | fights={len(cohort)} paths={paths}"
        )
        frames.append(
            bucket.simulate_round_rows(
                cohort,
                division,
                paths,
                seed + i * 100_000_000,
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--coarse-paths", type=int, default=8)
    parser.add_argument("--validation-paths", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/shared_power_decay_grid"),
    )
    args = parser.parse_args()

    # Patch only the diagnostic module's constructor reference. Frozen V1 source
    # and persisted model state are untouched.
    bucket.EventClockShadowKOKDModel = SharedRoundPowerShadow

    hist_rows, hist_summary = _historical_men(args.target_n)
    print("\nHISTORICAL MEN TARGETS")
    print(hist_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    grid = [(0.0, 0.0)]
    grid += [(fresh, decay) for fresh in (15.0, 25.0, 35.0) for decay in (15.0, 25.0, 35.0)]

    coarse_summaries = []
    ranking = []
    for fresh, decay in grid:
        sim_rows = _simulate_men(
            args.target_n,
            args.coarse_paths,
            args.seed,
            fresh,
            decay,
        )
        summary = _aggregate(sim_rows, "simulated", fresh, decay, "coarse")
        score = _score(summary, hist_summary)
        summary["score"] = score
        coarse_summaries.append(summary)
        ranking.append({
            "fresh_boost": fresh,
            "round_decay": decay,
            "score": score,
        })
        print(f"COARSE SCORE fresh={fresh:.1f} decay={decay:.1f}: {score:.6f}")

    ranking_df = pd.DataFrame(ranking).sort_values("score").reset_index(drop=True)
    finalists = ranking_df.head(args.top_k)[["fresh_boost", "round_decay"]].itertuples(index=False, name=None)

    validation_summaries = []
    for fresh, decay in finalists:
        sim_rows = _simulate_men(
            args.target_n,
            args.validation_paths,
            args.seed + 7_000_000_000,
            float(fresh),
            float(decay),
        )
        summary = _aggregate(sim_rows, "simulated", float(fresh), float(decay), "validation")
        score = _score(summary, hist_summary)
        summary["score"] = score
        validation_summaries.append(summary)
        print(f"VALIDATION SCORE fresh={fresh:.1f} decay={decay:.1f}: {score:.6f}")

    coarse_df = pd.concat(coarse_summaries, ignore_index=True)
    validation_df = pd.concat(validation_summaries, ignore_index=True)
    validation_rank = (
        validation_df[["fresh_boost", "round_decay", "score"]]
        .drop_duplicates()
        .sort_values("score")
        .reset_index(drop=True)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hist_summary.to_csv(args.out_dir / "historical_men_targets.csv", index=False)
    coarse_df.to_csv(args.out_dir / "coarse_round_metrics.csv", index=False)
    ranking_df.to_csv(args.out_dir / "coarse_ranking.csv", index=False)
    validation_df.to_csv(args.out_dir / "validation_round_metrics.csv", index=False)
    validation_rank.to_csv(args.out_dir / "validation_ranking.csv", index=False)

    print("\nCOARSE RANKING")
    print(ranking_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nVALIDATION RANKING")
    print(validation_rank.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nVALIDATION ROUND METRICS")
    print(validation_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
