"""Decompose Round-1 shock/KD generation for modern mature-fighter R1 KO bouts.

Primary cohort
--------------
- Actual UFC Round-1 KO/TKO bouts dated 2020-01-01 or later.
- Both fighters had at least 3 prior UFC fights before the bout.
- Leakage-safe pre-fight FSR snapshots only.

A same-size fixed random sample of non-R1-KO bouts is included only as a
comparison baseline. The diagnostic itself is centered on the actual R1-KO
cohort and simulates Round 1 only.

Questions
---------
1. Do actual R1-KO bouts generate enough simulated significant-strike exposure?
2. Do they generate more upper-tail shock than ordinary bouts?
3. Given severe shock, does the locked KD model convert it into knockdowns?
4. Is the failure before severe shock, at KD conversion, or after KD?

No simulator constants or FSR values are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_2020plus_mature_r1_severity_decomposition.parquet"
)
DEFAULT_PATHS_PER_BOUT = 25
DEFAULT_SEED = 20260810
HEARTBEAT_PATHS = 1000
STRONG = next(c for c in collapse.CANDIDATES if c.name == "strong")
SHOCK_THRESHOLDS = (0.03, 0.05, 0.08, 0.10)


class R1SeverityTraceSimulator(collapse.StaticFSRMCKOTKOV2KDCollapse):
    """Strong-collapse simulator with strike-level severity/KD trace capture."""

    def __init__(self, *args, **kwargs) -> None:
        # Round 1 only: every captured strike is an R1 strike.
        kwargs["rounds"] = 1
        super().__init__(*args, **kwargs)
        self.strike_trace: list[dict[str, float | int | bool]] = []

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            if self.finish is not None:
                break

            state = self.damage_state[defender]
            recent_kd_before = state.recent_knockdown
            reservoir_before = float(state.reservoir_current)
            reservoir_capacity = float(state.reservoir_capacity)

            raw_damage = self._draw_strike_damage(attacker)
            effective_damage = raw_damage
            if recent_kd_before:
                from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
                effective_damage *= ko.POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER

            state.reservoir_current = max(0.0, state.reservoir_current - effective_damage)
            shock_fraction = effective_damage / reservoir_capacity
            p_kd = self._knockdown_probability(defender, effective_damage)
            knockdown = bool(self.rng.random() < p_kd)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            attacker_stats.damage_dealt += effective_damage
            defender_stats.damage_absorbed += effective_damage
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage, effective_damage
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage, effective_damage
            )
            total_damage += effective_damage

            collapse_damage = 0.0
            if knockdown:
                collapse_fraction = self._kd_collapse_fraction(shock_fraction)
                collapse_damage = min(
                    collapse_fraction * reservoir_capacity,
                    state.reservoir_current,
                )
                state.reservoir_current = max(0.0, state.reservoir_current - collapse_damage)
                self.kd_collapse_damage_dealt[attacker] += collapse_damage
                attacker_stats.damage_dealt += collapse_damage
                defender_stats.damage_absorbed += collapse_damage
                total_damage += collapse_damage
                attacker_stats.knockdowns_scored += 1
                defender_stats.knockdowns_absorbed += 1

                from scripts.experimental import fsr_static_mc_damage_v1 as damage
                state.recent_knockdown_segments = max(
                    state.recent_knockdown_segments, damage.RECENT_KD_SEGMENTS
                )
                knockdowns += 1

            self.strike_trace.append(
                {
                    "attacker": attacker,
                    "defender": defender,
                    "raw_damage": float(raw_damage),
                    "effective_damage": float(effective_damage),
                    "shock_fraction": float(shock_fraction),
                    "p_kd": float(p_kd),
                    "knockdown": knockdown,
                    "recent_kd_before": bool(recent_kd_before),
                    "reservoir_fraction_before": reservoir_before / reservoir_capacity,
                    "reservoir_fraction_after_strike_and_collapse": (
                        float(state.reservoir_current) / reservoir_capacity
                    ),
                    "collapse_damage": float(collapse_damage),
                }
            )

            from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.finish = ko.FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=knockdown,
                    recent_kd_before=bool(recent_kd_before),
                )
                break

        return total_damage, knockdowns


def _prepare_cohorts(
    cohort: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = cohort[cohort["actual_r1_ko"].eq(1)].copy()
    controls = cohort[cohort["actual_r1_ko"].eq(0)].copy()
    if primary.empty:
        raise ValueError("No actual R1 KO/TKO bouts in eligible cohort.")
    if len(controls) < len(primary):
        raise ValueError("Not enough non-R1-KO bouts to construct comparison cohort.")
    controls = controls.sample(n=len(primary), random_state=seed, replace=False).copy()
    primary["cohort_group"] = "actual_r1_ko"
    controls["cohort_group"] = "comparison_non_r1_ko"
    return primary, controls


def _run_group(
    bouts: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(bouts) * paths_per_bout
    path_no = 0

    print(
        f"[R1 severity] group={bouts['cohort_group'].iloc[0]}; "
        f"bouts={len(bouts):,}; paths_per_bout={paths_per_bout}; total_paths={total_paths:,}",
        flush=True,
    )

    for bout_index, (_, bout) in enumerate(bouts.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]

        for path_index in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = R1SeverityTraceSimulator(
                red,
                blue,
                collapse=STRONG,
                seed=path_seed,
            )
            path = sim.run()
            trace = pd.DataFrame(sim.strike_trace)

            sig_landed = int(sim.stats[0].sig_landed + sim.stats[1].sig_landed)
            kd_total = int(sim.stats[0].knockdowns_scored + sim.stats[1].knockdowns_scored)
            r1_ko = int(path.finish is not None)

            row: dict[str, object] = {
                "bout_id": bout_id,
                "event_date": bout["event_date"],
                "cohort_group": bout["cohort_group"],
                "actual_r1_ko": int(bout["actual_r1_ko"]),
                "path_index": path_index,
                "path_seed": path_seed,
                "r1_sig_landed": sig_landed,
                "r1_kd": kd_total,
                "r1_any_kd": int(kd_total > 0),
                "r1_ko": r1_ko,
                "r1_landed_damage_events": len(trace),
                "r1_max_shock": float(trace["shock_fraction"].max()) if len(trace) else 0.0,
                "r1_mean_shock": float(trace["shock_fraction"].mean()) if len(trace) else 0.0,
                "r1_mean_p_kd_per_landed": float(trace["p_kd"].mean()) if len(trace) else 0.0,
            }

            for threshold in SHOCK_THRESHOLDS:
                label = int(round(threshold * 100))
                severe = trace[trace["shock_fraction"].ge(threshold)] if len(trace) else trace
                row[f"shock_ge_{label}pct_count"] = int(len(severe))
                row[f"any_shock_ge_{label}pct"] = int(len(severe) > 0)
                row[f"kd_on_shock_ge_{label}pct"] = (
                    int(severe["knockdown"].sum()) if len(severe) else 0
                )
                row[f"mean_p_kd_shock_ge_{label}pct"] = (
                    float(severe["p_kd"].mean()) if len(severe) else np.nan
                )

            rows.append(row)
            path_no += 1
            if path_no % HEARTBEAT_PATHS == 0 or path_no == total_paths:
                recent = pd.DataFrame(rows)
                print(
                    f"[R1 severity] {bouts['cohort_group'].iloc[0]} paths "
                    f"{path_no:,}/{total_paths:,}; bouts_started={bout_index:,}/{len(bouts):,}; "
                    f"P(KD)={recent['r1_any_kd'].mean():.2%}; "
                    f"P(KO)={recent['r1_ko'].mean():.2%}; "
                    f"mean max shock={recent['r1_max_shock'].mean():.4f}",
                    flush=True,
                )

    return pd.DataFrame(rows)


def _print_summary(paths: pd.DataFrame, primary_bouts: int, paths_per_bout: int) -> None:
    print("\n" + "=" * 124)
    print("ROUND-1 SEVERITY / KNOCKDOWN DECOMPOSITION — 2020+ MATURE FIGHTERS")
    print("=" * 124)
    print(f"primary actual R1 KO bouts: {primary_bouts:,}")
    print(f"comparison non-R1-KO bouts: {primary_bouts:,}")
    print(f"paths per bout: {paths_per_bout}")
    print("simulation horizon: Round 1 only")

    summary_rows: list[dict[str, object]] = []
    for group, g in paths.groupby("cohort_group", sort=False):
        row: dict[str, object] = {
            "group": group,
            "paths": len(g),
            "mean_sig_landed": g["r1_sig_landed"].mean(),
            "mean_max_shock": g["r1_max_shock"].mean(),
            "p_any_kd": g["r1_any_kd"].mean(),
            "mean_kd": g["r1_kd"].mean(),
            "p_r1_ko": g["r1_ko"].mean(),
        }
        for threshold in SHOCK_THRESHOLDS:
            label = int(round(threshold * 100))
            row[f"p_any_shock_ge_{label}pct"] = g[f"any_shock_ge_{label}pct"].mean()
            severe_count = int(g[f"shock_ge_{label}pct_count"].sum())
            severe_kd = int(g[f"kd_on_shock_ge_{label}pct"].sum())
            row[f"kd_conversion_shock_ge_{label}pct"] = (
                severe_kd / severe_count if severe_count else np.nan
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print("\nPRIMARY VS COMPARISON")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    primary = paths[paths["cohort_group"].eq("actual_r1_ko")]
    comparison = paths[paths["cohort_group"].eq("comparison_non_r1_ko")]
    print("\nPRIMARY R1-KO COHORT — BOTTLENECK READOUT")
    print(f"mean R1 significant strikes landed/path: {primary['r1_sig_landed'].mean():.3f}")
    print(f"mean maximum shock/path: {primary['r1_max_shock'].mean():.4f}")
    print(f"P(any simulated R1 KD): {primary['r1_any_kd'].mean():.2%}")
    print(f"P(simulated R1 KO): {primary['r1_ko'].mean():.2%}")
    for threshold in SHOCK_THRESHOLDS:
        label = int(round(threshold * 100))
        severe_count = int(primary[f"shock_ge_{label}pct_count"].sum())
        severe_kd = int(primary[f"kd_on_shock_ge_{label}pct"].sum())
        conversion = severe_kd / severe_count if severe_count else float("nan")
        print(
            f"shock >= {label}% capacity: "
            f"P(path has one)={primary[f'any_shock_ge_{label}pct'].mean():.2%}; "
            f"events={severe_count:,}; KD conversion={conversion:.2%}"
        )

    print("\nRELATIVE TRANSLATION INTO ACTUAL-R1-KO COHORT")
    for col, label in [
        ("r1_sig_landed", "sig landed"),
        ("r1_max_shock", "max shock"),
        ("r1_any_kd", "P(any KD)"),
        ("r1_ko", "P(R1 KO)"),
    ]:
        denom = comparison[col].mean()
        ratio = primary[col].mean() / denom if denom else float("nan")
        print(f"{label}: primary/comparison = {ratio:.3f}x")

    print("\nDECISION RULE")
    print("- Little/no lift in sig landed -> opportunity/pressure/precision translation is failing before severity.")
    print("- Sig landed lifts but high-shock incidence does not -> severity/power-tail generation is failing.")
    print("- High-shock incidence lifts but KD conversion stays weak -> locked shock-to-KD mapping is the bottleneck.")
    print("- KD incidence is reasonable but R1 KO remains low -> collapse/follow-up is the later bottleneck.")
    print("- Comparison bouts are diagnostic context only; the primary target is the actual R1-KO cohort.")
    print("- No simulator constants or FSR values are changed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose R1 severity/KD generation on modern mature-fighter actual R1 KO bouts"
    )
    parser.add_argument("--master", type=Path, default=modern.MASTER_PATH)
    parser.add_argument("--fsr-path", type=Path, default=modern.FSR_PATH)
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.paths_per_bout <= 0:
        raise ValueError("--paths-per-bout must be positive")

    master = modern._load_master(args.master)
    candidate = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(args.fsr_path, candidate)
    primary, controls = _prepare_cohorts(cohort, seed=args.seed)

    print(
        f"[R1 severity] eligible={len(cohort):,}; actual_R1_KO={len(primary):,}; "
        f"comparison={len(controls):,}; date={cohort['event_date'].min().date()} -> "
        f"{cohort['event_date'].max().date()}",
        flush=True,
    )

    primary_paths = _run_group(
        primary,
        pairs,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )
    comparison_paths = _run_group(
        controls,
        pairs,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed + 1,
    )
    paths = pd.concat([primary_paths, comparison_paths], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths.to_parquet(args.output, index=False)
    _print_summary(paths, len(primary), args.paths_per_bout)
    print(f"\n[R1 severity] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
