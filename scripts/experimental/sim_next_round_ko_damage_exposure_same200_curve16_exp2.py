"""Simulator next-round KO/TKO vs prior damage-exposure audit.

Research-only mirror of ``historical_next_round_ko_damage_exposure_same200.py``.

Configuration
-------------
- exact same first 200 mature 2020+ bouts
- 10 paths per bout
- contact sigma 0.80
- power magnitude scale 75
- base damage at power 50 = 1.18
- KD base -8.80
- KD shock coefficient 100
- KD depletion coefficient 0
- collapse scale 2.0
- collapse curvature 16.0
- fatigue exponent 2.0
- global stamina recovery 40% of missing
- global damage recovery 20% of missing

Unit of analysis
----------------
One simulated fighter entering R2 or R3. For each entry we record:
- cumulative significant strikes absorbed before the round;
- cumulative surviving KDs absorbed before the round;
- exact post-recovery reservoir fraction entering the round;
- whether that fighter is stopped by KO/TKO in that upcoming round;
- finish mechanism when stopped (direct strike vs terminal collapse).

The current simulator does not split significant strikes into head/body/leg, so
there is no simulator-side head-strike exposure metric yet. The historical head
strike audit remains useful as an external observable target, but this file does
not fabricate a head-strike proxy.

No simulator constants, FSR values, or production artifacts are persistently
modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_kdbase87_curve20_scale2_200 as run87
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse_mod
from scripts.experimental import fsr_static_mc_ko_tko_v3_1_rolling_fsr as v31
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810

KD_BASE_LOGIT = -8.80
COLLAPSE_SCALE = 2.0
COLLAPSE_CURVATURE = 16.0
FATIGUE_EXPONENT = 2.0

OUTPUT_DIR = Path("data/experimental")
DETAIL_PATH = OUTPUT_DIR / "sim_next_round_ko_damage_exposure_same200_curve16_exp2_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "sim_next_round_ko_damage_exposure_same200_curve16_exp2_summary.csv"
QUINTILE_PATH = OUTPUT_DIR / "sim_next_round_ko_damage_exposure_same200_curve16_exp2_quintiles.csv"
KD_PATH = OUTPUT_DIR / "sim_next_round_ko_damage_exposure_same200_curve16_exp2_kd.csv"
RESERVOIR_PATH = OUTPUT_DIR / "sim_next_round_ko_damage_exposure_same200_curve16_exp2_reservoir_quintiles.csv"


def _configure_candidate() -> None:
    run87.KD_BASE_LOGIT = KD_BASE_LOGIT
    run87.COLLAPSE_CURVATURE = COLLAPSE_CURVATURE
    run87.COLLAPSE = collapse_mod.CollapseCandidate(
        f"scale{COLLAPSE_SCALE:.1f}_curve{COLLAPSE_CURVATURE:.1f}",
        COLLAPSE_SCALE,
        COLLAPSE_CURVATURE,
    )
    # Process-local research override only.
    v31.FATIGUE_CURVE_EXPONENT = FATIGUE_EXPONENT


def _run_prefix(red, blue, *, rounds: int, seed: int, red_age, blue_age):
    sim = run87.AuditSim(
        red,
        blue,
        rounds=rounds,
        seed=seed,
        red_age=red_age,
        blue_age=blue_age,
    )
    path = sim.run()
    return sim, path


def _finish_round(path) -> int:
    if path.finish is None:
        return 0
    return int(getattr(path.finish, "round", 0) or 0)


def _finish_loser(path) -> int | None:
    if path.finish is None:
        return None
    loser = getattr(path.finish, "loser", None)
    return int(loser) if loser is not None else None


def _finish_mechanism(sim, path) -> str:
    if path.finish is None:
        return "none"
    if sim.terminal_collapse_finishes > 0:
        return "terminal_collapse"
    if sim.direct_strike_finishes > 0:
        return "direct_strike"
    return "other"


def _start_reservoir_fraction(sim, completed_round: int, fighter: int) -> float:
    """Exact post-recovery reservoir fraction after completed_round."""
    matches = [
        event
        for event in sim.round_recovery_events
        if int(event["after_round"]) == completed_round
        and int(event["fighter"]) == fighter
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one recovery event after R{completed_round} "
            f"for fighter {fighter}, found {len(matches)}"
        )
    event = matches[0]
    capacity = float(sim.damage_state[fighter].reservoir_capacity)
    after = float(event["reservoir_after"])
    return after / capacity if capacity > 0 else np.nan


def _entry_row(
    *,
    bout_id: str,
    path_idx: int,
    fighter: int,
    entering_round: int,
    prior_sim,
    upcoming_sim,
    upcoming_path,
) -> dict:
    opponent = 1 - fighter

    # Opponent's cumulative landed significant strikes are this fighter's
    # cumulative significant strikes absorbed.
    cum_sig_absorbed = int(prior_sim.stats[opponent].sig_landed)
    cum_kd_absorbed = int(prior_sim.stats[fighter].knockdowns_absorbed)

    reservoir_fraction = _start_reservoir_fraction(
        upcoming_sim,
        completed_round=entering_round - 1,
        fighter=fighter,
    )

    finish_round = _finish_round(upcoming_path)
    finish_loser = _finish_loser(upcoming_path)
    next_round_ko_loss = int(
        finish_round == entering_round and finish_loser == fighter
    )
    mechanism = (
        _finish_mechanism(upcoming_sim, upcoming_path)
        if next_round_ko_loss
        else "none"
    )

    return {
        "bout_id": bout_id,
        "path": path_idx,
        "fighter": fighter,
        "entering_round": entering_round,
        "cum_sig_absorbed": cum_sig_absorbed,
        "cum_kd_absorbed": cum_kd_absorbed,
        "reservoir_fraction": float(reservoir_fraction),
        "next_round_ko_loss": next_round_ko_loss,
        "next_round_finish_mechanism": mechanism,
    }


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for entering_round, g in detail.groupby("entering_round", sort=True):
        rows.append({
            "entering_round": int(entering_round),
            "fighter_entries": int(len(g)),
            "paths_reaching_round": int(g[["bout_id", "path"]].drop_duplicates().shape[0]),
            "next_round_ko_losses": int(g["next_round_ko_loss"].sum()),
            "fighter_next_round_ko_loss_rate": float(g["next_round_ko_loss"].mean()),
            "median_cum_sig_absorbed": float(g["cum_sig_absorbed"].median()),
            "p75_cum_sig_absorbed": float(g["cum_sig_absorbed"].quantile(0.75)),
            "p90_cum_sig_absorbed": float(g["cum_sig_absorbed"].quantile(0.90)),
            "median_cum_kd_absorbed": float(g["cum_kd_absorbed"].median()),
            "reservoir_fraction_mean": float(g["reservoir_fraction"].mean()),
            "reservoir_fraction_median": float(g["reservoir_fraction"].median()),
            "reservoir_fraction_p10": float(g["reservoir_fraction"].quantile(0.10)),
            "reservoir_fraction_p25": float(g["reservoir_fraction"].quantile(0.25)),
        })
    return pd.DataFrame(rows)


def _qcut_labels(series: pd.Series) -> pd.Series:
    # Rank first so repeated integer exposure values do not collapse qcut edges.
    # This intentionally creates approximately equal-sized empirical quintiles,
    # matching the historical audit's goal rather than fixed exposure thresholds.
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])


def _quintiles(detail: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict] = []
    for entering_round, g0 in detail.groupby("entering_round", sort=True):
        g = g0.copy()
        g["quintile"] = _qcut_labels(g[metric])
        for quintile, q in g.groupby("quintile", observed=True, sort=True):
            losses = q[q["next_round_ko_loss"].eq(1)]
            rows.append({
                "entering_round": int(entering_round),
                "metric": metric,
                "quintile": str(quintile),
                "fighter_entries": int(len(q)),
                "exposure_min": float(q[metric].min()),
                "exposure_median": float(q[metric].median()),
                "exposure_max": float(q[metric].max()),
                "reservoir_fraction_median": float(q["reservoir_fraction"].median()),
                "next_round_ko_losses": int(q["next_round_ko_loss"].sum()),
                "next_round_ko_loss_rate": float(q["next_round_ko_loss"].mean()),
                "terminal_collapse_losses": int(
                    (losses["next_round_finish_mechanism"] == "terminal_collapse").sum()
                ),
                "direct_strike_losses": int(
                    (losses["next_round_finish_mechanism"] == "direct_strike").sum()
                ),
            })
    return pd.DataFrame(rows)


def _kd_summary(detail: pd.DataFrame) -> pd.DataFrame:
    work = detail.copy()
    work["cum_kd_bucket"] = np.select(
        [work["cum_kd_absorbed"].eq(0), work["cum_kd_absorbed"].eq(1)],
        ["0", "1"],
        default="2+",
    )
    rows: list[dict] = []
    for (entering_round, bucket), g in work.groupby(
        ["entering_round", "cum_kd_bucket"], sort=True
    ):
        losses = g[g["next_round_ko_loss"].eq(1)]
        rows.append({
            "entering_round": int(entering_round),
            "cum_kd_absorbed": bucket,
            "fighter_entries": int(len(g)),
            "next_round_ko_losses": int(g["next_round_ko_loss"].sum()),
            "next_round_ko_loss_rate": float(g["next_round_ko_loss"].mean()),
            "median_cum_sig_absorbed": float(g["cum_sig_absorbed"].median()),
            "reservoir_fraction_median": float(g["reservoir_fraction"].median()),
            "terminal_collapse_losses": int(
                (losses["next_round_finish_mechanism"] == "terminal_collapse").sum()
            ),
            "direct_strike_losses": int(
                (losses["next_round_finish_mechanism"] == "direct_strike").sum()
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    _configure_candidate()

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(DEFAULT_BOUTS).reset_index(drop=True)
    total_paths = len(cohort) * DEFAULT_PATHS

    rng = np.random.default_rng(DEFAULT_SEED)
    seed_matrix = rng.integers(
        0, 2**31 - 1, size=(len(cohort), DEFAULT_PATHS), dtype=np.int64
    )

    rows: list[dict] = []
    completed = 0

    print("\n" + "=" * 158)
    print("SIMULATOR NEXT-ROUND KO/TKO vs PRIOR DAMAGE EXPOSURE — EXACT SAME 200 BOUTS x 10 PATHS")
    print("=" * 158)
    print("Unit: simulated fighter entering R2 or R3; exposure is what that fighter absorbed before the round")
    print("KO/TKO target: that specific fighter is stopped by KO/TKO in the upcoming round")
    print("Simulator has no head/body/leg strike split, so only total sig-strike and KD exposure are mirrored")
    print(f"fatigue exponent={v31.FATIGUE_CURVE_EXPONENT:.2f}; damage recovery={v33.GLOBAL_DAMAGE_RECOVERY_FRACTION:.0%}")
    print(f"KD base={run87.KD_BASE_LOGIT:.2f}; collapse scale={COLLAPSE_SCALE:.1f}; curvature={COLLAPSE_CURVATURE:.1f}")
    print("research only: all other simulator/FSR settings unchanged")

    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for path_idx, seed in enumerate(seed_matrix[bout_idx]):
            seed = int(seed)
            sim1, path1 = _run_prefix(
                red,
                blue,
                rounds=1,
                seed=seed,
                red_age=r_age,
                blue_age=b_age,
            )
            sim2, path2 = _run_prefix(
                red,
                blue,
                rounds=2,
                seed=seed,
                red_age=r_age,
                blue_age=b_age,
            )
            sim3, path3 = _run_prefix(
                red,
                blue,
                rounds=3,
                seed=seed,
                red_age=r_age,
                blue_age=b_age,
            )

            # A path enters R2 only if R1 did not finish it.
            if path1.finish is None:
                for fighter in (0, 1):
                    rows.append(
                        _entry_row(
                            bout_id=bout_id,
                            path_idx=path_idx,
                            fighter=fighter,
                            entering_round=2,
                            prior_sim=sim1,
                            upcoming_sim=sim2,
                            upcoming_path=path2,
                        )
                    )

            # A path enters R3 only if the two-round prefix did not finish it.
            if path2.finish is None:
                for fighter in (0, 1):
                    rows.append(
                        _entry_row(
                            bout_id=bout_id,
                            path_idx=path_idx,
                            fighter=fighter,
                            entering_round=3,
                            prior_sim=sim2,
                            upcoming_sim=sim3,
                            upcoming_path=path3,
                        )
                    )

            completed += 1
            if completed % 500 == 0 or completed == total_paths:
                print(f"paths {completed:,}/{total_paths:,}", flush=True)

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No R2/R3 fighter entries were generated")

    summary = _summary(detail)
    sig_quintiles = _quintiles(detail, "cum_sig_absorbed")
    reservoir_quintiles = _quintiles(detail, "reservoir_fraction")
    kd_summary = _kd_summary(detail)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    sig_quintiles.to_csv(QUINTILE_PATH, index=False)
    kd_summary.to_csv(KD_PATH, index=False)
    reservoir_quintiles.to_csv(RESERVOIR_PATH, index=False)

    print("\nROUND-ENTRY SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCUMULATIVE SIGNIFICANT-STRIKE EXPOSURE QUINTILES")
    print(sig_quintiles.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCUMULATIVE KD EXPOSURE")
    print(kd_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nRESERVOIR-FRACTION QUINTILES")
    print(reservoir_quintiles.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nHISTORICAL SAME-200 REFERENCE")
    print("R2 fighter next-round KO loss: 25/400 = 6.25%")
    print("R3 fighter next-round KO loss:  6/322 = 1.86%")
    print("Historical R3 prior-KD loss rates: 0 KD=1.41%; 1 KD=3.12%; 2+=14.29%")

    print("\nOUTPUTS")
    print(DETAIL_PATH)
    print(SUMMARY_PATH)
    print(QUINTILE_PATH)
    print(KD_PATH)
    print(RESERVOIR_PATH)
    print("\nResearch only: no production simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
