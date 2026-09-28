"""Diagnostics-only damage-reservoir audit for the locked curve-16 candidate.

Configuration:
- first 200 mature 2020+ bouts x 10 paths
- contact sigma 0.80
- power magnitude scale 75
- base damage at power 50 = 1.18
- KD base -8.80
- KD shock coefficient 100
- KD depletion coefficient 0
- collapse scale 2.0
- collapse curvature 16.0
- fatigue exponent 2.0
- max fatigue penalty 45
- global stamina recovery 40% of missing
- global damage recovery 20% of missing

No simulator or FSR constants/artifacts are modified persistently. The script
records damage-reservoir fractions at end R1/start R2, end R2/start R3, and
end R3, plus direct-strike vs terminal-collapse KO/TKO finishes by round.
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

DETAIL_PATH = Path("data/experimental/damage_reservoir_curve16_exp2_200_detail.csv")
SUMMARY_PATH = Path("data/experimental/damage_reservoir_curve16_exp2_200_summary.csv")
FINISH_PATH = Path("data/experimental/damage_reservoir_curve16_exp2_200_finish_mechanisms.csv")

CHECKPOINT_ORDER = ["end_r1", "start_r2", "end_r2", "start_r3", "end_r3"]


def _configure_candidate() -> None:
    run87.KD_BASE_LOGIT = KD_BASE_LOGIT
    run87.COLLAPSE_CURVATURE = COLLAPSE_CURVATURE
    run87.COLLAPSE = collapse_mod.CollapseCandidate(
        f"scale{COLLAPSE_SCALE:.1f}_curve{COLLAPSE_CURVATURE:.1f}",
        COLLAPSE_SCALE,
        COLLAPSE_CURVATURE,
    )
    # Research-process-local override only. The imported module is reset when
    # this Python process exits; no source constant is edited.
    v31.FATIGUE_CURVE_EXPONENT = FATIGUE_EXPONENT


def _append_recovery_checkpoints(rows: list[dict], sim, bout_id: str, path_idx: int) -> None:
    for event in sim.round_recovery_events:
        completed_round = int(event["after_round"])
        if completed_round not in (1, 2):
            continue
        fighter = int(event["fighter"])
        capacity = float(sim.damage_state[fighter].reservoir_capacity)
        before = float(event["reservoir_before"])
        after = float(event["reservoir_after"])
        rows.append({
            "bout_id": bout_id,
            "path": path_idx,
            "fighter": fighter,
            "checkpoint": f"end_r{completed_round}",
            "reservoir_capacity": capacity,
            "reservoir_current": before,
            "reservoir_fraction": before / capacity if capacity > 0 else np.nan,
        })
        rows.append({
            "bout_id": bout_id,
            "path": path_idx,
            "fighter": fighter,
            "checkpoint": f"start_r{completed_round + 1}",
            "reservoir_capacity": capacity,
            "reservoir_current": after,
            "reservoir_fraction": after / capacity if capacity > 0 else np.nan,
        })


def _append_end_r3(rows: list[dict], sim, path, bout_id: str, path_idx: int) -> None:
    # Only fighters that survive through the scheduled end of R3 have a true
    # end-R3 checkpoint. Finished paths are excluded rather than treated as zero.
    if path.finish is not None:
        return
    for fighter, state in enumerate(sim.damage_state):
        capacity = float(state.reservoir_capacity)
        current = float(state.reservoir_current)
        rows.append({
            "bout_id": bout_id,
            "path": path_idx,
            "fighter": int(fighter),
            "checkpoint": "end_r3",
            "reservoir_capacity": capacity,
            "reservoir_current": current,
            "reservoir_fraction": current / capacity if capacity > 0 else np.nan,
        })


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for checkpoint in CHECKPOINT_ORDER:
        g = detail.loc[detail["checkpoint"].eq(checkpoint)].copy()
        if g.empty:
            continue
        v = g["reservoir_fraction"].astype(float)
        rows.append({
            "checkpoint": checkpoint,
            "fighters": int(len(g)),
            "paths": int(g[["bout_id", "path"]].drop_duplicates().shape[0]),
            "reservoir_fraction_mean": float(v.mean()),
            "reservoir_fraction_p10": float(v.quantile(0.10)),
            "reservoir_fraction_p25": float(v.quantile(0.25)),
            "reservoir_fraction_median": float(v.median()),
            "reservoir_fraction_p75": float(v.quantile(0.75)),
            "reservoir_fraction_p90": float(v.quantile(0.90)),
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

    reservoir_rows: list[dict] = []
    finish_rows: list[dict] = []
    completed = 0

    print("\n" + "=" * 150)
    print("DAMAGE RESERVOIR DIAGNOSTIC — CURVE 16 / FATIGUE EXPONENT 2.0 — 200 BOUTS x 10 PATHS")
    print("=" * 150)
    print(f"KD base={run87.KD_BASE_LOGIT:.2f}; shock={run87.KD_SHOCK_COEFFICIENT:.0f}; depletion={run87.KD_DEPLETION_COEFFICIENT:.2f}")
    print(f"collapse scale={COLLAPSE_SCALE:.1f}; curvature={COLLAPSE_CURVATURE:.1f}")
    print(f"fatigue exponent={v31.FATIGUE_CURVE_EXPONENT:.2f}; max penalty={v31.MAX_FATIGUE_RATING_PENALTY:.1f}")
    print(f"damage recovery={v33.GLOBAL_DAMAGE_RECOVERY_FRACTION:.0%} of missing between rounds")
    print("end-R1/R2 values are exact pre-recovery reservoir states; start-R2/R3 are exact post-recovery states")
    print("end-R3 includes only paths surviving through the scheduled end of R3")
    print("diagnostics only: no production simulator or FSR artifact modified")

    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for path_idx, seed in enumerate(seed_matrix[bout_idx]):
            sim = run87.AuditSim(
                red,
                blue,
                rounds=3,
                seed=int(seed),
                red_age=r_age,
                blue_age=b_age,
            )
            path = sim.run()

            _append_recovery_checkpoints(reservoir_rows, sim, bout_id, path_idx)
            _append_end_r3(reservoir_rows, sim, path, bout_id, path_idx)

            finish_round = 0
            mechanism = "none"
            if path.finish is not None:
                finish_round = int(getattr(path.finish, "round", 0) or 0)
                if sim.terminal_collapse_finishes > 0:
                    mechanism = "terminal_collapse"
                elif sim.direct_strike_finishes > 0:
                    mechanism = "direct_strike"
                else:
                    mechanism = "other"
            finish_rows.append({
                "bout_id": bout_id,
                "path": path_idx,
                "finish_round": finish_round,
                "mechanism": mechanism,
            })

        completed += DEFAULT_PATHS
        if completed % 500 == 0 or bout_idx + 1 == len(cohort):
            print(f"paths {completed:,}/{total_paths:,}", flush=True)

    detail = pd.DataFrame(reservoir_rows)
    summary = _summarize(detail)
    finishes = pd.DataFrame(finish_rows)

    mechanism_rows: list[dict] = []
    for rnd in (1, 2, 3):
        g = finishes.loc[finishes["finish_round"].eq(rnd)]
        mechanism_rows.append({
            "round": rnd,
            "ko_tko_finishes": int(len(g)),
            "ko_rate_per_starting_path": float(len(g) / total_paths),
            "terminal_collapse": int((g["mechanism"] == "terminal_collapse").sum()),
            "direct_strike": int((g["mechanism"] == "direct_strike").sum()),
            "other": int((g["mechanism"] == "other").sum()),
            "terminal_share_of_round_finishes": float((g["mechanism"] == "terminal_collapse").mean()) if len(g) else np.nan,
            "direct_share_of_round_finishes": float((g["mechanism"] == "direct_strike").mean()) if len(g) else np.nan,
        })
    mechanism_summary = pd.DataFrame(mechanism_rows)

    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    mechanism_summary.to_csv(FINISH_PATH, index=False)

    print("\nRESERVOIR CHECKPOINT SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nKO/TKO FINISH MECHANISMS BY ROUND")
    print(mechanism_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved detail: {DETAIL_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved finish mechanisms: {FINISH_PATH}")
    print("Research only: no production simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
