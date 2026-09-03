"""Diagnostics-only fatigue checkpoint audit for the locked curve-16 power candidate.

Runs the same first 200 mature 2020+ bouts x 10 paths with the current R1
working lock:
- contact sigma 0.80
- power magnitude scale 75
- base damage at power 50 = 1.18
- KD base -8.80
- KD shock coefficient 100
- KD depletion coefficient 0
- collapse scale 2.0
- collapse curvature 16.0

No simulator/FSR constants are changed. The script only records the existing
stamina -> fatigue penalty -> effective striking-power state at six checkpoints:
start/end R1, start/end R2, start/end R3.
"""
from __future__ import annotations

from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_kdbase87_curve20_scale2_200 as run87
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse_mod
from scripts.experimental import fsr_static_mc_ko_tko_v3_1_rolling_fsr as v31

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810

KD_BASE_LOGIT = -8.80
COLLAPSE_SCALE = 2.0
COLLAPSE_CURVATURE = 16.0

DETAIL_PATH = Path("data/experimental/fatigue_checkpoint_curve16_200_detail.csv")
SUMMARY_PATH = Path("data/experimental/fatigue_checkpoint_curve16_200_summary.csv")

CHECKPOINT_ORDER = [
    "start_r1",
    "end_r1",
    "start_r2",
    "end_r2",
    "start_r3",
    "end_r3",
]


def _configure_locked_candidate() -> None:
    run87.KD_BASE_LOGIT = KD_BASE_LOGIT
    run87.COLLAPSE_CURVATURE = COLLAPSE_CURVATURE
    run87.COLLAPSE = collapse_mod.CollapseCandidate(
        f"scale{COLLAPSE_SCALE:.1f}_curve{COLLAPSE_CURVATURE:.1f}",
        COLLAPSE_SCALE,
        COLLAPSE_CURVATURE,
    )


def _fatigue_values(sim: run87.AuditSim, fighter: int, stamina_fraction: float) -> tuple[float, float, float]:
    """Recompute the existing V3.1 power-only fatigue mapping at a supplied stamina fraction."""
    base_power = float(sim.base_fighters[fighter]["striking_power"])
    resilience = float(sim.base_fighters[fighter][fsr32.STAMINA_PERFORMANCE_RESILIENCE])
    missing = float(np.clip(1.0 - stamina_fraction, 0.0, 1.0))
    resilience_multiplier = exp(-(resilience - 50.0) / v31.FATIGUE_RESILIENCE_SCALE)
    penalty = (
        (missing ** v31.FATIGUE_CURVE_EXPONENT)
        * v31.MAX_FATIGUE_RATING_PENALTY
        * resilience_multiplier
    )
    penalty = max(0.0, float(penalty))
    effective_power = max(v31.MIN_EFFECTIVE_FSR_RATING, base_power - penalty)
    return base_power, penalty, float(effective_power)


def _append_start_checkpoint(rows: list[dict], sim: run87.AuditSim, bout_id: str, path_idx: int, round_no: int) -> None:
    checkpoint = f"start_r{round_no}"
    events = [
        e for e in sim.effective_fsr_events
        if int(e["round"]) == round_no and int(e["segment"]) == 1
    ]
    for e in events:
        fighter = int(e["fighter"])
        rows.append({
            "bout_id": bout_id,
            "path": path_idx,
            "checkpoint": checkpoint,
            "fighter": fighter,
            "stamina_fraction": float(e["stamina_fraction"]),
            "fatigue_penalty": float(e["fatigue_penalty"]),
            "base_striking_power": float(sim.base_fighters[fighter]["striking_power"]),
            "effective_striking_power": float(e["effective_striking_power"]),
        })


def _append_end_checkpoint(rows: list[dict], sim: run87.AuditSim, path, bout_id: str, path_idx: int, round_no: int) -> None:
    checkpoint = f"end_r{round_no}"
    # The simulator uses 10-second segments, so a five-minute round ends at
    # segment 30. Use the engine constant rather than hard-coding a segment.
    matching = [
        e for e in path.events
        if int(e["round"]) == round_no
        and int(e["segment"]) == int(run87.base.SEGMENTS_PER_ROUND)
    ]
    if not matching:
        return
    e = matching[-1]
    for fighter, key in ((0, "red_stamina_after"), (1, "blue_stamina_after")):
        stamina_fraction = float(e[key])
        base_power, penalty, effective_power = _fatigue_values(sim, fighter, stamina_fraction)
        rows.append({
            "bout_id": bout_id,
            "path": path_idx,
            "checkpoint": checkpoint,
            "fighter": fighter,
            "stamina_fraction": stamina_fraction,
            "fatigue_penalty": penalty,
            "base_striking_power": base_power,
            "effective_striking_power": effective_power,
        })


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "stamina_fraction",
        "fatigue_penalty",
        "base_striking_power",
        "effective_striking_power",
    ]
    rows: list[dict] = []
    for checkpoint in CHECKPOINT_ORDER:
        g = detail.loc[detail["checkpoint"].eq(checkpoint)]
        if g.empty:
            continue
        row: dict[str, float | int | str] = {
            "checkpoint": checkpoint,
            "fighters": int(len(g)),
            "paths": int(g[["bout_id", "path"]].drop_duplicates().shape[0]),
        }
        for metric in metrics:
            values = g[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_p10"] = float(values.quantile(0.10))
            row[f"{metric}_p25"] = float(values.quantile(0.25))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_p75"] = float(values.quantile(0.75))
            row[f"{metric}_p90"] = float(values.quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    _configure_locked_candidate()

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(DEFAULT_BOUTS).reset_index(drop=True)
    total_paths = len(cohort) * DEFAULT_PATHS

    rng = np.random.default_rng(DEFAULT_SEED)
    seed_matrix = rng.integers(
        0,
        2**31 - 1,
        size=(len(cohort), DEFAULT_PATHS),
        dtype=np.int64,
    )

    rows: list[dict] = []
    finish_round_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    completed = 0

    print("\n" + "=" * 150)
    print("FATIGUE CHECKPOINT DIAGNOSTIC — LOCKED CURVE 16 — 200 BOUTS x 10 PATHS")
    print("=" * 150)
    print(f"segment seconds={run87.base.SEGMENT_SECONDS}; segments/round={run87.base.SEGMENTS_PER_ROUND}")
    print(f"KD base={run87.KD_BASE_LOGIT:.2f}; shock={run87.KD_SHOCK_COEFFICIENT:.0f}; depletion={run87.KD_DEPLETION_COEFFICIENT:.2f}")
    print(f"collapse scale={COLLAPSE_SCALE:.1f}; curvature={COLLAPSE_CURVATURE:.1f}")
    print(f"fatigue max penalty={v31.MAX_FATIGUE_RATING_PENALTY:.1f}; exponent={v31.FATIGUE_CURVE_EXPONENT:.2f}; resilience scale={v31.FATIGUE_RESILIENCE_SCALE:.1f}")
    print("global stamina recovery=40% of missing between rounds")
    print("diagnostics only: simulation behavior is unchanged")

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

            finish_round = 0
            if path.finish is not None:
                finish_round = int(getattr(path.finish, "round", 0) or 0)
            finish_round_counts[finish_round] = finish_round_counts.get(finish_round, 0) + 1

            for round_no in (1, 2, 3):
                _append_start_checkpoint(rows, sim, bout_id, path_idx, round_no)
                _append_end_checkpoint(rows, sim, path, bout_id, path_idx, round_no)

        completed += DEFAULT_PATHS
        if completed % 500 == 0 or bout_idx + 1 == len(cohort):
            print(f"paths {completed:,}/{total_paths:,}", flush=True)

    detail = pd.DataFrame(rows)
    summary = _summarize(detail)

    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nCHECKPOINT SUMMARY")
    display_cols = [
        "checkpoint",
        "paths",
        "stamina_fraction_mean",
        "stamina_fraction_p10",
        "stamina_fraction_p25",
        "stamina_fraction_median",
        "stamina_fraction_p75",
        "stamina_fraction_p90",
        "fatigue_penalty_mean",
        "fatigue_penalty_median",
        "fatigue_penalty_p90",
        "base_striking_power_mean",
        "effective_striking_power_mean",
        "effective_striking_power_p10",
        "effective_striking_power_median",
        "effective_striking_power_p90",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPATH REACH / FINISH COUNTS")
    print(f"total paths={total_paths}")
    print(f"R1 finishes={finish_round_counts.get(1, 0)}")
    print(f"R2 finishes={finish_round_counts.get(2, 0)}")
    print(f"R3 finishes={finish_round_counts.get(3, 0)}")
    print(f"no KO/TKO through R3={finish_round_counts.get(0, 0)}")

    print(f"\nSaved detail: {DETAIL_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print("Research only: no simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
