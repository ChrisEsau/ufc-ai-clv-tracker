"""Research-only V2 calibration for TD attempt frequency and TD success.

Refines V1 by:
- expanding TD-attempt baselines upward after the V1 optimum hit the grid edge;
- adding an absolute success-logit offset around the existing
  wrestling_conversion - opponent td_defense matchup;
- jointly scoring phase occupancy, TD attempts/round, and TD completion.

This remains a stationary research proxy. It does not modify the MC engine and
contains no dynamic-state logic.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from scripts.experimental import fsr_hierarchical_td_attempt_success_research_v1 as v1

# Narrow occupancy parameters around the good V1 region while expanding the
# attempt-frequency range that was still too low.
DISTANCE_CLINCH_BASES = (0.03, 0.04, 0.05, 0.06)
DISTANCE_TD_ATTEMPT_BASES = (0.08, 0.10, 0.12, 0.14, 0.16)
CLINCH_SEPARATE_BASES = (0.30, 0.40, 0.50, 0.60)
CLINCH_TD_ATTEMPT_BASES = (0.12, 0.16, 0.20, 0.24, 0.28)
GROUND_EXIT_BASES = (0.24, 0.28, 0.32, 0.36, 0.40)

# V1 centered a 50-vs-50 conversion matchup at 50% success, while the observed
# mean completion was ~38.5%. Negative logit offsets calibrate the absolute
# level without changing fighter ordering from the matchup edge.
TD_SUCCESS_LOGIT_OFFSETS = (-0.80, -0.60, -0.40, -0.20)


def _td_success(frame: pd.DataFrame, offset: float) -> np.ndarray:
    edge = (
        frame["wrestling_conversion"].to_numpy(dtype=float)
        - frame["opp_td_defense"].to_numpy(dtype=float)
    ) / v1.RATING_SCALE
    return v1._sigmoid(edge + float(offset))


def evaluate(
    frame: pd.DataFrame,
    params: tuple[float, float, float, float, float, float],
) -> dict[str, float]:
    dc_base, dta_base, cs_base, cta_base, ge_base, success_offset = params

    clinch_mod = v1._modifier(frame["clinch_pref"])
    wrestle_mod = v1._modifier(frame["wrestling_pref"])
    distance_exit_mod = v1._modifier(-frame["distance_pref"])
    td_success = _td_success(frame, success_offset)

    # Attempt hazards count every shot, successful or not.
    d_clinch_hazard = v1._clip_prob(
        dc_base * clinch_mod * np.sqrt(distance_exit_mod), 0.60
    )
    d_td_attempt_hazard = v1._clip_prob(dta_base * wrestle_mod, 0.70)
    c_td_attempt_hazard = v1._clip_prob(cta_base * wrestle_mod, 0.80)

    c_separate_hazard = v1._clip_prob(
        cs_base
        * v1._modifier(-frame["clinch_pref"])
        * np.exp(
            np.clip(-frame["control_matchup"].to_numpy(), -1.0, 1.0) * 0.15
        ),
        0.90,
    )

    ground_exit_mod = np.exp(
        np.clip(
            0.60 * frame["escape_matchup"].to_numpy()
            + 0.40 * frame["reversal_matchup"].to_numpy(),
            -1.5,
            1.5,
        )
    )
    ground_exit_hazard = v1._clip_prob(ge_base * ground_exit_mod, 0.90)

    # Only successful attempts change phase.
    d_to_g = d_td_attempt_hazard * td_success
    c_to_g = c_td_attempt_hazard * td_success
    d_to_c = d_clinch_hazard.copy()
    c_to_d = c_separate_hazard.copy()
    g_to_d = ground_exit_hazard

    d_sum = d_to_c + d_to_g
    over = d_sum > 0.95
    if np.any(over):
        scale = 0.95 / d_sum[over]
        d_to_c[over] *= scale
        d_to_g[over] *= scale

    c_sum = c_to_d + c_to_g
    over = c_sum > 0.95
    if np.any(over):
        scale = 0.95 / c_sum[over]
        c_to_d[over] *= scale
        c_to_g[over] *= scale

    occ_d, occ_c, occ_g = v1.stationary_distribution(
        d_to_c, d_to_g, c_to_d, c_to_g, g_to_d
    )

    td_attempt_rate = v1.SEGMENTS_PER_ROUND * (
        occ_d * d_td_attempt_hazard + occ_c * c_td_attempt_hazard
    )
    td_landed_rate = v1.SEGMENTS_PER_ROUND * (
        occ_d * d_to_g + occ_c * c_to_g
    )
    implied_completion = np.divide(
        td_landed_rate,
        td_attempt_rate,
        out=np.zeros_like(td_landed_rate),
        where=td_attempt_rate > 0,
    )

    obs_d = float(frame["rfs_phase_base_fight_distance_attempt_share"].mean())
    obs_c = float(frame["rfs_phase_base_fight_clinch_attempt_share"].mean())
    obs_g = float(frame["rfs_phase_base_fight_ground_attempt_share"].mean())
    obs_td = float(frame["rfs_phase_base_fight_td_attempts_per_round"].mean())
    completion_obs = pd.to_numeric(
        frame["rfs_phase_base_fight_td_completion_rate"], errors="coerce"
    ).dropna()
    obs_completion = float(completion_obs.mean())

    pred_d = float(occ_d.mean())
    pred_c = float(occ_c.mean())
    pred_g = float(occ_g.mean())
    pred_td = float(td_attempt_rate.mean())
    pred_completion = float(implied_completion[td_attempt_rate > 0].mean())

    errors = [
        ((pred_d - obs_d) / max(obs_d, 1e-6)) ** 2,
        ((pred_c - obs_c) / max(obs_c, 1e-6)) ** 2,
        ((pred_g - obs_g) / max(obs_g, 1e-6)) ** 2,
        ((pred_td - obs_td) / max(obs_td, 1e-6)) ** 2,
        ((pred_completion - obs_completion) / max(obs_completion, 1e-6)) ** 2,
    ]

    tmp = pd.DataFrame({
        "occ_d": occ_d,
        "occ_c": occ_c,
        "occ_g": occ_g,
        "td_rate": td_attempt_rate,
        "td_success": td_success,
        "obs_d": frame["rfs_phase_base_fight_distance_attempt_share"],
        "obs_c": frame["rfs_phase_base_fight_clinch_attempt_share"],
        "obs_g": frame["rfs_phase_base_fight_ground_attempt_share"],
        "obs_td": frame["rfs_phase_base_fight_td_attempts_per_round"],
    })

    return {
        "distance_clinch_base": dc_base,
        "distance_td_attempt_base": dta_base,
        "clinch_separate_base": cs_base,
        "clinch_td_attempt_base": cta_base,
        "ground_exit_base": ge_base,
        "td_success_logit_offset": success_offset,
        "mean_scale_error": float(np.mean(errors)),
        "mean_occ_distance": pred_d,
        "mean_occ_clinch": pred_c,
        "mean_occ_ground": pred_g,
        "mean_td_attempts_per_round": pred_td,
        "mean_implied_td_completion": pred_completion,
        "observed_mean_td_completion": obs_completion,
        "distance_spearman": float(tmp[["occ_d", "obs_d"]].corr(method="spearman").iloc[0, 1]),
        "clinch_spearman": float(tmp[["occ_c", "obs_c"]].corr(method="spearman").iloc[0, 1]),
        "ground_spearman": float(tmp[["occ_g", "obs_g"]].corr(method="spearman").iloc[0, 1]),
        "td_spearman": float(tmp[["td_rate", "obs_td"]].corr(method="spearman").iloc[0, 1]),
    }


def main() -> None:
    print(f"[TD split V2] loading FSR-26 from {v1.FSR_PATH}", flush=True)
    fsr = pd.read_parquet(v1.FSR_PATH)
    print(f"[TD split V2] loaded {len(fsr):,} FSR rows", flush=True)
    print(f"[TD split V2] loading RFS from {v1.RFS_PATH}", flush=True)
    rfs = pd.read_parquet(v1.RFS_PATH)
    print(f"[TD split V2] loaded {len(rfs):,} RFS rows", flush=True)
    frame = v1.build_research_frame(fsr, rfs)
    print(f"[TD split V2] research grain: {len(frame):,} rows", flush=True)

    grid = list(product(
        DISTANCE_CLINCH_BASES,
        DISTANCE_TD_ATTEMPT_BASES,
        CLINCH_SEPARATE_BASES,
        CLINCH_TD_ATTEMPT_BASES,
        GROUND_EXIT_BASES,
        TD_SUCCESS_LOGIT_OFFSETS,
    ))
    print(f"[TD split V2] sweeping {len(grid):,} candidates", flush=True)

    rows = []
    for i, params in enumerate(grid, 1):
        rows.append(evaluate(frame, params))
        if i == 1 or i % 1000 == 0 or i == len(grid):
            print(f"[TD split V2] candidate {i:,}/{len(grid):,}", flush=True)

    out = pd.DataFrame(rows).sort_values("mean_scale_error").reset_index(drop=True)
    print("\nTOP 15 REFINED TD CALIBRATION CANDIDATES")
    print(out.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOBSERVED POPULATION TARGETS")
    print(f"distance attempt share: {frame['rfs_phase_base_fight_distance_attempt_share'].mean():.4f}")
    print(f"clinch attempt share:   {frame['rfs_phase_base_fight_clinch_attempt_share'].mean():.4f}")
    print(f"ground attempt share:   {frame['rfs_phase_base_fight_ground_attempt_share'].mean():.4f}")
    print(f"TD attempts/round:      {frame['rfs_phase_base_fight_td_attempts_per_round'].mean():.4f}")
    print(f"mean TD completion:     {pd.to_numeric(frame['rfs_phase_base_fight_td_completion_rate'], errors='coerce').mean():.4f}")

    best = out.iloc[0]
    print("\nBEST REFINED TD CALIBRATION CANDIDATE")
    for c in [
        "distance_clinch_base",
        "distance_td_attempt_base",
        "clinch_separate_base",
        "clinch_td_attempt_base",
        "ground_exit_base",
        "td_success_logit_offset",
        "mean_occ_distance",
        "mean_occ_clinch",
        "mean_occ_ground",
        "mean_td_attempts_per_round",
        "mean_implied_td_completion",
        "observed_mean_td_completion",
        "mean_scale_error",
    ]:
        print(f"{c}: {best[c]:.4f}")

    print(
        "\nIMPORTANT: V2 is still a stationary static calibration proxy. "
        "If it jointly matches occupancy, attempt volume, and completion, these "
        "values are suitable starting priors for a rudimentary static-state MC."
    )


if __name__ == "__main__":
    main()
