"""Research-only hierarchical phase hazard calibration with TD attempt/success split.

Corrects the key limitation of the first hierarchical proxy: takedown attempts
are repeatable actions, while only successful takedowns transition to ground.

DISTANCE:
- clinch-entry hazard
- takedown-attempt hazard
- successful TD attempts transition to ground
- failed TD attempts remain at distance in this stationary proxy

CLINCH:
- separation hazard
- takedown-attempt hazard
- successful TD attempts transition to ground
- failed TD attempts remain in clinch in this stationary proxy

GROUND:
- exit hazard
- remainder stays ground

Takedown attempt frequency is driven primarily by wrestling_entry/style.
Takedown success is driven by wrestling_conversion vs opponent td_defense.
This remains research-only and does not modify the MC engine.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")

# Start around the occupancy solution from the first hierarchical experiment,
# but expand TD attempt baselines upward now that failed attempts no longer force
# a transition to ground.
DISTANCE_CLINCH_BASES = (0.03, 0.04, 0.05, 0.06)
DISTANCE_TD_ATTEMPT_BASES = (0.04, 0.06, 0.08, 0.10, 0.12)
CLINCH_SEPARATE_BASES = (0.20, 0.30, 0.40, 0.50)
CLINCH_TD_ATTEMPT_BASES = (0.08, 0.12, 0.16, 0.20, 0.24)
GROUND_EXIT_BASES = (0.16, 0.20, 0.24, 0.28, 0.32)

MODIFIER_SCALE = 6.0
RATING_SCALE = 12.0
SEGMENTS_PER_ROUND = 10.0


def _clip_prob(x: np.ndarray | float, high: float = 0.95):
    return np.clip(x, 0.0, high)


def _modifier(pref: pd.Series) -> np.ndarray:
    return np.exp(np.clip(pref.to_numpy(dtype=float), -8.0, 8.0) / MODIFIER_SCALE)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-x))


def _opponent(frame: pd.DataFrame, column: str) -> pd.Series:
    opp = frame[["fight_id", "fighter_id", column]].rename(
        columns={"fighter_id": "opponent_id", column: f"opp_{column}"}
    )
    pairs = frame[["fight_id", "fighter_id"]].merge(opp, on="fight_id")
    pairs = pairs[pairs["fighter_id"] != pairs["opponent_id"]]
    return pairs.set_index(["fight_id", "fighter_id"])[f"opp_{column}"]


def build_research_frame(fsr: pd.DataFrame, rfs: pd.DataFrame) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]
    fsr = fsr.copy()
    rfs = rfs.copy()
    for df in (fsr, rfs):
        df["fight_id"] = df["fight_id"].astype(str)
        df["fighter_id"] = df["fighter_id"].astype(str)

    needed = [
        *keys,
        "distance_striking_pressure",
        "clinch_striking_pressure",
        "wrestling_entry",
        "wrestling_conversion",
        "td_defense",
        "control_imposition",
        "control_resistance",
        "reversal_ability",
    ]
    f = fsr[needed].copy()

    f["distance_pref"] = (
        f["distance_striking_pressure"]
        - 0.5 * f["clinch_striking_pressure"]
        - 0.5 * f["wrestling_entry"]
    )
    f["clinch_pref"] = (
        f["clinch_striking_pressure"]
        - 0.5 * f["distance_striking_pressure"]
        - 0.5 * f["wrestling_entry"]
    )
    f["wrestling_pref"] = (
        0.75 * f["wrestling_entry"]
        + 0.25 * f["control_imposition"]
        - 0.5 * f["distance_striking_pressure"]
        - 0.5 * f["clinch_striking_pressure"]
    )
    for c in ("distance_pref", "clinch_pref", "wrestling_pref"):
        f[c] = f[c] - f[c].median()

    opp_td = _opponent(f, "td_defense")
    opp_ctrl = _opponent(f, "control_resistance")
    opp_rev = _opponent(f, "reversal_ability")
    idx = pd.MultiIndex.from_frame(f[keys])
    f["opp_td_defense"] = opp_td.reindex(idx).to_numpy()
    f["opp_control_resistance"] = opp_ctrl.reindex(idx).to_numpy()
    f["opp_reversal_ability"] = opp_rev.reindex(idx).to_numpy()

    f["control_matchup"] = (
        f["control_imposition"] - f["opp_control_resistance"]
    ) / RATING_SCALE
    f["escape_matchup"] = (
        f["control_resistance"] - f["control_imposition"]
    ) / RATING_SCALE
    f["reversal_matchup"] = (
        f["reversal_ability"] - f["control_imposition"]
    ) / RATING_SCALE

    # Execution probability is separate from attempt desire. A 50-vs-50 matchup
    # centers at 0.50 in this research proxy; absolute success calibration can be
    # refined later against observed completion rate.
    f["td_success_prob"] = _sigmoid(
        (
            f["wrestling_conversion"].to_numpy(dtype=float)
            - f["opp_td_defense"].to_numpy(dtype=float)
        ) / RATING_SCALE
    )

    outcome_cols = [
        *keys,
        "rfs_phase_base_fight_distance_attempt_share",
        "rfs_phase_base_fight_clinch_attempt_share",
        "rfs_phase_base_fight_ground_attempt_share",
        "rfs_phase_base_fight_td_attempts_per_round",
        "rfs_phase_base_fight_td_completion_rate",
    ]
    return f.merge(rfs[outcome_cols], on=keys, validate="one_to_one")


def stationary_distribution(
    d_to_c: np.ndarray,
    d_to_g: np.ndarray,
    c_to_d: np.ndarray,
    c_to_g: np.ndarray,
    g_to_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(d_to_c)
    d = np.full(n, 0.80)
    c = np.full(n, 0.10)
    g = np.full(n, 0.10)

    d_stay = 1.0 - d_to_c - d_to_g
    c_stay = 1.0 - c_to_d - c_to_g
    g_stay = 1.0 - g_to_d

    for _ in range(200):
        nd = d * d_stay + c * c_to_d + g * g_to_d
        nc = d * d_to_c + c * c_stay
        ng = d * d_to_g + c * c_to_g + g * g_stay
        total = nd + nc + ng
        nd, nc, ng = nd / total, nc / total, ng / total
        if np.max(np.abs(nd - d)) < 1e-10:
            d, c, g = nd, nc, ng
            break
        d, c, g = nd, nc, ng
    return d, c, g


def evaluate(frame: pd.DataFrame, params: tuple[float, float, float, float, float]) -> dict[str, float]:
    dc_base, dta_base, cs_base, cta_base, ge_base = params

    clinch_mod = _modifier(frame["clinch_pref"])
    wrestle_mod = _modifier(frame["wrestling_pref"])
    distance_exit_mod = _modifier(-frame["distance_pref"])
    td_success = frame["td_success_prob"].to_numpy(dtype=float)

    # Action hazards: these count attempts whether they succeed or fail.
    d_clinch_hazard = _clip_prob(dc_base * clinch_mod * np.sqrt(distance_exit_mod), 0.60)
    d_td_attempt_hazard = _clip_prob(dta_base * wrestle_mod, 0.60)
    c_td_attempt_hazard = _clip_prob(cta_base * wrestle_mod, 0.70)

    c_separate_hazard = _clip_prob(
        cs_base
        * _modifier(-frame["clinch_pref"])
        * np.exp(np.clip(-frame["control_matchup"].to_numpy(), -1.0, 1.0) * 0.15),
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
    ground_exit_hazard = _clip_prob(ge_base * ground_exit_mod, 0.90)

    # State-transition hazards use only successful TD attempts.
    d_to_g = d_td_attempt_hazard * td_success
    c_to_g = c_td_attempt_hazard * td_success
    d_to_c = d_clinch_hazard
    c_to_d = c_separate_hazard
    g_to_d = ground_exit_hazard

    # Keep competing successful transition hazards valid. Failed TD attempts are
    # self-transitions and therefore do not enter these sums.
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

    occ_d, occ_c, occ_g = stationary_distribution(
        d_to_c, d_to_g, c_to_d, c_to_g, g_to_d
    )

    # All attempts count, including failures.
    td_attempt_rate = SEGMENTS_PER_ROUND * (
        occ_d * d_td_attempt_hazard + occ_c * c_td_attempt_hazard
    )
    td_landed_rate = SEGMENTS_PER_ROUND * (
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
    )
    completion_obs = completion_obs[completion_obs.notna()]
    obs_completion = float(completion_obs.mean()) if len(completion_obs) else np.nan

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
    ]
    # Completion is diagnostic only in V1 because the current FSR conversion
    # matchup has not yet been absolute-probability calibrated.
    err = float(np.mean(errors))

    tmp = pd.DataFrame({
        "occ_d": occ_d,
        "occ_c": occ_c,
        "occ_g": occ_g,
        "td_rate": td_attempt_rate,
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
        "mean_scale_error": err,
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
    print(f"[TD split research] loading FSR-26 from {FSR_PATH}", flush=True)
    fsr = pd.read_parquet(FSR_PATH)
    print(f"[TD split research] loaded {len(fsr):,} FSR rows", flush=True)
    print(f"[TD split research] loading RFS from {RFS_PATH}", flush=True)
    rfs = pd.read_parquet(RFS_PATH)
    print(f"[TD split research] loaded {len(rfs):,} RFS rows", flush=True)
    frame = build_research_frame(fsr, rfs)
    print(f"[TD split research] research grain: {len(frame):,} rows", flush=True)

    grid = list(product(
        DISTANCE_CLINCH_BASES,
        DISTANCE_TD_ATTEMPT_BASES,
        CLINCH_SEPARATE_BASES,
        CLINCH_TD_ATTEMPT_BASES,
        GROUND_EXIT_BASES,
    ))
    print(f"[TD split research] sweeping {len(grid):,} candidates", flush=True)

    rows = []
    for i, params in enumerate(grid, 1):
        rows.append(evaluate(frame, params))
        if i == 1 or i % 500 == 0 or i == len(grid):
            print(f"[TD split research] candidate {i:,}/{len(grid):,}", flush=True)

    out = pd.DataFrame(rows).sort_values("mean_scale_error").reset_index(drop=True)

    print("\nTOP 15 TD-ATTEMPT/SUCCESS HAZARD CANDIDATES")
    print(out.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOBSERVED POPULATION TARGETS")
    print(f"distance attempt share: {frame['rfs_phase_base_fight_distance_attempt_share'].mean():.4f}")
    print(f"clinch attempt share:   {frame['rfs_phase_base_fight_clinch_attempt_share'].mean():.4f}")
    print(f"ground attempt share:   {frame['rfs_phase_base_fight_ground_attempt_share'].mean():.4f}")
    print(f"TD attempts/round:      {frame['rfs_phase_base_fight_td_attempts_per_round'].mean():.4f}")
    completion = pd.to_numeric(frame['rfs_phase_base_fight_td_completion_rate'], errors='coerce')
    print(f"mean TD completion:     {completion.mean():.4f}")

    best = out.iloc[0]
    print("\nBEST TD-ATTEMPT/SUCCESS RESEARCH CANDIDATE")
    for c in [
        "distance_clinch_base",
        "distance_td_attempt_base",
        "clinch_separate_base",
        "clinch_td_attempt_base",
        "ground_exit_base",
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
        "\nIMPORTANT: attempts and successful ground transitions are now separate. "
        "The selected baselines remain research priors for a future path simulator, "
        "not production constants. TD completion is diagnostic in this V1 sweep."
    )


if __name__ == "__main__":
    main()
