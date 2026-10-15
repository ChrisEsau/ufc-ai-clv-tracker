"""Research-only hierarchical transition hazard calibration for FSR-26.

This replaces the symmetric phase-choice proxy with a state-conditional hazard
model:

DISTANCE:
- clinch entry hazard
- takedown attempt hazard
- remainder stays at distance

CLINCH:
- separation hazard
- takedown attempt hazard
- remainder stays in clinch

GROUND:
- escape hazard
- reversal hazard
- remainder stays on ground

The script uses pre-fight FSR-26 snapshots as fighter-specific modifiers and
calibrates only population baseline hazards against historical aggregate phase
shares and takedown volume. It is research-only and does not modify the MC
engine.
"""

from __future__ import annotations

from itertools import product
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")

# Coarse population baseline hazards per 30-second segment. These are research
# candidates only; fighter-specific FSR modifiers are applied around them.
DISTANCE_CLINCH_BASES = (0.01, 0.02, 0.03, 0.04, 0.05)
DISTANCE_TD_BASES = (0.02, 0.04, 0.06, 0.08, 0.10)
CLINCH_SEPARATE_BASES = (0.20, 0.30, 0.40, 0.50, 0.60)
CLINCH_TD_BASES = (0.04, 0.08, 0.12, 0.16)
GROUND_EXIT_BASES = (0.08, 0.12, 0.16, 0.20, 0.24)

# Conservative multiplicative sensitivity from FSR relative preference units.
# exp(pref / MODIFIER_SCALE) keeps modifiers smooth and positive.
MODIFIER_SCALE = 6.0
SEGMENTS_PER_ROUND = 10.0


def _clip_prob(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 0.95)


def _modifier(pref: pd.Series) -> np.ndarray:
    return np.exp(np.clip(pref.to_numpy(dtype=float), -8.0, 8.0) / MODIFIER_SCALE)


def _opponent(frame: pd.DataFrame, column: str) -> pd.Series:
    opp = frame[["fight_id", "fighter_id", column]].rename(
        columns={"fighter_id": "opponent_id", column: f"opp_{column}"}
    )
    pairs = frame[["fight_id", "fighter_id"]].merge(opp, on="fight_id")
    pairs = pairs[pairs["fighter_id"] != pairs["opponent_id"]]
    return pairs.set_index(["fight_id", "fighter_id"])[f"opp_{column}"]


def build_research_frame(fsr: pd.DataFrame, rfs: pd.DataFrame) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]
    for df in (fsr, rfs):
        df["fight_id"] = df["fight_id"].astype(str)
        df["fighter_id"] = df["fighter_id"].astype(str)

    needed = [
        *keys,
        "distance_striking_pressure",
        "clinch_striking_pressure",
        "wrestling_entry",
        "control_imposition",
        "control_resistance",
        "td_defense",
        "reversal_ability",
    ]
    f = fsr[needed].copy()

    # Relative fighter-style preferences. These are not probabilities.
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

    # Center style utilities to keep multiplier scales interpretable.
    for c in ("distance_pref", "clinch_pref", "wrestling_pref"):
        f[c] = f[c] - f[c].median()

    opp_td = _opponent(f, "td_defense")
    opp_ctrl = _opponent(f, "control_resistance")
    opp_rev = _opponent(f, "reversal_ability")
    idx = pd.MultiIndex.from_frame(f[keys])
    f["opp_td_defense"] = opp_td.reindex(idx).to_numpy()
    f["opp_control_resistance"] = opp_ctrl.reindex(idx).to_numpy()
    f["opp_reversal_ability"] = opp_rev.reindex(idx).to_numpy()

    # Matchup modifiers used only where conceptually appropriate.
    f["td_matchup"] = (f["wrestling_entry"] - f["opp_td_defense"]) / 12.0
    f["control_matchup"] = (f["control_imposition"] - f["opp_control_resistance"]) / 12.0
    f["escape_matchup"] = (f["control_resistance"] - f["control_imposition"]) / 12.0
    f["reversal_matchup"] = (f["reversal_ability"] - f["control_imposition"]) / 12.0

    outcome_cols = [
        *keys,
        "rfs_phase_base_fight_distance_attempt_share",
        "rfs_phase_base_fight_clinch_attempt_share",
        "rfs_phase_base_fight_ground_attempt_share",
        "rfs_phase_base_fight_td_attempts_per_round",
    ]
    return f.merge(rfs[outcome_cols], on=keys, validate="one_to_one")


def stationary_distribution(
    d_to_c: np.ndarray,
    d_to_g: np.ndarray,
    c_to_d: np.ndarray,
    c_to_g: np.ndarray,
    g_to_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve per-row 3-state stationary occupancy by fixed-point iteration."""
    n = len(d_to_c)
    d = np.full(n, 0.80)
    c = np.full(n, 0.10)
    g = np.full(n, 0.10)

    d_stay = 1.0 - d_to_c - d_to_g
    c_stay = 1.0 - c_to_d - c_to_g
    g_stay = 1.0 - g_to_d

    for _ in range(150):
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
    dc_base, dt_base, cs_base, ct_base, ge_base = params

    clinch_mod = _modifier(frame["clinch_pref"])
    wrestle_mod = _modifier(frame["wrestling_pref"])
    distance_mod = _modifier(-frame["distance_pref"])

    # Entering clinch is primarily intrinsic style. Takedown initiation is
    # mostly wrestling tendency with only a mild matchup component.
    d_to_c = _clip_prob(dc_base * clinch_mod * np.sqrt(distance_mod))
    d_to_g = _clip_prob(dt_base * wrestle_mod * np.exp(np.clip(frame["td_matchup"].to_numpy(), -1.0, 1.0) * 0.10))

    # Clinch exits: separation is higher for low clinch preference / poor control
    # matchup; TD hazard uses the same wrestling tendency.
    c_to_d = _clip_prob(
        cs_base
        * _modifier(-frame["clinch_pref"])
        * np.exp(np.clip(-frame["control_matchup"].to_numpy(), -1.0, 1.0) * 0.15)
    )
    c_to_g = _clip_prob(ct_base * wrestle_mod)

    # Ground exit proxy combines escape and reversal pressure into one exit
    # hazard for occupancy calibration. The actual MC will separate those events.
    ground_exit_mod = np.exp(
        np.clip(
            0.60 * frame["escape_matchup"].to_numpy()
            + 0.40 * frame["reversal_matchup"].to_numpy(),
            -1.5,
            1.5,
        )
    )
    g_to_d = _clip_prob(ge_base * ground_exit_mod)

    # Competing hazards cannot exceed 0.95 in a segment.
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

    occ_d, occ_c, occ_g = stationary_distribution(d_to_c, d_to_g, c_to_d, c_to_g, g_to_d)

    # Expected TD attempts per round: attempts can originate from distance or
    # clinch, weighted by stationary occupancy.
    td_rate = SEGMENTS_PER_ROUND * (occ_d * d_to_g + occ_c * c_to_g)

    obs_d = float(frame["rfs_phase_base_fight_distance_attempt_share"].mean())
    obs_c = float(frame["rfs_phase_base_fight_clinch_attempt_share"].mean())
    obs_g = float(frame["rfs_phase_base_fight_ground_attempt_share"].mean())
    obs_td = float(frame["rfs_phase_base_fight_td_attempts_per_round"].mean())

    pred_d, pred_c, pred_g, pred_td = map(float, (occ_d.mean(), occ_c.mean(), occ_g.mean(), td_rate.mean()))

    # Relative squared error prevents TD units from dominating phase shares.
    err = np.mean([
        ((pred_d - obs_d) / max(obs_d, 1e-6)) ** 2,
        ((pred_c - obs_c) / max(obs_c, 1e-6)) ** 2,
        ((pred_g - obs_g) / max(obs_g, 1e-6)) ** 2,
        ((pred_td - obs_td) / max(obs_td, 1e-6)) ** 2,
    ])

    tmp = pd.DataFrame({
        "occ_d": occ_d,
        "occ_c": occ_c,
        "occ_g": occ_g,
        "td_rate": td_rate,
        "obs_d": frame["rfs_phase_base_fight_distance_attempt_share"],
        "obs_c": frame["rfs_phase_base_fight_clinch_attempt_share"],
        "obs_g": frame["rfs_phase_base_fight_ground_attempt_share"],
        "obs_td": frame["rfs_phase_base_fight_td_attempts_per_round"],
    })

    return {
        "distance_clinch_base": dc_base,
        "distance_td_base": dt_base,
        "clinch_separate_base": cs_base,
        "clinch_td_base": ct_base,
        "ground_exit_base": ge_base,
        "mean_scale_error": float(err),
        "mean_occ_distance": pred_d,
        "mean_occ_clinch": pred_c,
        "mean_occ_ground": pred_g,
        "mean_td_attempts_per_round": pred_td,
        "distance_spearman": float(tmp[["occ_d", "obs_d"]].corr(method="spearman").iloc[0, 1]),
        "clinch_spearman": float(tmp[["occ_c", "obs_c"]].corr(method="spearman").iloc[0, 1]),
        "ground_spearman": float(tmp[["occ_g", "obs_g"]].corr(method="spearman").iloc[0, 1]),
        "td_spearman": float(tmp[["td_rate", "obs_td"]].corr(method="spearman").iloc[0, 1]),
    }


def main() -> None:
    print(f"[hierarchical hazard] loading FSR-26 from {FSR_PATH}", flush=True)
    fsr = pd.read_parquet(FSR_PATH)
    print(f"[hierarchical hazard] loaded {len(fsr):,} FSR rows", flush=True)
    print(f"[hierarchical hazard] loading RFS from {RFS_PATH}", flush=True)
    rfs = pd.read_parquet(RFS_PATH)
    print(f"[hierarchical hazard] loaded {len(rfs):,} RFS rows", flush=True)
    frame = build_research_frame(fsr, rfs)
    print(f"[hierarchical hazard] research grain: {len(frame):,} rows", flush=True)

    grid = list(product(
        DISTANCE_CLINCH_BASES,
        DISTANCE_TD_BASES,
        CLINCH_SEPARATE_BASES,
        CLINCH_TD_BASES,
        GROUND_EXIT_BASES,
    ))
    print(f"[hierarchical hazard] sweeping {len(grid):,} candidates", flush=True)

    rows = []
    for i, params in enumerate(grid, 1):
        rows.append(evaluate(frame, params))
        if i == 1 or i % 250 == 0 or i == len(grid):
            print(f"[hierarchical hazard] candidate {i:,}/{len(grid):,}", flush=True)

    out = pd.DataFrame(rows).sort_values("mean_scale_error").reset_index(drop=True)

    print("\nTOP 15 HIERARCHICAL HAZARD CANDIDATES")
    print(out.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOBSERVED POPULATION TARGETS")
    print(f"distance attempt share: {frame['rfs_phase_base_fight_distance_attempt_share'].mean():.4f}")
    print(f"clinch attempt share:   {frame['rfs_phase_base_fight_clinch_attempt_share'].mean():.4f}")
    print(f"ground attempt share:   {frame['rfs_phase_base_fight_ground_attempt_share'].mean():.4f}")
    print(f"TD attempts/round:      {frame['rfs_phase_base_fight_td_attempts_per_round'].mean():.4f}")

    best = out.iloc[0]
    print("\nBEST HIERARCHICAL RESEARCH CANDIDATE")
    for c in [
        "distance_clinch_base",
        "distance_td_base",
        "clinch_separate_base",
        "clinch_td_base",
        "ground_exit_base",
        "mean_occ_distance",
        "mean_occ_clinch",
        "mean_occ_ground",
        "mean_td_attempts_per_round",
        "mean_scale_error",
    ]:
        print(f"{c}: {best[c]:.4f}")

    print(
        "\nIMPORTANT: this remains a stationary static calibration proxy. "
        "If the hierarchy can reproduce the population targets while preserving "
        "fighter rank signal, the selected hazards become starting priors for an "
        "actual segment-path simulator, not final production constants."
    )


if __name__ == "__main__":
    main()
