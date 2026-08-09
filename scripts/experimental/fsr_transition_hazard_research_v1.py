"""Research absolute 30-second transition hazards from FSR-26 phase utilities.

This script takes the previously selected static softmax temperature (4.0) and
researches how relative action-choice probabilities can be converted into
absolute per-segment hazards without changing the simulator engine.

Important distinction:
- softmax probabilities rank competing intentions;
- hazard scales determine how often the current phase actually changes.

The calibration targets are aggregate UFCStats/RFS observables only because
exact phase-entry/exit events and phase durations are not recorded directly.
Shadow/research only.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


FSR26_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
TEMPERATURE = 4.0
SEGMENT_SECONDS = 30.0

# Coarse grid only. These are research scales, not simulator contracts.
DISTANCE_EXIT_SCALES = (0.08, 0.12, 0.16, 0.20, 0.24)
CLINCH_EXIT_SCALES = (0.20, 0.30, 0.40, 0.50, 0.60)
WRESTLING_ATTEMPT_SCALES = (0.08, 0.12, 0.16, 0.20, 0.24)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    z = values / float(temperature)
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def build_opponents(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["fight_id"]
    cols = [
        "fight_id",
        "fighter_id",
        "td_defense",
        "control_resistance",
    ]
    opp = frame[cols].rename(
        columns={
            "fighter_id": "opponent_id",
            "td_defense": "opp_td_defense",
            "control_resistance": "opp_control_resistance",
        }
    )
    out = frame.merge(opp, on=keys, how="inner")
    return out[out["fighter_id"] != out["opponent_id"]].copy()


def construct_utilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the static research utilities supported by prior experiments."""
    out = frame.copy()

    d = out["distance_striking_pressure"] - 50.0
    c = out["clinch_striking_pressure"] - 50.0
    w = out["wrestling_entry"] - 50.0
    ctrl = out["control_imposition"] - 50.0
    opp_td = out["opp_td_defense"] - 50.0
    opp_ctrl_res = out["opp_control_resistance"] - 50.0

    # Distance: modest matchup adjustment improved realized distance share.
    out["u_distance"] = d - 0.50 * c - 0.50 * w + 0.10 * opp_td + 0.10 * opp_ctrl_res

    # Clinch: intrinsic clinch tendency performed best; do not over-adjust.
    out["u_clinch"] = c - 0.50 * d - 0.50 * w

    # Wrestling choice: control blend performed best for TD attempt tendency.
    out["u_wrestling"] = w + 0.25 * ctrl - 0.50 * d - 0.50 * c

    utilities = out[["u_distance", "u_clinch", "u_wrestling"]].to_numpy(float)
    probs = softmax(utilities, TEMPERATURE)
    out["p_distance_choice"] = probs[:, 0]
    out["p_clinch_choice"] = probs[:, 1]
    out["p_wrestling_choice"] = probs[:, 2]
    return out


def load_research_frame() -> pd.DataFrame:
    print(f"[hazard research] loading FSR-26 from {FSR26_PATH}", flush=True)
    fsr = pd.read_parquet(FSR26_PATH)
    print(f"[hazard research] loaded {len(fsr):,} FSR-26 rows", flush=True)

    print(f"[hazard research] loading RFS history from {RFS_PATH}", flush=True)
    rfs = pd.read_parquet(RFS_PATH)
    print(f"[hazard research] loaded {len(rfs):,} RFS rows", flush=True)

    for df in (fsr, rfs):
        df["fight_id"] = df["fight_id"].astype(str)
        df["fighter_id"] = df["fighter_id"].astype(str)

    needed_rfs = [
        "fight_id",
        "fighter_id",
        "rfs_phase_base_fight_distance_attempt_share",
        "rfs_phase_base_fight_clinch_attempt_share",
        "rfs_phase_base_fight_ground_attempt_share",
        "rfs_phase_base_fight_td_attempts_per_round",
        "rfs_phase_base_fight_control_seconds_per_round",
    ]
    research = fsr.merge(rfs[needed_rfs], on=["fight_id", "fighter_id"], validate="one_to_one")
    research = build_opponents(research)
    research = construct_utilities(research)

    for c in needed_rfs[2:]:
        research[c] = pd.to_numeric(research[c], errors="coerce")
    return research


def implied_metrics(
    frame: pd.DataFrame,
    distance_exit_scale: float,
    clinch_exit_scale: float,
    wrestling_attempt_scale: float,
) -> dict[str, float]:
    """Compute coarse static implied metrics, not a full fight simulation.

    This deliberately avoids pretending that exact historical transitions are
    observable. It uses hazard-scaled relative intentions to generate expected
    segment-level action rates, then compares rank/order and aggregate scale to
    observed phase proxies.
    """
    p_d = frame["p_distance_choice"].to_numpy(float)
    p_c = frame["p_clinch_choice"].to_numpy(float)
    p_w = frame["p_wrestling_choice"].to_numpy(float)

    # At distance, non-distance intent is gated by an absolute exit hazard.
    distance_exit = np.clip(distance_exit_scale * (p_c + p_w), 0.0, 0.95)
    distance_stay = 1.0 - distance_exit

    # Conditional split for exits from distance.
    denom_cw = np.maximum(p_c + p_w, 1e-12)
    d_to_c = distance_exit * p_c / denom_cw
    d_to_w = distance_exit * p_w / denom_cw

    # Clinch persistence gets its own absolute exit scale. Wrestling remains one
    # possible exit; separation-to-distance is the other.
    clinch_exit = np.clip(clinch_exit_scale * (p_d + p_w), 0.0, 0.95)
    clinch_stay = 1.0 - clinch_exit
    denom_dw = np.maximum(p_d + p_w, 1e-12)
    c_to_d = clinch_exit * p_d / denom_dw
    c_to_w = clinch_exit * p_w / denom_dw

    # Wrestling attempt hazard is independently scaled because desire to wrestle
    # is not the same quantity as attempts per 30-second segment.
    wrestle_hazard = np.clip(wrestling_attempt_scale * p_w, 0.0, 0.95)

    # Coarse occupancy scores. Ground receives weight from wrestling entries plus
    # persistence implied by control imposition. Normalize to form a stationary-
    # like proxy rather than claiming an exact Markov stationary distribution.
    ctrl = np.clip((frame["control_imposition"].to_numpy(float) - 40.0) / 20.0, 0.0, 1.0)
    ground_persist = 0.55 + 0.30 * ctrl

    score_distance = np.maximum(distance_stay + c_to_d, 1e-9)
    score_clinch = np.maximum(d_to_c + clinch_stay, 1e-9)
    score_ground = np.maximum((d_to_w + c_to_w) * (1.0 + ground_persist), 1e-9)
    total = score_distance + score_clinch + score_ground

    occ_d = score_distance / total
    occ_c = score_clinch / total
    occ_g = score_ground / total

    # Approximate attempts/round: ten 30-second segments per round.
    td_attempts_per_round = 10.0 * wrestle_hazard

    observed_d = frame["rfs_phase_base_fight_distance_attempt_share"]
    observed_c = frame["rfs_phase_base_fight_clinch_attempt_share"]
    observed_g = frame["rfs_phase_base_fight_ground_attempt_share"]
    observed_td = frame["rfs_phase_base_fight_td_attempts_per_round"]

    def spear(x: np.ndarray, y: pd.Series) -> float:
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(tmp) < 3:
            return float("nan")
        return float(tmp.corr(method="spearman").iloc[0, 1])

    # Scale error compares population means only; rank correlation evaluates
    # whether fighter ordering remains useful.
    result = {
        "distance_exit_scale": distance_exit_scale,
        "clinch_exit_scale": clinch_exit_scale,
        "wrestling_attempt_scale": wrestling_attempt_scale,
        "mean_occ_distance": float(np.mean(occ_d)),
        "mean_occ_clinch": float(np.mean(occ_c)),
        "mean_occ_ground": float(np.mean(occ_g)),
        "mean_td_attempts_per_round": float(np.mean(td_attempts_per_round)),
        "distance_spearman": spear(occ_d, observed_d),
        "clinch_spearman": spear(occ_c, observed_c),
        "ground_spearman": spear(occ_g, observed_g),
        "td_spearman": spear(td_attempts_per_round, observed_td),
        "observed_mean_distance_share": float(observed_d.mean()),
        "observed_mean_clinch_share": float(observed_c.mean()),
        "observed_mean_ground_share": float(observed_g.mean()),
        "observed_mean_td_attempts_per_round": float(observed_td.mean()),
    }

    result["mean_scale_error"] = float(
        abs(result["mean_occ_distance"] - result["observed_mean_distance_share"])
        + abs(result["mean_occ_clinch"] - result["observed_mean_clinch_share"])
        + abs(result["mean_occ_ground"] - result["observed_mean_ground_share"])
        + 0.10
        * abs(
            result["mean_td_attempts_per_round"]
            - result["observed_mean_td_attempts_per_round"]
        )
    )
    return result


def main() -> None:
    frame = load_research_frame()
    print(f"[hazard research] research grain: {len(frame):,} fighter-fight rows", flush=True)
    print("[hazard research] sweeping coarse hazard grid", flush=True)

    rows: list[dict[str, float]] = []
    total = len(DISTANCE_EXIT_SCALES) * len(CLINCH_EXIT_SCALES) * len(WRESTLING_ATTEMPT_SCALES)
    n = 0
    for d_scale in DISTANCE_EXIT_SCALES:
        for c_scale in CLINCH_EXIT_SCALES:
            for w_scale in WRESTLING_ATTEMPT_SCALES:
                rows.append(implied_metrics(frame, d_scale, c_scale, w_scale))
                n += 1
                if n == 1 or n % 25 == 0 or n == total:
                    print(f"[hazard research] grid {n}/{total}", flush=True)

    out = pd.DataFrame(rows).sort_values(
        ["mean_scale_error", "td_spearman", "distance_spearman"],
        ascending=[True, False, False],
    )

    cols = [
        "distance_exit_scale",
        "clinch_exit_scale",
        "wrestling_attempt_scale",
        "mean_scale_error",
        "mean_occ_distance",
        "mean_occ_clinch",
        "mean_occ_ground",
        "mean_td_attempts_per_round",
        "distance_spearman",
        "clinch_spearman",
        "ground_spearman",
        "td_spearman",
    ]

    print("\nTOP 15 COARSE HAZARD CANDIDATES")
    print(out[cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = out.iloc[0]
    print("\nOBSERVED POPULATION TARGETS")
    print(f"distance attempt share: {best['observed_mean_distance_share']:.4f}")
    print(f"clinch attempt share:   {best['observed_mean_clinch_share']:.4f}")
    print(f"ground attempt share:   {best['observed_mean_ground_share']:.4f}")
    print(f"TD attempts/round:      {best['observed_mean_td_attempts_per_round']:.4f}")

    print("\nBEST COARSE RESEARCH CANDIDATE")
    print(f"distance exit scale:      {best['distance_exit_scale']:.4f}")
    print(f"clinch exit scale:        {best['clinch_exit_scale']:.4f}")
    print(f"wrestling attempt scale:  {best['wrestling_attempt_scale']:.4f}")
    print(f"implied distance share:   {best['mean_occ_distance']:.4f}")
    print(f"implied clinch share:     {best['mean_occ_clinch']:.4f}")
    print(f"implied ground share:     {best['mean_occ_ground']:.4f}")
    print(f"implied TD attempts/round:{best['mean_td_attempts_per_round']:.4f}")

    print(
        "\nIMPORTANT: this is still a static aggregate calibration proxy, not the final "
        "segment transition engine. The next step is to use the best coarse scales as "
        "starting priors in an actual state-path simulation and calibrate occupancy, "
        "TD volume, control duration, and transition persistence jointly."
    )


if __name__ == "__main__":
    main()
