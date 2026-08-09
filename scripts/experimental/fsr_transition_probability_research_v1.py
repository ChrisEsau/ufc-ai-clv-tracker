"""Research-only calibration of static FSR phase utilities into action probabilities.

This script does NOT modify the simulator. It evaluates utility-to-probability
transforms for the currently supported distance-state actions:

- stay at distance
- enter clinch
- attempt takedown

The utilities come from the validated FSR-26 phase-preference research. We use
observed historical phase mix only as aggregate calibration targets; phase-share
statistics are NOT interpreted as literal 30-second transition probabilities.

Shadow/research only.
"""

from __future__ import annotations

from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")

TEMPERATURES = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

RFS_COLUMNS = {
    "distance_share": "rfs_phase_base_fight_distance_attempt_share",
    "clinch_share": "rfs_phase_base_fight_clinch_attempt_share",
    "td_rate": "rfs_phase_base_fight_td_attempts_per_round",
    "control_rate": "rfs_phase_base_fight_control_seconds_per_round",
}


def softmax3(a: np.ndarray, b: np.ndarray, c: np.ndarray, temperature: float):
    """Return row-wise softmax probabilities for three utilities."""
    z = np.column_stack([a, b, c]) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / ez.sum(axis=1, keepdims=True)
    return p[:, 0], p[:, 1], p[:, 2]


def derive_static_utilities(df: pd.DataFrame) -> pd.DataFrame:
    """Build the state-specific static utilities supported by prior research."""
    out = df.copy()

    d = out["distance_striking_pressure"] - 50.0
    c = out["clinch_striking_pressure"] - 50.0
    w = out["wrestling_entry"] - 50.0
    ctrl = out["control_imposition"] - 50.0
    opp_ctrl_res = out["opponent_control_resistance"] - 50.0

    # Distance and clinch use the matchup-adjusted / intrinsic conclusions from
    # phase-preference research. Wrestling desire uses control_blend because it
    # best tracked realized TD-attempt rate.
    out["utility_distance"] = d - 0.50 * c - 0.50 * w + 0.10 * opp_ctrl_res
    out["utility_clinch"] = c - 0.50 * d - 0.50 * w
    out["utility_wrestling"] = w + 0.25 * ctrl - 0.50 * d - 0.50 * c

    return out


def opponent_join(fsr: pd.DataFrame) -> pd.DataFrame:
    own = fsr.copy()
    opp = fsr[["fight_id", "fighter_id", "control_resistance"]].rename(
        columns={
            "fighter_id": "opponent_id",
            "control_resistance": "opponent_control_resistance",
        }
    )
    pairs = own.merge(opp, on="fight_id")
    pairs = pairs[pairs["fighter_id"].astype(str) != pairs["opponent_id"].astype(str)]
    return pairs.reset_index(drop=True)


def calibration_metrics(frame: pd.DataFrame, temperature: float) -> dict[str, float]:
    p_d, p_c, p_w = softmax3(
        frame["utility_distance"].to_numpy(),
        frame["utility_clinch"].to_numpy(),
        frame["utility_wrestling"].to_numpy(),
        temperature,
    )
    work = frame.copy()
    work["p_distance"] = p_d
    work["p_clinch"] = p_c
    work["p_wrestling"] = p_w

    return {
        "temperature": float(temperature),
        "distance_spearman": work[["p_distance", RFS_COLUMNS["distance_share"]]].corr(method="spearman").iloc[0, 1],
        "clinch_spearman": work[["p_clinch", RFS_COLUMNS["clinch_share"]]].corr(method="spearman").iloc[0, 1],
        "wrestling_td_spearman": work[["p_wrestling", RFS_COLUMNS["td_rate"]]].corr(method="spearman").iloc[0, 1],
        "wrestling_control_spearman": work[["p_wrestling", RFS_COLUMNS["control_rate"]]].corr(method="spearman").iloc[0, 1],
        "mean_p_distance": float(work["p_distance"].mean()),
        "mean_p_clinch": float(work["p_clinch"].mean()),
        "mean_p_wrestling": float(work["p_wrestling"].mean()),
        "p_distance_p05": float(work["p_distance"].quantile(0.05)),
        "p_distance_p95": float(work["p_distance"].quantile(0.95)),
        "p_clinch_p05": float(work["p_clinch"].quantile(0.05)),
        "p_clinch_p95": float(work["p_clinch"].quantile(0.95)),
        "p_wrestling_p05": float(work["p_wrestling"].quantile(0.05)),
        "p_wrestling_p95": float(work["p_wrestling"].quantile(0.95)),
    }


def print_probability_buckets(frame: pd.DataFrame, temperature: float) -> None:
    p_d, p_c, p_w = softmax3(
        frame["utility_distance"].to_numpy(),
        frame["utility_clinch"].to_numpy(),
        frame["utility_wrestling"].to_numpy(),
        temperature,
    )
    work = frame.copy()
    work["p_distance"] = p_d
    work["p_clinch"] = p_c
    work["p_wrestling"] = p_w

    specs = (
        ("DISTANCE", "p_distance", RFS_COLUMNS["distance_share"]),
        ("CLINCH", "p_clinch", RFS_COLUMNS["clinch_share"]),
        ("WRESTLING", "p_wrestling", RFS_COLUMNS["td_rate"]),
    )

    for label, pcol, target in specs:
        ranked = work.dropna(subset=[pcol, target]).copy()
        ranked["bucket"] = pd.qcut(
            ranked[pcol].rank(method="first"),
            q=7,
            labels=[f"Q{i}" for i in range(1, 8)],
        )
        summary = ranked.groupby("bucket", observed=False).agg(
            rows=("fighter_id", "size"),
            mean_probability=(pcol, "mean"),
            mean_realized=(target, "mean"),
        )
        print(f"\n{label} probability buckets")
        print(summary.to_string())


def main() -> None:
    print(f"[transition research] loading FSR-26 from {FSR_PATH}", flush=True)
    fsr = pd.read_parquet(FSR_PATH)
    print(f"[transition research] loaded {len(fsr):,} FSR-26 rows", flush=True)

    print(f"[transition research] loading RFS history from {RFS_PATH}", flush=True)
    rfs = pd.read_parquet(
        RFS_PATH,
        columns=["fight_id", "fighter_id", *RFS_COLUMNS.values()],
    )
    print(f"[transition research] loaded {len(rfs):,} RFS rows", flush=True)

    for frame in (fsr, rfs):
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)

    needed = [
        "fight_id",
        "fighter_id",
        "fighter_name",
        "distance_striking_pressure",
        "clinch_striking_pressure",
        "wrestling_entry",
        "control_imposition",
        "control_resistance",
    ]
    missing = [c for c in needed if c not in fsr.columns]
    if missing:
        raise RuntimeError(f"FSR-26 missing required phase utility columns: {missing}")

    print("[transition research] constructing opponent-aware utilities", flush=True)
    pairs = opponent_join(fsr[needed])
    research = pairs.merge(rfs, on=["fight_id", "fighter_id"], validate="one_to_one")
    research = derive_static_utilities(research)

    for c in RFS_COLUMNS.values():
        research[c] = pd.to_numeric(research[c], errors="coerce")

    rows = [calibration_metrics(research, t) for t in TEMPERATURES]
    table = pd.DataFrame(rows)

    print("\nSTATIC UTILITY -> SOFTMAX TEMPERATURE SWEEP")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Rank by an equal-weight correlation score. Temperature cannot change
    # Spearman ordering for a single utility in isolation, but with competing
    # softmax utilities it can slightly alter rankings and, more importantly,
    # controls probability concentration for later MC use.
    table["mean_rank_signal"] = table[[
        "distance_spearman",
        "clinch_spearman",
        "wrestling_td_spearman",
        "wrestling_control_spearman",
    ]].mean(axis=1)

    # Prefer the highest signal; break near-ties toward a less concentrated
    # transform (larger temperature) to avoid deterministic phase choices.
    best_signal = table["mean_rank_signal"].max()
    contenders = table[table["mean_rank_signal"] >= best_signal - 0.002]
    selected = contenders.sort_values("temperature", ascending=False).iloc[0]
    selected_t = float(selected["temperature"])

    print(f"\nSelected research temperature: {selected_t:.2f}")
    print("Reason: near-best rank signal with the least concentrated probabilities among near-ties.")
    print_probability_buckets(research, selected_t)

    print("\nIMPORTANT: these are relative action-choice probabilities, not yet calibrated")
    print("as literal 30-second transition hazards. Absolute hazard/persistence still requires")
    print("a simulator-level calibration step against aggregate phase occupancy and TD volume.")


if __name__ == "__main__":
    main()
