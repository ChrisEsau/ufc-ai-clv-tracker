"""Offline KD shock-response curve sweep for Damage Reservoir V1.

Purpose
-------
Use the strike-level shock audit artifact to study the *shape* of the KD
probability curve before changing the active simulator.

This follows the documented reservoir path:

    strike damage -> shock_fraction -> KD susceptibility

For each candidate shock coefficient, this script re-centers the KD baseline
logit so the mean expected KD probability remains equal to the current audited
KD-per-landed-strike rate. This prevents us from choosing a curve merely because
it creates more or fewer KDs overall.

The sweep then compares candidates on:
- KD probability at fixed shock levels;
- expected KD concentration in the upper shock tail;
- median shock among expected KD mass;
- power minus opponent KD-resistance separation;
- fresh vs depleted reservoir susceptibility;
- normal vs recent-KD susceptibility.

Important boundary
------------------
This is an OFFLINE curve-shape audit. It does not alter the simulator and does
not resimulate KD feedback into future recent-KD state. Any candidate selected
here must still pass a full path population validation before replacing the
active KD equation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage


INPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_shock_audit.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_kd_shock_curve_sweep.parquet"
)

# Current active value plus progressively steeper acute-shock candidates.
SHOCK_COEFFICIENTS = [12.0, 40.0, 80.0, 120.0, 160.0, 220.0]

# Fixed shock points from the observed audit scale.
SHOCK_POINTS = [0.005, 0.010, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200]

# Representative states for readable curve comparisons.
RESISTANCE_STATES = {
    "low_resistance_30": 30.0,
    "average_resistance_50": 50.0,
    "high_resistance_70": 70.0,
}
RESERVOIR_STATES = {
    "fresh_100pct": 1.00,
    "mid_50pct": 0.50,
    "depleted_25pct": 0.25,
}


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _base_terms(frame: pd.DataFrame) -> np.ndarray:
    resistance = pd.to_numeric(
        frame["defender_knockdown_resistance"], errors="coerce"
    ).fillna(50.0).to_numpy(dtype=float)
    reservoir_after = pd.to_numeric(
        frame["reservoir_fraction_after"], errors="coerce"
    ).fillna(1.0).to_numpy(dtype=float)
    recent = pd.to_numeric(frame["recent_kd_before"], errors="coerce").fillna(0.0)
    recent = recent.to_numpy(dtype=float)

    depletion = 1.0 - np.clip(reservoir_after, 0.0, 1.0)
    return (
        (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
        + damage.KD_DEPLETION_COEFFICIENT * depletion
        + damage.KD_RECENT_KD_LOGIT_BONUS * recent
    )


def _fit_intercept(
    shock: np.ndarray,
    base_terms: np.ndarray,
    shock_coefficient: float,
    target_mean: float,
) -> float:
    """Binary-search intercept so mean expected KD probability hits target."""
    low = -20.0
    high = 5.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        p = _sigmoid_array(mid + shock_coefficient * shock + base_terms)
        if float(p.mean()) < target_mean:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    cumulative = np.cumsum(w) / total
    idx = int(np.searchsorted(cumulative, q, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _rank_quintile(series: pd.Series) -> pd.Series:
    ranks = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    return pd.cut(
        ranks,
        bins=np.linspace(0.0, 1.0, 6),
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        include_lowest=True,
    )


def _candidate_probabilities(
    frame: pd.DataFrame,
    shock_coefficient: float,
    intercept: float,
) -> np.ndarray:
    shock = pd.to_numeric(frame["shock_fraction"], errors="coerce").fillna(0.0)
    shock = shock.to_numpy(dtype=float)
    return _sigmoid_array(intercept + shock_coefficient * shock + _base_terms(frame))


def _print_curve_table(intercepts: dict[float, float]) -> None:
    print("\nFIXED-SHOCK KD PROBABILITY CURVES")
    print("Representative state: KD resistance=50, fresh reservoir, no recent KD")
    rows: list[dict[str, float]] = []
    for coefficient in SHOCK_COEFFICIENTS:
        intercept = intercepts[coefficient]
        row: dict[str, float] = {
            "shock_coefficient": coefficient,
            "fitted_base_logit": intercept,
        }
        for shock in SHOCK_POINTS:
            p = float(_sigmoid_array(np.array([intercept + coefficient * shock]))[0])
            row[f"p_KD_at_{100*shock:g}pct_shock"] = p
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def _print_state_shift_table(intercepts: dict[float, float]) -> None:
    print("\nSTATE SHIFTS AT 3%, 5%, AND 10% SHOCK")
    rows: list[dict[str, float | str]] = []
    for coefficient in SHOCK_COEFFICIENTS:
        intercept = intercepts[coefficient]
        for resistance_label, resistance in RESISTANCE_STATES.items():
            for reservoir_label, reservoir_fraction in RESERVOIR_STATES.items():
                depletion = 1.0 - reservoir_fraction
                base_term = (
                    (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
                    + damage.KD_DEPLETION_COEFFICIENT * depletion
                )
                row: dict[str, float | str] = {
                    "shock_coefficient": coefficient,
                    "resistance_state": resistance_label,
                    "reservoir_state": reservoir_label,
                }
                for shock in (0.03, 0.05, 0.10):
                    logit = intercept + coefficient * shock + base_term
                    row[f"p_KD_{100*shock:g}pct_shock"] = float(
                        _sigmoid_array(np.array([logit]))[0]
                    )
                rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def _summarize_candidates(frame: pd.DataFrame, target_mean: float) -> pd.DataFrame:
    shock = pd.to_numeric(frame["shock_fraction"], errors="coerce").fillna(0.0)
    shock_arr = shock.to_numpy(dtype=float)
    base_terms = _base_terms(frame)

    edge = (
        pd.to_numeric(frame["attacker_power"], errors="coerce").fillna(50.0)
        - pd.to_numeric(
            frame["defender_knockdown_resistance"], errors="coerce"
        ).fillna(50.0)
    )
    edge_bucket = _rank_quintile(edge)

    reservoir_before = pd.to_numeric(
        frame["reservoir_fraction_before"], errors="coerce"
    ).fillna(1.0)
    fresh_mask = reservoir_before >= 0.75
    depleted_mask = reservoir_before < 0.25
    recent_mask = pd.to_numeric(frame["recent_kd_before"], errors="coerce").fillna(0) > 0
    normal_mask = ~recent_mask

    p99 = float(shock.quantile(0.99))
    p995 = float(shock.quantile(0.995))
    p999 = float(shock.quantile(0.999))

    rows: list[dict[str, float]] = []
    for coefficient in SHOCK_COEFFICIENTS:
        intercept = _fit_intercept(
            shock_arr,
            base_terms,
            coefficient,
            target_mean,
        )
        p = _sigmoid_array(intercept + coefficient * shock_arr + base_terms)

        q1 = edge_bucket == "Q1"
        q5 = edge_bucket == "Q5"
        q1_mean = float(p[q1.to_numpy()].mean())
        q5_mean = float(p[q5.to_numpy()].mean())

        fresh_mean = float(p[fresh_mask.to_numpy()].mean())
        depleted_mean = float(p[depleted_mask.to_numpy()].mean())
        normal_mean = float(p[normal_mask.to_numpy()].mean())
        recent_mean = float(p[recent_mask.to_numpy()].mean()) if recent_mask.any() else np.nan

        expected_mass = float(p.sum())
        def mass_share(mask: np.ndarray) -> float:
            return float(p[mask].sum() / expected_mass) if expected_mass > 0 else np.nan

        rows.append(
            {
                "shock_coefficient": coefficient,
                "fitted_base_logit": intercept,
                "mean_expected_kd_per_strike": float(p.mean()),
                "expected_median_shock_on_kd": _weighted_quantile(shock_arr, p, 0.50),
                "expected_p90_shock_on_kd": _weighted_quantile(shock_arr, p, 0.90),
                "KD_mass_share_at_ge_p99_shock": mass_share(shock_arr >= p99),
                "KD_mass_share_at_ge_p99_5_shock": mass_share(shock_arr >= p995),
                "KD_mass_share_at_ge_p99_9_shock": mass_share(shock_arr >= p999),
                "Q1_edge_expected_kd_per_strike": q1_mean,
                "Q5_edge_expected_kd_per_strike": q5_mean,
                "Q5_Q1_edge_ratio": q5_mean / q1_mean if q1_mean > 0 else np.nan,
                "fresh_expected_kd_per_strike": fresh_mean,
                "depleted_expected_kd_per_strike": depleted_mean,
                "depleted_fresh_ratio": (
                    depleted_mean / fresh_mean if fresh_mean > 0 else np.nan
                ),
                "normal_expected_kd_per_strike": normal_mean,
                "recent_expected_kd_per_strike": recent_mean,
                "recent_normal_ratio": (
                    recent_mean / normal_mean
                    if normal_mean > 0 and np.isfinite(recent_mean)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Damage V1 KD shock-response curve shapes"
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[KD shock curve sweep] loading strike audit from {args.input}", flush=True)
    frame = pd.read_parquet(args.input)
    required = {
        "shock_fraction",
        "attacker_power",
        "defender_knockdown_resistance",
        "reservoir_fraction_before",
        "reservoir_fraction_after",
        "recent_kd_before",
        "knockdown",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"shock audit artifact missing columns: {missing}")

    target_mean = float(pd.to_numeric(frame["knockdown"], errors="coerce").mean())
    shock_arr = pd.to_numeric(frame["shock_fraction"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_terms = _base_terms(frame)

    print("\n" + "=" * 132)
    print("DAMAGE RESERVOIR V1 — KD SHOCK RESPONSE CURVE SWEEP")
    print("=" * 132)
    print(f"landed significant strikes: {len(frame):,}")
    print(f"target overall KD per landed strike: {target_mean:.6f}")
    print(
        "boundary: every candidate is re-centered to the same overall KD rate; "
        "this compares curve shape only."
    )

    summary = _summarize_candidates(frame, target_mean)
    intercepts = {
        float(row["shock_coefficient"]): float(row["fitted_base_logit"])
        for _, row in summary.iterrows()
    }

    print("\nCANDIDATE SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    _print_curve_table(intercepts)
    _print_state_shift_table(intercepts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.output, index=False)
    print(f"\n[KD shock curve sweep] wrote {args.output}", flush=True)
    print(
        "\nRESEARCH BOUNDARY: do not modify the active KD equation from this "
        "offline sweep alone. Select finalists, then run full path population "
        "validation with dynamic recent-KD feedback."
    )


if __name__ == "__main__":
    main()
