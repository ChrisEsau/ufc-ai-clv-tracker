"""Audit whether striking_power acts mainly through the upper strike-damage tail.

This is a shadow/research audit. It reads the strike-level artifact produced by
``fsr_static_mc_damage_v1_population_audit.py`` and evaluates the existing V1
severity draw by attacker-power quintile.

The provisional reservoir-consumption study selected a 0.50 damage scale. The
scale is applied here only to report the candidate reservoir units; because it
is a constant multiplier, it cannot create or remove power-tail separation.

No simulator constants are modified by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STRIKE_AUDIT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_strike_audit.parquet"
)
DEFAULT_DAMAGE_SCALE = 0.50
POWER_LABELS = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]


def _power_bucket(series: pd.Series) -> pd.Series:
    """Create equal-count attacker-power quintiles with deterministic tie handling."""
    rank = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    return pd.cut(
        rank,
        bins=np.linspace(0.0, 1.0, 6),
        labels=POWER_LABELS,
        include_lowest=True,
        ordered=True,
    )


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(
        pd.DataFrame({"x": x[mask], "y": y[mask]})
        .corr(method="spearman")
        .iloc[0, 1]
    )


def run_audit(path: Path, damage_scale: float) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"attacker_power", "strike_damage"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"strike audit missing required columns: {missing}")
    if damage_scale <= 0:
        raise ValueError("damage_scale must be positive")

    work = frame[["attacker_power", "strike_damage"]].copy()
    work["attacker_power"] = pd.to_numeric(work["attacker_power"], errors="coerce")
    work["raw_strike_damage"] = pd.to_numeric(work["strike_damage"], errors="coerce")
    work = work.dropna().reset_index(drop=True)
    work["strike_damage"] = work["raw_strike_damage"] * float(damage_scale)
    work["power_bucket"] = _power_bucket(work["attacker_power"])

    global_p95 = float(work["strike_damage"].quantile(0.95))
    global_p99 = float(work["strike_damage"].quantile(0.99))
    global_p995 = float(work["strike_damage"].quantile(0.995))

    print("=" * 126)
    print("DAMAGE RESERVOIR V1 — STRIKING POWER TAIL AUDIT")
    print("=" * 126)
    print(f"strike rows: {len(work):,}")
    print(f"reporting damage scale: {damage_scale:.2f}")
    print(
        "global thresholds: "
        f"p95={global_p95:.4f}, p99={global_p99:.4f}, p99.5={global_p995:.4f}"
    )
    print(
        "Spearman attacker power vs strike damage: "
        f"{_safe_spearman(work['attacker_power'], work['strike_damage']):.5f}"
    )

    rows: list[dict[str, float | int | str]] = []
    for bucket in POWER_LABELS:
        g = work[work["power_bucket"] == bucket]
        rows.append(
            {
                "power_bucket": bucket,
                "landed_strikes": len(g),
                "mean_power": g["attacker_power"].mean(),
                "mean_damage": g["strike_damage"].mean(),
                "median_damage": g["strike_damage"].median(),
                "p90_damage": g["strike_damage"].quantile(0.90),
                "p95_damage": g["strike_damage"].quantile(0.95),
                "p99_damage": g["strike_damage"].quantile(0.99),
                "p995_damage": g["strike_damage"].quantile(0.995),
                "max_damage": g["strike_damage"].max(),
                "global_p95_exceed_rate": (g["strike_damage"] >= global_p95).mean(),
                "global_p99_exceed_rate": (g["strike_damage"] >= global_p99).mean(),
                "global_p995_exceed_rate": (g["strike_damage"] >= global_p995).mean(),
            }
        )

    result = pd.DataFrame(rows)
    print("\nPOWER QUINTILES")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    q1 = result.iloc[0]
    q5 = result.iloc[-1]
    print("\nQ5 / Q1 SEPARATION")
    for col in (
        "mean_damage",
        "median_damage",
        "p90_damage",
        "p95_damage",
        "p99_damage",
        "p995_damage",
        "global_p95_exceed_rate",
        "global_p99_exceed_rate",
        "global_p995_exceed_rate",
    ):
        denominator = float(q1[col])
        ratio = float(q5[col]) / denominator if denominator > 0 else float("nan")
        print(f"{col}: {ratio:.3f}x")

    print(
        "\nINTERPRETATION BOUNDARY: a valid tail-power mechanic should show much "
        "larger Q5/Q1 separation in p95/p99/tail exceedance than at the median. "
        "Do not tune KD constants from this audit."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Damage V1 striking-power tail behavior")
    parser.add_argument("--strike-audit-path", type=Path, default=STRIKE_AUDIT_PATH)
    parser.add_argument("--damage-scale", type=float, default=DEFAULT_DAMAGE_SCALE)
    args = parser.parse_args()

    print(f"[power tail audit] reading {args.strike_audit_path}", flush=True)
    run_audit(args.strike_audit_path, args.damage_scale)


if __name__ == "__main__":
    main()
