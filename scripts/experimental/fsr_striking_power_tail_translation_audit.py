"""Audit how the current striking-power FSR maps into strike severity and shock.

This is an isolated diagnostic of the existing Damage Reservoir V1 severity
architecture. It changes no FSR values and no simulator constants.

For fixed striking-power values, repeatedly draw landed-strike damage using the
current power-tail equations and report:
- tail-event probability;
- tail-magnitude multiplier;
- damage quantiles;
- shock quantiles assuming the canonical 100-unit reservoir;
- probability of shock >= 3%, 5%, 8%, 10%, and 15% capacity.

The goal is to determine whether realistic FSR power differences translate into
meaningfully different severe-strike distributions before changing any model
constants.
"""
from __future__ import annotations

import argparse
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage


DEFAULT_POWER_VALUES = (45.0, 48.0, 50.0, 53.0, 55.0, 58.0, 60.0, 62.0)
DEFAULT_DRAWS_PER_POWER = 250_000
DEFAULT_SEED = 20260810
CANONICAL_RESERVOIR_CAPACITY = 100.0
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_tail_translation_audit.parquet"
)
SHOCK_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15)


def _logit(p: float) -> float:
    return log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def tail_probability(power: float) -> float:
    """Exact current Damage V1 power-tail probability."""
    return _sigmoid(
        _logit(damage.POWER_TAIL_BASE_PROBABILITY)
        + (float(power) - 50.0) / damage.POWER_TAIL_RATING_SCALE
    )


def tail_magnitude_multiplier(power: float) -> float:
    """Exact current Damage V1 multiplier on the additive tail draw."""
    return exp((float(power) - 50.0) / damage.TAIL_MAGNITUDE_POWER_SCALE)


def _draw_damage(
    rng: np.random.Generator,
    power: float,
    draws: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized copy of the current _draw_strike_damage() architecture."""
    base_draw = rng.gamma(
        damage.BASE_SEVERITY_GAMMA_SHAPE,
        damage.BASE_SEVERITY_GAMMA_SCALE,
        size=draws,
    )

    p_tail = tail_probability(power)
    is_tail = rng.random(draws) < p_tail
    raw = base_draw.astype(float, copy=True)

    tail_count = int(is_tail.sum())
    if tail_count:
        tail = rng.gamma(
            damage.TAIL_SEVERITY_GAMMA_SHAPE,
            damage.TAIL_SEVERITY_GAMMA_SCALE,
            size=tail_count,
        )
        tail *= tail_magnitude_multiplier(power)
        raw[is_tail] += tail

    strike_damage = np.maximum(0.0, raw * damage.STRIKE_DAMAGE_SCALE)
    return strike_damage, is_tail


def _audit_power(
    rng: np.random.Generator,
    power: float,
    draws: int,
) -> dict[str, float | int]:
    strike_damage, is_tail = _draw_damage(rng, power, draws)
    shock = strike_damage / CANONICAL_RESERVOIR_CAPACITY

    row: dict[str, float | int] = {
        "striking_power": float(power),
        "draws": int(draws),
        "tail_probability_formula": tail_probability(power),
        "tail_event_rate_observed": float(is_tail.mean()),
        "tail_magnitude_multiplier": tail_magnitude_multiplier(power),
        "damage_mean": float(strike_damage.mean()),
        "damage_p50": float(np.quantile(strike_damage, 0.50)),
        "damage_p90": float(np.quantile(strike_damage, 0.90)),
        "damage_p95": float(np.quantile(strike_damage, 0.95)),
        "damage_p99": float(np.quantile(strike_damage, 0.99)),
        "damage_p999": float(np.quantile(strike_damage, 0.999)),
        "shock_mean": float(shock.mean()),
        "shock_p50": float(np.quantile(shock, 0.50)),
        "shock_p90": float(np.quantile(shock, 0.90)),
        "shock_p95": float(np.quantile(shock, 0.95)),
        "shock_p99": float(np.quantile(shock, 0.99)),
        "shock_p999": float(np.quantile(shock, 0.999)),
    }

    for threshold in SHOCK_THRESHOLDS:
        pct = int(round(threshold * 100))
        row[f"p_shock_ge_{pct}pct"] = float((shock >= threshold).mean())

    return row


def _print_results(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 150)
    print("STRIKING-POWER FSR -> CURRENT DAMAGE/SHOCK TAIL TRANSLATION")
    print("=" * 150)
    print(f"draws per fixed power value: {int(frame['draws'].iloc[0]):,}")
    print(f"canonical reservoir capacity: {CANONICAL_RESERVOIR_CAPACITY:.1f}")
    print("No simulator constants or FSR values are changed.\n")

    cols = [
        "striking_power",
        "tail_probability_formula",
        "tail_magnitude_multiplier",
        "damage_p50",
        "damage_p90",
        "damage_p95",
        "damage_p99",
        "shock_p99",
        "p_shock_ge_3pct",
        "p_shock_ge_5pct",
        "p_shock_ge_8pct",
        "p_shock_ge_10pct",
        "p_shock_ge_15pct",
    ]
    print("CURRENT TRANSLATION BY POWER")
    print(frame[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    low_power = 48.0 if frame["striking_power"].eq(48.0).any() else float(frame["striking_power"].min())
    high_power = 58.0 if frame["striking_power"].eq(58.0).any() else float(frame["striking_power"].max())
    low = frame.loc[frame["striking_power"].eq(low_power)].iloc[0]
    high = frame.loc[frame["striking_power"].eq(high_power)].iloc[0]

    print(f"\nREFERENCE SEPARATION: POWER {low_power:.0f} -> {high_power:.0f}")
    print(f"tail probability ratio: {high['tail_probability_formula'] / low['tail_probability_formula']:.3f}x")
    print(f"tail magnitude multiplier ratio: {high['tail_magnitude_multiplier'] / low['tail_magnitude_multiplier']:.3f}x")
    print(f"damage p99 ratio: {high['damage_p99'] / low['damage_p99']:.3f}x")
    for threshold in (0.05, 0.08, 0.10, 0.15):
        pct = int(round(threshold * 100))
        key = f"p_shock_ge_{pct}pct"
        denominator = float(low[key])
        ratio = float(high[key]) / denominator if denominator > 0 else float("nan")
        print(
            f"P(shock >= {pct}%): {float(low[key]):.4%} -> {float(high[key]):.4%} "
            f"({ratio:.3f}x)"
        )

    print("\nDECISION GUIDE")
    print("- Strong FSR signal but weak damage/shock separation -> simulator tail mapping is too compressed.")
    print("- Large shock separation already present -> look elsewhere before increasing tail sensitivity.")
    print("- A global Gamma change affects every fighter; power-sensitivity changes target separation by FSR.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit current striking-power FSR translation into strike damage and shock"
    )
    parser.add_argument(
        "--powers",
        type=float,
        nargs="+",
        default=list(DEFAULT_POWER_VALUES),
        help="Fixed striking-power FSR values to audit",
    )
    parser.add_argument("--draws-per-power", type=int, default=DEFAULT_DRAWS_PER_POWER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.draws_per_power <= 0:
        raise ValueError("--draws-per-power must be positive")

    rng = np.random.default_rng(args.seed)
    rows = []
    for power in args.powers:
        print(
            f"[power tail audit] power={power:.1f}; draws={args.draws_per_power:,}",
            flush=True,
        )
        rows.append(_audit_power(rng, power, args.draws_per_power))

    frame = pd.DataFrame(rows).sort_values("striking_power").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    _print_results(frame)
    print(f"\n[power tail audit] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
