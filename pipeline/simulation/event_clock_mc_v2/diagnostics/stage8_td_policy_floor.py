"""Diagnose whether Event Clock V2 TD over-selection comes from policy representation.

Measurement only. No simulator, FSR, timing, mechanics, or policy values are changed.
The diagnostic compares historical TD attempts with:
- raw validated FSR V3 takedown tendency,
- matchup-effective FSR V3 TD rate,
- TD completion skill,
- the current averaged BrainCapabilities.takedown value,
- the current neutral standing softmax TD choice probability.

It also reports the structurally implied per-opportunity TD choice probability:
    matchup TD rate per 15 / neutral brain opportunities per 15
where neutral opportunities per 15 are derived from the approved Stage 4 mean delay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.policy import (
    BrainDecisionContext,
    action_probabilities,
)
from pipeline.simulation.event_clock_mc_v2.brain.timing import (
    BrainTimingContext,
    expected_action_delay,
)
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Side
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    load_latest_profiles,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
)
from .stage6_real_causal_path import _capabilities
from .stage8_structural_population import (
    MASTER,
    ROUND_STATS,
    actual_side_totals,
    elapsed_seconds,
    pick_col,
    side_rows,
)


def _corr(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _mae(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.abs(x[mask] - y[mask]))) if mask.any() else float("nan")


def _percentile(series: pd.Series, value: float) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        raise RuntimeError("empty percentile reference")
    return float(np.mean(arr <= float(value)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=100)
    ap.add_argument("--output", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/stage8_td_policy_floor.json"))
    args = ap.parse_args()

    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    date_col = pick_col(master, "date", "event_date")
    master["_event_date"] = pd.to_datetime(master[date_col], errors="coerce").dt.normalize()
    master = master.dropna(subset=["_event_date"]).sort_values(["_event_date", "fight_id"], ascending=[False, False])

    round_stats = pd.read_parquet(ROUND_STATS).copy()
    rs_fight_col = pick_col(round_stats, "fight_id", "bout_id")
    available = set(round_stats[rs_fight_col].astype(str))
    snapshots = load_prefight_snapshots()
    latest = load_latest_profiles()
    reference = CapabilityReference.from_latest(latest)

    raw_td_reference = pd.to_numeric(latest["takedown_tendency"], errors="coerce")
    neutral_delay = expected_action_delay(FightState(), BrainTimingContext())
    neutral_opportunities_per15 = 900.0 / neutral_delay

    rows: list[dict[str, float | str]] = []
    fights_used = 0
    for _, fight in master.iterrows():
        if fights_used >= args.fights:
            break
        fight_id = str(fight["fight_id"])
        if fight_id not in available:
            continue
        horizon = elapsed_seconds(fight)
        if horizon <= 0:
            continue
        try:
            red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
            red_fsr, blue_fsr = historical_fighter_rows(
                snapshots,
                event_date=fight["_event_date"],
                fight_id=fight_id,
                fighter_ids=(red_id, blue_id),
            )
            side_rows(round_stats, fight_id, red_id, "red")
            side_rows(round_stats, fight_id, blue_id, "blue")
        except Exception:
            continue

        fights_used += 1
        for side, fighter_id, opponent_id, fighter_fsr, opponent_fsr, name in (
            (Side.RED, red_id, blue_id, red_fsr, blue_fsr, fight.get("r_name", red_id)),
            (Side.BLUE, blue_id, red_id, blue_fsr, red_fsr, fight.get("b_name", blue_id)),
        ):
            cap, runtime = _capabilities(fighter_fsr, opponent_fsr, reference)
            actual = actual_side_totals(side_rows(round_stats, fight_id, fighter_id, side.value))
            actual_td_per15 = actual["td_att"] * 900.0 / horizon
            distribution = action_probabilities(
                FightState(), side, cap, BrainDecisionContext()
            )
            current_td_probability = next(
                row.probability
                for row in distribution
                if row.action_family is ActionFamily.TAKEDOWN_ENTRY
            )
            implied_rate_from_current_policy = current_td_probability * neutral_opportunities_per15
            raw_tendency = float(fighter_fsr["takedown_tendency"])
            rows.append({
                "fight_id": fight_id,
                "event_date": str(fight["_event_date"].date()),
                "fighter_id": fighter_id,
                "opponent_id": opponent_id,
                "fighter_name": str(name),
                "actual_td_attempts_per15": float(actual_td_per15),
                "raw_takedown_tendency": raw_tendency,
                "raw_takedown_tendency_percentile": _percentile(raw_td_reference, raw_tendency),
                "matchup_takedown_rate_per15": float(runtime.takedown_rate_15m),
                "takedown_completion_probability": float(runtime.takedown_completion),
                "current_takedown_capability": float(cap.takedown),
                "current_neutral_td_choice_probability": float(current_td_probability),
                "current_policy_implied_td_rate_per15": float(implied_rate_from_current_policy),
                "fsr_rate_implied_choice_probability": float(runtime.takedown_rate_15m / neutral_opportunities_per15),
            })

    if not rows:
        raise RuntimeError("no complete historical observations")

    frame = pd.DataFrame(rows)
    actual = frame["actual_td_attempts_per15"].tolist()
    metrics = {}
    for column in (
        "raw_takedown_tendency",
        "raw_takedown_tendency_percentile",
        "matchup_takedown_rate_per15",
        "takedown_completion_probability",
        "current_takedown_capability",
        "current_neutral_td_choice_probability",
    ):
        metrics[column] = {
            "corr_with_actual_td_attempts_per15": _corr(actual, frame[column].tolist())
        }

    metrics["rate_calibration"] = {
        "actual_mean_td_attempts_per15": float(frame["actual_td_attempts_per15"].mean()),
        "fsr_matchup_rate_mean_per15": float(frame["matchup_takedown_rate_per15"].mean()),
        "current_policy_implied_mean_per15": float(frame["current_policy_implied_td_rate_per15"].mean()),
        "fsr_rate_mae_vs_actual": _mae(actual, frame["matchup_takedown_rate_per15"].tolist()),
        "current_policy_implied_mae_vs_actual": _mae(actual, frame["current_policy_implied_td_rate_per15"].tolist()),
        "neutral_mean_delay_seconds": neutral_delay,
        "neutral_opportunities_per15": neutral_opportunities_per15,
    }

    payload = {
        "diagnostic": "Stage 8 TD policy floor / representation audit",
        "fights": fights_used,
        "fighter_observations": len(frame),
        "measurement_only": True,
        "metrics": metrics,
        "rows": frame.to_dict(orient="records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 100)
    print("STAGE 8 TD POLICY FLOOR / REPRESENTATION AUDIT")
    print("=" * 100)
    print(f"fights={fights_used} fighter_observations={len(frame)}")
    print(json.dumps(metrics, indent=2))
    print("\nLOWEST RAW TD-TENDENCY OBSERVATIONS")
    columns = [
        "fighter_name",
        "actual_td_attempts_per15",
        "raw_takedown_tendency_percentile",
        "matchup_takedown_rate_per15",
        "current_takedown_capability",
        "current_neutral_td_choice_probability",
        "current_policy_implied_td_rate_per15",
    ]
    print(frame.sort_values("raw_takedown_tendency_percentile")[columns].head(20).to_string(index=False))
    print(f"\nWROTE {args.output}")


if __name__ == "__main__":
    main()
