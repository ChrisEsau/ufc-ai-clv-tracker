"""Audit canonical prefight physiology inputs without creating data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots


def _summary(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce")
    return {
        "count": int(clean.notna().sum()),
        "missing": int(clean.isna().sum()),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.10)),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = load_prefight_snapshots().copy()
    required = {
        "event_date",
        "fight_id",
        "fighter_id",
        "striking_power_v3",
        "damage_durability",
        "knockdown_resistance_v3",
        "stamina_capacity",
        "stamina_depletion_resistance",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"canonical artifact missing required traits: {sorted(missing)}")
    translated = pd.DataFrame(
        {
            "striking_power_log_effect": frame["striking_power_v3"],
            "damage_durability": frame["damage_durability"],
            "knockdown_resistance_log_effect": frame["knockdown_resistance_v3"],
            "stamina_capacity": frame["stamina_capacity"],
            "stamina_depletion_resistance": frame["stamina_depletion_resistance"],
        }
    )
    payload = {
        "rows": len(frame),
        "historical_key_duplicates": int(
            frame.duplicated(["event_date", "fight_id", "fighter_id"]).sum()
        ),
        "traits": {column: _summary(translated[column]) for column in translated},
        "correlations": translated.corr().to_dict(),
    }
    if payload["historical_key_duplicates"]:
        raise AssertionError("duplicate historical prefight physiology keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
