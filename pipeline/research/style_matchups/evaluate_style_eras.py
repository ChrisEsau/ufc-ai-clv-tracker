"""Evaluate UFC style-matchup ROI by historical era.

Research-only runner. It consumes the already-built style_matchup_fights.parquet
artifact produced by evaluate_style_matchups.py and writes era-segmented ROI
reports under data/research/style_matchups/.

Example:
    python -m pipeline.research.style_matchups.evaluate_style_eras \
      --config pipeline/research/style_matchups/style_config.yaml \
      --min-fights 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = "pipeline/research/style_matchups/style_config.yaml"
DEFAULT_MIN_FIGHTS = 20
DEFAULT_ERAS = "2010-2016:2010-01-01:2016-12-31,2017-2020:2017-01-01:2020-12-31,2021-present:2021-01-01:2099-12-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UFC style matchup ROI by era.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--min-fights", type=int, default=DEFAULT_MIN_FIGHTS)
    parser.add_argument(
        "--eras",
        default=DEFAULT_ERAS,
        help="Comma-separated era specs in label:start_date:end_date format.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Style research config not found: {p}")
    config = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Style research config must be a mapping: {p}")
    return config


def parse_eras(raw: str) -> list[dict[str, Any]]:
    eras: list[dict[str, Any]] = []
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec:
            continue
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid era spec: {spec}. Expected label:start_date:end_date")
        label, start, end = parts
        eras.append({"era": label, "start": pd.Timestamp(start), "end": pd.Timestamp(end)})
    if not eras:
        raise ValueError("At least one era spec is required.")
    return eras


def american_return(odds: float, stake: float = 100.0) -> float:
    if pd.isna(odds) or odds == 0:
        return float("nan")
    return stake * odds / 100.0 if odds > 0 else stake * 100.0 / abs(odds)


def assign_era(fights: pd.DataFrame, eras: list[dict[str, Any]]) -> pd.Series:
    labels = pd.Series(pd.NA, index=fights.index, dtype="object")
    fight_dates = pd.to_datetime(fights["date"], errors="coerce")
    for era in eras:
        mask = fight_dates.between(era["start"], era["end"], inclusive="both")
        labels.loc[mask] = era["era"]
    return labels


def add_profit_columns(fights: pd.DataFrame) -> pd.DataFrame:
    required = {"r_odds", "b_odds", "red_win", "blue_win"}
    missing = sorted(required.difference(fights.columns))
    if missing:
        raise ValueError(f"style_matchup_fights missing columns required for ROI: {missing}")

    out = fights.copy()
    out["r_odds"] = pd.to_numeric(out["r_odds"], errors="coerce")
    out["b_odds"] = pd.to_numeric(out["b_odds"], errors="coerce")
    out["red_win"] = pd.to_numeric(out["red_win"], errors="coerce")
    out["blue_win"] = pd.to_numeric(out["blue_win"], errors="coerce")
    out = out.dropna(subset=["r_odds", "b_odds", "red_win", "blue_win"])

    stake = 100.0
    out["red_profit"] = [
        american_return(odds, stake) if win == 1 else -stake
        for odds, win in zip(out["r_odds"], out["red_win"])
    ]
    out["blue_profit"] = [
        american_return(odds, stake) if win == 1 else -stake
        for odds, win in zip(out["b_odds"], out["blue_win"])
    ]
    out["red_is_underdog"] = out["r_odds"] > out["b_odds"]
    out["underdog_won"] = out["red_win"].where(out["red_is_underdog"], out["blue_win"])
    dog_odds = out["r_odds"].where(out["red_is_underdog"], out["b_odds"])
    out["underdog_profit"] = [
        american_return(odds, stake) if win == 1 else -stake
        for odds, win in zip(dog_odds, out["underdog_won"])
    ]
    return out


def build_era_roi_report(fights: pd.DataFrame, min_fights: int) -> pd.DataFrame:
    grouped = fights.groupby(["era", "style_matchup_key"], dropna=False)
    report = grouped.agg(
        fights=("fight_id", "count"),
        red_win_rate=("red_win", "mean"),
        avg_r_odds=("r_odds", "mean"),
        avg_b_odds=("b_odds", "mean"),
        flat_bet_red_profit=("red_profit", "sum"),
        flat_bet_blue_profit=("blue_profit", "sum"),
        underdog_win_rate=("underdog_won", "mean"),
        underdog_profit=("underdog_profit", "sum"),
        red_underdog_rate=("red_is_underdog", "mean"),
    ).reset_index()
    stake = 100.0
    report["flat_bet_red_roi"] = report["flat_bet_red_profit"] / (report["fights"] * stake)
    report["flat_bet_blue_roi"] = report["flat_bet_blue_profit"] / (report["fights"] * stake)
    report["underdog_roi"] = report["underdog_profit"] / (report["fights"] * stake)
    report = report[report["fights"] >= min_fights].copy()
    return report.sort_values(["era", "flat_bet_red_roi", "fights"], ascending=[True, False, False])


def build_era_summary(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame()
    idx = report.groupby("era")["flat_bet_red_roi"].idxmax()
    best = report.loc[idx].copy().sort_values("era")
    return best[[
        "era",
        "style_matchup_key",
        "fights",
        "red_win_rate",
        "flat_bet_red_profit",
        "flat_bet_red_roi",
        "underdog_win_rate",
        "underdog_profit",
        "underdog_roi",
    ]]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    research_dir = Path(str((config.get("outputs") or {}).get("research_dir", "data/research/style_matchups")))
    fights_path = research_dir / "style_matchup_fights.parquet"
    if not fights_path.exists():
        raise FileNotFoundError(
            f"Style matchup fights not found: {fights_path}. Run evaluate_style_matchups first."
        )

    eras = parse_eras(args.eras)
    fights = pd.read_parquet(fights_path)
    if "date" not in fights.columns:
        raise ValueError("style_matchup_fights missing date column required for era analysis.")

    fights = add_profit_columns(fights)
    fights["era"] = assign_era(fights, eras)
    unmatched = int(fights["era"].isna().sum())
    fights = fights.dropna(subset=["era"]).copy()

    era_report = build_era_roi_report(fights, args.min_fights)
    era_summary = build_era_summary(era_report)

    report_path = research_dir / "style_matchup_era_roi_report.csv"
    summary_csv_path = research_dir / "style_matchup_era_summary.csv"
    summary_json_path = research_dir / "style_matchup_era_summary.json"

    era_report.to_csv(report_path, index=False)
    era_summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(
        json.dumps(
            {
                "input_fights_path": str(fights_path),
                "eras": [
                    {"era": e["era"], "start": str(e["start"].date()), "end": str(e["end"].date())}
                    for e in eras
                ],
                "min_fights": int(args.min_fights),
                "matched_fights": int(len(fights)),
                "unmatched_fights": unmatched,
                "era_report_rows": int(len(era_report)),
                "summary_rows": int(len(era_summary)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved era ROI report: {report_path}")
    print(f"Saved era summary   : {summary_csv_path}")
    print(f"Saved summary json  : {summary_json_path}")
    print("DONE")


if __name__ == "__main__":
    main()
