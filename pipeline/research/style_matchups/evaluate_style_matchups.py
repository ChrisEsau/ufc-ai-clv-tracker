"""Evaluate historical UFC style-cluster matchups.

Run from the repository root after style clustering:

    python -m pipeline.research.style_matchups.evaluate_style_matchups

This runner is research-only. It joins fighter-level style cluster assignments
back to historical fight rows and produces oriented matchup win-rate reports.
It does not modify production fighter-state, feature-view, model, or dashboard
code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = "pipeline/research/style_matchups/style_config.yaml"
DEFAULT_MIN_FIGHTS = 50


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the style matchup evaluator."""

    parser = argparse.ArgumentParser(description="Evaluate UFC style matchup history.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the style-matchup research YAML config.",
    )
    parser.add_argument(
        "--min-fights",
        type=int,
        default=DEFAULT_MIN_FIGHTS,
        help="Minimum fights required for edge-report rows.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the style-matchup research config."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Style research config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Style research config must be a mapping: {path}")
    return config


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present in the dataframe."""

    for column in candidates:
        if column in df.columns:
            return column
    return None


def _require_columns(df: pd.DataFrame, columns: list[str], *, label: str) -> None:
    """Raise if any required columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _style_label_frame(assignments_df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Return one side's fight-level style labels keyed for master join."""

    if side not in {"r", "b"}:
        raise ValueError(f"Unsupported side: {side}")

    _require_columns(
        assignments_df,
        ["fight_id", "fighter_id", "style_cluster", "style_cluster_label"],
        label="style_cluster_assignments",
    )
    out = assignments_df.loc[
        :,
        ["fight_id", "fighter_id", "style_cluster", "style_cluster_label"],
    ].copy()
    out = out.rename(
        columns={
            "fighter_id": f"{side}_id",
            "style_cluster": f"{side}_style_cluster",
            "style_cluster_label": f"{side}_style_label",
        }
    )
    return out


def build_fight_style_matchups(master_df: pd.DataFrame, assignments_df: pd.DataFrame) -> pd.DataFrame:
    """Join red/blue style cluster assignments onto master fight rows."""

    _require_columns(master_df, ["fight_id", "r_id", "b_id"], label="ufc_master")
    target_column = _first_existing(master_df, ["target", "red_win", "r_win"])
    if target_column is None:
        raise ValueError("ufc_master missing target column. Expected one of: target, red_win, r_win")

    red_styles = _style_label_frame(assignments_df, "r")
    blue_styles = _style_label_frame(assignments_df, "b")

    context_columns = [
        column
        for column in [
            "event_id",
            "event_name",
            "fight_id",
            "date",
            "r_id",
            "b_id",
            "r_name",
            "b_name",
            "division",
            "title_fight",
            "method",
            target_column,
        ]
        if column in master_df.columns
    ]
    fight_df = master_df.loc[:, context_columns].copy()
    fight_df = fight_df.merge(red_styles, on=["fight_id", "r_id"], how="left")
    fight_df = fight_df.merge(blue_styles, on=["fight_id", "b_id"], how="left")

    missing_style = fight_df[["r_style_cluster", "b_style_cluster"]].isna().any(axis=1)
    if missing_style.any():
        print(f"Rows missing style assignments: {int(missing_style.sum())}")
    fight_df = fight_df.loc[~missing_style].copy()

    fight_df["red_win"] = pd.to_numeric(fight_df[target_column], errors="coerce")
    fight_df = fight_df[fight_df["red_win"].isin([0, 1])].copy()
    fight_df["red_win"] = fight_df["red_win"].astype(int)
    fight_df["blue_win"] = 1 - fight_df["red_win"]

    fight_df["style_matchup_key"] = fight_df["r_style_label"] + "_vs_" + fight_df["b_style_label"]
    fight_df["reverse_style_matchup_key"] = fight_df["b_style_label"] + "_vs_" + fight_df["r_style_label"]
    fight_df["same_style_matchup"] = fight_df["r_style_label"] == fight_df["b_style_label"]
    return fight_df


def build_matchup_matrix(fight_df: pd.DataFrame) -> pd.DataFrame:
    """Build oriented style matchup win-rate matrix."""

    grouped = fight_df.groupby(["r_style_label", "b_style_label", "style_matchup_key"], dropna=False)
    matrix_df = grouped.agg(
        fights=("fight_id", "count"),
        red_wins=("red_win", "sum"),
        blue_wins=("blue_win", "sum"),
        red_win_rate=("red_win", "mean"),
        same_style_matchup=("same_style_matchup", "max"),
    ).reset_index()
    matrix_df["blue_win_rate"] = 1.0 - matrix_df["red_win_rate"]
    matrix_df = matrix_df.sort_values(["fights", "red_win_rate"], ascending=[False, False])
    return matrix_df


def build_edge_report(matrix_df: pd.DataFrame, *, baseline_red_win_rate: float, min_fights: int) -> pd.DataFrame:
    """Build an oriented edge report against the overall red-side baseline."""

    edge_df = matrix_df[matrix_df["fights"] >= min_fights].copy()
    edge_df["baseline_red_win_rate"] = baseline_red_win_rate
    edge_df["red_win_rate_edge"] = edge_df["red_win_rate"] - baseline_red_win_rate
    edge_df["abs_red_win_rate_edge"] = edge_df["red_win_rate_edge"].abs()
    edge_df = edge_df.sort_values(["abs_red_win_rate_edge", "fights"], ascending=[False, False])
    return edge_df


def build_underdog_report(fight_df: pd.DataFrame) -> pd.DataFrame:
    """Build underdog report when odds columns are available.

    The current master artifact may not contain historical odds. If no supported
    odds columns are present, return an empty explanatory dataframe instead of
    inventing ROI.
    """

    r_odds_col = _first_existing(fight_df, ["r_odds", "red_odds", "r_moneyline", "red_moneyline"])
    b_odds_col = _first_existing(fight_df, ["b_odds", "blue_odds", "b_moneyline", "blue_moneyline"])
    if r_odds_col is None or b_odds_col is None:
        return pd.DataFrame(
            [
                {
                    "status": "not_available",
                    "reason": "No supported historical odds columns found in matchup fight dataframe.",
                    "supported_red_columns": "r_odds|red_odds|r_moneyline|red_moneyline",
                    "supported_blue_columns": "b_odds|blue_odds|b_moneyline|blue_moneyline",
                }
            ]
        )

    odds_df = fight_df.copy()
    odds_df["r_odds_numeric"] = pd.to_numeric(odds_df[r_odds_col], errors="coerce")
    odds_df["b_odds_numeric"] = pd.to_numeric(odds_df[b_odds_col], errors="coerce")
    odds_df = odds_df.dropna(subset=["r_odds_numeric", "b_odds_numeric"])
    if odds_df.empty:
        return pd.DataFrame([{"status": "not_available", "reason": "Odds columns exist but contain no numeric odds."}])

    odds_df["red_is_underdog"] = odds_df["r_odds_numeric"] > odds_df["b_odds_numeric"]
    odds_df["underdog_won"] = odds_df["red_win"].where(odds_df["red_is_underdog"], odds_df["blue_win"])
    report = odds_df.groupby("style_matchup_key").agg(
        fights=("fight_id", "count"),
        underdog_win_rate=("underdog_won", "mean"),
        red_underdog_rate=("red_is_underdog", "mean"),
    ).reset_index()
    return report.sort_values(["underdog_win_rate", "fights"], ascending=[False, False])


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    outputs = config.get("outputs") or {}
    inputs = config.get("inputs") or {}

    research_dir = Path(str(outputs.get("research_dir", "data/research/style_matchups")))
    assignments_path = research_dir / "style_cluster_assignments.parquet"
    master_path = Path(str(inputs.get("master_path", "data/master/ufc_master.parquet")))

    fight_output_path = research_dir / "style_matchup_fights.parquet"
    matrix_path = research_dir / "style_matchup_matrix.csv"
    edge_path = research_dir / "style_matchup_edge_report.csv"
    underdog_path = research_dir / "style_matchup_underdogs.csv"
    summary_path = research_dir / "style_matchup_summary.json"

    if not assignments_path.exists():
        raise FileNotFoundError(f"Style cluster assignments not found: {assignments_path}")
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    print("=" * 80)
    print("EVALUATE UFC STYLE MATCHUPS")
    print("=" * 80)
    print(f"Master path     : {master_path}")
    print(f"Assignments path: {assignments_path}")
    print(f"Minimum fights  : {args.min_fights}")

    master_df = pd.read_parquet(master_path)
    assignments_df = pd.read_parquet(assignments_path)
    print(f"Master shape     : {master_df.shape}")
    print(f"Assignments shape: {assignments_df.shape}")

    fight_df = build_fight_style_matchups(master_df, assignments_df)
    matrix_df = build_matchup_matrix(fight_df)
    baseline_red_win_rate = float(fight_df["red_win"].mean()) if not fight_df.empty else 0.0
    edge_df = build_edge_report(matrix_df, baseline_red_win_rate=baseline_red_win_rate, min_fights=args.min_fights)
    underdog_df = build_underdog_report(fight_df)

    research_dir.mkdir(parents=True, exist_ok=True)
    fight_df.to_parquet(fight_output_path, index=False)
    matrix_df.to_csv(matrix_path, index=False)
    edge_df.to_csv(edge_path, index=False)
    underdog_df.to_csv(underdog_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "fight_count": int(len(fight_df)),
                "baseline_red_win_rate": baseline_red_win_rate,
                "matchup_count": int(len(matrix_df)),
                "edge_report_rows": int(len(edge_df)),
                "minimum_fights": int(args.min_fights),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved fight joins : {fight_output_path}")
    print(f"Saved matrix      : {matrix_path}")
    print(f"Saved edge report : {edge_path}")
    print(f"Saved underdogs   : {underdog_path}")
    print(f"Saved summary     : {summary_path}")
    print("DONE")


if __name__ == "__main__":
    main()
