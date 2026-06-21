"""Build the standalone style-clustering research dataset.

Run from the repository root:

    python -m pipeline.research.style_matchups.build_style_dataset

This runner is intentionally research-only. It reads the fighter-state history
artifact, keeps only point-in-time fighter snapshots with enough prior UFC
history, selects style-relevant columns, and writes a clean parquet dataset for
clustering experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = "pipeline/research/style_matchups/style_config.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the research dataset builder."""

    parser = argparse.ArgumentParser(description="Build UFC style-matchup research dataset.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the style-matchup research YAML config.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the style-matchup research config."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Style research config not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Style research config must be a mapping: {path}")

    required_blocks = ["inputs", "outputs", "filters", "identity_columns", "style_columns"]
    missing_blocks = [block for block in required_blocks if block not in config]
    if missing_blocks:
        raise ValueError(f"Style research config missing blocks: {missing_blocks}")

    input_path = (config.get("inputs") or {}).get("fighter_state_history_path")
    output_path = (config.get("outputs") or {}).get("style_fighter_snapshots_path")
    if not input_path:
        raise ValueError("Style research config missing inputs.fighter_state_history_path")
    if not output_path:
        raise ValueError("Style research config missing outputs.style_fighter_snapshots_path")

    return config


def _existing_columns(df: pd.DataFrame, requested_columns: list[str]) -> list[str]:
    """Return requested columns that are present in the dataframe."""

    return [column for column in requested_columns if column in df.columns]


def _missing_columns(df: pd.DataFrame, requested_columns: list[str]) -> list[str]:
    """Return requested columns that are absent from the dataframe."""

    return [column for column in requested_columns if column not in df.columns]


def build_style_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Build the style research dataframe from fighter-state history."""

    input_path = Path(str(config["inputs"]["fighter_state_history_path"]))
    if not input_path.exists():
        raise FileNotFoundError(f"Fighter-state history artifact not found: {input_path}")

    min_fights = int((config.get("filters") or {}).get("min_fights", 0))
    identity_columns = [str(column) for column in config.get("identity_columns", [])]
    style_columns = [str(column) for column in config.get("style_columns", [])]
    allow_missing = bool((config.get("missingness") or {}).get("allow_missing_style_columns", False))

    print("=" * 80)
    print("BUILD UFC STYLE MATCHUP RESEARCH DATASET")
    print("=" * 80)
    print(f"Input path : {input_path}")
    print(f"Min fights : {min_fights}")

    history_df = pd.read_parquet(input_path)
    print(f"Input shape: {history_df.shape}")

    if "fights" not in history_df.columns:
        raise ValueError("Fighter-state history is missing required filter column: fights")

    missing_style_columns = _missing_columns(history_df, style_columns)
    if missing_style_columns and not allow_missing:
        raise ValueError(f"Missing configured style columns: {missing_style_columns}")
    if missing_style_columns:
        print(f"Missing style columns skipped: {missing_style_columns}")

    available_identity_columns = _existing_columns(history_df, identity_columns)
    available_style_columns = _existing_columns(history_df, style_columns)
    if not available_style_columns:
        raise ValueError("No configured style columns were found in fighter-state history.")

    fight_count_filter = pd.to_numeric(history_df["fights"], errors="coerce").fillna(0) >= min_fights
    filtered_history_df = history_df.loc[fight_count_filter].copy()

    output_columns = list(dict.fromkeys([*available_identity_columns, *available_style_columns]))
    style_df = filtered_history_df.loc[:, output_columns].copy()

    # Convert style signal columns to numeric so clustering can consume them
    # directly in the next research step.
    for column in available_style_columns:
        style_df[column] = pd.to_numeric(style_df[column], errors="coerce")

    style_df = style_df.sort_values(
        [column for column in ["fighter_id", "fight_date", "source_row_index", "fight_id"] if column in style_df.columns]
    ).reset_index(drop=True)

    print(f"Output shape: {style_df.shape}")
    print(f"Style columns used: {len(available_style_columns)}")
    print(f"Identity columns used: {len(available_identity_columns)}")

    missing_rate = style_df[available_style_columns].isna().mean().sort_values(ascending=False)
    print("Top style missingness rates:")
    for column, rate in missing_rate.head(10).items():
        print(f"  {column}: {rate:.4f}")

    return style_df


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    output_path = Path(str(config["outputs"]["style_fighter_snapshots_path"]))

    style_df = build_style_dataset(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    style_df.to_parquet(output_path, index=False)
    print(f"Saved style research dataset: {output_path}")
    print("DONE")


if __name__ == "__main__":
    main()
