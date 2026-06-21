"""Build continuous style-score research features from locked k=5 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = "pipeline/research/style_matchups/style_config.yaml"
DEFAULT_WEIGHTS_PATH = "data/research/style_matchups/style_score_weights.yaml"

EDGE_PAIRS = {
    "style_edge_ko_finisher_vs_decision_technician": ("ko_finisher", "decision_technician"),
    "style_edge_decision_technician_vs_submission_grappler": ("decision_technician", "submission_grappler"),
    "style_edge_control_wrestler_vs_ko_finisher": ("control_wrestler", "ko_finisher"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build continuous style score features.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must be a mapping: {p}")
    return data


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return ((values - values.mean()) / std).fillna(0.0)


def build_scores(snapshots: pd.DataFrame, weights_payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    style_weights = weights_payload.get("style_score_weights") or {}
    if not style_weights:
        raise ValueError("No style_score_weights found.")

    out = snapshots.copy()
    needed_features = sorted({feature for spec in style_weights.values() for feature in (spec.get("weights") or {})})
    missing = [feature for feature in needed_features if feature not in out.columns]
    if missing:
        raise ValueError(f"Missing weighted style features: {missing}")

    z = {feature: zscore(out[feature]) for feature in needed_features}
    score_cols: list[str] = []
    for style_name, spec in style_weights.items():
        score_col = f"{style_name}_score"
        score = pd.Series(0.0, index=out.index)
        for feature, weight in (spec.get("weights") or {}).items():
            score += float(weight) * z[feature]
        out[score_col] = score
        score_cols.append(score_col)

    out["primary_style"] = out[score_cols].idxmax(axis=1).str.replace("_score", "", regex=False)
    out["primary_style_score"] = out[score_cols].max(axis=1)
    out["style_score_spread"] = out[score_cols].max(axis=1) - out[score_cols].min(axis=1)
    return out, score_cols


def make_side(df: pd.DataFrame, side: str, score_cols: list[str]) -> pd.DataFrame:
    keep = ["fight_id", "fighter_id", "primary_style", "primary_style_score", "style_score_spread"] + score_cols
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    rename = {
        "fighter_id": f"{side}_id",
        "primary_style": f"{side}_primary_style",
        "primary_style_score": f"{side}_primary_style_score",
        "style_score_spread": f"{side}_style_score_spread",
    }
    rename.update({col: f"{side}_{col}" for col in score_cols})
    return out.rename(columns=rename)


def build_matchups(scores: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    if "corner" not in scores.columns:
        raise ValueError("corner column required.")
    corner = scores["corner"].astype(str).str.lower()
    red = make_side(scores[corner.isin(["red", "r"])], "red", score_cols)
    blue = make_side(scores[corner.isin(["blue", "b"])], "blue", score_cols)
    out = red.merge(blue, on="fight_id", how="inner")

    for score_col in score_cols:
        style = score_col.replace("_score", "")
        out[f"{style}_score_diff"] = out[f"red_{score_col}"] - out[f"blue_{score_col}"]
        out[f"{style}_score_abs_diff"] = out[f"{style}_score_diff"].abs()

    out["primary_style_matchup"] = out["red_primary_style"] + "_vs_" + out["blue_primary_style"]
    out["same_primary_style"] = out["red_primary_style"] == out["blue_primary_style"]

    for feature_name, (red_style, blue_style) in EDGE_PAIRS.items():
        direct = out[f"red_{red_style}_score"] * out[f"blue_{blue_style}_score"]
        reverse = out[f"red_{blue_style}_score"] * out[f"blue_{red_style}_score"]
        out[feature_name] = direct
        out[f"{feature_name}_reverse"] = reverse
        out[f"{feature_name}_net"] = direct - reverse
    return out


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    weights = load_yaml(args.weights_path)
    outputs = config.get("outputs") or {}
    research_dir = Path(str(outputs.get("research_dir", "data/research/style_matchups")))
    snapshots_path = Path(str(outputs.get("style_fighter_snapshots_path", research_dir / "style_fighter_snapshots.parquet")))

    snapshots = pd.read_parquet(snapshots_path)
    scores, score_cols = build_scores(snapshots, weights)
    keep_ids = [c for c in (config.get("identity_columns") or []) if c in scores.columns]
    score_out = scores[keep_ids + ["primary_style", "primary_style_score", "style_score_spread"] + score_cols].copy()
    matchups = build_matchups(score_out, score_cols)

    research_dir.mkdir(parents=True, exist_ok=True)
    score_out.to_parquet(research_dir / "style_fighter_scores.parquet", index=False)
    score_out.head(5000).to_csv(research_dir / "style_fighter_scores.csv", index=False)
    matchups.to_parquet(research_dir / "style_matchup_score_features.parquet", index=False)
    matchups.head(5000).to_csv(research_dir / "style_matchup_score_features.csv", index=False)
    (research_dir / "style_score_feature_summary.json").write_text(
        json.dumps(
            {
                "fighter_score_rows": int(len(score_out)),
                "matchup_feature_rows": int(len(matchups)),
                "score_columns": score_cols,
                "edge_pairs": EDGE_PAIRS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("DONE")


if __name__ == "__main__":
    main()
