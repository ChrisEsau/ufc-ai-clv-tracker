from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.common.paths import PREDICTIONS_DIR
from pipeline.modeling.model_config import load_model_config
from pipeline.modeling.model_loader import load_model_bundle
from pipeline.modeling.model_registry import get_model_entry, load_model_registry, resolve_selected_model_id


DEFAULT_MODEL_FAMILY = "moneyline"
DEFAULT_LIVE_FEATURES_PATH = PREDICTIONS_DIR / "live_model_features.parquet"


class FeatureInspectionError(RuntimeError):
    """Raised when live fight features cannot be inspected."""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect zero/nonzero model features for a live fight row.",
    )
    parser.add_argument(
        "--fighter",
        required=True,
        help="Fighter name substring to locate in red_fighter or blue_fighter.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Explicit model ID. Overrides UFC_MODEL_ID and registry active model.",
    )
    parser.add_argument(
        "--model-family",
        default=DEFAULT_MODEL_FAMILY,
        help="Model family used when --model-id is not supplied.",
    )
    parser.add_argument(
        "--market-key",
        default=None,
        help="Optional market key for market-scoped active model resolution, e.g. goes_distance.",
    )
    parser.add_argument(
        "--registry-path",
        default="configs/models/model_registry.yaml",
        help="Path to model registry YAML.",
    )
    parser.add_argument(
        "--live-features-path",
        default=str(DEFAULT_LIVE_FEATURES_PATH),
        help="Path to live_model_features.parquet.",
    )
    parser.add_argument(
        "--show-zero",
        type=int,
        default=80,
        help="Number of zero features to print.",
    )
    parser.add_argument(
        "--show-nonzero",
        type=int,
        default=80,
        help="Number of nonzero features to print.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    live_features_path = Path(args.live_features_path)
    if not live_features_path.exists():
        raise FeatureInspectionError(f"Live features not found: {live_features_path}")

    live_features = pd.read_parquet(live_features_path)
    row = _find_fight_row(live_features, args.fighter)

    registry = load_model_registry(args.registry_path)
    model_id = resolve_selected_model_id(
        model_family=args.model_family,
        registry=registry,
        model_id=args.model_id,
        market_key=args.market_key,
    )
    model_entry = get_model_entry(model_id, registry)
    model_config = load_model_config(Path(model_entry["config_path"]), require_prediction=True)
    model_bundle = load_model_bundle(model_config)
    feature_columns = model_bundle.feature_columns

    missing_columns = [column for column in feature_columns if column not in live_features.columns]
    if missing_columns:
        raise FeatureInspectionError(f"Live features missing model columns: {missing_columns[:25]}")

    feature_values = row[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    zero_features = feature_values[feature_values == 0.0].sort_index()
    nonzero_features = feature_values[feature_values != 0.0].sort_values(key=lambda s: s.abs(), ascending=False)

    print("=" * 80)
    print("LIVE FIGHT FEATURE INSPECTION")
    print("=" * 80)
    print(f"Model family: {args.model_family}")
    print(f"Market key: {args.market_key or 'registry primary'}")
    print(f"Model ID: {model_id}")
    print(f"Live features path: {live_features_path}")
    print(f"Event: {row.get('event_name', '')}")
    print(f"Fight ID: {row.get('fight_id', '')}")
    print(f"Red: {row.get('red_fighter', '')} ({row.get('red_fighter_id', '')})")
    print(f"Blue: {row.get('blue_fighter', '')} ({row.get('blue_fighter_id', '')})")

    print("\nFeature QA:")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Nonzero features: {len(nonzero_features)}")
    print(f"Zero features: {len(zero_features)}")
    print(f"Feature completeness: {len(nonzero_features) / len(feature_columns):.3f}")
    for column in [
        "nonzero_feature_count",
        "zero_feature_pct",
        "feature_match_type",
        "passes_model_data_quality",
        "passes_feature_validation",
    ]:
        if column in row.index:
            print(f"{column}: {row.get(column)}")

    family_summary = _build_family_summary(feature_values)
    print("\nFeature family summary:")
    print(family_summary.to_string(index=False))

    print(f"\nTop {args.show_nonzero} nonzero model features by absolute value:")
    print(_series_to_frame(nonzero_features.head(args.show_nonzero), "feature", "value").to_string(index=False))

    print(f"\nFirst {args.show_zero} zero model features:")
    print(_series_to_frame(zero_features.head(args.show_zero), "feature", "value").to_string(index=False))



def _find_fight_row(live_features: pd.DataFrame, fighter_query: str) -> pd.Series:
    query = fighter_query.lower().strip()
    if not query:
        raise FeatureInspectionError("--fighter cannot be blank.")

    red = live_features.get("red_fighter", pd.Series("", index=live_features.index)).astype("string").fillna("").str.lower()
    blue = live_features.get("blue_fighter", pd.Series("", index=live_features.index)).astype("string").fillna("").str.lower()
    matches = live_features[red.str.contains(query, regex=False) | blue.str.contains(query, regex=False)].copy()

    if matches.empty:
        raise FeatureInspectionError(f"No live feature row found for fighter query: {fighter_query}")

    if len(matches) > 1:
        print("Multiple rows matched; using first row. Matches:")
        display_columns = [column for column in ["event_name", "fight_id", "red_fighter", "blue_fighter"] if column in matches.columns]
        print(matches[display_columns].to_string(index=False))

    return matches.iloc[0]



def _build_family_summary(feature_values: pd.Series) -> pd.DataFrame:
    rows = []
    for feature, value in feature_values.items():
        family = _feature_family(feature)
        rows.append({"family": family, "feature": feature, "is_nonzero": float(value) != 0.0})

    df = pd.DataFrame(rows)
    summary = df.groupby("family", dropna=False).agg(
        total_features=("feature", "count"),
        nonzero_features=("is_nonzero", "sum"),
    ).reset_index()
    summary["zero_features"] = summary["total_features"] - summary["nonzero_features"]
    summary["completeness"] = summary["nonzero_features"] / summary["total_features"]
    summary = summary.sort_values(["completeness", "total_features"], ascending=[True, False])
    return summary



def _feature_family(feature: str) -> str:
    lower = feature.lower()
    if "elo" in lower:
        return "elo"
    if "ewm" in lower:
        return "ewm"
    if "streak" in lower or "win_rate" in lower or "recent" in lower:
        return "recent_form"
    if "age" in lower or "height" in lower or "reach" in lower or "stance" in lower:
        return "physical"
    if "sig" in lower or "str" in lower or "strike" in lower or "sapm" in lower or "slpm" in lower:
        return "striking"
    if "td" in lower or "takedown" in lower or "sub" in lower or "grappl" in lower:
        return "grappling"
    if "ko" in lower or "finish" in lower or "dec" in lower or "round" in lower or "time" in lower:
        return "finish_duration"
    if lower.startswith("r_") or lower.startswith("b_"):
        return "side_raw"
    if "diff" in lower:
        return "diff"
    return "other"



def _series_to_frame(series: pd.Series, name_column: str, value_column: str) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=[name_column, value_column])
    return series.rename_axis(name_column).reset_index(name=value_column)



if __name__ == "__main__":
    main()
