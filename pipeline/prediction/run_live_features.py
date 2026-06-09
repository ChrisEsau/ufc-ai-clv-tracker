from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.modeling.model_config import load_model_config
from pipeline.modeling.model_loader import load_model_bundle
from pipeline.modeling.model_registry import (
    get_model_entry,
    load_model_registry,
    resolve_selected_model_id,
)
from pipeline.prediction.live_feature_builder import (
    build_live_model_features,
    write_live_feature_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build live model-ready features.")
    parser.add_argument("--model-family", default="moneyline")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--registry-path", default="configs/models/model_registry.yaml")
    parser.add_argument(
        "--live-feature-output-path",
        default="data/predictions/live_model_features.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_model_registry(args.registry_path)
    model_id = resolve_selected_model_id(
        model_family=args.model_family,
        registry=registry,
        model_id=args.model_id,
    )
    model_entry = get_model_entry(model_id, registry)
    model_config = load_model_config(Path(model_entry["config_path"]), require_prediction=True)
    model_bundle = load_model_bundle(model_config, prefer_calibrated=True)

    result = build_live_model_features(feature_columns=model_bundle.feature_columns)
    write_live_feature_outputs(
        result,
        live_feature_output_path=args.live_feature_output_path,
    )

    print("=" * 80)
    print("UFC LIVE FEATURES V2")
    print("=" * 80)
    print(f"Model ID: {model_id}")
    print(f"Feature count: {len(model_bundle.feature_columns)}")
    print(f"Live feature rows: {len(result.live_feature_df)}")
    print(f"Live feature output: {args.live_feature_output_path}")
    print("Live Features V2 complete.")


if __name__ == "__main__":
    main()
