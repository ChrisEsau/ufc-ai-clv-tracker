from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from pipeline.common.paths import (
    AUDITS_DIR,
    CURRENT_FIGHTER_FEATURES_PATH,
    LIVE_CARD_PATH,
    PREDICTIONS_DIR,
)
from pipeline.features.views.moneyline import get_state_feature_columns
from pipeline.modeling.model_loader import load_model_bundle
from pipeline.modeling.model_config import load_model_config
from pipeline.prediction.live_feature_builder import build_live_model_features
from ufc_feature_engineering import add_v5_engineered_features

DEFAULT_MODEL_CONFIG = Path("configs/models/moneyline_xgboost_v6_dev.yaml")
DEFAULT_OUTPUT_PATH = AUDITS_DIR / "ufc_live_feature_formula_parity.parquet"
DEFAULT_PREVIEW_PATH = AUDITS_DIR / "ufc_live_feature_formula_parity_preview.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare live bridge features to moneyline feature-view formulas.")
    parser.add_argument("--model-config-path", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--live-card-path", default=str(LIVE_CARD_PATH))
    parser.add_argument("--current-fighter-features-path", default=str(CURRENT_FIGHTER_FEATURES_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-path", default=str(DEFAULT_PREVIEW_PATH))
    parser.add_argument("--fighter-name-contains", default="")
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args()


def _load_model_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a dict: {path}")
    return config


def _standardize_current_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    id_col = None
    for candidate in ["fighter_id", "id", "ufcstats_fighter_id"]:
        if candidate in out.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError("Current fighter features missing fighter ID column.")
    if id_col != "fighter_id":
        out["fighter_id"] = out[id_col]
    out["fighter_id"] = out["fighter_id"].astype(str)
    return out


def _build_formula_features(live_card_df: pd.DataFrame, current_state_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    state_df = _standardize_current_state(current_state_df)
    state_columns = get_state_feature_columns(state_df)

    red = state_df[["fighter_id", *state_columns]].copy()
    blue = state_df[["fighter_id", *state_columns]].copy()

    red = red.rename(columns={"fighter_id": "red_fighter_id"})
    blue = blue.rename(columns={"fighter_id": "blue_fighter_id"})

    red_rename = {}
    blue_rename = {}
    for column in state_columns:
        if column.startswith("ewm_"):
            red_rename[column] = f"r_{column}"
            blue_rename[column] = f"b_{column}"
        elif column.startswith("form_delta_"):
            base = column.replace("form_delta_", "", 1)
            red_rename[column] = f"r_recent_form_{base}"
            blue_rename[column] = f"b_recent_form_{base}"
        else:
            red_rename[column] = f"r_pre_{column}"
            blue_rename[column] = f"b_pre_{column}"

    red = red.rename(columns=red_rename)
    blue = blue.rename(columns=blue_rename)

    out = live_card_df.copy()
    out["red_fighter_id"] = out["red_fighter_id"].astype(str)
    out["blue_fighter_id"] = out["blue_fighter_id"].astype(str)
    out = out.merge(red, on="red_fighter_id", how="left")
    out = out.merge(blue, on="blue_fighter_id", how="left")

    new_cols = {}
    for column in state_columns:
        if column.startswith("ewm_"):
            stat = column.replace("ewm_", "", 1)
            r_col = f"r_ewm_{stat}"
            b_col = f"b_ewm_{stat}"
            diff_col = f"ewm_{stat}_diff"
        elif column.startswith("form_delta_"):
            stat = column.replace("form_delta_", "", 1)
            r_col = f"r_recent_form_{stat}"
            b_col = f"b_recent_form_{stat}"
            diff_col = f"recent_form_{stat}_diff"
        else:
            r_col = f"r_pre_{column}"
            b_col = f"b_pre_{column}"
            diff_col = f"{column}_diff"

        if r_col in out.columns and b_col in out.columns:
            new_cols[diff_col] = pd.to_numeric(out[r_col], errors="coerce").fillna(0.0) - pd.to_numeric(out[b_col], errors="coerce").fillna(0.0)

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    out = add_v5_engineered_features(out)
    missing = [feature for feature in feature_columns if feature not in out.columns]
    if missing:
        raise ValueError(f"Formula feature frame missing model features: {missing}")
    return out


def _maybe_filter_fights(df: pd.DataFrame, text: str) -> pd.DataFrame:
    if not text:
        return df
    needle = text.lower()
    mask = (
        df.get("red_fighter", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(needle, na=False)
        | df.get("blue_fighter", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(needle, na=False)
        | df.get("r_name", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(needle, na=False)
        | df.get("b_name", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(needle, na=False)
    )
    return df[mask].copy()


def main() -> None:
    args = parse_args()
    model_config = _load_model_config(Path(args.model_config_path))
    model_bundle = load_model_bundle(model_config)
    feature_columns = model_bundle.feature_columns

    live_result = build_live_model_features(
        feature_columns=feature_columns,
        live_card_path=args.live_card_path,
        current_fighter_features_path=args.current_fighter_features_path,
    )
    live_features = live_result.live_feature_df.copy()
    live_card = pd.read_parquet(args.live_card_path)
    current_state = pd.read_parquet(args.current_fighter_features_path)

    formula_features = _build_formula_features(live_card, current_state, feature_columns)

    live_features = _maybe_filter_fights(live_features, args.fighter_name_contains)
    formula_features = formula_features[formula_features["fight_id"].isin(live_features["fight_id"].astype(str))].copy()

    live_index = live_features.set_index("fight_id")
    formula_index = formula_features.set_index("fight_id")

    rows = []
    for fight_id in live_index.index:
        if fight_id not in formula_index.index:
            continue
        live_row = live_index.loc[fight_id]
        formula_row = formula_index.loc[fight_id]
        if isinstance(live_row, pd.DataFrame):
            live_row = live_row.iloc[0]
        if isinstance(formula_row, pd.DataFrame):
            formula_row = formula_row.iloc[0]
        for feature in feature_columns:
            live_value = pd.to_numeric(pd.Series([live_row.get(feature)]), errors="coerce").iloc[0]
            formula_value = pd.to_numeric(pd.Series([formula_row.get(feature)]), errors="coerce").iloc[0]
            live_value = 0.0 if pd.isna(live_value) else float(live_value)
            formula_value = 0.0 if pd.isna(formula_value) else float(formula_value)
            abs_diff = abs(live_value - formula_value)
            rows.append({
                "fight_id": fight_id,
                "event_name": live_row.get("event_name"),
                "red_fighter": live_row.get("red_fighter"),
                "blue_fighter": live_row.get("blue_fighter"),
                "feature": feature,
                "live_value": live_value,
                "formula_value": formula_value,
                "abs_diff": abs_diff,
                "matches": abs_diff < 1e-9,
            })

    audit = pd.DataFrame(rows).sort_values(["abs_diff", "fight_id", "feature"], ascending=[False, True, True])
    output_path = Path(args.output_path)
    preview_path = Path(args.preview_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(output_path, index=False)
    audit.head(args.top_n).to_csv(preview_path, index=False)

    print("=" * 80)
    print("LIVE FEATURE FORMULA PARITY")
    print("=" * 80)
    print("Model ID:", model_bundle.model_id)
    print("Feature count:", len(feature_columns))
    print("Live rows checked:", live_features["fight_id"].nunique())
    print("Audit rows:", len(audit))
    print("Mismatched feature rows:", int((~audit["matches"]).sum()) if not audit.empty else 0)
    print("Top differences:")
    print(audit.head(args.top_n).to_string(index=False))
    print("Saved audit:", output_path)
    print("Saved preview:", preview_path)


if __name__ == "__main__":
    main()
