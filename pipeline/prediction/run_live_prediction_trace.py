from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import AUDITS_DIR, PREDICTIONS_DIR

DEFAULT_LIVE_FEATURES_PATH = PREDICTIONS_DIR / "live_model_features.parquet"
DEFAULT_MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"
DEFAULT_OUTPUT_PATH = AUDITS_DIR / "ufc_live_prediction_trace.parquet"
DEFAULT_PREVIEW_PATH = AUDITS_DIR / "ufc_live_prediction_trace_preview.csv"

KEY_FEATURES = [
    "splm",
    "sapm",
    "td_avg",
    "sub_avg",
    "ctrl_per_min",
    "ctrl_against_per_min",
    "ewm_splm",
    "ewm_sapm",
    "ewm_td_avg",
    "ewm_sub_avg",
    "recent_splm",
    "recent_sapm",
    "recent_td_avg",
    "recent_form_win_pct",
    "elo",
    "ewm_elo",
    "win_pct",
    "ewm_win_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace live prediction inputs and outcome probabilities for a fight.")
    parser.add_argument("--fighter-a", default="Michael Chandler")
    parser.add_argument("--fighter-b", default="Mauricio Ruffy")
    parser.add_argument("--live-features-path", default=str(DEFAULT_LIVE_FEATURES_PATH))
    parser.add_argument("--model-outcomes-path", default=str(DEFAULT_MODEL_OUTCOMES_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-path", default=str(DEFAULT_PREVIEW_PATH))
    return parser.parse_args()


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def contains_fighter(row: pd.Series, fighter: str) -> bool:
    target = norm(fighter)
    name_cols = [
        "red_fighter",
        "blue_fighter",
        "r_name",
        "b_name",
        "fighter_name",
        "outcome_label",
        "outcome_label_model",
    ]
    return any(target in norm(row.get(col)) for col in name_cols if col in row.index)


def filter_fight(df: pd.DataFrame, fighter_a: str, fighter_b: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask_a = df.apply(lambda row: contains_fighter(row, fighter_a), axis=1)
    mask_b = df.apply(lambda row: contains_fighter(row, fighter_b), axis=1)
    return df.loc[mask_a & mask_b].copy()


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def build_feature_trace(live_row: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for base in KEY_FEATURES:
        red_col = first_existing(
            live_row.to_frame().T,
            [f"r_state_{base}", f"r_pre_{base}", f"r_{base}"],
        )
        blue_col = first_existing(
            live_row.to_frame().T,
            [f"b_state_{base}", f"b_pre_{base}", f"b_{base}"],
        )
        diff_col = f"{base}_diff"
        rows.append(
            {
                "row_type": "feature",
                "feature": base,
                "red_value": live_row.get(red_col) if red_col else pd.NA,
                "blue_value": live_row.get(blue_col) if blue_col else pd.NA,
                "diff_value": live_row.get(diff_col) if diff_col in live_row.index else pd.NA,
                "red_source_column": red_col or "",
                "blue_source_column": blue_col or "",
                "diff_source_column": diff_col if diff_col in live_row.index else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    live_features_path = Path(args.live_features_path)
    model_outcomes_path = Path(args.model_outcomes_path)

    if not live_features_path.exists():
        raise FileNotFoundError(f"Live features file not found: {live_features_path}")
    if not model_outcomes_path.exists():
        raise FileNotFoundError(f"Model outcomes file not found: {model_outcomes_path}")

    live_features = pd.read_parquet(live_features_path)
    model_outcomes = pd.read_parquet(model_outcomes_path)

    live_match = filter_fight(live_features, args.fighter_a, args.fighter_b)
    if live_match.empty:
        raise ValueError(
            f"No live feature rows matched fighter_a={args.fighter_a!r} and fighter_b={args.fighter_b!r}. "
            f"Available columns: {list(live_features.columns)}"
        )

    fight_ids = set(live_match["fight_id"].astype(str)) if "fight_id" in live_match.columns else set()
    outcome_match = model_outcomes.copy()
    if fight_ids and "fight_id" in outcome_match.columns:
        outcome_match = outcome_match[outcome_match["fight_id"].astype(str).isin(fight_ids)].copy()
    else:
        outcome_match = filter_fight(model_outcomes, args.fighter_a, args.fighter_b)

    live_row = live_match.iloc[0]
    feature_trace = build_feature_trace(live_row)

    context_cols = [
        "event_name",
        "fight_id",
        "red_fighter",
        "blue_fighter",
        "red_fighter_id",
        "blue_fighter_id",
        "feature_match_type",
        "feature_count_expected",
        "feature_count_actual",
        "nonzero_feature_count",
        "zero_feature_pct",
    ]
    context = {col: live_row.get(col) for col in context_cols if col in live_row.index}
    for key, value in context.items():
        feature_trace[key] = value

    outcome_rows = []
    outcome_cols = [
        "model_id",
        "prediction_run_id",
        "event_name",
        "fight_id",
        "outcome_label",
        "outcome_label_model",
        "fighter_id",
        "outcome_fighter_id",
        "model_probability",
        "raw_probability",
        "confidence_score",
        "model_confidence",
    ]
    for _, row in outcome_match.iterrows():
        outcome_rows.append(
            {
                "row_type": "outcome",
                "feature": "model_outcome",
                **{col: row.get(col) for col in outcome_cols if col in row.index},
            }
        )
    outcome_trace = pd.DataFrame(outcome_rows)

    combined = pd.concat([feature_trace, outcome_trace], ignore_index=True, sort=False)

    output_path = Path(args.output_path)
    preview_path = Path(args.preview_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    combined.to_csv(preview_path, index=False)

    print("=" * 80)
    print("LIVE PREDICTION TRACE")
    print("=" * 80)
    print("Fighter A:", args.fighter_a)
    print("Fighter B:", args.fighter_b)
    print("Live features path:", live_features_path)
    print("Model outcomes path:", model_outcomes_path)
    print("Live feature rows:", len(live_features))
    print("Matched live rows:", len(live_match))
    print("Matched outcome rows:", len(outcome_match))
    print()
    print("========== LIVE FEATURE CONTEXT ==========")
    for key, value in context.items():
        print(f"{key}: {value}")
    print()
    print("========== KEY FEATURE TRACE ==========")
    print(feature_trace[["feature", "red_value", "blue_value", "diff_value", "red_source_column", "blue_source_column", "diff_source_column"]].to_string(index=False))
    print()
    print("========== MODEL OUTCOMES ==========")
    if outcome_match.empty:
        print("No matching model outcome rows found.")
    else:
        print(outcome_match[[col for col in outcome_cols if col in outcome_match.columns]].to_string(index=False))
    print()
    print("Saved trace:", output_path)
    print("Saved preview:", preview_path)


if __name__ == "__main__":
    main()
