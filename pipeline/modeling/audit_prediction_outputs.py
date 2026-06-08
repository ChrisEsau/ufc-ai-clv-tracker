from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.common.paths import LIVE_CARD_PATH, LIVE_FEATURE_AUDIT_PATH, PREDICTIONS_DIR


DEFAULT_MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"
DEFAULT_LIVE_FEATURES_PATH = PREDICTIONS_DIR / "live_model_features.parquet"


class PredictionAuditError(RuntimeError):
    """Raised when Prediction V2 outputs cannot be audited."""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Prediction V2 output artifacts.",
    )
    parser.add_argument(
        "--live-card-path",
        default=str(LIVE_CARD_PATH),
        help="Path to raw live card parquet.",
    )
    parser.add_argument(
        "--model-outcomes-path",
        default=str(DEFAULT_MODEL_OUTCOMES_PATH),
        help="Path to model_outcomes.parquet.",
    )
    parser.add_argument(
        "--live-features-path",
        default=str(DEFAULT_LIVE_FEATURES_PATH),
        help="Path to live_model_features.parquet.",
    )
    parser.add_argument(
        "--feature-audit-path",
        default=str(LIVE_FEATURE_AUDIT_PATH),
        help="Path to live feature audit parquet.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of rows/model picks to print.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    live_card_path = Path(args.live_card_path)
    model_outcomes_path = Path(args.model_outcomes_path)
    live_features_path = Path(args.live_features_path)
    feature_audit_path = Path(args.feature_audit_path)

    print("=" * 80)
    print("PREDICTION V2 OUTPUT AUDIT")
    print("=" * 80)

    if live_card_path.exists():
        live_card = pd.read_parquet(live_card_path)
        print("\n" + "=" * 80)
        print("RAW LIVE CARD REVIEW")
        print("=" * 80)
        print(f"Live card path: {live_card_path}")
        _audit_live_card(live_card, top_n=args.top_n)
    else:
        print(f"\nLive card not found: {live_card_path}")

    outcomes = _read_required_parquet(model_outcomes_path, "model outcomes")
    print("\n" + "=" * 80)
    print("MODEL OUTCOMES")
    print("=" * 80)
    print(f"Model outcomes path: {model_outcomes_path}")
    _audit_outcomes(outcomes, top_n=args.top_n)

    if live_features_path.exists():
        live_features = pd.read_parquet(live_features_path)
        print("\n" + "=" * 80)
        print("LIVE FEATURE OUTPUT")
        print("=" * 80)
        _audit_live_features(live_features)
    else:
        print(f"\nLive features not found: {live_features_path}")

    if feature_audit_path.exists():
        feature_audit = pd.read_parquet(feature_audit_path)
        print("\n" + "=" * 80)
        print("LIVE FEATURE AUDIT")
        print("=" * 80)
        _audit_feature_audit(feature_audit)
    else:
        print(f"\nFeature audit not found: {feature_audit_path}")

    print("\nAudit complete.")



def _read_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise PredictionAuditError(f"Missing {label}: {path}")
    return pd.read_parquet(path)



def _audit_live_card(df: pd.DataFrame, *, top_n: int) -> None:
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate column names: {int(df.columns.duplicated().sum())}")
    print(f"Unique fights: {_safe_nunique(df, 'fight_id')}")
    print(f"Unique events: {_safe_nunique(df, 'event_id')}")

    print("\nColumns:")
    print(", ".join(str(column) for column in df.columns))

    identity_candidates = [
        "event_id",
        "event_name",
        "commence_time",
        "date",
        "fight_id",
        "red_fighter_id",
        "blue_fighter_id",
        "r_id",
        "b_id",
        "red_fighter",
        "blue_fighter",
        "r_name",
        "b_name",
        "fighter_1",
        "fighter_2",
    ]
    present_identity = [column for column in identity_candidates if column in df.columns]
    if present_identity:
        print("\nIdentity/display column null or blank counts:")
        for column in present_identity:
            print(f"  {column}: {_missing_or_blank_count(df[column])}")

    for column in ["event_name", "event_id"]:
        if column in df.columns:
            print(f"\nTop {column} counts:")
            print(df[column].value_counts(dropna=False).head(top_n).to_string())

    if "fight_id" in df.columns:
        duplicate_fight_rows = df[df["fight_id"].duplicated(keep=False)].copy()
        print(f"\nDuplicate fight_id rows: {len(duplicate_fight_rows)}")
        if len(duplicate_fight_rows) > 0:
            display_columns = [
                column for column in [
                    "event_name",
                    "event_id",
                    "commence_time",
                    "date",
                    "fight_id",
                    "red_fighter",
                    "blue_fighter",
                    "r_name",
                    "b_name",
                    "red_fighter_id",
                    "blue_fighter_id",
                    "r_id",
                    "b_id",
                ]
                if column in duplicate_fight_rows.columns
            ]
            print("\nDuplicate fight_id sample:")
            print(duplicate_fight_rows[display_columns].head(top_n).to_string(index=False))

    id_pairs = _resolve_live_card_id_columns(df)
    if id_pairs:
        red_id_col, blue_id_col = id_pairs
        missing_red = _missing_or_blank_mask(df[red_id_col])
        missing_blue = _missing_or_blank_mask(df[blue_id_col])
        missing_either = missing_red | missing_blue
        print(f"\nUsing live-card ID columns: red={red_id_col}, blue={blue_id_col}")
        print(f"Missing red IDs: {int(missing_red.sum())}")
        print(f"Missing blue IDs: {int(missing_blue.sum())}")
        print(f"Rows missing either fighter ID: {int(missing_either.sum())}")

        if missing_either.any():
            display_columns = [
                column for column in [
                    "event_name",
                    "event_id",
                    "commence_time",
                    "date",
                    "fight_id",
                    "red_fighter",
                    "blue_fighter",
                    "r_name",
                    "b_name",
                    red_id_col,
                    blue_id_col,
                ]
                if column in df.columns
            ]
            print("\nRows missing fighter IDs:")
            print(df.loc[missing_either, display_columns].head(top_n).to_string(index=False))
    else:
        print("\nNo recognized red/blue fighter ID columns found in live card.")

    name_pairs = _resolve_live_card_name_columns(df)
    if name_pairs:
        red_name_col, blue_name_col = name_pairs
        missing_red_name = _missing_or_blank_mask(df[red_name_col])
        missing_blue_name = _missing_or_blank_mask(df[blue_name_col])
        missing_either_name = missing_red_name | missing_blue_name
        print(f"\nUsing live-card name columns: red={red_name_col}, blue={blue_name_col}")
        print(f"Missing red names: {int(missing_red_name.sum())}")
        print(f"Missing blue names: {int(missing_blue_name.sum())}")
        print(f"Rows missing either fighter name: {int(missing_either_name.sum())}")



def _audit_outcomes(df: pd.DataFrame, *, top_n: int) -> None:
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique fights: {_safe_nunique(df, 'fight_id')}")
    print(f"Unique events: {_safe_nunique(df, 'event_id')}")
    print(f"Models: {_safe_unique(df, 'model_id')}")
    print(f"Markets: {_safe_unique(df, 'market_key')}")

    _print_missing_required(df, required_columns=[
        "prediction_run_id",
        "model_id",
        "fight_id",
        "market_key",
        "outcome_label",
        "model_probability",
        "is_model_pick",
        "model_confidence",
    ])

    if "model_probability" in df.columns:
        probability = pd.to_numeric(df["model_probability"], errors="coerce")
        print("\nProbability summary:")
        print(probability.describe().to_string())
        print(f"Probability nulls: {int(probability.isna().sum())}")
        print(f"Probability < 0: {int((probability < 0).sum())}")
        print(f"Probability > 1: {int((probability > 1).sum())}")

    if "outcome_label" in df.columns:
        labels = df["outcome_label"].astype("string").fillna("").str.strip()
        print(f"Blank outcome labels: {int(labels.eq('').sum())}")
        print(f"Fallback fighter_id labels: {int(labels.str.startswith('fighter_id:').sum())}")

    if "is_model_pick" in df.columns:
        picks = df[df["is_model_pick"].astype(bool)].copy()
        print(f"\nModel pick rows: {len(picks)}")
        print(f"Expected pick rows if one pick per fight: {_safe_nunique(df, 'fight_id')}")

        display_columns = [
            "event_name",
            "fight_id",
            "red_fighter",
            "blue_fighter",
            "outcome_label",
            "model_probability",
            "model_confidence",
            "passes_model_data_quality",
            "passes_feature_validation",
        ]
        display_columns = [column for column in display_columns if column in picks.columns]
        if display_columns:
            picks = picks.sort_values("model_confidence", ascending=False) if "model_confidence" in picks.columns else picks
            print(f"\nTop {top_n} model picks:")
            print(picks[display_columns].head(top_n).to_string(index=False))

    duplicate_keys = [
        column for column in ["prediction_run_id", "model_id", "fight_id", "market_key", "outcome_label"]
        if column in df.columns
    ]
    if duplicate_keys:
        duplicate_count = int(df.duplicated(subset=duplicate_keys).sum())
        print(f"\nDuplicate outcome keys: {duplicate_count}")



def _audit_live_features(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate column names: {int(df.columns.duplicated().sum())}")
    print(f"Unique fights: {_safe_nunique(df, 'fight_id')}")

    for column in [
        "feature_count_expected",
        "feature_count_actual",
        "nonzero_feature_count",
        "zero_feature_pct",
    ]:
        if column in df.columns:
            print(f"\n{column} summary:")
            print(pd.to_numeric(df[column], errors="coerce").describe().to_string())

    for column in ["passes_feature_validation", "passes_model_data_quality", "feature_match_type"]:
        if column in df.columns:
            print(f"\n{column} counts:")
            print(df[column].value_counts(dropna=False).to_string())



def _audit_feature_audit(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique fights: {_safe_nunique(df, 'fight_id')}")

    for column in ["red_feature_match", "blue_feature_match", "feature_match_type"]:
        if column in df.columns:
            print(f"\n{column} counts:")
            print(df[column].value_counts(dropna=False).to_string())

    for column in ["passes_feature_validation", "passes_model_data_quality"]:
        if column in df.columns:
            print(f"\n{column} counts:")
            print(df[column].value_counts(dropna=False).to_string())

    if {"event_name", "red_fighter", "blue_fighter", "zero_feature_pct"}.issubset(df.columns):
        display = df.sort_values("zero_feature_pct", ascending=False)[[
            "event_name",
            "red_fighter",
            "blue_fighter",
            "zero_feature_pct",
            "feature_match_type",
        ]]
        print("\nHighest zero-feature percentage rows:")
        print(display.head(25).to_string(index=False))



def _print_missing_required(df: pd.DataFrame, *, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print(f"\nMissing required columns: {missing_columns}")
        return

    print("\nRequired column null/blank counts:")
    for column in required_columns:
        print(f"  {column}: {_missing_or_blank_count(df[column])}")



def _resolve_live_card_id_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    red_candidates = ["red_fighter_id", "r_id", "red_id"]
    blue_candidates = ["blue_fighter_id", "b_id", "blue_id"]
    red = _first_existing(df, red_candidates)
    blue = _first_existing(df, blue_candidates)
    if red and blue:
        return red, blue
    return None



def _resolve_live_card_name_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    red_candidates = ["red_fighter", "r_name", "red_name", "fighter_1", "fighter_a"]
    blue_candidates = ["blue_fighter", "b_name", "blue_name", "fighter_2", "fighter_b"]
    red = _first_existing(df, red_candidates)
    blue = _first_existing(df, blue_candidates)
    if red and blue:
        return red, blue
    return None



def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None



def _missing_or_blank_count(series: pd.Series) -> int:
    return int(_missing_or_blank_mask(series).sum())



def _missing_or_blank_mask(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("").str.strip()
    return series.isna() | values.eq("") | values.str.lower().isin({"nan", "none", "null", "nat"})



def _safe_nunique(df: pd.DataFrame, column: str) -> int | str:
    if column not in df.columns:
        return "missing"
    return int(df[column].nunique(dropna=True))



def _safe_unique(df: pd.DataFrame, column: str) -> list[str] | str:
    if column not in df.columns:
        return "missing"
    return sorted(str(value) for value in df[column].dropna().unique())



if __name__ == "__main__":
    main()
