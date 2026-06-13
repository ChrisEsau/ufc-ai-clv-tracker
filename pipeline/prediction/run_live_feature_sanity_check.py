# ============================================================
# pipeline/prediction/run_live_feature_sanity_check.py
# ============================================================

"""Inspect live red-minus-blue feature orientation for suspicious moneyline rows.

This read-only diagnostic combines:

* live model features from data/predictions/live_model_features.parquet
* model outcomes from data/predictions/model_outcomes.parquet
* betting outcomes from data/predictions/betting_outcomes.parquet
* moneyline sanity flags from data/audits/ufc_moneyline_outcome_sanity_check.parquet

The goal is to determine whether large market/model disagreements are caused by
live feature orientation/state problems or by legitimate model disagreement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import AUDITS_DIR, BETTING_OUTCOMES_PATH, PREDICTIONS_DIR, ensure_data_dirs

DEFAULT_LIVE_FEATURES_PATH = PREDICTIONS_DIR / "live_model_features.parquet"
DEFAULT_MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"
DEFAULT_MONEYLINE_SANITY_PATH = AUDITS_DIR / "ufc_moneyline_outcome_sanity_check.parquet"
DEFAULT_OUTPUT_PATH = AUDITS_DIR / "ufc_live_feature_sanity_check.parquet"

KEY_FEATURES = [
    "elo_diff",
    "ewm_elo_diff",
    "avg_opponent_elo_diff",
    "ewm_avg_opponent_elo_diff",
    "best_win_elo_diff",
    "worst_loss_elo_diff",
    "recent_form_elo_diff",
    "recent_form_win_pct_diff",
    "win_pct_diff",
    "wins_diff",
    "losses_diff",
    "age_diff",
    "reach_diff",
    "height_diff",
    "weight_diff",
    "striking_edge",
    "grappling_edge",
    "wrestling_mismatch_diff",
    "submission_mismatch_diff",
    "chin_risk_diff",
]

OUTPUT_BASE_COLUMNS = [
    "diagnostic_run_id",
    "diagnostic_timestamp",
    "fight_id",
    "event_name",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "red_feature_match",
    "blue_feature_match",
    "feature_match_type",
    "feature_count_expected",
    "feature_count_actual",
    "nonzero_feature_count",
    "zero_feature_pct",
    "passes_feature_validation",
    "passes_model_data_quality",
    "red_model_probability",
    "blue_model_probability",
    "model_favorite_side",
    "model_favorite_name",
    "red_american_odds",
    "blue_american_odds",
    "red_implied_probability",
    "blue_implied_probability",
    "market_favorite_side",
    "market_favorite_name",
    "model_favorite_is_market_underdog",
    "possible_pair_probability_inversion",
    "large_edge_flag",
    "review_reason",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build live feature sanity diagnostic.")
    parser.add_argument("--live-features-path", default=str(DEFAULT_LIVE_FEATURES_PATH))
    parser.add_argument("--model-outcomes-path", default=str(DEFAULT_MODEL_OUTCOMES_PATH))
    parser.add_argument("--betting-outcomes-path", default=str(BETTING_OUTCOMES_PATH))
    parser.add_argument("--moneyline-sanity-path", default=str(DEFAULT_MONEYLINE_SANITY_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Only output fights flagged for review by the moneyline sanity diagnostic.",
    )
    return parser.parse_args()


def _utc_run() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("live_feature_sanity_%Y%m%d_%H%M%S"), now.isoformat()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def _to_float(value: Any) -> float | None:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return None
    return float(converted)


def _side_from_outcome(row: pd.Series) -> str | None:
    outcome_side = str(row.get("outcome_side", "")).strip().lower()
    if outcome_side in {"red", "r", "positive"}:
        return "red"
    if outcome_side in {"blue", "b", "negative"}:
        return "blue"

    outcome_fighter_id = str(row.get("outcome_fighter_id", "")).strip()
    red_id = str(row.get("red_fighter_id", "")).strip()
    blue_id = str(row.get("blue_fighter_id", "")).strip()
    if outcome_fighter_id and outcome_fighter_id == red_id:
        return "red"
    if outcome_fighter_id and outcome_fighter_id == blue_id:
        return "blue"
    return None


def _summarize_model_outcomes(model_df: pd.DataFrame) -> pd.DataFrame:
    model = model_df[model_df.get("market_key", "").astype(str).str.lower() == "moneyline"].copy()
    if model.empty:
        return pd.DataFrame(columns=["fight_id", "red_model_probability", "blue_model_probability"])

    model["_side"] = model.apply(_side_from_outcome, axis=1)
    model["_prob"] = pd.to_numeric(model.get("model_probability"), errors="coerce")

    pivot = model.pivot_table(index="fight_id", columns="_side", values="_prob", aggfunc="first").reset_index()
    pivot.columns.name = None
    if "red" not in pivot.columns:
        pivot["red"] = pd.NA
    if "blue" not in pivot.columns:
        pivot["blue"] = pd.NA
    return pivot.rename(columns={"red": "red_model_probability", "blue": "blue_model_probability"})[
        ["fight_id", "red_model_probability", "blue_model_probability"]
    ]


def _summarize_betting_outcomes(betting_df: pd.DataFrame) -> pd.DataFrame:
    betting = betting_df[betting_df.get("market_key", "").astype(str).str.lower() == "moneyline"].copy()
    if betting.empty:
        return pd.DataFrame(
            columns=[
                "fight_id",
                "red_american_odds",
                "blue_american_odds",
                "red_implied_probability",
                "blue_implied_probability",
            ]
        )

    betting["_side"] = betting.apply(_side_from_outcome, axis=1)
    betting["_american_odds"] = pd.to_numeric(betting.get("american_odds"), errors="coerce")
    betting["_implied_probability"] = pd.to_numeric(betting.get("implied_probability"), errors="coerce")

    odds = betting.pivot_table(index="fight_id", columns="_side", values="_american_odds", aggfunc="first").reset_index()
    odds.columns.name = None
    implied = betting.pivot_table(index="fight_id", columns="_side", values="_implied_probability", aggfunc="first").reset_index()
    implied.columns.name = None

    for frame in [odds, implied]:
        if "red" not in frame.columns:
            frame["red"] = pd.NA
        if "blue" not in frame.columns:
            frame["blue"] = pd.NA

    odds = odds.rename(columns={"red": "red_american_odds", "blue": "blue_american_odds"})
    implied = implied.rename(columns={"red": "red_implied_probability", "blue": "blue_implied_probability"})
    return odds[["fight_id", "red_american_odds", "blue_american_odds"]].merge(
        implied[["fight_id", "red_implied_probability", "blue_implied_probability"]],
        on="fight_id",
        how="outer",
    )


def _summarize_moneyline_sanity(sanity_df: pd.DataFrame) -> pd.DataFrame:
    if sanity_df.empty or "fight_id" not in sanity_df.columns:
        return pd.DataFrame(
            columns=[
                "fight_id",
                "possible_pair_probability_inversion",
                "large_edge_flag",
                "review_reason",
            ]
        )

    grouped = sanity_df.groupby("fight_id", dropna=False).agg(
        possible_pair_probability_inversion=("possible_pair_probability_inversion", lambda s: bool(s.fillna(False).any())),
        large_edge_flag=("large_edge_flag", lambda s: bool(s.fillna(False).any())),
        review_reason=("sanity_notes", lambda s: " | ".join(sorted(set(str(v) for v in s.dropna() if str(v).strip())))),
    )
    return grouped.reset_index()


def _favorite_side(red_probability: Any, blue_probability: Any) -> str | None:
    red = _to_float(red_probability)
    blue = _to_float(blue_probability)
    if red is None or blue is None:
        return None
    if red > blue:
        return "red"
    if blue > red:
        return "blue"
    return "tie"


def _side_name(row: pd.Series, side: str | None) -> str | None:
    if side == "red":
        return row.get("red_fighter")
    if side == "blue":
        return row.get("blue_fighter")
    return None


def _add_feature_interpretation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature in KEY_FEATURES:
        if feature not in out.columns:
            out[feature] = pd.NA

    feature_value_columns = [f"feature__{feature}" for feature in KEY_FEATURES]
    for feature, output_col in zip(KEY_FEATURES, feature_value_columns):
        out[output_col] = pd.to_numeric(out[feature], errors="coerce")

    positive_counts = []
    negative_counts = []
    zero_counts = []
    for _, row in out.iterrows():
        values = pd.to_numeric(row[feature_value_columns], errors="coerce").dropna()
        positive_counts.append(int((values > 0).sum()))
        negative_counts.append(int((values < 0).sum()))
        zero_counts.append(int((values == 0).sum()))

    out["key_feature_positive_count"] = positive_counts
    out["key_feature_negative_count"] = negative_counts
    out["key_feature_zero_count"] = zero_counts
    out["key_feature_direction"] = out.apply(
        lambda row: "red" if row["key_feature_positive_count"] > row["key_feature_negative_count"]
        else "blue" if row["key_feature_negative_count"] > row["key_feature_positive_count"]
        else "mixed_or_neutral",
        axis=1,
    )
    return out


def build_live_feature_sanity(
    *,
    live_features_df: pd.DataFrame,
    model_outcomes_df: pd.DataFrame,
    betting_outcomes_df: pd.DataFrame,
    moneyline_sanity_df: pd.DataFrame,
    diagnostic_run_id: str,
    diagnostic_timestamp: str,
    review_only: bool,
) -> pd.DataFrame:
    base_cols = [
        "fight_id",
        "event_name",
        "red_fighter",
        "blue_fighter",
        "red_fighter_id",
        "blue_fighter_id",
        "red_feature_match",
        "blue_feature_match",
        "feature_match_type",
        "feature_count_expected",
        "feature_count_actual",
        "nonzero_feature_count",
        "zero_feature_pct",
        "passes_feature_validation",
        "passes_model_data_quality",
    ]
    for col in base_cols:
        if col not in live_features_df.columns:
            live_features_df[col] = pd.NA

    out = live_features_df[base_cols + [col for col in KEY_FEATURES if col in live_features_df.columns]].copy()
    out = out.merge(_summarize_model_outcomes(model_outcomes_df), on="fight_id", how="left")
    out = out.merge(_summarize_betting_outcomes(betting_outcomes_df), on="fight_id", how="left")
    out = out.merge(_summarize_moneyline_sanity(moneyline_sanity_df), on="fight_id", how="left")

    out["model_favorite_side"] = out.apply(
        lambda row: _favorite_side(row.get("red_model_probability"), row.get("blue_model_probability")),
        axis=1,
    )
    out["model_favorite_name"] = out.apply(lambda row: _side_name(row, row.get("model_favorite_side")), axis=1)
    out["market_favorite_side"] = out.apply(
        lambda row: _favorite_side(row.get("red_implied_probability"), row.get("blue_implied_probability")),
        axis=1,
    )
    out["market_favorite_name"] = out.apply(lambda row: _side_name(row, row.get("market_favorite_side")), axis=1)
    out["model_favorite_is_market_underdog"] = (
        out["model_favorite_side"].notna()
        & out["market_favorite_side"].notna()
        & out["model_favorite_side"].ne(out["market_favorite_side"])
    )

    out["possible_pair_probability_inversion"] = out["possible_pair_probability_inversion"].fillna(False).astype(bool)
    out["large_edge_flag"] = out["large_edge_flag"].fillna(False).astype(bool)
    out["review_reason"] = out["review_reason"].fillna("")
    out = _add_feature_interpretation(out)

    out.insert(0, "diagnostic_timestamp", diagnostic_timestamp)
    out.insert(0, "diagnostic_run_id", diagnostic_run_id)

    if review_only:
        out = out[
            out["possible_pair_probability_inversion"]
            | out["large_edge_flag"]
            | out["model_favorite_is_market_underdog"]
        ].copy()

    ordered = OUTPUT_BASE_COLUMNS + [
        "key_feature_positive_count",
        "key_feature_negative_count",
        "key_feature_zero_count",
        "key_feature_direction",
    ] + [f"feature__{feature}" for feature in KEY_FEATURES]
    for column in ordered:
        if column not in out.columns:
            out[column] = pd.NA
    return out[ordered]


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()
    diagnostic_run_id, diagnostic_timestamp = _utc_run()

    print("=" * 80)
    print("LIVE FEATURE SANITY CHECK")
    print("=" * 80)
    print("Diagnostic run ID:", diagnostic_run_id)
    print("Live features path:", args.live_features_path)
    print("Model outcomes path:", args.model_outcomes_path)
    print("Betting outcomes path:", args.betting_outcomes_path)
    print("Moneyline sanity path:", args.moneyline_sanity_path)
    print("Output path:", args.output_path)

    live_features_df = _load_required_parquet(Path(args.live_features_path), "Live features")
    model_outcomes_df = _load_required_parquet(Path(args.model_outcomes_path), "Model outcomes")
    betting_outcomes_df = _load_required_parquet(Path(args.betting_outcomes_path), "Betting outcomes")
    moneyline_sanity_df = _load_required_parquet(Path(args.moneyline_sanity_path), "Moneyline sanity")

    diagnostic_df = build_live_feature_sanity(
        live_features_df=live_features_df,
        model_outcomes_df=model_outcomes_df,
        betting_outcomes_df=betting_outcomes_df,
        moneyline_sanity_df=moneyline_sanity_df,
        diagnostic_run_id=diagnostic_run_id,
        diagnostic_timestamp=diagnostic_timestamp,
        review_only=bool(args.review_only),
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_df.to_parquet(output_path, index=False)

    print()
    print("========== LIVE FEATURE SANITY SUMMARY ==========")
    print("Rows:", len(diagnostic_df))
    print("Review-only:", bool(args.review_only))
    if not diagnostic_df.empty:
        print("Possible pair inversions:", int(diagnostic_df["possible_pair_probability_inversion"].fillna(False).sum()))
        print("Model favorite market underdog:", int(diagnostic_df["model_favorite_is_market_underdog"].fillna(False).sum()))
        print("Feature direction counts:")
        print(diagnostic_df["key_feature_direction"].value_counts(dropna=False).to_string())
        preview_cols = [
            "fight_id",
            "red_fighter",
            "blue_fighter",
            "model_favorite_name",
            "market_favorite_name",
            "key_feature_direction",
            "feature__elo_diff",
            "feature__ewm_elo_diff",
            "feature__recent_form_win_pct_diff",
            "feature__age_diff",
            "possible_pair_probability_inversion",
            "review_reason",
        ]
        print("Review preview:")
        print(diagnostic_df[preview_cols].head(20).to_string(index=False))
    print()
    print("File saved:", output_path)


if __name__ == "__main__":
    main()
