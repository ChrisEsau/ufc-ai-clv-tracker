# ============================================================
# pipeline/betting/run_moneyline_outcome_sanity_check.py
# ============================================================

"""Diagnose moneyline outcome probability/odds alignment.

This read-only diagnostic is intended to answer a narrower question than the
join-key diagnostic:

    Are model probabilities, market odds, and outcome fighter identities aligned
    on the joined moneyline rows?

It does not alter betting outcomes. It loads the same model/market inputs used by
Betting Outcomes V2, joins them on the canonical V2 key, and writes an inspectable
parquet audit with row-level and fight-pair-level sanity flags.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.betting.betting_joiner import prepare_market_outcomes, prepare_model_predictions
from pipeline.betting.betting_schema import JOIN_KEYS
from pipeline.betting.run_betting_outcomes_v2 import (
    DEFAULT_REGISTRY_PATH,
    _load_model_outcomes,
)
from pipeline.common.paths import AUDITS_DIR, MARKET_OUTCOMES_PATH, ensure_data_dirs

DEFAULT_OUTPUT_PATH = AUDITS_DIR / "ufc_moneyline_outcome_sanity_check.parquet"

OUTPUT_COLUMNS = [
    "sanity_run_id",
    "sanity_timestamp",
    "fight_id",
    "event_name",
    "red_fighter",
    "blue_fighter",
    "bookmaker",
    "market_key",
    "outcome_join_key",
    "outcome_label_model",
    "outcome_label_market",
    "outcome_fighter_id_model",
    "outcome_fighter_id_market",
    "fighter_id_match",
    "model_probability",
    "opponent_model_probability",
    "american_odds",
    "implied_probability",
    "edge",
    "abs_assigned_market_gap",
    "abs_opponent_market_gap",
    "opponent_probability_closer_to_market",
    "model_probability_sum",
    "market_implied_sum",
    "joined_moneyline_outcomes_in_group",
    "model_favorite_label",
    "market_favorite_label",
    "model_favorite_is_market_underdog",
    "possible_pair_probability_inversion",
    "large_edge_flag",
    "sanity_status",
    "sanity_notes",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a moneyline outcome probability/odds sanity diagnostic."
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Model registry YAML used to load model-scoped predictions.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["production", "all", "single"],
        default="production",
        help="Same model mode used by Betting Outcomes V2.",
    )
    parser.add_argument(
        "--market-outcomes-path",
        default=str(MARKET_OUTCOMES_PATH),
        help="Market outcomes parquet path.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output parquet path for the diagnostic.",
    )
    parser.add_argument(
        "--large-edge-threshold",
        type=float,
        default=0.20,
        help="Absolute edge threshold used to flag unusually large model/market gaps.",
    )
    return parser.parse_args()


def _utc_run() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("moneyline_sanity_%Y%m%d_%H%M%S"), now.isoformat()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def _first_available(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return pd.NA


def _prepare_joined_moneyline(model_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    model = prepare_model_predictions(model_df)
    market = prepare_market_outcomes(market_df)

    model = model[model["market_key"].astype(str).str.lower() == "moneyline"].copy()
    market = market[market["market_key"].astype(str).str.lower() == "moneyline"].copy()

    joined = model.merge(
        market,
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_model", "_market"),
    )
    return joined


def _base_rows(joined: pd.DataFrame, *, sanity_run_id: str, sanity_timestamp: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        model_probability = pd.to_numeric(row.get("model_probability"), errors="coerce")
        implied_probability = pd.to_numeric(row.get("implied_probability"), errors="coerce")
        outcome_fighter_id_model = _first_available(row, "outcome_fighter_id_model", "outcome_fighter_id")
        outcome_fighter_id_market = _first_available(row, "outcome_fighter_id_market", "outcome_fighter_id")

        rows.append(
            {
                "sanity_run_id": sanity_run_id,
                "sanity_timestamp": sanity_timestamp,
                "fight_id": row.get("fight_id"),
                "event_name": _first_available(row, "event_name_model", "event_name_market", "event_name"),
                "red_fighter": _first_available(row, "red_fighter_model", "red_fighter_market", "red_fighter"),
                "blue_fighter": _first_available(row, "blue_fighter_model", "blue_fighter_market", "blue_fighter"),
                "bookmaker": _first_available(row, "bookmaker_market", "bookmaker"),
                "market_key": row.get("market_key"),
                "outcome_join_key": row.get("outcome_join_key"),
                "outcome_label_model": _first_available(row, "outcome_label_model", "outcome_label"),
                "outcome_label_market": _first_available(row, "outcome_label_market", "outcome_label"),
                "outcome_fighter_id_model": outcome_fighter_id_model,
                "outcome_fighter_id_market": outcome_fighter_id_market,
                "fighter_id_match": str(outcome_fighter_id_model) == str(outcome_fighter_id_market),
                "model_probability": model_probability,
                "american_odds": pd.to_numeric(row.get("american_odds"), errors="coerce"),
                "implied_probability": implied_probability,
                "edge": model_probability - implied_probability,
            }
        )
    return pd.DataFrame(rows)


def _attach_pair_diagnostics(rows: pd.DataFrame, *, large_edge_threshold: float) -> pd.DataFrame:
    if rows.empty:
        return rows

    out = rows.copy()
    pair_keys = ["fight_id", "bookmaker", "market_key"]

    # Defaults for groups that do not have exactly two moneyline outcomes.
    out["opponent_model_probability"] = pd.NA
    out["abs_assigned_market_gap"] = (out["model_probability"] - out["implied_probability"]).abs()
    out["abs_opponent_market_gap"] = pd.NA
    out["opponent_probability_closer_to_market"] = False
    out["model_probability_sum"] = pd.NA
    out["market_implied_sum"] = pd.NA
    out["joined_moneyline_outcomes_in_group"] = 0
    out["model_favorite_label"] = pd.NA
    out["market_favorite_label"] = pd.NA
    out["model_favorite_is_market_underdog"] = False
    out["possible_pair_probability_inversion"] = False
    out["large_edge_flag"] = out["edge"].abs().ge(float(large_edge_threshold))
    out["sanity_status"] = "review"
    out["sanity_notes"] = ""

    for _, group in out.groupby(pair_keys, dropna=False):
        idx = group.index.tolist()
        group_size = len(group)
        out.loc[idx, "joined_moneyline_outcomes_in_group"] = group_size

        if group_size != 2:
            out.loc[idx, "sanity_notes"] = f"Expected 2 joined moneyline outcomes for pair; found {group_size}."
            continue

        model_probs = pd.to_numeric(group["model_probability"], errors="coerce")
        implied_probs = pd.to_numeric(group["implied_probability"], errors="coerce")
        out.loc[idx, "model_probability_sum"] = float(model_probs.sum())
        out.loc[idx, "market_implied_sum"] = float(implied_probs.sum())

        first_idx, second_idx = idx
        first_prob = out.at[first_idx, "model_probability"]
        second_prob = out.at[second_idx, "model_probability"]
        out.at[first_idx, "opponent_model_probability"] = second_prob
        out.at[second_idx, "opponent_model_probability"] = first_prob

        out.loc[idx, "abs_opponent_market_gap"] = (
            pd.to_numeric(out.loc[idx, "opponent_model_probability"], errors="coerce")
            - pd.to_numeric(out.loc[idx, "implied_probability"], errors="coerce")
        ).abs()
        out.loc[idx, "opponent_probability_closer_to_market"] = (
            pd.to_numeric(out.loc[idx, "abs_opponent_market_gap"], errors="coerce")
            < pd.to_numeric(out.loc[idx, "abs_assigned_market_gap"], errors="coerce")
        )

        model_fav_idx = model_probs.idxmax()
        market_fav_idx = implied_probs.idxmax()
        model_fav_label = out.at[model_fav_idx, "outcome_label_model"]
        market_fav_label = out.at[market_fav_idx, "outcome_label_market"]
        out.loc[idx, "model_favorite_label"] = model_fav_label
        out.loc[idx, "market_favorite_label"] = market_fav_label
        out.loc[idx, "model_favorite_is_market_underdog"] = model_fav_idx != market_fav_idx

        possible_inversion = bool(out.loc[idx, "opponent_probability_closer_to_market"].all())
        out.loc[idx, "possible_pair_probability_inversion"] = possible_inversion

        notes: list[str] = []
        if not bool(out.loc[idx, "fighter_id_match"].all()):
            notes.append("Model and market outcome fighter IDs do not all match.")
        if possible_inversion:
            notes.append("Opponent model probabilities are closer to market probabilities for both outcomes.")
        if bool(out.loc[idx, "large_edge_flag"].any()):
            notes.append("At least one outcome has a large model-vs-market edge.")
        if bool(out.loc[idx, "model_favorite_is_market_underdog"].iloc[0]):
            notes.append("Model favorite is market underdog.")

        if notes:
            out.loc[idx, "sanity_status"] = "review"
            out.loc[idx, "sanity_notes"] = " ".join(notes)
        else:
            out.loc[idx, "sanity_status"] = "pass"
            out.loc[idx, "sanity_notes"] = "No row-level alignment issue detected."

    return out


def build_moneyline_sanity_check(
    *,
    model_df: pd.DataFrame,
    market_df: pd.DataFrame,
    sanity_run_id: str,
    sanity_timestamp: str,
    large_edge_threshold: float,
) -> pd.DataFrame:
    joined = _prepare_joined_moneyline(model_df, market_df)
    rows = _base_rows(joined, sanity_run_id=sanity_run_id, sanity_timestamp=sanity_timestamp)
    rows = _attach_pair_diagnostics(rows, large_edge_threshold=large_edge_threshold)

    for column in OUTPUT_COLUMNS:
        if column not in rows.columns:
            rows[column] = pd.NA
    return rows[OUTPUT_COLUMNS]


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()
    sanity_run_id, sanity_timestamp = _utc_run()

    print("=" * 80)
    print("MONEYLINE OUTCOME SANITY CHECK")
    print("=" * 80)
    print("Sanity run ID:", sanity_run_id)
    print("Model mode:", args.model_mode)
    print("Registry path:", args.registry_path)
    print("Market outcomes path:", args.market_outcomes_path)
    print("Output path:", args.output_path)

    model_df, selected_models = _load_model_outcomes(
        registry_path=Path(args.registry_path),
        model_mode=args.model_mode,
    )
    market_df = _load_required_parquet(Path(args.market_outcomes_path), "Market outcomes")

    sanity_df = build_moneyline_sanity_check(
        model_df=model_df,
        market_df=market_df,
        sanity_run_id=sanity_run_id,
        sanity_timestamp=sanity_timestamp,
        large_edge_threshold=float(args.large_edge_threshold),
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanity_df.to_parquet(output_path, index=False)

    print()
    print("========== MONEYLINE SANITY SUMMARY ==========")
    print("Selected models:", ", ".join(str(row.get("model_id")) for row in selected_models))
    print("Model rows:", len(model_df))
    print("Market rows:", len(market_df))
    print("Joined moneyline rows:", len(sanity_df))
    if not sanity_df.empty:
        print("Unique joined fights:", sanity_df["fight_id"].nunique(dropna=True))
        print("Sanity status counts:")
        print(sanity_df["sanity_status"].value_counts(dropna=False).to_string())
        print("Possible pair inversions:", int(sanity_df["possible_pair_probability_inversion"].fillna(False).sum()))
        print("Fighter ID mismatches:", int((~sanity_df["fighter_id_match"].fillna(False)).sum()))
        print("Large-edge rows:", int(sanity_df["large_edge_flag"].fillna(False).sum()))
        preview_cols = [
            "fight_id",
            "outcome_label_model",
            "outcome_label_market",
            "model_probability",
            "american_odds",
            "implied_probability",
            "edge",
            "possible_pair_probability_inversion",
            "sanity_notes",
        ]
        print("Review preview:")
        print(sanity_df[sanity_df["sanity_status"] == "review"][preview_cols].head(20).to_string(index=False))
    print()
    print("File saved:", output_path)


if __name__ == "__main__":
    main()
