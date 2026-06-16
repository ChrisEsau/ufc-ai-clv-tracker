from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.outcome_join import build_outcome_join_key
from pipeline.common.paths import MARKET_DIR, ensure_data_dirs

DEFAULT_MARKET_PATH = MARKET_DIR / "historical_market_outcomes.parquet"
SYNTHETIC_MARKETS = {"goes_distance", "inside_distance"}
METHOD_MARKETS = {"win_by_decision", "win_by_ko_tko_dq", "win_by_submission"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append synthetic goes_distance and inside_distance rows to historical market outcomes."
    )
    parser.add_argument("--market-path", default=str(DEFAULT_MARKET_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_MARKET_PATH))
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        default=True,
        help="Remove existing synthetic distance rows before appending regenerated rows.",
    )
    return parser.parse_args()


def american_to_implied_probability(odds: Any) -> float | None:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) == 0:
        return None
    value = float(value)
    return 100.0 / (value + 100.0) if value > 0 else abs(value) / (abs(value) + 100.0)


def implied_probability_to_american(probability: Any) -> float | None:
    prob = pd.to_numeric(pd.Series([probability]), errors="coerce").iloc[0]
    if pd.isna(prob):
        return None
    prob = float(prob)
    if prob <= 0.0 or prob >= 1.0:
        return None
    if prob >= 0.5:
        return round(-100.0 * prob / (1.0 - prob), 6)
    return round(100.0 * (1.0 - prob) / prob, 6)


def profit_per_100(odds: Any) -> float | None:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) == 0:
        return None
    value = float(value)
    return value if value > 0 else 10000.0 / abs(value)


def normalize_probability(probability: float | None) -> float | None:
    if probability is None or pd.isna(probability):
        return None
    probability = float(probability)
    if probability <= 0.0:
        return None
    return min(probability, 0.999999)


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else pd.NA


def is_decision_finish(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return "decision" in text or text in {"dec", "d"}


def build_synthetic_rows(market_df: pd.DataFrame, run_id: str, timestamp: str) -> pd.DataFrame:
    source = market_df[market_df["market_key"].isin(METHOD_MARKETS)].copy()
    if source.empty:
        return pd.DataFrame(columns=market_df.columns)

    grouped = source.groupby("fight_id", dropna=False)
    rows: list[dict[str, Any]] = []

    for fight_id, group in grouped:
        decision = group[group["market_key"] == "win_by_decision"].copy()
        ko_tko = group[group["market_key"] == "win_by_ko_tko_dq"].copy()
        submission = group[group["market_key"] == "win_by_submission"].copy()

        decision_prob = normalize_probability(decision["implied_probability"].sum(skipna=True))
        inside_prob = normalize_probability(
            pd.concat([ko_tko["implied_probability"], submission["implied_probability"]]).sum(skipna=True)
        )

        template = group.iloc[0]
        decision_won = bool(decision["won"].fillna(False).any()) if not decision.empty else None
        inside_won = bool(
            pd.concat([ko_tko["won"], submission["won"]]).fillna(False).any()
        ) if not ko_tko.empty or not submission.empty else None

        if decision_prob is not None:
            rows.append(
                build_synthetic_row(
                    template=template,
                    run_id=run_id,
                    timestamp=timestamp,
                    market_key="goes_distance",
                    outcome_label="goes_distance",
                    implied_probability=decision_prob,
                    won=decision_won,
                    component_rows=len(decision),
                )
            )
        if inside_prob is not None:
            rows.append(
                build_synthetic_row(
                    template=template,
                    run_id=run_id,
                    timestamp=timestamp,
                    market_key="inside_distance",
                    outcome_label="inside_distance",
                    implied_probability=inside_prob,
                    won=inside_won,
                    component_rows=len(ko_tko) + len(submission),
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=market_df.columns)
    for column in market_df.columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[market_df.columns]


def build_synthetic_row(
    *,
    template: pd.Series,
    run_id: str,
    timestamp: str,
    market_key: str,
    outcome_label: str,
    implied_probability: float,
    won: bool | None,
    component_rows: int,
) -> dict[str, Any]:
    american_odds = implied_probability_to_american(implied_probability)
    return {
        "historical_market_run_id": run_id,
        "historical_market_timestamp": timestamp,
        "fight_id": template.get("fight_id"),
        "date": template.get("date"),
        "event_name": template.get("event_name"),
        "market_key": market_key,
        "bookmaker": "synthetic_legacy_consensus",
        "outcome_join_key": build_outcome_join_key(
            market_key="goes_distance",
            outcome_label=outcome_label,
        ),
        "outcome_fighter_id": pd.NA,
        "outcome_label": outcome_label,
        "outcome_side": "yes" if outcome_label == "goes_distance" else "no",
        "canonical_side": outcome_label,
        "legacy_side": "synthetic",
        "american_odds": american_odds,
        "implied_probability": implied_probability,
        "profit_per_100": profit_per_100(american_odds),
        "won": won,
        "result_status": "graded" if won is not None else "ungraded",
        "source": "synthetic_from_method_props",
        "mapping_method": f"synthetic_{component_rows}_components",
        "legacy_row_number": pd.NA,
        "legacy_r_name": template.get("legacy_r_name"),
        "legacy_b_name": template.get("legacy_b_name"),
        "legacy_winner_side": template.get("legacy_winner_side"),
    }


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    market_path = Path(args.market_path)
    output_path = Path(args.output_path)
    market_df = pd.read_parquet(market_path)

    required = ["fight_id", "market_key", "implied_probability", "won"]
    missing = [column for column in required if column not in market_df.columns]
    if missing:
        raise ValueError(f"Historical market file missing required columns: {missing}")

    run_time = datetime.now(timezone.utc)
    run_id = run_time.strftime("synthetic_distance_markets_%Y%m%d_%H%M%S")
    timestamp = run_time.isoformat()

    base_df = market_df.copy()
    if args.overwrite_existing:
        base_df = base_df[~base_df["market_key"].isin(SYNTHETIC_MARKETS)].copy()

    synthetic_df = build_synthetic_rows(base_df, run_id, timestamp)
    combined = pd.concat([base_df, synthetic_df], ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    print("=" * 80)
    print("BUILD SYNTHETIC DISTANCE MARKETS")
    print("=" * 80)
    print(f"Input rows      : {len(market_df)}")
    print(f"Base rows       : {len(base_df)}")
    print(f"Synthetic rows  : {len(synthetic_df)}")
    print(f"Output rows     : {len(combined)}")
    print("Rows by market:")
    print(combined["market_key"].value_counts(dropna=False).to_string())
    print(f"Saved           : {output_path}")


if __name__ == "__main__":
    main()
