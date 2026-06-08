from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    AUDITS_DIR,
    CURRENT_FIGHTER_FEATURES_PATH,
    LIVE_CARD_PATH,
)


FIGHTER_STATE_COVERAGE_AUDIT_PATH = AUDITS_DIR / "ufc_live_fighter_state_coverage_audit.parquet"
FIGHTER_STATE_MISSING_PATH = AUDITS_DIR / "ufc_live_fighter_state_missing.parquet"
FIGHTER_STATE_FIGHT_AUDIT_PATH = AUDITS_DIR / "ufc_live_fighter_state_fight_audit.parquet"

FIGHTER_ID_CANDIDATES = ["fighter_id", "id", "ufcstats_fighter_id"]


class FighterStateAuditError(RuntimeError):
    """Raised when fighter state coverage cannot be audited."""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit live-card fighter coverage against current fighter features.",
    )
    parser.add_argument(
        "--live-card-path",
        default=str(LIVE_CARD_PATH),
        help="Path to ufc_live_card.parquet.",
    )
    parser.add_argument(
        "--current-fighter-features-path",
        default=str(CURRENT_FIGHTER_FEATURES_PATH),
        help="Path to ufc_current_fighter_features.parquet.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of detail rows to print.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    live_card_path = Path(args.live_card_path)
    current_features_path = Path(args.current_fighter_features_path)

    live_card = _read_required_parquet(live_card_path, "live card")
    current_features = _read_required_parquet(current_features_path, "current fighter features")

    fighter_id_column = _find_first_existing_column(current_features, FIGHTER_ID_CANDIDATES)
    if fighter_id_column is None:
        raise FighterStateAuditError(
            "Current fighter features must include one fighter ID column. "
            f"Checked: {FIGHTER_ID_CANDIDATES}"
        )

    fighter_rows = _build_live_fighter_rows(live_card)
    coverage = _build_fighter_coverage(
        fighter_rows=fighter_rows,
        current_features=current_features,
        fighter_id_column=fighter_id_column,
    )
    fight_audit = _build_fight_audit(live_card=live_card, coverage=coverage)

    _write_outputs(coverage=coverage, fight_audit=fight_audit)
    _print_report(
        live_card=live_card,
        current_features=current_features,
        fighter_rows=fighter_rows,
        coverage=coverage,
        fight_audit=fight_audit,
        fighter_id_column=fighter_id_column,
        top_n=args.top_n,
    )



def _read_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FighterStateAuditError(f"Missing {label}: {path}")
    return pd.read_parquet(path)



def _build_live_fighter_rows(live_card: pd.DataFrame) -> pd.DataFrame:
    required = [
        "event_id",
        "event_name",
        "fight_id",
        "red_fighter",
        "blue_fighter",
        "red_fighter_id",
        "blue_fighter_id",
    ]
    missing = [column for column in required if column not in live_card.columns]
    if missing:
        raise FighterStateAuditError(f"Live card missing required columns: {missing}")

    red = live_card[[
        "event_id",
        "event_name",
        "fight_id",
        "red_fighter",
        "red_fighter_id",
    ]].copy()
    red = red.rename(columns={
        "red_fighter": "fighter_name",
        "red_fighter_id": "fighter_id",
    })
    red["side"] = "red"

    blue = live_card[[
        "event_id",
        "event_name",
        "fight_id",
        "blue_fighter",
        "blue_fighter_id",
    ]].copy()
    blue = blue.rename(columns={
        "blue_fighter": "fighter_name",
        "blue_fighter_id": "fighter_id",
    })
    blue["side"] = "blue"

    fighters = pd.concat([red, blue], ignore_index=True)
    fighters["fighter_id"] = _normalize_id_series(fighters["fighter_id"])
    fighters["fighter_name"] = fighters["fighter_name"].astype("string").fillna("").str.strip()

    return fighters



def _build_fighter_coverage(
    *,
    fighter_rows: pd.DataFrame,
    current_features: pd.DataFrame,
    fighter_id_column: str,
) -> pd.DataFrame:
    features = current_features.copy()
    features[fighter_id_column] = _normalize_id_series(features[fighter_id_column])

    feature_columns = [column for column in features.columns if column != fighter_id_column]
    numeric_feature_columns = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]

    if numeric_feature_columns:
        numeric_matrix = features[numeric_feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        features["state_numeric_feature_count"] = len(numeric_feature_columns)
        features["state_nonzero_numeric_feature_count"] = (numeric_matrix != 0).sum(axis=1)
        features["state_zero_numeric_feature_pct"] = 1.0 - (
            features["state_nonzero_numeric_feature_count"] / len(numeric_feature_columns)
        )
    else:
        features["state_numeric_feature_count"] = 0
        features["state_nonzero_numeric_feature_count"] = 0
        features["state_zero_numeric_feature_pct"] = pd.NA

    feature_summary_columns = [
        fighter_id_column,
        "state_numeric_feature_count",
        "state_nonzero_numeric_feature_count",
        "state_zero_numeric_feature_pct",
    ]
    feature_summary_columns += [
        column for column in ["fighter_name", "name", "r_name", "b_name"]
        if column in features.columns and column not in feature_summary_columns
    ]

    feature_summary = features[feature_summary_columns].drop_duplicates(subset=[fighter_id_column]).copy()

    coverage = fighter_rows.merge(
        feature_summary,
        left_on="fighter_id",
        right_on=fighter_id_column,
        how="left",
    )
    coverage["has_fighter_state"] = coverage[fighter_id_column].notna()
    coverage["missing_reason"] = coverage["has_fighter_state"].map({True: "matched", False: "missing_from_current_fighter_features"})

    return coverage



def _build_fight_audit(*, live_card: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    red = coverage[coverage["side"] == "red"][[
        "fight_id",
        "fighter_id",
        "fighter_name",
        "has_fighter_state",
        "state_nonzero_numeric_feature_count",
        "state_zero_numeric_feature_pct",
    ]].rename(columns={
        "fighter_id": "red_fighter_id",
        "fighter_name": "red_fighter",
        "has_fighter_state": "red_has_fighter_state",
        "state_nonzero_numeric_feature_count": "red_state_nonzero_numeric_feature_count",
        "state_zero_numeric_feature_pct": "red_state_zero_numeric_feature_pct",
    })

    blue = coverage[coverage["side"] == "blue"][[
        "fight_id",
        "fighter_id",
        "fighter_name",
        "has_fighter_state",
        "state_nonzero_numeric_feature_count",
        "state_zero_numeric_feature_pct",
    ]].rename(columns={
        "fighter_id": "blue_fighter_id",
        "fighter_name": "blue_fighter",
        "has_fighter_state": "blue_has_fighter_state",
        "state_nonzero_numeric_feature_count": "blue_state_nonzero_numeric_feature_count",
        "state_zero_numeric_feature_pct": "blue_state_zero_numeric_feature_pct",
    })

    fight_base_columns = [
        column for column in ["event_id", "event_name", "event_date", "fight_id", "weight_class"]
        if column in live_card.columns
    ]
    fight_audit = live_card[fight_base_columns].drop_duplicates(subset=["fight_id"]).copy()
    fight_audit = fight_audit.merge(red, on="fight_id", how="left")
    fight_audit = fight_audit.merge(blue, on="fight_id", how="left")
    fight_audit["both_fighters_matched"] = fight_audit["red_has_fighter_state"].fillna(False) & fight_audit["blue_has_fighter_state"].fillna(False)
    fight_audit["partial_match"] = fight_audit["red_has_fighter_state"].fillna(False) ^ fight_audit["blue_has_fighter_state"].fillna(False)
    fight_audit["neither_matched"] = (~fight_audit["red_has_fighter_state"].fillna(False)) & (~fight_audit["blue_has_fighter_state"].fillna(False))

    return fight_audit



def _write_outputs(*, coverage: pd.DataFrame, fight_audit: pd.DataFrame) -> None:
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)

    coverage.to_parquet(FIGHTER_STATE_COVERAGE_AUDIT_PATH, index=False)
    coverage[~coverage["has_fighter_state"]].to_parquet(FIGHTER_STATE_MISSING_PATH, index=False)
    fight_audit.to_parquet(FIGHTER_STATE_FIGHT_AUDIT_PATH, index=False)



def _print_report(
    *,
    live_card: pd.DataFrame,
    current_features: pd.DataFrame,
    fighter_rows: pd.DataFrame,
    coverage: pd.DataFrame,
    fight_audit: pd.DataFrame,
    fighter_id_column: str,
    top_n: int,
) -> None:
    print("=" * 80)
    print("LIVE FIGHTER STATE COVERAGE AUDIT")
    print("=" * 80)
    print(f"Live fights: {live_card['fight_id'].nunique() if 'fight_id' in live_card.columns else 'missing fight_id'}")
    print(f"Live fighter slots: {len(fighter_rows)}")
    print(f"Unique live fighters: {fighter_rows['fighter_id'].nunique(dropna=True)}")
    print(f"Current fighter feature rows: {len(current_features)}")
    print(f"Current fighter ID column: {fighter_id_column}")

    print("\nFighter slot coverage:")
    print(coverage["has_fighter_state"].value_counts(dropna=False).to_string())

    missing = coverage[~coverage["has_fighter_state"]].copy()
    print(f"\nMissing fighter-state slots: {len(missing)}")
    print(f"Unique missing fighters: {missing['fighter_id'].nunique(dropna=True)}")

    if len(missing) > 0:
        display_columns = [
            "event_name",
            "fight_id",
            "side",
            "fighter_name",
            "fighter_id",
            "missing_reason",
        ]
        print(f"\nMissing fighters first {top_n} rows:")
        print(missing[display_columns].head(top_n).to_string(index=False))

    print("\nFight-level coverage:")
    for column in ["both_fighters_matched", "partial_match", "neither_matched"]:
        print(f"\n{column} counts:")
        print(fight_audit[column].value_counts(dropna=False).to_string())

    problem_fights = fight_audit[~fight_audit["both_fighters_matched"]].copy()
    print(f"\nProblem fights: {len(problem_fights)}")
    if len(problem_fights) > 0:
        display_columns = [
            column for column in [
                "event_name",
                "fight_id",
                "red_fighter",
                "blue_fighter",
                "red_fighter_id",
                "blue_fighter_id",
                "red_has_fighter_state",
                "blue_has_fighter_state",
                "both_fighters_matched",
                "partial_match",
                "neither_matched",
            ]
            if column in problem_fights.columns
        ]
        print(f"\nProblem fights first {top_n} rows:")
        print(problem_fights[display_columns].head(top_n).to_string(index=False))

    matched = coverage[coverage["has_fighter_state"]].copy()
    if len(matched) > 0 and "state_zero_numeric_feature_pct" in matched.columns:
        matched["state_zero_numeric_feature_pct"] = pd.to_numeric(matched["state_zero_numeric_feature_pct"], errors="coerce")
        print("\nMatched fighter numeric feature zero pct summary:")
        print(matched["state_zero_numeric_feature_pct"].describe().to_string())

        display_columns = [
            "event_name",
            "fight_id",
            "side",
            "fighter_name",
            "fighter_id",
            "state_nonzero_numeric_feature_count",
            "state_zero_numeric_feature_pct",
        ]
        print(f"\nWorst matched fighter-state coverage first {top_n} rows:")
        print(
            matched.sort_values("state_zero_numeric_feature_pct", ascending=False)[display_columns]
            .head(top_n)
            .to_string(index=False)
        )

    print("\nAudit outputs written:")
    print(f"  {FIGHTER_STATE_COVERAGE_AUDIT_PATH}")
    print(f"  {FIGHTER_STATE_MISSING_PATH}")
    print(f"  {FIGHTER_STATE_FIGHT_AUDIT_PATH}")



def _normalize_id_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("").str.strip()
    values = values.mask(values.str.lower().isin({"", "nan", "none", "null", "nat", "<na>"}), pd.NA)
    return values



def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None



if __name__ == "__main__":
    main()
