from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import AUDITS_DIR, MARKET_DIR, MASTER_PATH, ensure_data_dirs

DEFAULT_INPUT_PATH = Path("ufc-master w odds.csv")
DEFAULT_MONEYLINE_OUTPUT_PATH = MARKET_DIR / "historical_moneyline_odds.parquet"
DEFAULT_MARKET_OUTPUT_PATH = MARKET_DIR / "historical_market_outcomes.parquet"
DEFAULT_AUDIT_PATH = AUDITS_DIR / "ufc_historical_odds_mapping_audit.parquet"

PROP_ODDS = {
    "win_by_decision": {"red": "r_dec_odds", "blue": "b_dec_odds"},
    "win_by_submission": {"red": "r_sub_odds", "blue": "b_sub_odds"},
    "win_by_ko_tko_dq": {"red": "r_ko_odds", "blue": "b_ko_odds"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generic historical market outcomes from legacy CSV.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--master-path", default=str(MASTER_PATH))
    parser.add_argument("--moneyline-output-path", default=str(DEFAULT_MONEYLINE_OUTPUT_PATH))
    parser.add_argument("--market-output-path", default=str(DEFAULT_MARKET_OUTPUT_PATH))
    parser.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    return parser.parse_args()


def normalize_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def implied_prob(odds: Any) -> float | None:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) == 0:
        return None
    value = float(value)
    return 100.0 / (value + 100.0) if value > 0 else abs(value) / (abs(value) + 100.0)


def profit_per_100(odds: Any) -> float | None:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) == 0:
        return None
    value = float(value)
    return value if value > 0 else 10000.0 / abs(value)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_legacy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    base_required = ["R_fighter", "B_fighter", "date", "R_odds", "B_odds", "Winner", "finish"]
    require_columns(df, base_required, "legacy odds CSV")
    out = pd.DataFrame()
    out["legacy_row_number"] = range(1, len(df) + 1)
    out["legacy_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    out["legacy_r_name"] = df["R_fighter"].astype(str)
    out["legacy_b_name"] = df["B_fighter"].astype(str)
    out["legacy_r_norm"] = out["legacy_r_name"].map(normalize_name)
    out["legacy_b_norm"] = out["legacy_b_name"].map(normalize_name)
    out["legacy_r_odds"] = pd.to_numeric(df["R_odds"], errors="coerce")
    out["legacy_b_odds"] = pd.to_numeric(df["B_odds"], errors="coerce")
    out["legacy_winner_side"] = df["Winner"].astype(str).str.strip().str.lower()
    out["legacy_finish"] = df["finish"].astype(str).str.strip().str.lower()
    for market_cols in PROP_ODDS.values():
        for column in market_cols.values():
            out[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.NA
    return out


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = ["fight_id", "date", "r_name", "b_name", "r_id", "b_id", "winner_id"]
    require_columns(df, required, "master parquet")
    optional = [column for column in ["event_name", "location"] if column in df.columns]
    out = df[required + optional].copy()
    out["master_date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["master_r_norm"] = out["r_name"].map(normalize_name)
    out["master_b_norm"] = out["b_name"].map(normalize_name)
    return out


def map_rows(legacy: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    direct = legacy.merge(
        master,
        left_on=["legacy_date", "legacy_r_norm", "legacy_b_norm"],
        right_on=["master_date", "master_r_norm", "master_b_norm"],
        how="left",
    )
    direct["mapping_method"] = direct["fight_id"].notna().map(lambda ok: "direct" if ok else "unmatched")
    matched = direct[direct["fight_id"].notna()].copy()
    unmatched = direct.loc[direct["fight_id"].isna(), legacy.columns].copy()
    reversed_match = unmatched.merge(
        master,
        left_on=["legacy_date", "legacy_r_norm", "legacy_b_norm"],
        right_on=["master_date", "master_b_norm", "master_r_norm"],
        how="left",
    )
    reversed_match["mapping_method"] = reversed_match["fight_id"].notna().map(lambda ok: "reversed" if ok else "unmatched")
    return pd.concat([matched, reversed_match], ignore_index=True, sort=False)


def winner_is_side(row: pd.Series, legacy_side: str) -> bool | None:
    winner = str(row.get("legacy_winner_side", "")).strip().lower()
    if winner in {"red", "r"}:
        return legacy_side == "red"
    if winner in {"blue", "b"}:
        return legacy_side == "blue"
    return None


def method_won(row: pd.Series, legacy_side: str, market_key: str) -> bool | None:
    side_win = winner_is_side(row, legacy_side)
    if side_win is None:
        return None
    if not side_win:
        return False
    finish = str(row.get("legacy_finish", "")).lower()
    if market_key == "win_by_decision":
        return "dec" in finish
    if market_key == "win_by_submission":
        return "sub" in finish
    if market_key == "win_by_ko_tko_dq":
        return any(token in finish for token in ["ko", "tko", "dq"])
    return None


def side_specs(row: pd.Series) -> list[tuple[str, Any, Any, str]]:
    reversed_order = row.get("mapping_method") == "reversed"
    return [
        ("red", row.get("r_id"), row.get("r_name"), "blue" if reversed_order else "red"),
        ("blue", row.get("b_id"), row.get("b_name"), "red" if reversed_order else "blue"),
    ]


def add_row(rows: list[dict[str, Any]], row: pd.Series, run_id: str, ts: str, market_key: str, side: str, fighter_id: Any, label: Any, legacy_side: str, odds: Any, won: bool | None) -> None:
    rows.append({
        "historical_market_run_id": run_id,
        "historical_market_timestamp": ts,
        "fight_id": row.get("fight_id"),
        "date": row.get("master_date"),
        "event_name": row.get("event_name"),
        "market_key": market_key,
        "bookmaker": "legacy_consensus",
        "outcome_join_key": str(fighter_id),
        "outcome_fighter_id": fighter_id,
        "outcome_label": label,
        "outcome_side": side,
        "canonical_side": side,
        "legacy_side": legacy_side,
        "american_odds": odds,
        "implied_probability": implied_prob(odds),
        "profit_per_100": profit_per_100(odds),
        "won": won,
        "result_status": "graded" if won is not None else "ungraded",
        "source": "ufc-master w odds.csv",
        "mapping_method": row.get("mapping_method"),
        "legacy_row_number": row.get("legacy_row_number"),
        "legacy_r_name": row.get("legacy_r_name"),
        "legacy_b_name": row.get("legacy_b_name"),
        "legacy_winner_side": row.get("legacy_winner_side"),
    })


def build_market_outcomes(mapped: pd.DataFrame, run_id: str, ts: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in mapped[mapped["fight_id"].notna()].iterrows():
        for side, fighter_id, label, legacy_side in side_specs(row):
            moneyline_col = "legacy_r_odds" if legacy_side == "red" else "legacy_b_odds"
            add_row(rows, row, run_id, ts, "moneyline", side, fighter_id, label, legacy_side, row.get(moneyline_col), winner_is_side(row, legacy_side))
            for market_key, cols in PROP_ODDS.items():
                odds_col = cols[legacy_side]
                add_row(rows, row, run_id, ts, market_key, side, fighter_id, label, legacy_side, row.get(odds_col), method_won(row, legacy_side, market_key))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[out["american_odds"].notna()].copy()
    return out


def build_audit(mapped: pd.DataFrame, run_id: str, ts: str) -> pd.DataFrame:
    audit = mapped.copy()
    audit.insert(0, "mapping_run_id", run_id)
    audit.insert(1, "mapping_timestamp", ts)
    audit["mapped"] = audit["fight_id"].notna()
    audit["has_valid_moneyline_odds"] = audit["legacy_r_odds"].notna() & audit["legacy_b_odds"].notna()
    audit["winner_side_valid"] = audit["legacy_winner_side"].isin(["red", "blue", "r", "b"])
    keep = ["mapping_run_id", "mapping_timestamp", "legacy_row_number", "legacy_date", "legacy_r_name", "legacy_b_name", "legacy_r_odds", "legacy_b_odds", "legacy_winner_side", "legacy_finish", "fight_id", "event_name", "date", "r_name", "b_name", "r_id", "b_id", "winner_id", "mapping_method", "mapped", "has_valid_moneyline_odds", "winner_side_valid"]
    for column in keep:
        if column not in audit.columns:
            audit[column] = pd.NA
    return audit[keep]


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    run_time = datetime.now(timezone.utc)
    run_id = run_time.strftime("historical_market_outcomes_%Y%m%d_%H%M%S")
    ts = run_time.isoformat()
    legacy = load_legacy(Path(args.input_path))
    master = load_master(Path(args.master_path))
    mapped = map_rows(legacy, master)
    audit = build_audit(mapped, run_id, ts)
    market = build_market_outcomes(mapped, run_id, ts)
    moneyline = market[market["market_key"] == "moneyline"].copy()
    Path(args.moneyline_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.market_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_path).parent.mkdir(parents=True, exist_ok=True)
    moneyline.to_parquet(args.moneyline_output_path, index=False)
    market.to_parquet(args.market_output_path, index=False)
    audit.to_parquet(args.audit_path, index=False)
    print("=" * 80)
    print("BUILD HISTORICAL MARKET OUTCOMES")
    print("=" * 80)
    print("Legacy rows:", len(legacy))
    print("Master rows:", len(master))
    print("Mapped rows:", int(audit["mapped"].sum()))
    print("Unmapped rows:", int((~audit["mapped"]).sum()))
    print("Moneyline rows:", len(moneyline))
    print("All market outcome rows:", len(market))
    print("Rows by market:")
    print(market["market_key"].value_counts(dropna=False).to_string())
    print("Mapping methods:")
    print(audit["mapping_method"].value_counts(dropna=False).to_string())
    print("Saved moneyline odds:", args.moneyline_output_path)
    print("Saved historical market outcomes:", args.market_output_path)
    print("Saved mapping audit:", args.audit_path)


if __name__ == "__main__":
    main()
