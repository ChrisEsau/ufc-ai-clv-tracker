from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from pipeline.common.paths import AUDITS_DIR, MASTER_PATH, ensure_data_dirs
from utils.bankroll_artifacts import american_profit, is_open_result, load_bet_ledger, normalize_ledger, save_bet_ledger

BET_SETTLEMENT_AUDIT_PATH = AUDITS_DIR / "ufc_bet_settlement_audit.parquet"

AUDIT_COLUMNS = [
    "run_id",
    "run_timestamp",
    "bet_id",
    "fight_id",
    "event_id",
    "event_name",
    "bet_side",
    "bet_side_id",
    "matched_winner",
    "matched_winner_id",
    "match_status",
    "settlement_status",
    "bet_result",
    "stake",
    "odds_taken",
    "profit_loss",
    "dry_run",
    "error",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master dataset: {MASTER_PATH}")
    master = pd.read_parquet(MASTER_PATH)
    if "fight_id" not in master.columns:
        raise ValueError("Master dataset does not contain fight_id column.")
    return master


def _winner_from_master(row: pd.Series) -> tuple[str, str]:
    winner = _safe_text(row.get("winner"))
    winner_id = _safe_text(row.get("winner_id"))
    return winner, winner_id


def _side_matches_winner(bet: pd.Series, master_row: pd.Series) -> bool | None:
    fighter_id = _safe_text(bet.get("fighter_id"))
    fighter = _safe_text(bet.get("fighter"))
    winner, winner_id = _winner_from_master(master_row)

    if fighter_id and winner_id:
        return fighter_id == winner_id
    if fighter and winner:
        return fighter.casefold() == winner.casefold()
    return None


def _settle_one_open_bet(bet: pd.Series, master: pd.DataFrame, dry_run: bool) -> dict[str, Any]:
    fight_id = _safe_text(bet.get("fight_id"))
    bet_id = _safe_text(bet.get("bet_id"))
    stake = _safe_float(bet.get("stake"))
    odds_taken = _safe_float(bet.get("odds_taken"))

    audit = {
        "bet_id": bet_id,
        "fight_id": fight_id,
        "event_id": _safe_text(bet.get("event_id")),
        "event_name": _safe_text(bet.get("event_name")),
        "bet_side": _safe_text(bet.get("fighter")),
        "bet_side_id": _safe_text(bet.get("fighter_id")),
        "matched_winner": "",
        "matched_winner_id": "",
        "match_status": "not_started",
        "settlement_status": "not_started",
        "bet_result": "",
        "stake": stake,
        "odds_taken": odds_taken,
        "profit_loss": 0.0,
        "dry_run": dry_run,
        "error": "",
    }

    if not fight_id:
        audit.update(match_status="skipped_missing_fight_id", settlement_status="skipped_missing_fight_id")
        return audit
    if stake is None or stake < 0 or odds_taken is None or odds_taken == 0:
        audit.update(match_status="skipped_invalid_odds_or_stake", settlement_status="skipped_invalid_odds_or_stake")
        return audit

    matches = master[master["fight_id"].astype(str).str.strip() == fight_id]
    if matches.empty:
        audit.update(match_status="skipped_no_master_match", settlement_status="skipped_no_master_match")
        return audit
    if len(matches) > 1:
        audit.update(match_status="skipped_multiple_master_matches", settlement_status="skipped_multiple_master_matches")
        return audit

    master_row = matches.iloc[0]
    winner, winner_id = _winner_from_master(master_row)
    audit["matched_winner"] = winner
    audit["matched_winner_id"] = winner_id
    audit["match_status"] = "matched_by_fight_id"

    side_won = _side_matches_winner(bet, master_row)
    if side_won is None:
        audit.update(settlement_status="skipped_missing_bet_side", error="Could not compare bet side to winner by fighter_id or fighter name.")
        return audit

    result = "Win" if side_won else "Loss"
    profit_loss = american_profit(stake, odds_taken, result)
    audit.update(
        settlement_status="settled_win" if side_won else "settled_loss",
        bet_result=result,
        profit_loss=profit_loss,
    )
    return audit


def _write_audit(rows: list[dict[str, Any]]) -> pd.DataFrame:
    audit = pd.DataFrame(rows)
    for column in AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = pd.NA
    audit = audit[AUDIT_COLUMNS]
    BET_SETTLEMENT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(BET_SETTLEMENT_AUDIT_PATH, index=False)
    return audit


def run_settle_open_bets(*, dry_run: bool = False) -> pd.DataFrame:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_timestamp = _now_iso()

    print("=" * 80)
    print("UFC SETTLE OPEN BETS")
    print("=" * 80)
    print("Run ID:", run_id)
    print("Dry run:", dry_run)

    ensure_data_dirs()
    ledger = normalize_ledger(load_bet_ledger())
    master = _load_master()

    if ledger.empty:
        audit = _write_audit([
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "match_status": "skipped_empty_ledger",
                "settlement_status": "skipped_empty_ledger",
                "dry_run": dry_run,
            }
        ])
        print("Ledger is empty. Saved audit:", BET_SETTLEMENT_AUDIT_PATH)
        return audit

    open_mask = ledger["result"].apply(is_open_result)
    open_bets = ledger[open_mask].copy()
    print("Ledger bets:", len(ledger))
    print("Open bets:", len(open_bets))

    audit_rows: list[dict[str, Any]] = []
    settled_updates: dict[int, dict[str, Any]] = {}

    for idx, bet in open_bets.iterrows():
        audit = _settle_one_open_bet(bet, master, dry_run=dry_run)
        audit["run_id"] = run_id
        audit["run_timestamp"] = run_timestamp
        audit_rows.append(audit)

        if audit["settlement_status"] in {"settled_win", "settled_loss"}:
            settled_updates[idx] = {
                "result": audit["bet_result"],
                "profit_loss": audit["profit_loss"],
                "settled_timestamp": run_timestamp,
                "notes": "Auto-settled by Monday Reset using fight_id match.",
            }

    if not audit_rows:
        audit_rows.append(
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "match_status": "skipped_no_open_bets",
                "settlement_status": "skipped_no_open_bets",
                "dry_run": dry_run,
            }
        )

    audit = _write_audit(audit_rows)

    if settled_updates and not dry_run:
        for idx, updates in settled_updates.items():
            for column, value in updates.items():
                ledger.loc[idx, column] = value
        save_bet_ledger(normalize_ledger(ledger))

    settled_count = int(audit["settlement_status"].isin(["settled_win", "settled_loss", "settled_push"]).sum())
    skipped_count = int(len(audit) - settled_count)

    print("Settled bets:", settled_count)
    print("Skipped bets:", skipped_count)
    print("Saved audit:", BET_SETTLEMENT_AUDIT_PATH)
    if dry_run:
        print("Dry run only. Ledger was not modified.")

    print("========== SETTLEMENT SUMMARY ==========")
    print(
        audit[["bet_id", "fight_id", "bet_side", "matched_winner", "settlement_status", "bet_result", "profit_loss", "error"]]
        .to_string(index=False)
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically settle open moneyline bets from completed master results.")
    parser.add_argument("--dry-run", action="store_true", help="Write settlement audit but do not modify the ledger.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_settle_open_bets(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
