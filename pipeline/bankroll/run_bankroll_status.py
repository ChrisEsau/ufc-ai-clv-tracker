from __future__ import annotations

from pipeline.common.paths import BANKROLL_SNAPSHOTS_PATH, OPEN_BETS_PATH, ensure_data_dirs
from utils.bankroll_artifacts import build_bankroll_snapshot, derive_open_bets, load_bet_ledger


def main() -> None:
    ensure_data_dirs()
    ledger = load_bet_ledger()
    open_bets = derive_open_bets(ledger)
    snapshot = build_bankroll_snapshot(ledger)

    open_bets.to_parquet(OPEN_BETS_PATH, index=False)
    snapshot.to_parquet(BANKROLL_SNAPSHOTS_PATH, index=False)

    row = snapshot.iloc[0].to_dict()
    print("========== BANKROLL STATUS ==========")
    print(f"Ledger bets: {len(ledger)}")
    print(f"Open bets: {len(open_bets)}")
    print(f"Current bankroll: {row['current_bankroll']:.2f}")
    print(f"Available bankroll: {row['available_bankroll']:.2f}")
    print(f"Open risk: {row['open_risk']:.2f}")
    print(f"ROI: {row['roi']:.4f}")
    print(f"Saved open bets: {OPEN_BETS_PATH}")
    print(f"Saved snapshot: {BANKROLL_SNAPSHOTS_PATH}")


if __name__ == "__main__":
    main()
