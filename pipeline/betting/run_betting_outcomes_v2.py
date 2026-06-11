# ============================================================
# pipeline/betting/run_betting_outcomes_v2.py
# ============================================================

"""Build generic outcome-level betting opportunities.

Betting Outcomes V2 joins prediction outcomes to market outcomes using the
canonical ID-based key:

    fight_id + market_key + outcome_join_key

This runner intentionally remains thin. Schema, math, join, and audit logic live
in dedicated modules so the betting board can evolve from a single-model runner
into a registry-driven multi-model aggregator without turning this file back into
a large orchestration script.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.betting.betting_audit import build_betting_audit
from pipeline.betting.betting_joiner import build_betting_outcomes
from pipeline.common.paths import (
    BETTING_OUTCOMES_AUDIT_PATH,
    BETTING_OUTCOMES_PATH,
    MARKET_OUTCOMES_PATH,
    PREDICTIONS_DIR,
    ensure_data_dirs,
)
from pipeline.common.risk_settings import load_risk_settings


MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"


def _utc_run() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("betting_%Y%m%d_%H%M%S"), now.isoformat()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def main() -> None:
    print("=" * 80)
    print("UFC BETTING OUTCOMES V2")
    print("=" * 80)

    ensure_data_dirs()
    betting_run_id, betting_timestamp = _utc_run()

    model_df = _load_required_parquet(MODEL_OUTCOMES_PATH, "Model outcomes")
    market_df = _load_required_parquet(MARKET_OUTCOMES_PATH, "Market outcomes")
    settings = load_risk_settings()

    print("Betting run ID:", betting_run_id)
    print("Model outcomes path:", MODEL_OUTCOMES_PATH)
    print("Market outcomes path:", MARKET_OUTCOMES_PATH)
    print("Model rows:", len(model_df))
    print("Market rows:", len(market_df))
    print("Risk settings:", asdict(settings))

    betting_df = build_betting_outcomes(
        model_df=model_df,
        market_df=market_df,
        settings=settings,
        betting_run_id=betting_run_id,
        betting_timestamp=betting_timestamp,
    )
    audit_df = build_betting_audit(
        model_df=model_df,
        market_df=market_df,
        betting_df=betting_df,
        betting_run_id=betting_run_id,
        betting_timestamp=betting_timestamp,
    )

    betting_df.to_parquet(BETTING_OUTCOMES_PATH, index=False)
    audit_df.to_parquet(BETTING_OUTCOMES_AUDIT_PATH, index=False)

    print()
    print("========== BETTING OUTCOMES V2 SUMMARY ==========")
    print("Joined rows:", len(betting_df))
    print("Bet candidates:", int(betting_df["is_bet_candidate"].fillna(False).sum()) if not betting_df.empty else 0)
    print("Validation passes:", bool(audit_df["passes_validation"].iloc[0]))
    print()
    print("Files saved:")
    print(BETTING_OUTCOMES_PATH)
    print(BETTING_OUTCOMES_AUDIT_PATH)


if __name__ == "__main__":
    main()
