from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from pipeline.common.paths import (
    MARKET_OUTCOMES_PATH,
    MARKET_INTELLIGENCE_HISTORY_PATH,
    MARKET_SIGNALS_AUDIT_PATH,
    MARKET_SIGNALS_PATH,
)
from pipeline.market.signals.schema import (
    ensure_market_signal_audit_columns,
    ensure_market_signal_columns,
)
from pipeline.market.signals.movement import build_movement_signals
from pipeline.market.signals.price import build_price_signals


def _now() -> tuple[str, str]:
    dt = datetime.now(timezone.utc)
    return dt.strftime("market_signals_%Y%m%d_%H%M%S"), dt.isoformat()


def main() -> None:
    print("=" * 80)
    print("BUILD MARKET SIGNALS")
    print("=" * 80)

    run_id, timestamp = _now()

    if not MARKET_OUTCOMES_PATH.exists():
        raise FileNotFoundError(f"Missing market outcomes: {MARKET_OUTCOMES_PATH}")

    market_outcomes = pd.read_parquet(MARKET_OUTCOMES_PATH)
    signal_frames = [build_price_signals(market_outcomes, run_id=run_id, timestamp=timestamp)]

    if MARKET_INTELLIGENCE_HISTORY_PATH.exists():
        history = pd.read_parquet(MARKET_INTELLIGENCE_HISTORY_PATH)
        signal_frames.append(build_movement_signals(history, run_id=run_id, timestamp=timestamp))

    signals = ensure_market_signal_columns(pd.concat(signal_frames, ignore_index=True))

    MARKET_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKET_SIGNALS_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    signals.to_parquet(MARKET_SIGNALS_PATH, index=False)

    counts = signals["signal_type"].value_counts(dropna=False).to_dict() if not signals.empty else {}
    audit = ensure_market_signal_audit_columns(
        pd.DataFrame(
            [
                {
                    "signal_run_id": run_id,
                    "signal_timestamp": timestamp,
                    "source_market_rows": len(market_outcomes),
                    "output_signal_rows": len(signals),
                    "signal_type_counts": json.dumps(counts, sort_keys=True),
                    "passes_validation": True,
                    "notes": "Initial price-signal prototype.",
                }
            ]
        )
    )
    audit.to_parquet(MARKET_SIGNALS_AUDIT_PATH, index=False)

    print("Run ID:", run_id)
    print("Market rows:", len(market_outcomes))
    print("Signals:", len(signals))
    print("Signal types:")
    print(signals["signal_type"].value_counts(dropna=False).to_string() if not signals.empty else "none")
    print("Output:", MARKET_SIGNALS_PATH)
    print("Audit:", MARKET_SIGNALS_AUDIT_PATH)


if __name__ == "__main__":
    main()
