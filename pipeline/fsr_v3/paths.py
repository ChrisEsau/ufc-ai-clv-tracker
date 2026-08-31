"""FSR V3-owned output paths.

These paths deliberately do not modify pipeline.common.paths so FSR V2 and
Event Clock V1 remain untouched and runnable exactly as before.
"""

from pathlib import Path

FSR_V3_DIR = Path("data/fsr_v3")
FSR_V3_HISTORY_DIR = FSR_V3_DIR / "history"
FSR_V3_PREFIGHT_SNAPSHOTS_PATH = FSR_V3_DIR / "fsr_v3_prefight_snapshots.parquet"
FSR_V3_LATEST_PATH = FSR_V3_DIR / "fsr_v3_latest.parquet"
FSR_V3_PREFIGHT_UNCERTAINTY_PATH = FSR_V3_DIR / "fsr_v3_prefight_uncertainty.parquet"

TAKEDOWN_TENDENCY_HISTORY_PATH = FSR_V3_HISTORY_DIR / "takedown_tendency.parquet"
TAKEDOWN_SUPPRESSION_HISTORY_PATH = FSR_V3_HISTORY_DIR / "takedown_suppression.parquet"
TAKEDOWN_EFFECTIVENESS_HISTORY_PATH = FSR_V3_HISTORY_DIR / "takedown_effectiveness.parquet"

STANDING_TENDENCY_HISTORY_PATH = FSR_V3_HISTORY_DIR / "standing_striking_tendency.parquet"
STANDING_SUPPRESSION_HISTORY_PATH = FSR_V3_HISTORY_DIR / "standing_striking_suppression.parquet"
STANDING_EFFECTIVENESS_HISTORY_PATH = FSR_V3_HISTORY_DIR / "standing_striking_effectiveness.parquet"

GROUND_TENDENCY_HISTORY_PATH = FSR_V3_HISTORY_DIR / "ground_striking_tendency.parquet"
GROUND_SUPPRESSION_HISTORY_PATH = FSR_V3_HISTORY_DIR / "ground_striking_suppression.parquet"
GROUND_EFFECTIVENESS_HISTORY_PATH = FSR_V3_HISTORY_DIR / "ground_striking_effectiveness.parquet"

POWER_HISTORY_PATH = FSR_V3_HISTORY_DIR / "striking_power_v3.parquet"
ESCAPE_HISTORY_PATH = FSR_V3_HISTORY_DIR / "escape_effectiveness_v3.parquet"
KD_RESISTANCE_HISTORY_PATH = FSR_V3_HISTORY_DIR / "knockdown_resistance_v3.parquet"
