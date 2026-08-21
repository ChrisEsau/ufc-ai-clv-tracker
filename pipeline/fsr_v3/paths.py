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

GROUND_TENDENCY_HISTORY_PATH = FSR_V3_HISTORY_DIR / "ground_striking_tendency.parquet"
GROUND_SUPPRESSION_HISTORY_PATH = FSR_V3_HISTORY_DIR / "ground_striking_suppression.parquet"
GROUND_EFFECTIVENESS_HISTORY_PATH = FSR_V3_HISTORY_DIR / "ground_striking_effectiveness.parquet"
