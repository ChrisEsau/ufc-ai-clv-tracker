"""FSR V3 — validated hierarchical fighter-state ratings.

FSR V3 is intentionally isolated from FSR V2.  The existing V2 package and
published parquet files remain the frozen comparison baseline.
"""

from .config import FSRV3Config

__all__ = ["FSRV3Config"]
