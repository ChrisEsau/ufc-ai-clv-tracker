"""Run submission-conversion trait audit with V2 comparators rebuilt from raw data."""
from __future__ import annotations

from pathlib import Path

from pipeline.fsr_v3.research import active_trait_audit_sources as sources
from pipeline.fsr_v3.research import submission_conversion_trait_audit as audit


def main():
    path = Path("/tmp/fsr_v3_active_audit_submission_conversion_legacy.parquet")
    sources.legacy_submission_conversion_prefight().to_parquet(path, index=False)
    audit.FSR_V3_PREFIGHT_SNAPSHOTS_PATH = path
    audit.main()


if __name__ == "__main__":
    main()
