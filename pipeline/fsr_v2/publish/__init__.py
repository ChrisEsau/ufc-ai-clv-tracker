"""FSR V2 history publication helpers and module CLI."""

from pipeline.common.paths import FSR_V2_LATEST_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v2.physical import (
    PHYSICAL_COLUMNS,
    attach_physical_latest,
    attach_physical_prefight,
    build_physical_snapshots,
)
from .snapshots import assemble_latest, assemble_prefight, load_histories


def main() -> None:
    histories = load_histories()
    prefight = assemble_prefight(histories)
    latest = assemble_latest(histories)

    physical = build_physical_snapshots()
    prefight = attach_physical_prefight(prefight, physical.prefight)
    latest = attach_physical_latest(latest, physical.latest)

    FSR_V2_PREFIGHT_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    prefight.to_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH, index=False)
    latest.to_parquet(FSR_V2_LATEST_PATH, index=False)
    print(
        f"published {len(prefight):,} prefight rows and {len(latest):,} latest profiles "
        f"with {len(PHYSICAL_COLUMNS)} preserved physical/stamina fields"
    )
