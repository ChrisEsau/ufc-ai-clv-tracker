"""FSR V2 history publication helpers and module CLI."""

from pipeline.common.paths import FSR_V2_LATEST_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from .snapshots import assemble_latest, assemble_prefight, load_histories


def main() -> None:
    histories = load_histories()
    prefight = assemble_prefight(histories)
    latest = assemble_latest(histories)
    FSR_V2_PREFIGHT_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    prefight.to_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH, index=False)
    latest.to_parquet(FSR_V2_LATEST_PATH, index=False)
    print(f"published {len(prefight):,} prefight rows and {len(latest):,} latest profiles")
