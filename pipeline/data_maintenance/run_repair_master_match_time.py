from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.common.fight_time import needs_elapsed_match_time_repair, repair_elapsed_match_time
from pipeline.common.paths import AUDITS_DIR, MASTER_PATH, ensure_data_dirs, master_backup_path

AUDIT_PATH = AUDITS_DIR / "ufc_master_match_time_repair_audit.parquet"


def main() -> None:
    ensure_data_dirs()
    run_id = datetime.now(timezone.utc).strftime("match_time_repair_%Y%m%d_%H%M%S")
    backup_path = master_backup_path(run_id)

    master = pd.read_parquet(MASTER_PATH)
    repair_mask = needs_elapsed_match_time_repair(master)

    audit_cols = [
        "event_name",
        "date",
        "fight_id",
        "r_name",
        "b_name",
        "method",
        "finish_round",
        "match_time_sec",
    ]
    audit_cols = [col for col in audit_cols if col in master.columns]
    audit = master.loc[repair_mask, audit_cols].copy()
    audit.insert(0, "repair_run_id", run_id)
    audit["old_match_time_sec"] = pd.to_numeric(audit["match_time_sec"], errors="coerce") if "match_time_sec" in audit.columns else pd.NA

    repaired = repair_elapsed_match_time(master)
    if not audit.empty:
        repaired_values = repaired.loc[repair_mask, "match_time_sec"].reset_index(drop=True)
        audit["new_match_time_sec"] = repaired_values.values
        audit["match_time_delta_sec"] = audit["new_match_time_sec"] - audit["old_match_time_sec"]
    else:
        audit["new_match_time_sec"] = pd.Series(dtype="float64")
        audit["match_time_delta_sec"] = pd.Series(dtype="float64")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    master.to_parquet(backup_path, index=False)
    repaired.to_parquet(MASTER_PATH, index=False)
    audit.to_parquet(AUDIT_PATH, index=False)

    print("=" * 80)
    print("REPAIR MASTER MATCH TIME")
    print("=" * 80)
    print("Master path:", MASTER_PATH)
    print("Backup path:", backup_path)
    print("Audit path:", AUDIT_PATH)
    print("Rows:", len(master))
    print("Rows repaired:", int(repair_mask.sum()))
    if not audit.empty:
        print("Preview:")
        print(audit.head(25).to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
