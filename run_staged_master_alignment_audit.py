import pandas as pd

from pipeline.common.paths import MASTER_PATH, STAGED_FIGHT_DETAILS_PATH

STAGED_PATH = STAGED_FIGHT_DETAILS_PATH

master = pd.read_parquet(MASTER_PATH)
staged = pd.read_parquet(STAGED_PATH)

master_cols = set(master.columns)
staged_cols = set(staged.columns)

direct_matches = sorted(
    master_cols.intersection(staged_cols)
)

missing = sorted(
    master_cols - staged_cols
)

extra = sorted(
    staged_cols - master_cols
)

audit_rows = []

for col in sorted(master_cols):

    if col in staged_cols:
        status = "DIRECT_MATCH"

    elif (
        col.startswith("r_")
        or col.startswith("b_")
    ):
        status = "LIKELY_DERIVABLE_OR_PROFILE"

    else:
        status = "MISSING"

    audit_rows.append({
        "column": col,
        "status": status
    })

audit_df = pd.DataFrame(audit_rows)

print("========== ALIGNMENT AUDIT ==========")
print("Master columns:", len(master_cols))
print("Staged columns:", len(staged_cols))
print("Direct matches:", len(direct_matches))
print("Missing:", len(missing))
print("Extra:", len(extra))

audit_df.to_parquet(
    "./ufc_alignment_audit.parquet",
    index=False
)

print()
print("Saved: ./ufc_alignment_audit.parquet")