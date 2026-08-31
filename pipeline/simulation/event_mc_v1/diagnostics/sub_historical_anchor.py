"""Descriptive historical submission anchor from authoritative master data."""

import json

import pandas as pd

from pipeline.common.paths import MASTER_PATH


def historical_submission_anchor(path=MASTER_PATH):
    fights = pd.read_parquet(path).drop_duplicates("fight_id")
    fights = fights[fights["winner"].notna()]
    method = fights["method"].astype(str).str.upper()
    submission = method.str.contains("SUBMISSION", regex=False)
    red_attempts = pd.to_numeric(fights["r_sub_att"], errors="coerce").fillna(0)
    blue_attempts = pd.to_numeric(fights["b_sub_att"], errors="coerce").fillna(0)
    attempts = red_attempts + blue_attempts
    attempted = attempts > 0
    return {
        "fights": len(fights),
        "submission_finish_rate": float(submission.mean()),
        "submission_attempts_per_fight": float(attempts.mean()),
        "share_with_attempt": float(attempted.mean()),
        "submission_finish_given_recorded_attempt": float(submission[attempted].mean()) if attempted.any() else None,
    }


if __name__ == "__main__":
    print(json.dumps(historical_submission_anchor(), indent=2, sort_keys=True))
