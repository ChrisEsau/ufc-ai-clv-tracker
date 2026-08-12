"""Validate incremental held-out value of age for the striking-power event model.

Compares an identical chronological holdout logistic model with and without age.
No FSR or simulator artifacts are modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROWS_PATH = Path("data/experimental/striking_power_age_effect/striking_power_age_effect_rows.csv")
OUT_PATH = Path("data/experimental/fsr_age_modifier_calibration/striking_power_age_incremental_value.csv")
AGE_CENTER = 30.0
FIT_AGE_MIN = 21.0
FIT_AGE_MAX = 39.0


def _split(frame: pd.DataFrame):
    dates = np.sort(frame["event_date"].dropna().unique())
    idx = max(1, int(len(dates) * 0.80)) - 1
    split = pd.Timestamp(dates[idx])
    train = frame.loc[frame["event_date"] <= split].copy()
    test = frame.loc[frame["event_date"] > split].copy()
    return train, test, split


def _design(frame: pd.DataFrame, degree: int) -> np.ndarray:
    cols = [
        frame["striking_power"].to_numpy(float),
        frame["sig_str_landed"].to_numpy(float),
        np.log1p(frame["prior_ufc_fights"].to_numpy(float)),
    ]
    x = frame["age"].to_numpy(float) - AGE_CENTER
    for p in range(1, degree + 1):
        cols.append(x ** p)
    return np.column_stack(cols)


def main() -> None:
    if not ROWS_PATH.exists():
        raise FileNotFoundError(ROWS_PATH)
    frame = pd.read_csv(ROWS_PATH)
    for c in ("age", "striking_power", "sig_str_landed", "prior_ufc_fights", "power_event_int"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame = frame.dropna(subset=[
        "age", "striking_power", "sig_str_landed", "prior_ufc_fights", "power_event_int", "event_date"
    ])
    frame = frame.loc[frame["age"].between(FIT_AGE_MIN, FIT_AGE_MAX)].copy()

    train, test, split = _split(frame)
    y_train = train["power_event_int"].to_numpy(int)
    y_test = test["power_event_int"].to_numpy(int)

    rows = []
    for degree in (0, 1, 2, 3):
        model = LogisticRegression(C=1000.0, max_iter=10000, solver="lbfgs")
        model.fit(_design(train, degree), y_train)
        p = model.predict_proba(_design(test, degree))[:, 1]
        rows.append({
            "model": "no_age" if degree == 0 else f"age_degree_{degree}",
            "age_degree": degree,
            "split_date": split,
            "train_rows": len(train),
            "test_rows": len(test),
            "logloss": float(log_loss(y_test, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y_test, p)),
            "auc": float(roc_auc_score(y_test, p)),
        })

    out = pd.DataFrame(rows)
    base_loss = float(out.loc[out["age_degree"].eq(0), "logloss"].iloc[0])
    out["logloss_improvement_vs_no_age"] = base_loss - out["logloss"]
    out["relative_logloss_improvement_pct"] = 100.0 * out["logloss_improvement_vs_no_age"] / base_loss
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("\nSTRIKING POWER AGE INCREMENTAL VALUE")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nwrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
