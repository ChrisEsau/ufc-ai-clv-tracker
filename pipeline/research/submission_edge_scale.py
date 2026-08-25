from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH


def main() -> None:
    paired = build_paired_rounds()
    snapshots = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    snapshots["event_date"] = pd.to_datetime(snapshots["event_date"]).dt.normalize()
    paired["event_date"] = pd.to_datetime(paired["event_date"]).dt.normalize()
    keys = ["event_date", "fight_id", "fighter_id"]
    ff = paired.groupby(keys, as_index=False).agg(
        submission_attempts=("effective_submission_attempts", "sum"),
        submission_finish=("submission_finish", "max"),
        opponent_id=("opponent_id", "first"),
    )
    ff["log_attempts"] = np.log1p(ff["submission_attempts"])
    own = snapshots[keys + ["submission_offense", "submission_conversion_baseline"]].copy()
    frame = ff.merge(own, on=keys, how="inner", validate="one_to_one")
    opp = snapshots[["event_date", "fight_id", "fighter_id", "submission_defense"]].rename(
        columns={"fighter_id": "opponent_id", "submission_defense": "opponent_submission_defense"}
    )
    frame = frame.merge(opp, on=["event_date", "fight_id", "opponent_id"], how="left", validate="one_to_one")
    frame["offense_defense_edge"] = frame["submission_offense"] - frame["opponent_submission_defense"]
    p = np.clip(frame["submission_conversion_baseline"].astype(float), 1e-8, 1 - 1e-8)
    frame["baseline_logit"] = np.log(p / (1 - p))
    frame = frame.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)
    cutoff = frame["event_date"].quantile(0.70, interpolation="nearest")
    train = frame[(frame.event_date < cutoff) & (frame.submission_attempts > 0)].dropna(
        subset=["log_attempts", "baseline_logit", "offense_defense_edge"]
    )
    features = ["log_attempts", "baseline_logit", "offense_defense_edge"]
    x = train[features].astype(float).to_numpy()
    y = train["submission_finish"].astype(int).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    model.fit(x, y)
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    raw = lr.coef_[0] / scaler.scale_
    raw_intercept = float(lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_))
    out = {
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "n_train": int(len(train)),
        "features": features,
        "means": {k: float(v) for k, v in zip(features, scaler.mean_)},
        "scales": {k: float(v) for k, v in zip(features, scaler.scale_)},
        "standardized_coefficients": {k: float(v) for k, v in zip(features, lr.coef_[0])},
        "raw_coefficients": {k: float(v) for k, v in zip(features, raw)},
        "standardized_intercept": float(lr.intercept_[0]),
        "raw_intercept": raw_intercept,
    }
    print(json.dumps(out, indent=2))
    with open("data/diagnostics/submission_edge_scale.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
