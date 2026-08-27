"""Fast runner for the Event2 KO hazard v2 grouped-strike study.

Uses exactly the same fit and metrics as run_event2_ko_hazard_v2_study, while
replacing only the fight-cluster bootstrap implementation with an algebraically
equivalent pre-aggregated version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pipeline.research.run_event2_ko_hazard_v2_study as study


def fast_bootstrap_prediction_delta(
    test: pd.DataFrame,
    base_p: np.ndarray,
    alt_p: np.ndarray,
    n_boot: int,
) -> dict:
    y = test["target_ko_round"].to_numpy(float)
    base = np.clip(np.asarray(base_p, float), 1e-15, 1 - 1e-15)
    alt = np.clip(np.asarray(alt_p, float), 1e-15, 1 - 1e-15)

    base_loss = -(y * np.log(base) + (1.0 - y) * np.log1p(-base))
    alt_loss = -(y * np.log(alt) + (1.0 - y) * np.log1p(-alt))
    base_brier = (base - y) ** 2
    alt_brier = (alt - y) ** 2

    x = pd.DataFrame({
        "fight_id": test["fight_id"].astype(str).to_numpy(),
        "dll_sum": alt_loss - base_loss,
        "dbr_sum": alt_brier - base_brier,
        "n": np.ones(len(test), dtype=float),
    })
    g = x.groupby("fight_id", sort=False, as_index=False).agg(
        dll_sum=("dll_sum", "sum"),
        dbr_sum=("dbr_sum", "sum"),
        n=("n", "sum"),
    )
    dll_f = g["dll_sum"].to_numpy(float)
    dbr_f = g["dbr_sum"].to_numpy(float)
    n_f = g["n"].to_numpy(float)

    rng = np.random.default_rng(20260826)
    nfights = len(g)
    dll, dbr = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nfights, size=nfights)
        denom = float(n_f[idx].sum())
        dll[i] = float(dll_f[idx].sum() / denom)
        dbr[i] = float(dbr_f[idx].sum() / denom)

    def summarize(a: np.ndarray) -> dict:
        return {
            "mean": float(a.mean()),
            "p2_5": float(np.quantile(a, 0.025)),
            "p97_5": float(np.quantile(a, 0.975)),
            "improvement_share": float(np.mean(a < 0)),
        }

    return {
        "delta_round_log_loss": summarize(dll),
        "delta_round_brier": summarize(dbr),
    }


if __name__ == "__main__":
    study.bootstrap_prediction_delta = fast_bootstrap_prediction_delta
    study.main()
